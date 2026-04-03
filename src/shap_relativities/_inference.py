"""
SHAPInference: asymptotically valid confidence intervals for global SHAP feature importance.

Implements the de-biased U-statistic estimator from:
  Whitehouse, Sawarni, Syrgkanis (2026). "Statistical Inference and Learning
  for SHAP." arXiv:2602.10532.

The key insight: theta_p = E[|phi_a(X)|^p] cannot be estimated by a naive
sample mean — the SHAP values phi_a are themselves estimated from the data,
introducing a bias that does not vanish at the sqrt(n) rate. The paper's
solution is a Neyman-orthogonal (de-biased) estimator using cross-fitting.

Usage
-----
>>> import numpy as np
>>> si = SHAPInference(shap_values, y, feature_names=["age", "ncd", "area"])
>>> si.fit()
>>> si.importance_table()

IMPORTANT: Valid inference requires interventional SHAP values. Path-dependent
TreeSHAP (the default in shap's TreeExplainer) computes a different functional
and CIs from this class will not have valid coverage with it.

Theory notes
------------
For p >= 2, the influence function is:

  rho_a(Z_i) = |phi_a(X_i)|^p + alpha_hat(X_i) * (Y_i - mu_hat(X_i))

where mu_hat = E[Y|X] and alpha_hat = E[gamma_p(X) | X] (the sensitivity
correction). The U-statistic decomposes to the O(n) sample mean of rho_a.

For 1 <= p < 2, |phi|^p is not twice differentiable at zero, so we substitute
the smoothed approximation phi_{p,beta}(x) = |x|^p * tanh(beta * |x|^{2-p}).
As beta -> inf (at rate calibrated by n), the smoothed version recovers the
original estimand.

This is NOT a bootstrap. It is a first-principles analytical variance estimate.
Bootstrap on SHAP values would be invalid because it ignores the uncertainty
from estimating mu, which is the dominant source of bias for small-p cases.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

try:
    from sklearn.base import clone
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Smoothing utilities for p < 2
# ---------------------------------------------------------------------------

def _smoothed_phi(phi: np.ndarray, p: float, beta: float) -> np.ndarray:
    """
    Smoothed version of |phi|^p for use when p < 2.

    phi_{p,beta}(x) = |x|^p * tanh(beta * |x|^{2-p})

    As beta -> inf this recovers |x|^p. The smoothing removes the
    non-differentiability at x=0 that breaks the Neyman-orthogonality
    argument for 1 <= p < 2.
    """
    abs_phi = np.abs(phi)
    # Clip to avoid overflow in tanh for very large abs_phi * beta
    arg = np.clip(beta * abs_phi ** (2 - p), -500, 500)
    return abs_phi ** p * np.tanh(arg)


def _smoothed_gamma_deriv(phi: np.ndarray, p: float, beta: float) -> np.ndarray:
    """
    Derivative d/dphi of smoothed_phi(phi, p, beta).

    d/dx [|x|^p * tanh(beta * |x|^{2-p})]
      = sign(x) * [p|x|^{p-1} * tanh(beta*|x|^{2-p})
                   + beta*(2-p)*|x| * sech^2(beta*|x|^{2-p})]

    This is the gamma function (sensitivity of importance to SHAP values)
    used in the de-biased score for p < 2.
    """
    abs_phi = np.abs(phi)
    sign_phi = np.sign(phi)
    arg = np.clip(beta * abs_phi ** (2 - p), -500, 500)
    tanh_term = np.tanh(arg)
    # sech^2 = 1 - tanh^2, more numerically stable than 1/cosh^2
    sech2_term = 1.0 - tanh_term ** 2
    return sign_phi * (
        p * abs_phi ** (p - 1) * tanh_term
        + beta * (2 - p) * abs_phi * sech2_term
    )


def _default_beta_n(n: int, p: float, delta: float = 1.0) -> float:
    """
    Default smoothing parameter for p < 2.

    From Theorem 3.7: beta_n must grow faster than n^{(2-p)/(2*(p+delta))}.
    We use this rate directly with a small constant. delta=1 assumes the
    SHAP density near zero is Lipschitz — conservative but reasonable for
    insurance pricing features.
    """
    return float(n ** ((2 - p) / (2 * (p + delta))))


def _unsmoothed_gamma(phi: np.ndarray, p: float) -> np.ndarray:
    """
    Gamma function for p >= 2: derivative of |phi|^p with respect to phi.

    gamma_p(phi) = p * sign(phi) * |phi|^{p-1}
    """
    return p * np.sign(phi) * np.abs(phi) ** (p - 1)


# ---------------------------------------------------------------------------
# Default nuisance estimator
# ---------------------------------------------------------------------------

def _make_nuisance_estimator(spec: str | Any) -> Any:
    """
    Build a nuisance estimator from a string shorthand or return the object as-is.

    'gradient_boosting' → HistGradientBoostingRegressor with sensible defaults.
    HistGBT is the right choice here: it handles missing values natively,
    achieves n^{-1/4} rate on smooth functions, and is fast enough for n=100k.
    """
    if not _SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for SHAPInference. "
            "Install with: pip install shap-relativities[ml]"
        )
    if isinstance(spec, str):
        if spec == "gradient_boosting":
            return HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                random_state=0,
            )
        raise ValueError(
            f"Unknown nuisance_estimator string '{spec}'. "
            "Use 'gradient_boosting' or pass an sklearn-compatible estimator."
        )
    return spec


# ---------------------------------------------------------------------------
# Per-feature estimation
# ---------------------------------------------------------------------------

def _fit_single_feature(
    phi_col: np.ndarray,
    y: np.ndarray,
    shap_matrix: np.ndarray,
    p: float,
    beta_n: float,
    kf: Any,
    nuisance_estimator: Any,
    alpha_estimator: Any,
) -> tuple[float, float, np.ndarray]:
    """
    Estimate theta_hat_p, SE, and influence function vector for one feature.

    Args:
        phi_col:           SHAP values for the target feature, shape (n,)
        y:                 Observed outcomes, shape (n,)
        shap_matrix:       Full SHAP matrix (n, d) used as X for nuisance models.
                           Using SHAP values as features means the nuisance model
                           operates on the same information space as the SHAP
                           computation, which tends to give better convergence.
        p:                 Importance power (>= 1)
        beta_n:            Smoothing parameter (only used for p < 2)
        kf:                Fitted KFold splitter (already initialised with folds)
        nuisance_estimator: Prototype estimator for mu and gamma
        alpha_estimator:   Prototype estimator for alpha

    Returns:
        (theta_hat, se, rho) where rho is shape (n,)
    """
    n = len(phi_col)
    gamma_hat = np.zeros(n)
    mu_hat = np.zeros(n)
    alpha_hat = np.zeros(n)

    indices = np.arange(n)

    for train_idx, val_idx in kf.split(indices):
        phi_train = phi_col[train_idx]
        y_train = y[train_idx]
        X_train = shap_matrix[train_idx]
        X_val = shap_matrix[val_idx]

        # --- mu_hat: E[Y | X] ---
        # We regress y on the full SHAP matrix. Using SHAP values as input
        # to the nuisance models is a deliberate design choice: SHAP values
        # already encode the model's view of feature contributions, so the
        # nuisance model can focus on the residual structure.
        mu_model = clone(nuisance_estimator).fit(X_train, y_train)
        mu_hat[val_idx] = mu_model.predict(X_val)

        # --- gamma_hat: sensitivity of |phi|^p to phi ---
        if p >= 2.0:
            gamma_labels_train = _unsmoothed_gamma(phi_train, p)
        else:
            gamma_labels_train = _smoothed_gamma_deriv(phi_train, p, beta_n)

        gamma_model = clone(nuisance_estimator).fit(X_train, gamma_labels_train)
        gamma_hat[val_idx] = gamma_model.predict(X_val)

        # --- alpha_hat: response correction ---
        # Full alpha_p requires a density ratio omega_S over all subsets S.
        # Under the independence approximation (omega_S = 1), this simplifies
        # to E[gamma_hat * (Y - mu_hat) | X]. We regress the product
        # gamma_hat * residual on X.
        #
        # This is the "simplified alpha" described in the spec (section 8,
        # open question 1). For correlated features, this introduces a small
        # second-order bias that is dominated by the n^{-1/2} term.
        #
        # Note: we use gamma_hat from the TRAINING fold (evaluated on train_idx)
        # to avoid data leakage when constructing alpha labels.
        gamma_train_pred = gamma_model.predict(X_train)
        # Use in-sample mu predictions for alpha label construction.
        # mu_hat[train_idx] is not yet filled (cross-fitting fills val_idx only).
        # In-sample predictions are over-fitted but the resulting bias in alpha is
        # second-order and does not affect the sqrt(n) rate of the estimator.
        mu_train_pred = mu_model.predict(X_train)
        residuals_train = y_train - mu_train_pred
        alpha_labels_train = gamma_train_pred * residuals_train
        alpha_model = clone(alpha_estimator).fit(X_train, alpha_labels_train)
        alpha_hat[val_idx] = alpha_model.predict(X_val)

    # --- Influence function per observation ---
    # rho_a(Z_i) = importance_term(phi_i) + alpha_hat(X_i) * (Y_i - mu_hat(X_i))
    #
    # The gamma * (psi_a - phi) term from the full score (section 2.2) averages
    # to zero in the population by Neyman orthogonality, so we drop it here.
    # This O(n) approximation is exact in expectation and introduces only
    # second-order error in the variance.
    if p >= 2.0:
        importance_term = np.abs(phi_col) ** p
    else:
        importance_term = _smoothed_phi(phi_col, p, beta_n)

    rho = importance_term + alpha_hat * (y - mu_hat)

    theta_hat = float(np.mean(rho))
    sigma2_hat = float(np.var(rho, ddof=1))
    se = float(np.sqrt(sigma2_hat / n))

    return theta_hat, se, rho


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SHAPInference:
    """
    Asymptotically valid confidence intervals for global SHAP feature importance.

    Implements the de-biased U-statistic estimator from Whitehouse, Sawarni,
    Syrgkanis (2026), arXiv:2602.10532. Provides CIs for theta_p = E[|phi_a(X)|^p]
    for any p >= 1 for each feature.

    The most common use cases are:
      p=1: mean absolute SHAP (standard SHAP importance bar chart)
      p=2: mean squared SHAP (variance-like, cleaner theory)

    IMPORTANT: Valid inference requires interventional SHAP values, not the
    default path-dependent TreeSHAP. Use SHAPRelativities with
    feature_perturbation='interventional' before calling SHAPInference, or
    pass interventional SHAP values directly.

    Args:
        shap_values: np.ndarray of shape (n_obs, n_features). SHAP values,
            one column per feature. Must be interventional SHAP for theoretical
            validity. Path-dependent SHAP will produce point estimates and
            intervals, but coverage guarantees do not hold.
        y: np.ndarray of shape (n_obs,). Observed outcomes (claim counts or
            claim amounts). Required for the alpha nuisance correction.
        feature_names: List[str] of length n_features. Column names.
        p: float >= 1. Power for importance measure. Default 2.0 (mean squared
            SHAP). p=1 gives mean absolute SHAP (the standard bar chart metric)
            but requires smoothing. p=2 has cleaner asymptotic theory.
        n_folds: int >= 2. Number of cross-fitting folds. Default 5. More folds
            reduce bias from the cross-fitting but increase compute.
        nuisance_estimator: str or sklearn estimator. Used for mu_hat (E[Y|X])
            and gamma_hat. Default 'gradient_boosting' uses
            HistGradientBoostingRegressor.
        alpha_estimator: str or sklearn estimator. Used for alpha_hat. Defaults
            to same as nuisance_estimator.
        beta_n: float or None. Smoothing parameter for p < 2. If None, computed
            as n^{(2-p)/(2*(p+1))} (assumes delta=1). Only used when p < 2.
        ci_level: float. Two-sided confidence level. Default 0.95.
        n_jobs: int. Placeholder for future parallelism. Currently unused;
            features are estimated sequentially.
        random_state: int or None. Controls fold splitting for reproducibility.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> shap_vals = rng.normal(size=(500, 3))
    >>> y = rng.poisson(1.0, size=500).astype(float)
    >>> si = SHAPInference(shap_vals, y, feature_names=["a", "b", "c"], p=2)
    >>> si.fit()
    SHAPInference(n_obs=500, n_features=3, p=2.0, n_folds=5)
    >>> tbl = si.importance_table()
    """

    def __init__(
        self,
        shap_values: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        p: float = 2.0,
        n_folds: int = 5,
        nuisance_estimator: str | Any = "gradient_boosting",
        alpha_estimator: str | Any = "gradient_boosting",
        beta_n: float | None = None,
        ci_level: float = 0.95,
        n_jobs: int = 1,
        random_state: int | None = None,
    ) -> None:
        # --- Input validation ---
        shap_values = np.asarray(shap_values, dtype=float)
        y = np.asarray(y, dtype=float)

        if shap_values.ndim != 2:
            raise ValueError(
                f"shap_values must be 2D array (n_obs, n_features), "
                f"got shape {shap_values.shape}"
            )
        if y.ndim != 1:
            raise ValueError(f"y must be 1D array, got shape {y.shape}")
        if shap_values.shape[0] != len(y):
            raise ValueError(
                f"shap_values and y must have the same number of observations. "
                f"Got shap_values.shape[0]={shap_values.shape[0]} and len(y)={len(y)}."
            )
        if len(feature_names) != shap_values.shape[1]:
            raise ValueError(
                f"len(feature_names)={len(feature_names)} must equal "
                f"shap_values.shape[1]={shap_values.shape[1]}."
            )
        if p < 1.0:
            raise ValueError(f"p must be >= 1. Got p={p}.")
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2. Got n_folds={n_folds}.")
        if not (0.0 < ci_level < 1.0):
            raise ValueError(f"ci_level must be in (0, 1). Got ci_level={ci_level}.")
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("feature_names must be unique.")

        self.shap_values = shap_values
        self.y = y
        self.feature_names = list(feature_names)
        self.p = float(p)
        self.n_folds = n_folds
        self.nuisance_estimator = nuisance_estimator
        self.alpha_estimator = alpha_estimator
        self.beta_n = beta_n
        self.ci_level = ci_level
        self.n_jobs = n_jobs
        self.random_state = random_state

        # Fitted attributes — populated by fit()
        self._theta_hat: np.ndarray | None = None   # shape (n_features,)
        self._se: np.ndarray | None = None          # shape (n_features,)
        self._rho: np.ndarray | None = None         # shape (n_obs, n_features)
        self._is_fitted: bool = False

    def __repr__(self) -> str:
        n, d = self.shap_values.shape
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"SHAPInference(n_obs={n}, n_features={d}, "
            f"p={self.p}, n_folds={self.n_folds}, status={status})"
        )

    def fit(self) -> "SHAPInference":
        """
        Estimate nuisance functions via cross-fitting and compute de-biased
        theta_hat_p for each feature.

        The algorithm:
        1. Split observations into n_folds folds.
        2. For each fold, train mu_hat, gamma_hat, alpha_hat on the complement.
        3. Evaluate nuisances on the held-out fold.
        4. Assemble full-data nuisance predictions.
        5. Compute the influence function rho_a for each feature.
        6. theta_hat = mean(rho_a), SE = sqrt(var(rho_a) / n).

        Returns:
            self, for method chaining.
        """
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn >= 1.3 is required for SHAPInference. "
                "Install with: pip install shap-relativities[ml]"
            )

        n, d = self.shap_values.shape

        # Warn if p < 2: smoothing is active
        effective_beta_n = self.beta_n
        if self.p < 2.0:
            if effective_beta_n is None:
                effective_beta_n = _default_beta_n(n, self.p)
            warnings.warn(
                f"p={self.p} < 2: using smoothed estimator phi_{{p,beta}} "
                f"with beta_n={effective_beta_n:.3f}. "
                "Coverage is asymptotically valid but may be approximate for "
                "features with many near-zero SHAP values. "
                "Consider p=2 for cleaner guarantees.",
                UserWarning,
                stacklevel=2,
            )
        else:
            effective_beta_n = 0.0  # unused but keeps type consistent

        nu_est = _make_nuisance_estimator(self.nuisance_estimator)
        al_est = _make_nuisance_estimator(self.alpha_estimator)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        theta_hats = np.zeros(d)
        ses = np.zeros(d)
        rhos = np.zeros((n, d))

        for j in range(d):
            theta_j, se_j, rho_j = _fit_single_feature(
                phi_col=self.shap_values[:, j],
                y=self.y,
                shap_matrix=self.shap_values,
                p=self.p,
                beta_n=effective_beta_n,
                kf=kf,
                nuisance_estimator=nu_est,
                alpha_estimator=al_est,
            )
            theta_hats[j] = theta_j
            ses[j] = se_j
            rhos[:, j] = rho_j

        self._theta_hat = theta_hats
        self._se = ses
        self._rho = rhos
        self._is_fitted = True

        return self

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before accessing results.")

    def importance_table(self) -> pl.DataFrame:
        """
        Return feature importance estimates with confidence intervals.

        All theta_hat values are theoretically non-negative (they estimate
        E[|phi|^p]), but may be slightly negative for features with very
        small true importance — this is expected sampling variability.

        Returns:
            Polars DataFrame with columns:
              feature:          Feature name
              theta_hat:        Point estimate of E[|phi_a(X)|^p]
              theta_lower:      Lower CI bound
              theta_upper:      Upper CI bound
              sigma_hat:        sqrt(Var[rho_a]) — asymptotic std dev
              se:               Standard error = sigma_hat / sqrt(n)
              rank:             Rank by theta_hat (1 = most important)
              rank_lower:       Conservative rank (using theta_lower)
              rank_upper:       Optimistic rank (using theta_upper)
              p_value_nonzero:  Two-sided p-value for H0: theta_p = 0
        """
        self._check_fitted()
        assert self._theta_hat is not None
        assert self._se is not None
        assert self._rho is not None

        n = self.shap_values.shape[0]
        z = float(stats.norm.ppf((1 + self.ci_level) / 2))

        theta = self._theta_hat
        se = self._se
        lower = theta - z * se
        upper = theta + z * se

        # sigma_hat = SE * sqrt(n) = sqrt(Var[rho])
        sigma_hat = se * np.sqrt(n)

        # Ranks (1 = most important)
        rank = _dense_rank_descending(theta)
        rank_lower = _dense_rank_descending(lower)  # conservative: lower bound
        rank_upper = _dense_rank_descending(upper)  # optimistic: upper bound

        # p-value: two-sided test H0: theta_p = 0
        # Under H0, z_stat = theta_hat / SE ~ N(0,1) asymptotically
        with np.errstate(divide="ignore", invalid="ignore"):
            z_stat = np.where(se > 0, theta / se, np.inf)
        p_values = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_stat)))

        return pl.DataFrame({
            "feature": self.feature_names,
            "theta_hat": theta.tolist(),
            "theta_lower": lower.tolist(),
            "theta_upper": upper.tolist(),
            "sigma_hat": sigma_hat.tolist(),
            "se": se.tolist(),
            # _dense_rank_descending already returns list[int]; no .tolist() needed
            "rank": rank,
            "rank_lower": rank_lower,
            "rank_upper": rank_upper,
            "p_value_nonzero": p_values.tolist(),
        }).sort("rank")

    def ranking_ci(self, feature_a: str, feature_b: str) -> dict[str, float]:
        """
        Test whether feature_a has strictly higher importance than feature_b.

        Uses the joint asymptotic distribution of (theta_hat_a, theta_hat_b).
        The covariance is estimated from influence function cross-products,
        which accounts for the fact that both features' rho vectors are
        computed from the same observations.

        H0: theta_a = theta_b
        H1: theta_a > theta_b  (one-sided)

        Args:
            feature_a: Name of the first feature.
            feature_b: Name of the second feature.

        Returns:
            dict with:
              diff:      theta_hat_a - theta_hat_b
              se_diff:   Standard error of the difference
              z_stat:    Standardised test statistic
              p_value:   One-sided p-value for H1: theta_a > theta_b
              ci_lower:  Lower bound on (theta_a - theta_b) at self.ci_level
              ci_upper:  Upper bound on (theta_a - theta_b) at self.ci_level
        """
        self._check_fitted()
        assert self._theta_hat is not None
        assert self._rho is not None

        if feature_a not in self.feature_names:
            raise ValueError(f"feature_a='{feature_a}' not in feature_names.")
        if feature_b not in self.feature_names:
            raise ValueError(f"feature_b='{feature_b}' not in feature_names.")

        j_a = self.feature_names.index(feature_a)
        j_b = self.feature_names.index(feature_b)

        n = self.shap_values.shape[0]
        theta_a = float(self._theta_hat[j_a])
        theta_b = float(self._theta_hat[j_b])

        rho_a = self._rho[:, j_a]
        rho_b = self._rho[:, j_b]

        # Var(theta_hat_a - theta_hat_b) = Var(mean(rho_a - rho_b)) / n
        # = Var(rho_a - rho_b) / n
        diff_rho = rho_a - rho_b
        var_diff = float(np.var(diff_rho, ddof=1)) / n
        se_diff = float(np.sqrt(max(var_diff, 0.0)))

        diff = theta_a - theta_b
        # Guard: when both feature arguments are identical, diff==0 and SE==0.
        # 0/0 is indeterminate; the correct result is z_stat=0, p_value=1.
        if diff == 0.0 and se_diff == 0.0:
            z_stat = 0.0
            p_value = 1.0
        else:
            z_stat = diff / se_diff if se_diff > 0 else float("inf")
            # One-sided p-value for H1: theta_a > theta_b
            p_value = float(1.0 - stats.norm.cdf(z_stat))

        # Two-sided CI on the difference
        z_ci = float(stats.norm.ppf((1 + self.ci_level) / 2))
        ci_lower = diff - z_ci * se_diff
        ci_upper = diff + z_ci * se_diff

        return {
            "diff": diff,
            "se_diff": se_diff,
            "z_stat": z_stat,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    def plot_importance(
        self,
        top_n: int | None = None,
        ax: Any | None = None,
        sort: bool = True,
    ) -> Any:
        """
        Bar chart of theta_hat with CI error bars.

        Styled for insurance governance presentations: clean background,
        coloured bars by significance (dark blue = CI excludes zero, grey =
        CI includes zero), error bars at self.ci_level.

        Args:
            top_n: Show only top N features by theta_hat. None shows all.
            ax: matplotlib Axes. If None, creates a new figure.
            sort: Sort by theta_hat descending. Default True.

        Returns:
            The matplotlib Axes object.
        """
        self._check_fitted()

        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for plot_importance(). "
                "Install with: pip install shap-relativities[plot]"
            ) from e

        tbl = self.importance_table()
        if sort:
            tbl = tbl.sort("theta_hat", descending=True)
        if top_n is not None:
            tbl = tbl.head(top_n)

        features = tbl["feature"].to_list()
        theta = tbl["theta_hat"].to_numpy()
        lower = tbl["theta_lower"].to_numpy()
        upper = tbl["theta_upper"].to_numpy()

        err_low = theta - lower
        err_high = upper - theta

        # Colour by significance: CI excludes zero => dark teal, else grey
        significant = lower > 0
        colours = ["#1a6b7c" if s else "#9e9e9e" for s in significant]

        if ax is None:
            fig, ax = plt.subplots(figsize=(max(6, len(features) * 0.6 + 2), 5))

        y_pos = np.arange(len(features))
        ax.barh(
            y_pos,
            theta,
            xerr=[err_low, err_high],
            color=colours,
            capsize=4,
            edgecolor="white",
            linewidth=0.5,
            error_kw={"elinewidth": 1.2, "capthick": 1.2, "ecolor": "#555555"},
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=10)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

        p_label = f"E[|φ(X)|^{{{self.p:.4g}}}]"
        ax.set_xlabel(p_label, fontsize=11)
        ax.set_title(
            f"Global SHAP Feature Importance ({int(self.ci_level * 100)}% CI)",
            fontsize=12,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#f8f8f8")
        if hasattr(ax, "figure") and ax.figure is not None:
            ax.figure.set_facecolor("white")

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#1a6b7c", label="CI excludes 0"),
            Patch(facecolor="#9e9e9e", label="CI includes 0"),
        ]
        ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

        return ax

    @property
    def influence_matrix(self) -> np.ndarray:
        """
        Influence function matrix, shape (n_obs, n_features).

        rho[i, j] is observation i's contribution to theta_hat_j.
        Observations with large |rho[i, j]| are high-leverage for feature j's
        importance estimate — useful for identifying influential policies in
        governance reviews.
        """
        self._check_fitted()
        assert self._rho is not None
        return self._rho.copy()


# ---------------------------------------------------------------------------
# Helper: dense rank descending
# ---------------------------------------------------------------------------

def _dense_rank_descending(values: np.ndarray) -> list[int]:
    """
    Rank values from largest to smallest. Ties share the same rank.
    Returns a list of ints with 1 = most important.
    """
    order = np.argsort(-values)  # descending
    ranks = np.empty(len(values), dtype=int)
    ranks[order[0]] = 1
    for k in range(1, len(order)):
        if values[order[k]] == values[order[k - 1]]:
            ranks[order[k]] = ranks[order[k - 1]]
        else:
            ranks[order[k]] = k + 1
    return ranks.tolist()
