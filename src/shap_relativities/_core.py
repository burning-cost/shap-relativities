"""
Core SHAPRelativities class.

Converts a trained tree model's SHAP values into actuarial-grade multiplicative
rating relativities. Output is directly comparable to GLM exp(beta) relativities:
a table of (feature, level, relativity) triples where the base level = 1.0 and
relativities multiply together to give the model's expected prediction.

Supports CatBoost models with log-link objectives (Poisson, Tweedie, Gamma).
CatBoost is the default and recommended model - it handles categorical features
natively without encoding, which removes a common source of information loss in
insurance pricing models. Will not work correctly with linear-link models.

Known limitations
-----------------
- SHAP attribution for correlated features is not uniquely defined. Correlated
  features share attribution in a way that depends on tree split order. Document
  this when presenting results.
- CLT confidence intervals capture data uncertainty only - not model uncertainty
  from the GBM fitting process.
- TreeSHAP allocates interaction effects back to individual features by default.
  Use shap_interaction_values() if you need pure main effects, but note that
  this is O(n * p^2) and quickly becomes infeasible.

Data handling
-------------
X may be a Polars DataFrame or a pandas DataFrame. Internally, all data
operations use Polars. Conversion to pandas is done only when calling shap's
TreeExplainer, which requires a pandas DataFrame for column names and dtype
inference. The output of extract_relativities() is a Polars DataFrame.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import polars as pl

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

from ._aggregation import aggregate_categorical, aggregate_continuous
from ._normalisation import normalise_base_level, normalise_mean
from ._validation import (
    CheckResult,
    check_feature_coverage,
    check_reconstruction,
    check_sparse_levels,
)


# Output column order - consistent with what pricing teams expect
_RELATIVITY_COLUMNS = [
    "feature", "level", "relativity", "lower_ci", "upper_ci",
    "mean_shap", "shap_std", "n_obs", "exposure_weight",
]


def _to_polars(X: Any) -> pl.DataFrame:
    """Convert X to a Polars DataFrame regardless of input type."""
    try:
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            return pl.from_pandas(X)
    except ImportError:
        pass
    if isinstance(X, pl.DataFrame):
        return X
    raise TypeError(
        f"X must be a Polars or pandas DataFrame, got {type(X).__name__}. "
        "Install polars or pandas as appropriate."
    )


def _to_pandas(X: pl.DataFrame):
    """Convert a Polars DataFrame to pandas for shap/catboost bridging."""
    import pandas as pd  # noqa: F401 - required for SHAP
    return X.to_pandas()


class SHAPRelativities:
    """
    Extract multiplicative rating relativities from a tree model via SHAP.

    Workflow::

        sr = SHAPRelativities(
            model=catboost_model,
            X=df.select(["area", "ncd_years", "has_convictions"]),
            exposure=df["exposure"],
            categorical_features=["area", "ncd_years"],
        )
        sr.fit()
        rels = sr.extract_relativities(
            normalise_to="base_level",
            base_levels={"area": "A", "ncd_years": 0},
        )

    Args:
        model: A trained CatBoost model. Must use a log-link objective (Poisson,
            Tweedie, Gamma). CatBoost is the recommended default - it handles
            categoricals natively.
        X: Feature matrix. Use training data for in-sample relativities, or a
            representative holdout sample for out-of-sample. Polars DataFrames
            are preferred; pandas DataFrames are accepted and converted
            internally.
        exposure: Earned policy years (or other volume measure). Used as
            observation weights throughout. If None, all observations are
            weighted equally.
        categorical_features: Features to aggregate by level (bar-chart style).
            If None, all non-numeric columns are treated as categorical.
        continuous_features: Features to leave as per-observation points.
            If None, all numeric columns are treated as continuous.
        feature_perturbation: "tree_path_dependent" (default, fast, no
            background data needed) or "interventional" (corrects for feature
            correlation, needs background_data).
        background_data: Required only if feature_perturbation="interventional".
        n_background_samples: Number of background samples for interventional
            SHAP. Default 1000.
        annualise_exposure: If True and exposure is provided, subtract mean
            log(exposure) from the expected_value to give an annualised
            baseline. Default True.
    """

    def __init__(
        self,
        model: Any,
        X: Any,
        exposure: Any = None,
        categorical_features: list[str] | None = None,
        continuous_features: list[str] | None = None,
        background_data: Any = None,
        feature_perturbation: str = "tree_path_dependent",
        n_background_samples: int = 1000,
        annualise_exposure: bool = True,
    ) -> None:
        if not _SHAP_AVAILABLE:
            raise ImportError(
                "shap is required for SHAPRelativities. "
                "Install it with: uv add 'shap-relativities[ml]'"
            )

        self._model = model
        self._X: pl.DataFrame = _to_polars(X)
        self._background_data = (
            _to_polars(background_data) if background_data is not None else None
        )

        # Normalise exposure to a numpy array
        if exposure is None:
            self._exposure: np.ndarray | None = None
        elif isinstance(exposure, np.ndarray):
            self._exposure = exposure
        elif isinstance(exposure, pl.Series):
            self._exposure = exposure.to_numpy()
        else:
            # pd.Series or similar
            self._exposure = np.asarray(exposure)

        # Validate exposure length matches X
        if self._exposure is not None and len(self._exposure) != len(self._X):
            raise ValueError(
                f"exposure length ({len(self._exposure)}) does not match "
                f"X length ({len(self._X)}). Both must have the same number of rows."
            )

        self._feature_perturbation = feature_perturbation
        self._n_background_samples = n_background_samples
        self._annualise_exposure = annualise_exposure

        # Classify features
        self._categorical_features = categorical_features or self._infer_categorical()
        self._continuous_features = continuous_features or self._infer_continuous()

        # Populated by fit()
        self._shap_values: np.ndarray | None = None
        self._expected_value: float | None = None
        self._is_fitted: bool = False

    def _infer_categorical(self) -> list[str]:
        numeric_types = (
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64,
        )
        return [
            c for c in self._X.columns
            if not isinstance(self._X[c].dtype, numeric_types)
        ]

    def _infer_continuous(self) -> list[str]:
        numeric_types = (
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64,
        )
        return [
            c for c in self._X.columns
            if isinstance(self._X[c].dtype, numeric_types)
            and c not in (self._categorical_features or [])
        ]

    def fit(self) -> "SHAPRelativities":
        """
        Compute SHAP values for all features in X.

        Must be called before extract_relativities(). Calling fit() again
        recomputes SHAP values (e.g. after changing X or the background data).

        The feature matrix is converted to pandas internally for shap's
        TreeExplainer. The conversion is a necessary bridge - shap requires
        pandas for column name handling.

        Returns:
            Self, for method chaining.
        """
        # Convert to pandas for shap - unavoidable bridge
        X_pd = _to_pandas(self._X)

        bg_data = None
        if self._feature_perturbation == "interventional":
            if self._background_data is not None:
                bg_data = _to_pandas(self._background_data)
            else:
                n_bg = min(self._n_background_samples, len(X_pd))
                bg_data = shap.sample(X_pd, n_bg)

        explainer = shap.TreeExplainer(
            self._model,
            data=bg_data,
            feature_perturbation=self._feature_perturbation,
            model_output="raw",
        )

        raw = explainer.shap_values(X_pd)

        # Some models return a list when there is a single output
        if isinstance(raw, list):
            if len(raw) == 1:
                raw = raw[0]
            else:
                raise ValueError(
                    f"Model has {len(raw)} outputs. SHAPRelativities supports "
                    "single-output models only."
                )

        self._shap_values = raw

        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev = float(ev[0])
        self._expected_value = float(ev)

        self._is_fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before using this method.")

    def shap_values(self) -> np.ndarray:
        """
        Raw SHAP values, shape (n_obs, n_features), in log space.

        Returns:
            Array of shape (n_obs, n_features).
        """
        self._check_fitted()
        return self._shap_values  # type: ignore[return-value]

    def baseline(self) -> float:
        """
        exp(expected_value) - the base rate in prediction space.

        If annualise_exposure=True and exposure was provided, this is adjusted
        for the average log-exposure offset so it represents an annualised rate.

        Returns:
            Base rate as a float.
        """
        self._check_fitted()
        ev = self._expected_value  # type: ignore[assignment]

        if self._annualise_exposure and self._exposure is not None:
            mean_log_exp = float(np.mean(np.log(np.clip(self._exposure, 1e-9, None))))
            ev = ev - mean_log_exp

        return float(np.exp(ev))

    def extract_relativities(
        self,
        normalise_to: str = "base_level",
        base_levels: dict[str, str | float | int] | None = None,
        ci_method: str = "clt",
        n_bootstrap: int = 200,
        ci_level: float = 0.95,
    ) -> pl.DataFrame:
        """
        Extract multiplicative relativities from SHAP values.

        Args:
            normalise_to: "base_level" (base level for each feature gets
                relativity = 1.0) or "mean" (exposure-weighted portfolio
                mean = 1.0).
            base_levels: Mapping of feature -> base level value. Required for
                categorical features when normalise_to="base_level". Continuous
                features automatically use mean normalisation regardless of
                this setting.
            ci_method: "clt" (CLT approximation, default, fast) or "none" (no
                CIs). "bootstrap" is not yet implemented.
            n_bootstrap: Ignored unless ci_method="bootstrap".
            ci_level: Two-sided confidence level. Default 0.95.

        Returns:
            Polars DataFrame with columns: feature, level, relativity,
            lower_ci, upper_ci, mean_shap, shap_std, n_obs, exposure_weight.
            One row per (feature, level) combination.
        """
        self._check_fitted()

        _VALID_CI_METHODS = {"clt", "bootstrap", "none"}
        if ci_method not in _VALID_CI_METHODS:
            raise ValueError(
                f"Unknown ci_method {ci_method!r}. "
                f"Valid options are: {sorted(_VALID_CI_METHODS)}."
            )

        if ci_method == "bootstrap":
            raise NotImplementedError(
                "Bootstrap CIs are not yet implemented. Use ci_method='clt'."
            )

        base_levels = base_levels or {}
        weights = (
            self._exposure if self._exposure is not None
            else np.ones(len(self._X))
        )

        feature_names = self._X.columns
        shap_vals = self._shap_values  # type: ignore[assignment]

        parts: list[pl.DataFrame] = []

        for i, feat in enumerate(feature_names):
            feat_vals = self._X[feat].to_numpy()
            shap_col = shap_vals[:, i]

            is_categorical = feat in self._categorical_features

            if is_categorical:
                agg = aggregate_categorical(feat, feat_vals, shap_col, weights)
            else:
                agg = aggregate_continuous(feat, feat_vals, shap_col, weights)

            # Normalisation
            if normalise_to == "base_level" and is_categorical:
                base = base_levels.get(feat)
                if base is None:
                    # Fall back to the level with the smallest mean_shap as
                    # a sensible default (closest to intercept)
                    base = agg.sort("mean_shap")["level"][0]
                    warnings.warn(
                        f"No base level specified for '{feat}'. "
                        f"Using '{base}' (lowest mean SHAP) as base.",
                        UserWarning,
                        stacklevel=2,
                    )

                if ci_method == "none":
                    base_key = str(base)
                    base_rows = agg.filter(pl.col("level") == base_key)
                    base_shap = base_rows["mean_shap"][0]
                    agg = agg.with_columns([
                        (pl.col("mean_shap") - base_shap).exp().alias("relativity"),
                        pl.lit(float("nan")).alias("lower_ci"),
                        pl.lit(float("nan")).alias("upper_ci"),
                    ])
                else:
                    agg = normalise_base_level(agg, base, ci_level=ci_level)

            else:
                # Mean normalisation for continuous features, or when
                # normalise_to="mean" for any feature
                if ci_method == "none":
                    total_weight = agg["exposure_weight"].sum()
                    portfolio_mean = float(
                        (agg["mean_shap"] * agg["exposure_weight"]).sum()
                        / total_weight
                    ) if total_weight > 0 else 0.0
                    agg = agg.with_columns([
                        (pl.col("mean_shap") - portfolio_mean).exp().alias("relativity"),
                        pl.lit(float("nan")).alias("lower_ci"),
                        pl.lit(float("nan")).alias("upper_ci"),
                    ])
                else:
                    agg = normalise_mean(agg, ci_level=ci_level)

            parts.append(agg)

        # Cast the 'level' column to Utf8 in every part before concat.
        # Categorical features produce level as Utf8; continuous features
        # produce level as Float64. pl.concat with how="diagonal" cannot
        # unify mismatched types for the same column name, so we normalise
        # here rather than requiring callers to pre-cast their feature columns.
        parts = [
            p.with_columns(pl.col("level").cast(pl.String))
            if "level" in p.columns else p
            for p in parts
        ]

        result = pl.concat(parts, how="diagonal")

        # Ensure standard column order (wsq_weight is internal, not exported)
        available = [c for c in _RELATIVITY_COLUMNS if c in result.columns]
        return result.select(available)

    def extract_continuous_curve(
        self,
        feature: str,
        n_points: int = 100,
        smooth_method: str = "loess",
    ) -> pl.DataFrame:
        """
        Smoothed relativity curve for a continuous feature.

        Args:
            feature: Feature name. Must be in continuous_features.
            n_points: Number of points in the output curve (not the input
                data).
            smooth_method: "loess" (locally weighted regression, requires
                statsmodels), "isotonic" (monotone curve via isotonic
                regression), or "none" (raw per-observation relativities).

        Returns:
            Polars DataFrame with columns: feature_value, relativity,
            lower_ci, upper_ci.

        Raises:
            ValueError: If feature is not in X or smooth_method is unknown.
        """
        self._check_fitted()

        if feature not in self._X.columns:
            raise ValueError(f"Feature '{feature}' not in X.")

        feat_idx = self._X.columns.index(feature)
        feat_vals = self._X[feature].to_numpy().astype(float)
        shap_col = self._shap_values[:, feat_idx]  # type: ignore[index]
        weights = (
            self._exposure if self._exposure is not None
            else np.ones(len(self._X))
        )

        # Exposure-weighted mean over the actual data distribution
        portfolio_mean = np.average(shap_col, weights=weights)
        relativities = np.exp(shap_col - portfolio_mean)

        grid = np.linspace(feat_vals.min(), feat_vals.max(), n_points)

        if smooth_method == "none":
            order = np.argsort(feat_vals)
            return pl.DataFrame({
                "feature_value": feat_vals[order],
                "relativity": relativities[order],
                "lower_ci": np.full(len(feat_vals), float("nan")),
                "upper_ci": np.full(len(feat_vals), float("nan")),
            })

        elif smooth_method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(feat_vals, shap_col, sample_weight=weights)
            smoothed_shap = ir.predict(grid)

            # P1-4 fix: normalise the smoothed curve so the exposure-weighted
            # geometric mean of relativities = 1.0.
            # The smooth is on the data, evaluated on a uniform grid. Subtracting
            # portfolio_mean (computed on the data distribution) would be correct
            # only if the grid were distributed like the data — it isn't.
            # Instead, compute the data-distribution-weighted mean of the smoothed
            # curve at the original data points, then use that as the reference.
            smoothed_at_data = ir.predict(feat_vals)
            weighted_mean_smoothed = np.average(smoothed_at_data, weights=weights)
            smoothed_rel = np.exp(smoothed_shap - weighted_mean_smoothed)

            return pl.DataFrame({
                "feature_value": grid,
                "relativity": smoothed_rel,
                "lower_ci": np.full(n_points, float("nan")),
                "upper_ci": np.full(n_points, float("nan")),
            })

        elif smooth_method == "loess":
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess

                smoothed_shap = lowess(
                    shap_col, feat_vals, frac=0.3, it=3,
                    xvals=grid, is_sorted=False,
                )

                # P1-4 fix: compute the smoothed values at original data points
                # so the normalisation is data-distribution-weighted, not
                # grid-uniform. Use the same lowess parameters.
                smoothed_at_data = lowess(
                    shap_col, feat_vals, frac=0.3, it=3,
                    xvals=feat_vals, is_sorted=False,
                )
                weighted_mean_smoothed = np.average(smoothed_at_data, weights=weights)
                smoothed_rel = np.exp(smoothed_shap - weighted_mean_smoothed)

                return pl.DataFrame({
                    "feature_value": grid,
                    "relativity": smoothed_rel,
                    "lower_ci": np.full(n_points, float("nan")),
                    "upper_ci": np.full(n_points, float("nan")),
                })
            except ImportError:
                warnings.warn(
                    "statsmodels not installed; falling back to smooth_method='none'.",
                    UserWarning,
                    stacklevel=2,
                )
                return self.extract_continuous_curve(
                    feature, n_points=n_points, smooth_method="none"
                )

        else:
            raise ValueError(
                f"Unknown smooth_method '{smooth_method}'. "
                "Choose from: 'loess', 'isotonic', 'none'."
            )

    def validate(self) -> dict[str, CheckResult]:
        """
        Run diagnostic checks on the SHAP computation.

        Checks performed:

        1. reconstruction: exp(shap.sum(1) + expected_value) should match
           model predictions within tolerance. Material failure here indicates
           the explainer was set up incorrectly.

        2. feature_coverage: every feature in X should appear in the SHAP
           output. Currently always passes given TreeExplainer's API.

        3. sparse_levels: warns if any categorical level has fewer than 30
           observations. CLT CIs will be unreliable for these levels.

        Returns:
            Dict with keys "reconstruction", "feature_coverage",
            "sparse_levels". Each value is a CheckResult(passed, value,
            message).
        """
        self._check_fitted()

        X_pd = _to_pandas(self._X)

        # Get model predictions for reconstruction check
        preds = None
        if self._model is not None:
            try:
                preds = self._model.predict(X_pd)
            except Exception:
                preds = None

        results: dict[str, CheckResult] = {}

        if preds is not None:
            results["reconstruction"] = check_reconstruction(
                self._shap_values,  # type: ignore[arg-type]
                self._expected_value,  # type: ignore[arg-type]
                preds,
                tolerance=1e-4,
            )
        else:
            results["reconstruction"] = CheckResult(
                passed=False,
                value=float("nan"),
                message="Could not obtain model predictions for reconstruction check.",
            )

        feature_names = self._X.columns
        results["feature_coverage"] = check_feature_coverage(
            feature_names, feature_names
        )

        # Check sparse levels for categorical features
        weights = (
            self._exposure if self._exposure is not None
            else np.ones(len(self._X))
        )

        sparse_parts: list[pl.DataFrame] = []
        for feat in self._categorical_features:
            if feat not in self._X.columns:
                continue
            feat_idx = self._X.columns.index(feat)
            agg = aggregate_categorical(
                feat,
                self._X[feat].to_numpy(),
                self._shap_values[:, feat_idx],  # type: ignore[index]
                weights,
            )
            sparse_parts.append(agg)

        if sparse_parts:
            all_agg = pl.concat(sparse_parts, how="diagonal")
            results["sparse_levels"] = check_sparse_levels(all_agg)
        else:
            results["sparse_levels"] = CheckResult(
                passed=True, value=0.0,
                message="No categorical features to check."
            )

        return results

    def plot_relativities(
        self,
        features: list[str] | None = None,
        show_ci: bool = True,
        figsize: tuple[int, int] = (12, 8),
    ) -> None:
        """
        Plot relativities as bar charts (categorical) or line charts (continuous).

        Args:
            features: Subset of features to plot. Defaults to all features.
            show_ci: Whether to show confidence intervals. Default True.
            figsize: Overall figure size.
        """
        self._check_fitted()

        from ._plotting import plot_relativities as _plot

        rels = self.extract_relativities()
        _plot(
            rels,
            categorical_features=self._categorical_features,
            continuous_features=self._continuous_features,
            features=features,
            show_ci=show_ci,
            figsize=figsize,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialisable representation of the fitted object.

        Stores SHAP values, expected value, feature names, and feature
        classification. Does not store the original model or X DataFrame.

        Returns:
            Dict suitable for JSON serialisation.
        """
        self._check_fitted()
        return {
            "shap_values": self._shap_values.tolist(),  # type: ignore[union-attr]
            "expected_value": self._expected_value,
            "feature_names": self._X.columns,
            "categorical_features": self._categorical_features,
            "continuous_features": self._continuous_features,
            "X_values": {c: self._X[c].to_list() for c in self._X.columns},
            "exposure": (
                self._exposure.tolist()
                if self._exposure is not None else None
            ),
            "annualise_exposure": self._annualise_exposure,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SHAPRelativities":
        """
        Reconstruct a fitted SHAPRelativities from to_dict() output.

        The reconstructed object has no model attached, so validate() and
        plot_relativities() still work but fit() cannot be re-run.

        Args:
            data: Output of to_dict().

        Returns:
            Fitted SHAPRelativities instance.
        """
        # P1-2 fix: use feature_names to control column ordering in X.
        # Without this, tools that sort JSON object keys (REST APIs, some
        # pretty-printers) reorder X_values, misaligning columns with the
        # shap_values matrix columns.
        feature_names: list[str] = data.get("feature_names", list(data["X_values"].keys()))
        X = pl.DataFrame({k: data["X_values"][k] for k in feature_names})

        exposure = (
            np.array(data["exposure"]) if data.get("exposure") is not None
            else None
        )

        # Create a minimal instance without a real model
        instance = cls.__new__(cls)
        instance._model = None
        instance._X = X
        instance._exposure = exposure
        instance._categorical_features = data.get("categorical_features", [])
        instance._continuous_features = data.get("continuous_features", [])
        instance._feature_perturbation = "tree_path_dependent"
        instance._background_data = None
        instance._n_background_samples = 1000
        instance._annualise_exposure = data.get("annualise_exposure", True)
        instance._shap_values = np.array(data["shap_values"])
        instance._expected_value = float(data["expected_value"])
        instance._is_fitted = True

        return instance
