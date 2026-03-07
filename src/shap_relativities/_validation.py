"""
Diagnostic checks for SHAP relativity computations.

The validate() method on SHAPRelativities returns a dict mapping check name
to a named tuple of (passed, value, message). Callers should inspect this dict
before relying on extracted relativities.

Design rationale
----------------
Validation is separate from computation so that the main fit() path stays
clean and fast. Validation can be expensive (e.g. predict call) and is opt-in.

All functions accept Polars DataFrames where a DataFrame is required.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
import polars as pl


class CheckResult(NamedTuple):
    passed: bool
    value: float
    message: str


def check_reconstruction(
    shap_values: np.ndarray,
    expected_value: float,
    predictions: np.ndarray,
    tolerance: float = 1e-4,
) -> CheckResult:
    """
    Verify that exp(shap_values.sum(axis=1) + expected_value) matches
    model predictions to within tolerance.

    TreeSHAP satisfies the efficiency axiom: the sum of SHAP values plus the
    expected value should exactly reconstruct the model's raw output. Any
    material deviation suggests the explainer was constructed incorrectly
    (e.g. wrong model_output setting).

    Args:
        shap_values: SHAP values in log space, shape (n_obs, n_features).
        expected_value: Explainer expected value (intercept in log space).
        predictions: Model predictions in response space, shape (n_obs,).
        tolerance: Maximum acceptable absolute difference.

    Returns:
        CheckResult with passed=True if max error is below tolerance.
    """
    reconstructed = np.exp(shap_values.sum(axis=1) + expected_value)
    max_diff = float(np.abs(reconstructed - predictions).max())
    passed = max_diff < tolerance
    msg = (
        f"Max absolute reconstruction error: {max_diff:.2e}."
        + ("" if passed else f" Exceeds tolerance {tolerance:.2e}.")
    )
    return CheckResult(passed=passed, value=max_diff, message=msg)


def check_feature_coverage(
    shap_feature_names: list[str],
    expected_features: list[str],
) -> CheckResult:
    """
    Verify that every feature in X has a corresponding SHAP column.

    Args:
        shap_feature_names: Feature names from the explainer.
        expected_features: Feature names from X.

    Returns:
        CheckResult with passed=True if no features are missing.
    """
    missing = set(expected_features) - set(shap_feature_names)
    passed = len(missing) == 0
    msg = (
        "All features covered by SHAP."
        if passed
        else f"Features missing from SHAP output: {sorted(missing)}"
    )
    return CheckResult(passed=passed, value=float(len(missing)), message=msg)


def check_sparse_levels(
    aggregated: pl.DataFrame,
    min_obs: int = 30,
) -> CheckResult:
    """
    Warn if any factor level has fewer than min_obs observations.

    CLT confidence intervals become unreliable for small samples. This check
    flags levels where the CI should be treated with caution rather than as
    a hard threshold.

    Args:
        aggregated: Output from aggregate_categorical, with n_obs column.
        min_obs: Minimum observation count per level. Default 30 (CLT rule
            of thumb).

    Returns:
        CheckResult with passed=False if any level has fewer than min_obs
        observations.
    """
    if "n_obs" not in aggregated.columns:
        return CheckResult(passed=True, value=0.0, message="No n_obs column to check.")

    sparse = aggregated.filter(pl.col("n_obs") < min_obs)
    n_sparse = len(sparse)
    passed = n_sparse == 0
    if not passed:
        levels = sparse["level"].to_list() if "level" in sparse.columns else "unknown"
        msg = (
            f"{n_sparse} factor level(s) have fewer than {min_obs} observations. "
            f"CLT CIs will be unreliable. Levels: {levels}"
        )
        warnings.warn(msg, UserWarning, stacklevel=3)
    else:
        msg = f"All factor levels have >= {min_obs} observations."

    return CheckResult(passed=passed, value=float(n_sparse), message=msg)
