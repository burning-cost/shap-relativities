"""
Aggregation of SHAP values by feature level.

Categorical features are grouped by level and exposure-weighted statistics
are computed. Continuous features are returned as per-observation points
suitable for smoothing downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_categorical(
    feature: str,
    feature_values: np.ndarray,
    shap_col: np.ndarray,
    weights: np.ndarray,
) -> pd.DataFrame:
    """
    Aggregate SHAP values by categorical level using exposure weights.

    For each unique level of the feature, computes the weighted mean SHAP
    value, weighted standard deviation, observation count, and total exposure
    weight. These statistics are used downstream for normalisation and CI
    computation.

    Parameters
    ----------
    feature : str
        Feature name (used as a column label in the output).
    feature_values : np.ndarray
        Per-observation feature values (strings or integers).
    shap_col : np.ndarray
        Per-observation SHAP values in log space.
    weights : np.ndarray
        Per-observation exposure weights (e.g. earned years).

    Returns
    -------
    pd.DataFrame
        Columns: feature, level, mean_shap, shap_std, n_obs, exposure_weight.
        One row per unique level.
    """
    df = pd.DataFrame({
        "level": feature_values,
        "shap": shap_col,
        "w": weights,
    })

    def _stats(g: pd.DataFrame) -> pd.Series:
        w = g["w"].values
        s = g["shap"].values
        w_mean = np.average(s, weights=w)
        w_var = np.average((s - w_mean) ** 2, weights=w)
        return pd.Series({
            "mean_shap": w_mean,
            "shap_std": np.sqrt(w_var),
            "n_obs": float(len(g)),
            "exposure_weight": float(w.sum()),
        })

    result = df.groupby("level", sort=True).apply(_stats).reset_index()
    result.insert(0, "feature", feature)
    return result


def aggregate_continuous(
    feature: str,
    feature_values: np.ndarray,
    shap_col: np.ndarray,
    weights: np.ndarray,
) -> pd.DataFrame:
    """
    Return per-observation SHAP values for a continuous feature.

    Unlike categorical aggregation, continuous features are not grouped —
    each observation is returned as its own row. Smoothing and binning are
    handled separately (see extract_continuous_curve on the main class).

    Parameters
    ----------
    feature : str
        Feature name.
    feature_values : np.ndarray
        Per-observation feature values (floats or ints).
    shap_col : np.ndarray
        Per-observation SHAP values in log space.
    weights : np.ndarray
        Per-observation exposure weights.

    Returns
    -------
    pd.DataFrame
        Columns: feature, level (numeric), mean_shap (= shap, no averaging),
        n_obs (= 1), exposure_weight (= weight).
    """
    return pd.DataFrame({
        "feature": feature,
        "level": feature_values.astype(float),
        "mean_shap": shap_col,
        "shap_std": np.zeros(len(shap_col)),
        "n_obs": np.ones(len(shap_col)),
        "exposure_weight": weights,
    })
