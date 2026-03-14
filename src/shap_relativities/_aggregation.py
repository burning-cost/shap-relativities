"""
Aggregation of SHAP values by feature level.

Categorical features are grouped by level and exposure-weighted statistics
are computed. Continuous features are returned as per-observation points
suitable for smoothing downstream.

All output is Polars DataFrames. The caller (SHAPRelativities) converts
the combined result to pandas only if a downstream library requires it.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def aggregate_categorical(
    feature: str,
    feature_values: np.ndarray,
    shap_col: np.ndarray,
    weights: np.ndarray,
) -> pl.DataFrame:
    """
    Aggregate SHAP values by categorical level using exposure weights.

    For each unique level of the feature, computes the weighted mean SHAP
    value, weighted standard deviation, observation count, and total exposure
    weight. These statistics are used downstream for normalisation and CI
    computation.

    Args:
        feature: Feature name (used as a column label in the output).
        feature_values: Per-observation feature values (strings or integers).
        shap_col: Per-observation SHAP values in log space.
        weights: Per-observation exposure weights (e.g. earned years).

    Returns:
        Polars DataFrame with columns: feature, level, mean_shap, shap_std,
        n_obs, exposure_weight. One row per unique level.
    """
    # Build a Polars frame for groupby. Store levels as strings internally
    # so that mixed-type levels (int codes, string labels) concatenate cleanly.
    df = pl.DataFrame({
        "level": feature_values.astype(str),
        "shap": shap_col.astype(float),
        "w": weights.astype(float),
    })

    # Weighted mean and weighted variance per level
    agg = (
        df.group_by("level")
        .agg([
            pl.len().alias("n_obs"),
            pl.col("w").sum().alias("exposure_weight"),
            # Weighted mean: sum(w * x) / sum(w)
            (pl.col("w") * pl.col("shap")).sum().alias("_wsum"),
            (pl.col("w")).sum().alias("_wtot"),
            # Weighted sum of squares for variance: sum(w * x^2)
            (pl.col("w") * pl.col("shap") ** 2).sum().alias("_wsq"),
        ])
        .with_columns([
            (pl.col("_wsum") / pl.col("_wtot")).alias("mean_shap"),
        ])
        .with_columns([
            # Weighted variance: E[x^2] - E[x]^2
            (
                (pl.col("_wsq") / pl.col("_wtot")) - pl.col("mean_shap") ** 2
            ).clip(lower_bound=0.0).sqrt().alias("shap_std"),
        ])
        .drop(["_wsum", "_wtot", "_wsq"])
        .sort("level")
    )

    result = agg.with_columns(pl.lit(feature).alias("feature"))

    # Reorder columns
    return result.select(
        ["feature", "level", "mean_shap", "shap_std", "n_obs", "exposure_weight"]
    )


def aggregate_continuous(
    feature: str,
    feature_values: np.ndarray,
    shap_col: np.ndarray,
    weights: np.ndarray,
) -> pl.DataFrame:
    """
    Return per-observation SHAP values for a continuous feature.

    Unlike categorical aggregation, continuous features are not grouped -
    each observation is returned as its own row. Smoothing and binning are
    handled separately (see extract_continuous_curve on the main class).

    Args:
        feature: Feature name.
        feature_values: Per-observation feature values (floats or ints).
        shap_col: Per-observation SHAP values in log space.
        weights: Per-observation exposure weights.

    Returns:
        Polars DataFrame with columns: feature, level (float), mean_shap,
        shap_std (zeros), n_obs (ones), exposure_weight.
    """
    n = len(feature_values)
    # Cast level to str and n_obs to UInt32 so the schema matches
    # aggregate_categorical() output. This allows pl.concat(how="diagonal")
    # to stack both feature types without type conflicts.
    return pl.DataFrame({
        "feature": [feature] * n,
        "level": feature_values.astype(str),
        "mean_shap": shap_col.astype(float),
        "shap_std": np.zeros(n),
        "n_obs": np.ones(n, dtype=np.uint32),
        "exposure_weight": weights.astype(float),
    })
