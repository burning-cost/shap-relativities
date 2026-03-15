"""
Aggregation of SHAP values by feature level.

Categorical features are grouped by level and exposure-weighted statistics
are computed. Continuous features are returned as per-observation points
suitable for smoothing downstream.

All output is Polars DataFrames. The caller (SHAPRelativities) converts
the combined result to pandas only if a downstream library requires it.
"""

from __future__ import annotations

import warnings

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
    value, weighted standard deviation, observation count, total exposure
    weight, and sum of squared weights (used downstream for effective sample
    size in CI computation). These statistics are used downstream for
    normalisation and CI computation.

    Levels with zero total weight are excluded from the result and a warning
    is emitted. A zero-weight level would produce NaN mean_shap which would
    silently poison all relativities if the level is selected as the base.

    Args:
        feature: Feature name (used as a column label in the output).
        feature_values: Per-observation feature values (strings or integers).
        shap_col: Per-observation SHAP values in log space.
        weights: Per-observation exposure weights (e.g. earned years).

    Returns:
        Polars DataFrame with columns: feature, level, mean_shap, shap_std,
        n_obs, exposure_weight, wsq_weight. One row per unique level with
        positive total weight.
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
            # Sum of squared weights — needed for effective sample size
            # n_eff = sum(w)^2 / sum(w^2)
            (pl.col("w") ** 2).sum().alias("wsq_weight"),
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

    # P1-1: guard against zero-weight levels — they produce NaN mean_shap
    # which silently poisons all relativities if used as the base level.
    zero_weight = agg.filter(pl.col("exposure_weight") == 0.0)
    if len(zero_weight) > 0:
        bad_levels = zero_weight["level"].to_list()
        warnings.warn(
            f"Feature '{feature}': level(s) {bad_levels} have zero total exposure "
            "weight and will be excluded from relativities. "
            "Check your weights array for observations with exposure=0.",
            UserWarning,
            stacklevel=2,
        )
        agg = agg.filter(pl.col("exposure_weight") > 0.0)

    result = agg.with_columns(pl.lit(feature).alias("feature"))

    # Reorder columns
    return result.select(
        ["feature", "level", "mean_shap", "shap_std", "n_obs", "exposure_weight", "wsq_weight"]
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

    For continuous features each observation is its own group, so wsq_weight
    equals w^2 and n_eff = w^2 / w^2 = 1 (which is correct — each point is
    a single observation).

    Args:
        feature: Feature name.
        feature_values: Per-observation feature values (floats or ints).
        shap_col: Per-observation SHAP values in log space.
        weights: Per-observation exposure weights.

    Returns:
        Polars DataFrame with columns: feature, level (float), mean_shap,
        shap_std (zeros), n_obs (ones), exposure_weight, wsq_weight.
    """
    n = len(feature_values)
    w = weights.astype(float)
    # Cast level to str and n_obs to UInt32 so the schema matches
    # aggregate_categorical() output. This allows pl.concat(how="diagonal")
    # to stack both feature types without type conflicts.
    return pl.DataFrame({
        "feature": [feature] * n,
        "level": feature_values.astype(str),
        "mean_shap": shap_col.astype(float),
        "shap_std": np.zeros(n),
        "n_obs": np.ones(n, dtype=np.uint32),
        "exposure_weight": w,
        # For a single observation: wsq_weight = w^2, n_eff = w^2/w^2 = 1
        "wsq_weight": w ** 2,
    })
