"""
Normalisation and confidence interval computation for SHAP relativities.

Two normalisation modes are supported:

base_level
    The named base level for each feature gets relativity = 1.0. This matches
    GLM convention where exp(beta_base) cancels out. Only applicable to
    categorical features; for continuous features use mean normalisation.

mean
    The exposure-weighted mean relativity across all levels = 1.0. Useful for
    portfolio benchmarking and for continuous features.

CLT confidence intervals
    SE = shap_std / sqrt(n_obs), so CI = exp(mean_shap +/- z * SE - norm_shap).
    These quantify data uncertainty only - they do not reflect model uncertainty
    from the GBM fitting process. Use bootstrap CIs across model refits for
    full uncertainty, but that is expensive.

All functions operate on Polars DataFrames and return Polars DataFrames.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import scipy.stats


def normalise_base_level(
    result: pl.DataFrame,
    base_level: str | float | int,
    ci_level: float = 0.95,
) -> pl.DataFrame:
    """
    Normalise so the base level receives relativity = 1.0.

    Args:
        result: Output from aggregate_categorical or aggregate_continuous.
            Must have columns: mean_shap, shap_std, n_obs, exposure_weight,
            level. The level column contains string values (as produced by
            aggregate_categorical).
        base_level: The level value to use as the reference (relativity = 1.0).
        ci_level: Two-sided confidence level, e.g. 0.95 for 95% intervals.

    Returns:
        Input DataFrame with added columns: relativity, lower_ci, upper_ci.

    Raises:
        ValueError: If base_level is not found in the result's level column.
    """
    base_key = str(base_level)
    base_rows = result.filter(pl.col("level") == base_key)
    if len(base_rows) == 0:
        levels = result["level"].to_list()
        raise ValueError(
            f"Base level '{base_level}' not found in levels: {levels}"
        )

    base_shap = base_rows["mean_shap"][0]

    z = scipy.stats.norm.ppf((1 + ci_level) / 2)

    se = pl.col("shap_std") / pl.col("n_obs").clip(lower_bound=1).sqrt()

    result = result.with_columns([
        (pl.col("mean_shap") - base_shap).exp().alias("relativity"),
        (pl.col("mean_shap") - z * se - base_shap).exp().alias("lower_ci"),
        (pl.col("mean_shap") + z * se - base_shap).exp().alias("upper_ci"),
    ])

    return result


def normalise_mean(
    result: pl.DataFrame,
    ci_level: float = 0.95,
) -> pl.DataFrame:
    """
    Normalise so the exposure-weighted mean relativity = 1.0.

    Args:
        result: Output from aggregate_categorical or aggregate_continuous.
        ci_level: Two-sided confidence level.

    Returns:
        Input DataFrame with added columns: relativity, lower_ci, upper_ci.
    """
    # Exposure-weighted mean of mean_shap
    total_weight = result["exposure_weight"].sum()
    if total_weight == 0:
        portfolio_mean_shap = 0.0
    else:
        portfolio_mean_shap = float(
            (result["mean_shap"] * result["exposure_weight"]).sum() / total_weight
        )

    z = scipy.stats.norm.ppf((1 + ci_level) / 2)

    se = pl.col("shap_std") / pl.col("n_obs").clip(lower_bound=1).sqrt()

    result = result.with_columns([
        (pl.col("mean_shap") - portfolio_mean_shap).exp().alias("relativity"),
        (pl.col("mean_shap") - z * se - portfolio_mean_shap).exp().alias("lower_ci"),
        (pl.col("mean_shap") + z * se - portfolio_mean_shap).exp().alias("upper_ci"),
    ])

    return result
