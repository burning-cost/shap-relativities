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
    SE = shap_std / sqrt(n_eff), where n_eff = sum(w)^2 / sum(w^2) is the
    effective sample size that accounts for non-uniform exposure weights.
    Using raw n_obs (count of observations) understates SE when weights vary —
    which is systematic for UK motor books where short-term cancellations are
    concentrated in high-risk segments.

    For base_level normalisation the CI also includes the estimation variance
    of the base level: se_combined = sqrt(se_L^2 + se_base^2).

    These CIs quantify data uncertainty only — they do not reflect model
    uncertainty from the GBM fitting process.

All functions operate on Polars DataFrames and return Polars DataFrames.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import scipy.stats


def _effective_se(result: pl.DataFrame) -> pl.Expr:
    """
    Polars expression for the standard error of mean_shap using effective
    sample size.

    n_eff = sum(w)^2 / sum(w^2) = exposure_weight^2 / wsq_weight

    This corrects for non-uniform exposure weights. For equal weights (all
    w=1), n_eff = n_obs exactly. For mixed weights, n_eff < n_obs, giving
    wider and more honest CIs.

    The wsq_weight column is produced by aggregate_categorical. If it is
    absent (e.g. a manually constructed DataFrame), we fall back to raw n_obs.
    """
    if "wsq_weight" in result.columns:
        # n_eff = exposure_weight^2 / wsq_weight, clipped at 1 to avoid division by zero
        n_eff = (
            pl.col("exposure_weight") ** 2
            / pl.col("wsq_weight").clip(lower_bound=1e-12)
        ).clip(lower_bound=1.0)
    else:
        # Fallback: raw observation count (equal-weight assumption)
        n_eff = pl.col("n_obs").cast(pl.Float64).clip(lower_bound=1.0)

    return pl.col("shap_std") / n_eff.sqrt()


def _base_se(base_rows: pl.DataFrame) -> float:
    """
    Standard error of the base level's mean_shap estimate, as a Python float.

    Used by normalise_base_level to include base level estimation uncertainty
    in the CI of all non-base levels.
    """
    shap_std = float(base_rows["shap_std"][0])
    if "wsq_weight" in base_rows.columns:
        exposure_w = float(base_rows["exposure_weight"][0])
        wsq_w = float(base_rows["wsq_weight"][0])
        if wsq_w > 0:
            n_eff = max(1.0, (exposure_w ** 2) / wsq_w)
        else:
            n_eff = 1.0
    else:
        n_obs = float(base_rows["n_obs"][0])
        n_eff = max(1.0, n_obs)
    return shap_std / math.sqrt(n_eff)


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
            aggregate_categorical). If wsq_weight is present it is used for
            effective sample size; otherwise raw n_obs is used.
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

    base_shap = float(base_rows["mean_shap"][0])
    base_se = _base_se(base_rows)

    z = scipy.stats.norm.ppf((1 + ci_level) / 2)

    # SE per level using effective sample size
    se_L = _effective_se(result)

    # P0-1 fix: CI must account for uncertainty in BOTH mean_shap_L and base_shap.
    # Var(mean_L - base) = Var(mean_L) + Var(base), so
    # se_combined = sqrt(se_L^2 + se_base^2)
    se_combined = (se_L ** 2 + pl.lit(base_se ** 2)).sqrt()

    result = result.with_columns([
        (pl.col("mean_shap") - base_shap).exp().alias("relativity"),
        (pl.col("mean_shap") - z * se_combined - base_shap).exp().alias("lower_ci"),
        (pl.col("mean_shap") + z * se_combined - base_shap).exp().alias("upper_ci"),
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
            If wsq_weight is present it is used for effective sample size;
            otherwise raw n_obs is used.
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

    # P0-3 fix: use effective sample size SE
    se = _effective_se(result)

    result = result.with_columns([
        (pl.col("mean_shap") - portfolio_mean_shap).exp().alias("relativity"),
        (pl.col("mean_shap") - z * se - portfolio_mean_shap).exp().alias("lower_ci"),
        (pl.col("mean_shap") + z * se - portfolio_mean_shap).exp().alias("upper_ci"),
    ])

    return result
