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
    SE = shap_std / sqrt(n_obs), so CI = exp(mean_shap ± z * SE - norm_shap).
    These quantify data uncertainty only — they do not reflect model uncertainty
    from the GBM fitting process. Use bootstrap CIs across model refits for
    full uncertainty, but that is expensive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats


def normalise_base_level(
    result: pd.DataFrame,
    base_level: str | float | int,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """
    Normalise so the base level receives relativity = 1.0.

    Parameters
    ----------
    result : pd.DataFrame
        Output from aggregate_categorical or aggregate_continuous.
        Must have columns: mean_shap, shap_std, n_obs, exposure_weight, level.
    base_level : str | float | int
        The level value to use as the reference (relativity = 1.0).
    ci_level : float
        Two-sided confidence level, e.g. 0.95 for 95% intervals.

    Returns
    -------
    pd.DataFrame
        Input with added columns: relativity, lower_ci, upper_ci.

    Raises
    ------
    ValueError
        If base_level is not found in the result's level column.
    """
    result = result.copy()

    base_mask = result["level"].astype(str) == str(base_level)
    if not base_mask.any():
        raise ValueError(
            f"Base level '{base_level}' not found in levels: "
            f"{result['level'].tolist()}"
        )

    base_shap = result.loc[base_mask, "mean_shap"].values[0]
    result["relativity"] = np.exp(result["mean_shap"] - base_shap)

    z = scipy.stats.norm.ppf((1 + ci_level) / 2)
    se = result["shap_std"] / np.sqrt(result["n_obs"].clip(lower=1))
    result["lower_ci"] = np.exp(result["mean_shap"] - z * se - base_shap)
    result["upper_ci"] = np.exp(result["mean_shap"] + z * se - base_shap)

    return result


def normalise_mean(
    result: pd.DataFrame,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """
    Normalise so the exposure-weighted mean relativity = 1.0.

    Parameters
    ----------
    result : pd.DataFrame
        Output from aggregate_categorical or aggregate_continuous.
    ci_level : float
        Two-sided confidence level.

    Returns
    -------
    pd.DataFrame
        Input with added columns: relativity, lower_ci, upper_ci.
    """
    result = result.copy()

    portfolio_mean_shap = np.average(
        result["mean_shap"], weights=result["exposure_weight"]
    )
    result["relativity"] = np.exp(result["mean_shap"] - portfolio_mean_shap)

    z = scipy.stats.norm.ppf((1 + ci_level) / 2)
    se = result["shap_std"] / np.sqrt(result["n_obs"].clip(lower=1))
    result["lower_ci"] = np.exp(result["mean_shap"] - z * se - portfolio_mean_shap)
    result["upper_ci"] = np.exp(result["mean_shap"] + z * se - portfolio_mean_shap)

    return result
