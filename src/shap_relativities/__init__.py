"""
SHAP Relativities - actuarial-grade multiplicative relativities from tree models.

Converts a trained GBM's SHAP values into the same format as GLM exp(beta)
relativities: a table of (feature, level, relativity) triples where the base
level is 1.0 and relativities multiply together to give the model's expected
prediction.

Typical usage
-------------
>>> from shap_relativities import SHAPRelativities
>>> sr = SHAPRelativities(model, X, exposure=df["exposure"],
...                       categorical_features=["area", "ncd_years"])
>>> sr.fit()
>>> rels = sr.extract_relativities(
...     normalise_to="base_level",
...     base_levels={"area": "A", "ncd_years": 0},
... )

Or use the convenience wrapper for one-liners:

>>> from shap_relativities import extract_relativities
>>> rels = extract_relativities(model, X, exposure=df["exposure"],
...                             categorical_features=["area"])
"""

from __future__ import annotations

from typing import Any

import polars as pl

from ._core import SHAPRelativities

__all__ = ["SHAPRelativities", "extract_relativities"]

__version__ = "0.2.3"


def extract_relativities(
    model: Any,
    X: Any,
    exposure: Any = None,
    categorical_features: list[str] | None = None,
    base_levels: dict[str, str | float | int] | None = None,
    ci_method: str = "clt",
) -> pl.DataFrame:
    """
    One-shot extraction of SHAP relativities from a tree model.

    Wraps SHAPRelativities.fit() and extract_relativities() for cases where
    you don't need the intermediate object.

    Args:
        model: Trained CatBoost model with a log-link objective (Poisson,
            Tweedie, or Gamma). CatBoost is the recommended choice - it handles
            categorical features natively without encoding.
        X: Feature matrix. Accepts a Polars or pandas DataFrame. Polars is
            preferred; pandas is accepted and converted internally.
        exposure: Earned policy years. If None, all observations are equally
            weighted.
        categorical_features: Features to aggregate by level. If None, all
            non-numeric columns are treated as categorical.
        base_levels: Base level for each categorical feature (gets
            relativity = 1.0).
        ci_method: "clt" (default) or "none".

    Returns:
        Polars DataFrame with columns: feature, level, relativity, lower_ci,
        upper_ci, mean_shap, shap_std, n_obs, exposure_weight.
    """
    sr = SHAPRelativities(model, X, exposure, categorical_features)
    sr.fit()
    return sr.extract_relativities(base_levels=base_levels, ci_method=ci_method)
