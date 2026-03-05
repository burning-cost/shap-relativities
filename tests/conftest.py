"""
Shared test fixtures for shap-relativities.

Includes a minimal synthetic motor data generator with known DGP parameters,
so the test suite is fully self-contained and doesn't depend on external packages.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# True DGP parameters — the ground truth that SHAP relativities should recover
# ---------------------------------------------------------------------------

TRUE_FREQ_PARAMS = {
    "intercept": -3.2,
    "vehicle_group": 0.025,
    "driver_age_young": 0.55,
    "driver_age_old": 0.30,
    "ncd_years": -0.12,
    "area_B": 0.10,
    "area_C": 0.20,
    "area_D": 0.35,
    "area_E": 0.50,
    "area_F": 0.65,
    "has_convictions": 0.45,
}

TRUE_SEV_PARAMS = {
    "intercept": 7.8,
    "vehicle_group": 0.018,
    "driver_age_young": 0.25,
}

AREA_BANDS = ["A", "B", "C", "D", "E", "F"]
AREA_PROBS = [0.12, 0.18, 0.25, 0.22, 0.14, 0.09]


def _driver_age_effect(ages: np.ndarray) -> np.ndarray:
    effect = np.zeros(len(ages))
    effect[ages < 25] = TRUE_FREQ_PARAMS["driver_age_young"]
    effect[ages >= 70] = TRUE_FREQ_PARAMS["driver_age_old"]
    blend_mask = (ages >= 25) & (ages < 30)
    blend_factor = (30 - ages[blend_mask]) / 5.0
    effect[blend_mask] = TRUE_FREQ_PARAMS["driver_age_young"] * blend_factor
    return effect


def generate_motor_data(n_policies: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic UK motor data with known DGP.

    Minimal version for testing — produces the columns needed by the test suite.
    """
    rng = np.random.default_rng(seed)

    # Driver age distribution
    driver_ages = np.concatenate([
        rng.integers(17, 25, size=int(n_policies * 0.12)),
        rng.integers(25, 40, size=int(n_policies * 0.30)),
        rng.integers(40, 60, size=int(n_policies * 0.35)),
        rng.integers(60, 75, size=int(n_policies * 0.18)),
        rng.integers(75, 86, size=int(n_policies * 0.05)),
    ])[:n_policies]
    rng.shuffle(driver_ages)

    # NCD years: 0-5
    max_exp = np.clip(driver_ages - 17, 0, 40).astype(int)
    driver_experience = np.array([rng.integers(0, max(1, m + 1)) for m in max_exp])
    ncd_max = np.clip(driver_experience // 2, 0, 5).astype(int)
    ncd_years = np.array([rng.integers(0, max(1, m + 1)) for m in ncd_max])

    # Convictions
    conviction_probs = np.where(driver_ages < 30, 0.12, 0.04)
    has_convictions = rng.random(n_policies) < conviction_probs
    conviction_points = np.where(
        has_convictions, rng.choice([3, 3, 3, 6, 6, 9], size=n_policies), 0
    )

    # Vehicle group: 1-50
    vehicle_group = np.clip(rng.normal(loc=25, scale=12, size=n_policies).astype(int), 1, 50)

    # Area
    area = rng.choice(AREA_BANDS, size=n_policies, p=AREA_PROBS)

    # Exposure: mostly 1.0, ~8% cancellations
    exposure = np.ones(n_policies)
    is_cancel = rng.random(n_policies) < 0.08
    exposure[is_cancel] = rng.uniform(0.05, 0.90, size=is_cancel.sum())

    # Frequency from true DGP
    log_lambda = (
        TRUE_FREQ_PARAMS["intercept"]
        + TRUE_FREQ_PARAMS["vehicle_group"] * vehicle_group
        + _driver_age_effect(driver_ages)
        + TRUE_FREQ_PARAMS["ncd_years"] * ncd_years
        + TRUE_FREQ_PARAMS["has_convictions"] * has_convictions.astype(float)
    )

    area_effect = np.zeros(n_policies)
    for band, key in [("B", "area_B"), ("C", "area_C"), ("D", "area_D"),
                      ("E", "area_E"), ("F", "area_F")]:
        area_effect[area == band] = TRUE_FREQ_PARAMS[key]
    log_lambda += area_effect
    log_lambda += np.log(np.clip(exposure, 1e-6, None))

    claim_count = rng.poisson(np.exp(log_lambda))

    # Severity from true DGP
    log_mu = (
        TRUE_SEV_PARAMS["intercept"]
        + TRUE_SEV_PARAMS["vehicle_group"] * vehicle_group
        + TRUE_SEV_PARAMS["driver_age_young"] * (driver_ages < 25).astype(float)
    )
    gamma_shape = 2.0
    gamma_mean = np.exp(log_mu)
    gamma_scale = gamma_mean / gamma_shape

    incurred = np.zeros(n_policies)
    has_claims = claim_count > 0
    if has_claims.any():
        for i in np.where(has_claims)[0]:
            per_claim = rng.gamma(shape=gamma_shape, scale=gamma_scale[i], size=int(claim_count[i]))
            incurred[i] = per_claim.sum()

    return pd.DataFrame({
        "vehicle_group": vehicle_group,
        "driver_age": driver_ages,
        "ncd_years": ncd_years,
        "conviction_points": conviction_points,
        "area": area,
        "exposure": exposure,
        "claim_count": claim_count,
        "incurred": incurred,
    })
