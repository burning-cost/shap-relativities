"""
Synthetic UK motor insurance dataset.

Generates realistic policy and claims data for testing and experimentation.
The data generating process (DGP) uses known true parameters so that fitted
GLMs can be validated against the ground truth.

Design decisions
----------------
The DGP is deliberately close to the kind of data a UK pricing actuary would
encounter on a mid-sized personal lines portfolio. Rating factor distributions
match observed UK market patterns: skewed towards younger drivers for frequency,
high NCD penetration in mature books, ABI group distribution weighted towards
the lower-middle bands.

The true frequency model is:
    log(lambda) = log(exposure) + beta_0
                + beta_vehicle_group * vehicle_group_scaled
                + beta_driver_age * f(driver_age)
                + beta_ncd * ncd_years
                + beta_area * area_effect
                + beta_convictions * has_convictions

The true severity model is:
    log(mu) = gamma_0
            + gamma_vehicle_group * vehicle_group_scaled
            + gamma_driver_age * young_driver_flag

These parameters are exported as ``TRUE_FREQ_PARAMS`` and ``TRUE_SEV_PARAMS``
so test code can check GLM recovery.
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta
from typing import Final

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# True DGP parameters — these are what a correctly specified GLM should recover
# ---------------------------------------------------------------------------

TRUE_FREQ_PARAMS: Final[dict[str, float]] = {
    "intercept": -3.2,          # log baseline frequency; portfolio average ~10% per year
    "vehicle_group": 0.025,     # per ABI group unit (1-50)
    "driver_age_young": 0.55,   # additional log-frequency for drivers under 25
    "driver_age_old": 0.30,     # additional log-frequency for drivers over 70
    "ncd_years": -0.12,         # per year of NCD — strong inverse relationship
    "area_B": 0.10,
    "area_C": 0.20,
    "area_D": 0.35,
    "area_E": 0.50,
    "area_F": 0.65,
    "has_convictions": 0.45,    # any SP30/CU80 etc.
}

TRUE_SEV_PARAMS: Final[dict[str, float]] = {
    "intercept": 7.8,           # log baseline severity ~£2440
    "vehicle_group": 0.018,     # per ABI group unit
    "driver_age_young": 0.25,   # young drivers have higher severity too
}

# ABI area bands — loosely mapped to urban density / theft / accident rates
# A = rural/low risk (e.g. parts of Scotland), F = inner city high risk
AREA_BANDS: Final[list[str]] = ["A", "B", "C", "D", "E", "F"]

# Approximate distribution of UK motor policies by area band
AREA_PROBS: Final[list[float]] = [0.12, 0.18, 0.25, 0.22, 0.14, 0.09]


def _driver_age_effect(ages: np.ndarray) -> np.ndarray:
    """
    Non-linear driver age effect on log-frequency.

    Young drivers (<25) have elevated frequency. Very old drivers (70+) also
    show elevated frequency. Mid-range (25-70) is the base.
    """
    effect = np.zeros(len(ages))
    effect[ages < 25] = TRUE_FREQ_PARAMS["driver_age_young"]
    effect[ages >= 70] = TRUE_FREQ_PARAMS["driver_age_old"]
    # Smooth transition: 25-30 blend down from young effect
    blend_mask = (ages >= 25) & (ages < 30)
    blend_factor = (30 - ages[blend_mask]) / 5.0
    effect[blend_mask] = TRUE_FREQ_PARAMS["driver_age_young"] * blend_factor
    return effect


def _generate_policies(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate the policy characteristics table.

    All policies span a 5-year window (2019-2023) with realistic inception
    date spread. Exposure is not all 1.0: ~8% are cancellations (short-term),
    ~5% are mid-term inceptions in the last year of the window.
    """
    # Policy dates: 5 policy years, 2019-2023
    # Inception dates spread across all 5 years
    base_date = date(2019, 1, 1)
    total_days = 5 * 365  # rough span

    inception_days = rng.integers(0, total_days, size=n)
    inception_dates = [base_date + timedelta(days=int(d)) for d in inception_days]

    # Standard term: 12 months (annual policy). ~8% cancel early.
    is_cancellation = rng.random(n) < 0.08
    cancel_fraction = rng.uniform(0.05, 0.90, n)  # cancelled at X% through term

    expiry_dates = []
    for i, inc in enumerate(inception_dates):
        # 12-month policy
        try:
            full_expiry = date(inc.year + 1, inc.month, inc.day)
        except ValueError:
            # 29 Feb edge case
            full_expiry = date(inc.year + 1, inc.month, inc.day - 1)

        if is_cancellation[i]:
            term_days = (full_expiry - inc).days
            actual_days = max(1, int(term_days * cancel_fraction[i]))
            expiry_dates.append(inc + timedelta(days=actual_days))
        else:
            expiry_dates.append(full_expiry)

    # Driver age: realistic UK motor book distribution
    # Weighted towards 30-60 bracket (bulk of book), long tail young and old
    driver_ages = np.concatenate([
        rng.integers(17, 25, size=int(n * 0.12)),   # young drivers ~12%
        rng.integers(25, 40, size=int(n * 0.30)),   # young adult
        rng.integers(40, 60, size=int(n * 0.35)),   # middle-aged
        rng.integers(60, 75, size=int(n * 0.18)),   # older
        rng.integers(75, 86, size=int(n * 0.05)),   # elderly
    ])
    driver_ages = driver_ages[:n]
    rng.shuffle(driver_ages)

    # Driver experience: correlated with age but not perfectly
    # Can't have more experience than years since 17
    max_exp = np.clip(driver_ages - 17, 0, 40).astype(int)
    driver_experience = np.array([
        rng.integers(0, max(1, m + 1)) for m in max_exp
    ])

    # NCD years: inversely correlated with age (older drivers have more NCD)
    # UK scale: 0-5, with 5+ treated as max
    # Young drivers mostly 0-2, experienced drivers 3-5
    ncd_max = np.clip(driver_experience // 2, 0, 5).astype(int)
    ncd_years = np.array([
        rng.integers(0, max(1, m + 1)) for m in ncd_max
    ])

    ncd_protected = (ncd_years >= 4) & (rng.random(n) < 0.35)

    # Conviction points: small minority, correlated with young drivers
    conviction_probs = np.where(driver_ages < 30, 0.12, 0.04)
    has_convictions = rng.random(n) < conviction_probs
    conviction_points = np.where(
        has_convictions,
        rng.choice([3, 3, 3, 6, 6, 9], size=n),  # SP30 most common
        0
    )

    # Annual mileage: log-normal, ~10,000 miles mean
    annual_mileage = np.clip(
        rng.lognormal(mean=9.0, sigma=0.5, size=n).astype(int),
        2000, 30000
    )

    # Vehicle age: 0-20, skewed towards 3-10 year range
    vehicle_age = np.clip(
        rng.gamma(shape=2.5, scale=3.0, size=n).astype(int),
        0, 20
    )

    # ABI vehicle group: 1-50, weighted towards groups 15-35
    vehicle_group = np.clip(
        rng.normal(loc=25, scale=12, size=n).astype(int),
        1, 50
    )

    # Area: A-F with realistic distribution
    area = rng.choice(AREA_BANDS, size=n, p=AREA_PROBS)

    # Occupation class: 1-5 (1=low risk professional, 5=high risk)
    occupation_class = rng.choice(
        [1, 2, 3, 4, 5],
        size=n,
        p=[0.20, 0.30, 0.28, 0.15, 0.07]
    )

    # Policy type: Comp vs TPFT
    # Younger/lower-value cars more likely TPFT
    tpft_prob = np.where(driver_ages < 25, 0.35, 0.15)
    policy_type = np.where(rng.random(n) < tpft_prob, "TPFT", "Comp")

    df = pd.DataFrame({
        "inception_date": inception_dates,
        "expiry_date": expiry_dates,
        "vehicle_age": vehicle_age,
        "vehicle_group": vehicle_group,
        "driver_age": driver_ages,
        "driver_experience": driver_experience,
        "ncd_years": ncd_years,
        "ncd_protected": ncd_protected,
        "conviction_points": conviction_points,
        "annual_mileage": annual_mileage,
        "area": area,
        "occupation_class": occupation_class,
        "policy_type": policy_type,
    })

    return df


def _calculate_earned_exposure(df: pd.DataFrame) -> pd.Series:
    """
    Calculate earned exposure in years for each policy row.

    Simple single-year version used internally during dataset generation.
    The full multi-year splitting is handled by external tooling.
    """
    days = (
        pd.to_datetime(df["expiry_date"]) - pd.to_datetime(df["inception_date"])
    ).dt.days
    return (days / 365.25).clip(lower=0.0)


def _generate_claims(
    df: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.Series, pd.Series]:
    """
    Generate claim counts and incurred amounts from the true DGP.

    Frequency: Poisson with log-linear predictor (see TRUE_FREQ_PARAMS).
    Severity: Gamma with log-linear predictor (see TRUE_SEV_PARAMS), conditional
    on at least one claim.

    Returns
    -------
    claim_count : pd.Series of int
    incurred : pd.Series of float (0.0 where claim_count == 0)
    """
    n = len(df)

    # --- Frequency predictor ---
    log_lambda = (
        TRUE_FREQ_PARAMS["intercept"]
        + TRUE_FREQ_PARAMS["vehicle_group"] * df["vehicle_group"].values
        + _driver_age_effect(df["driver_age"].values)
        + TRUE_FREQ_PARAMS["ncd_years"] * df["ncd_years"].values
        + TRUE_FREQ_PARAMS["has_convictions"] * (df["conviction_points"].values > 0).astype(float)
    )

    # Area effect
    area_effect = np.zeros(n)
    for band, key in [("B", "area_B"), ("C", "area_C"), ("D", "area_D"),
                      ("E", "area_E"), ("F", "area_F")]:
        area_effect[df["area"].values == band] = TRUE_FREQ_PARAMS[key]
    log_lambda += area_effect

    # Exposure offset: policies with less than a full year have lower claim count
    exposure = df["exposure"].values
    log_lambda += np.log(np.clip(exposure, 1e-6, None))

    # Poisson claim counts
    lambda_vals = np.exp(log_lambda)
    claim_count = rng.poisson(lambda_vals)

    # --- Severity predictor ---
    log_mu = (
        TRUE_SEV_PARAMS["intercept"]
        + TRUE_SEV_PARAMS["vehicle_group"] * df["vehicle_group"].values
        + TRUE_SEV_PARAMS["driver_age_young"] * (df["driver_age"].values < 25).astype(float)
    )

    # Gamma severity: shape parameter (dispersion ~0.5, shape ~2)
    # Gamma mean = exp(log_mu), shape = 2 → CV ≈ 0.71
    gamma_shape = 2.0
    gamma_mean = np.exp(log_mu)
    gamma_scale = gamma_mean / gamma_shape

    # Only generate severity where there are claims
    has_claims = claim_count > 0
    incurred = np.zeros(n)

    if has_claims.any():
        # For policies with multiple claims, sum of Gamma(shape, scale) per claim
        # = Gamma(count * shape, scale) when independent and same parameters
        # We simulate per-claim for accuracy
        for i in np.where(has_claims)[0]:
            per_claim = rng.gamma(
                shape=gamma_shape,
                scale=gamma_scale[i],
                size=int(claim_count[i])
            )
            incurred[i] = per_claim.sum()

    return pd.Series(claim_count, dtype=int), pd.Series(incurred, dtype=float)


def load_motor(
    n_policies: int = 50_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load a synthetic UK motor insurance dataset.

    Generates ``n_policies`` rows with realistic UK personal lines motor
    characteristics and simulated claims from a known data generating process.
    Because the true parameters are known (see ``TRUE_FREQ_PARAMS`` and
    ``TRUE_SEV_PARAMS``), you can fit GLMs and validate coefficient recovery.

    Parameters
    ----------
    n_policies : int
        Number of policies to generate. Default 50,000 gives stable GLM
        estimates. Use 5,000-10,000 for quick tests.
    seed : int
        Random seed for reproducibility. Changing this seed gives a different
        but equally valid synthetic portfolio.

    Returns
    -------
    pd.DataFrame
        One row per policy with columns:

        - ``policy_id`` : int — sequential identifier
        - ``inception_date`` : date — policy start
        - ``expiry_date`` : date — policy end (may be < 12 months for cancellations)
        - ``accident_year`` : int — year of inception (used for cohort splits)
        - ``vehicle_age`` : int — 0-20 years
        - ``vehicle_group`` : int — ABI group 1-50
        - ``driver_age`` : int — 17-85
        - ``driver_experience`` : int — years licensed
        - ``ncd_years`` : int — 0-5 (UK NCD scale)
        - ``ncd_protected`` : bool
        - ``conviction_points`` : int — total endorsement points
        - ``annual_mileage`` : int — 2,000-30,000 miles
        - ``area`` : str — ABI area band A-F
        - ``occupation_class`` : int — 1-5
        - ``policy_type`` : str — 'Comp' or 'TPFT'
        - ``claim_count`` : int — number of claims in period
        - ``incurred`` : float — total incurred cost (0.0 if no claims)
        - ``exposure`` : float — earned years (< 1.0 for cancellations/short periods)

    Examples
    --------
    >>> df = load_motor(n_policies=10_000, seed=0)
    >>> df.shape[0]
    10000
    >>> df["claim_count"].mean()  # roughly 6-8% claim rate
    # ~0.07

    Notes
    -----
    The data generating process is documented in ``TRUE_FREQ_PARAMS`` and
    ``TRUE_SEV_PARAMS``. Policies span accident years 2019-2023. Exposure varies:
    cancellations and mid-term inceptions create sub-annual exposures.

    There are no missing values. Real data always has missing values; use this
    dataset for algorithm testing, not for missing-data workflows.
    """
    rng = np.random.default_rng(seed)

    df = _generate_policies(n_policies, rng)

    # Exposure must be calculated before claim generation (frequency uses it)
    df["exposure"] = _calculate_earned_exposure(df)

    # Accident year is the inception year for this dataset
    df["accident_year"] = pd.to_datetime(df["inception_date"]).dt.year

    # Generate claims from the true DGP
    df["claim_count"], df["incurred"] = _generate_claims(df, rng)

    # Add policy_id and reorder columns to match the schema
    df.insert(0, "policy_id", np.arange(1, n_policies + 1))

    column_order = [
        "policy_id",
        "inception_date",
        "expiry_date",
        "accident_year",
        "vehicle_age",
        "vehicle_group",
        "driver_age",
        "driver_experience",
        "ncd_years",
        "ncd_protected",
        "conviction_points",
        "annual_mileage",
        "area",
        "occupation_class",
        "policy_type",
        "claim_count",
        "incurred",
        "exposure",
    ]
    df = df[column_order].copy()

    # Enforce dtypes
    df["policy_id"] = df["policy_id"].astype(int)
    df["inception_date"] = pd.to_datetime(df["inception_date"]).dt.date
    df["expiry_date"] = pd.to_datetime(df["expiry_date"]).dt.date
    df["accident_year"] = df["accident_year"].astype(int)
    df["vehicle_age"] = df["vehicle_age"].astype(int)
    df["vehicle_group"] = df["vehicle_group"].astype(int)
    df["driver_age"] = df["driver_age"].astype(int)
    df["driver_experience"] = df["driver_experience"].astype(int)
    df["ncd_years"] = df["ncd_years"].astype(int)
    df["ncd_protected"] = df["ncd_protected"].astype(bool)
    df["conviction_points"] = df["conviction_points"].astype(int)
    df["annual_mileage"] = df["annual_mileage"].astype(int)
    df["claim_count"] = df["claim_count"].astype(int)
    df["incurred"] = df["incurred"].astype(float)
    df["exposure"] = df["exposure"].astype(float)

    return df.reset_index(drop=True)
