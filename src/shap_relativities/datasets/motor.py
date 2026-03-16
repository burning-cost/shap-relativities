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

The true frequency model is::

    log(lambda) = log(exposure) + beta_0
                + beta_vehicle_group * vehicle_group_scaled
                + beta_driver_age * f(driver_age)
                + beta_ncd * ncd_years
                + beta_area * area_effect
                + beta_convictions * has_convictions

The true severity model is::

    log(mu) = gamma_0
            + gamma_vehicle_group * vehicle_group_scaled
            + gamma_driver_age * young_driver_flag

These parameters are exported as ``TRUE_FREQ_PARAMS`` and ``TRUE_SEV_PARAMS``
so test code can check GLM recovery.

Output
------
load_motor() returns a Polars DataFrame. All data manipulation uses Polars
internally; numpy is used only for numerical generation via numpy's random
number generator.
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta
from typing import Final

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# True DGP parameters -- these are what a correctly specified GLM should recover
# ---------------------------------------------------------------------------

TRUE_FREQ_PARAMS: Final[dict[str, float]] = {
    "intercept": -3.2,          # log baseline frequency; portfolio average ~10% per year
    "vehicle_group": 0.025,     # per ABI group unit (1-50)
    "driver_age_young": 0.55,   # additional log-frequency for drivers under 25
    "driver_age_old": 0.30,     # additional log-frequency for drivers over 70
    "ncd_years": -0.12,         # per year of NCD; strong inverse relationship
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

# ABI area bands -- loosely mapped to urban density / theft / accident rates
# A = rural/low risk (e.g. parts of Scotland), F = inner city high risk
AREA_BANDS: Final[list[str]] = ["A", "B", "C", "D", "E", "F"]

# Approximate distribution of UK motor policies by area band
AREA_PROBS: Final[list[float]] = [0.12, 0.18, 0.25, 0.22, 0.14, 0.09]


def _driver_age_effect(ages: np.ndarray) -> np.ndarray:
    """
    Non-linear driver age effect on log-frequency.

    Young drivers (<25) have elevated frequency. Very old drivers (70+) also
    show elevated frequency. Mid-range (25-70) is the base.

    Args:
        ages: Array of driver ages.

    Returns:
        Array of log-frequency offsets.
    """
    effect = np.zeros(len(ages))
    effect[ages < 25] = TRUE_FREQ_PARAMS["driver_age_young"]
    effect[ages >= 70] = TRUE_FREQ_PARAMS["driver_age_old"]
    # Smooth transition: 25-30 blend down from young effect
    blend_mask = (ages >= 25) & (ages < 30)
    blend_factor = (30 - ages[blend_mask]) / 5.0
    effect[blend_mask] = TRUE_FREQ_PARAMS["driver_age_young"] * blend_factor
    return effect


def _generate_policies(n: int, rng: np.random.Generator) -> dict:
    """
    Generate the policy characteristics table as a dict of arrays.

    All policies span a 5-year window (2019-2023) with realistic inception
    date spread. Exposure is not all 1.0: ~8% are cancellations (short-term),
    ~5% are mid-term inceptions in the last year of the window.

    Args:
        n: Number of policies to generate.
        rng: Numpy random generator instance.

    Returns:
        Dict of column name to array/list.
    """
    # Policy dates: 5 policy years, 2019-2023
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
    # P1-3 fix: int() truncation of proportional sizes can make the total < n.
    # At small n (e.g. n=10): sizes may sum to 8, leaving driver_ages[:n] to
    # silently return fewer than n elements. Add the remainder to segment [1]
    # (young adult) so the total is always exactly n.
    _age_props = [0.12, 0.30, 0.35, 0.18, 0.05]
    _age_ranges = [(17, 25), (25, 40), (40, 60), (60, 75), (75, 86)]
    sizes = [int(n * p) for p in _age_props]
    sizes[1] += n - sum(sizes)  # absorb rounding remainder into young-adult segment
    driver_ages = np.concatenate([
        rng.integers(lo, hi, size=s)
        for (lo, hi), s in zip(_age_ranges, sizes)
    ])
    rng.shuffle(driver_ages)

    # Driver experience: correlated with age but not perfectly
    max_exp = np.clip(driver_ages - 17, 0, 40).astype(int)
    driver_experience = np.array([
        rng.integers(0, max(1, m + 1)) for m in max_exp
    ])

    # NCD years: inversely correlated with age
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
        rng.choice([3, 3, 3, 6, 6, 9], size=n),
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

    # Occupation class: 1-5
    occupation_class = rng.choice(
        [1, 2, 3, 4, 5],
        size=n,
        p=[0.20, 0.30, 0.28, 0.15, 0.07]
    )

    # Policy type: Comp vs TPFT
    tpft_prob = np.where(driver_ages < 25, 0.35, 0.15)
    policy_type = np.where(rng.random(n) < tpft_prob, "TPFT", "Comp")

    return {
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
        "area": area.tolist(),
        "occupation_class": occupation_class,
        "policy_type": policy_type.tolist(),
    }


def _calculate_earned_exposure(inception_dates: list, expiry_dates: list) -> np.ndarray:
    """
    Calculate earned exposure in years for each policy row.

    Args:
        inception_dates: List of policy inception dates.
        expiry_dates: List of policy expiry dates.

    Returns:
        Array of earned exposure in years, clipped to >= 0.
    """
    days = np.array([
        (exp - inc).days for inc, exp in zip(inception_dates, expiry_dates)
    ], dtype=float)
    return np.clip(days / 365.25, 0.0, None)


def _generate_claims(
    data: dict,
    exposure: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate claim counts and incurred amounts from the true DGP.

    Args:
        data: Policy characteristics dict from _generate_policies.
        exposure: Earned exposure array.
        rng: Numpy random generator instance.

    Returns:
        Tuple of (claim_count, incurred) arrays.
    """
    n = len(exposure)

    vehicle_group = np.asarray(data["vehicle_group"])
    driver_ages = np.asarray(data["driver_age"])
    ncd_years = np.asarray(data["ncd_years"])
    conviction_points = np.asarray(data["conviction_points"])
    area = np.asarray(data["area"])

    # Frequency predictor
    log_lambda = (
        TRUE_FREQ_PARAMS["intercept"]
        + TRUE_FREQ_PARAMS["vehicle_group"] * vehicle_group
        + _driver_age_effect(driver_ages)
        + TRUE_FREQ_PARAMS["ncd_years"] * ncd_years
        + TRUE_FREQ_PARAMS["has_convictions"] * (conviction_points > 0).astype(float)
    )

    # Area effect
    area_effect = np.zeros(n)
    for band, key in [("B", "area_B"), ("C", "area_C"), ("D", "area_D"),
                      ("E", "area_E"), ("F", "area_F")]:
        area_effect[area == band] = TRUE_FREQ_PARAMS[key]
    log_lambda += area_effect
    log_lambda += np.log(np.clip(exposure, 1e-6, None))

    # Poisson claim counts
    lambda_vals = np.exp(log_lambda)
    claim_count = rng.poisson(lambda_vals)

    # Severity predictor
    log_mu = (
        TRUE_SEV_PARAMS["intercept"]
        + TRUE_SEV_PARAMS["vehicle_group"] * vehicle_group
        + TRUE_SEV_PARAMS["driver_age_young"] * (driver_ages < 25).astype(float)
    )

    gamma_shape = 2.0
    gamma_mean = np.exp(log_mu)
    gamma_scale = gamma_mean / gamma_shape

    has_claims = claim_count > 0
    incurred = np.zeros(n)

    if has_claims.any():
        for i in np.where(has_claims)[0]:
            per_claim = rng.gamma(
                shape=gamma_shape,
                scale=gamma_scale[i],
                size=int(claim_count[i])
            )
            incurred[i] = per_claim.sum()

    return claim_count, incurred


def load_motor(
    n_policies: int = 50_000,
    seed: int = 42,
) -> pl.DataFrame:
    """
    Load a synthetic UK motor insurance dataset.

    Generates ``n_policies`` rows with realistic UK personal lines motor
    characteristics and simulated claims from a known data generating process.
    Because the true parameters are known (see ``TRUE_FREQ_PARAMS`` and
    ``TRUE_SEV_PARAMS``), you can fit GLMs and validate coefficient recovery.

    Args:
        n_policies: Number of policies to generate. Default 50,000 gives
            stable GLM estimates. Use 5,000-10,000 for quick tests.
        seed: Random seed for reproducibility. Changing this seed gives a
            different but equally valid synthetic portfolio.

    Returns:
        Polars DataFrame with one row per policy and columns:

        - ``policy_id``: Int64, sequential identifier
        - ``inception_date``: Date, policy start
        - ``expiry_date``: Date, policy end (may be < 12 months for
          cancellations)
        - ``accident_year``: Int64, year of inception (used for cohort splits)
        - ``vehicle_age``: Int64, 0-20 years
        - ``vehicle_group``: Int64, ABI group 1-50
        - ``driver_age``: Int64, 17-85
        - ``driver_experience``: Int64, years licensed
        - ``ncd_years``: Int64, 0-5 (UK NCD scale)
        - ``ncd_protected``: Boolean
        - ``conviction_points``: Int64, total endorsement points
        - ``annual_mileage``: Int64, 2,000-30,000 miles
        - ``area``: Utf8, ABI area band A-F
        - ``occupation_class``: Int64, 1-5
        - ``policy_type``: Utf8, 'Comp' or 'TPFT'
        - ``claim_count``: Int64, number of claims in period
        - ``incurred``: Float64, total incurred cost (0.0 if no claims)
        - ``exposure``: Float64, earned years (< 1.0 for cancellations)

    Examples:
        >>> df = load_motor(n_policies=10_000, seed=0)
        >>> df.shape[0]
        10000
        >>> df["claim_count"].mean()  # roughly 6-8% claim rate
        # ~0.07
    """
    rng = np.random.default_rng(seed)

    policy_data = _generate_policies(n_policies, rng)

    inception_dates = policy_data["inception_date"]
    expiry_dates = policy_data["expiry_date"]
    exposure = _calculate_earned_exposure(inception_dates, expiry_dates)

    accident_year = np.array([d.year for d in inception_dates], dtype=int)

    claim_count, incurred = _generate_claims(policy_data, exposure, rng)

    df = pl.DataFrame({
        "policy_id": np.arange(1, n_policies + 1, dtype=int),
        "inception_date": inception_dates,
        "expiry_date": expiry_dates,
        "accident_year": accident_year,
        "vehicle_age": policy_data["vehicle_age"].astype(int),
        "vehicle_group": policy_data["vehicle_group"].astype(int),
        "driver_age": policy_data["driver_age"].astype(int),
        "driver_experience": policy_data["driver_experience"].astype(int),
        "ncd_years": policy_data["ncd_years"].astype(int),
        "ncd_protected": policy_data["ncd_protected"],
        "conviction_points": policy_data["conviction_points"].astype(int),
        "annual_mileage": policy_data["annual_mileage"].astype(int),
        "area": policy_data["area"],
        "occupation_class": policy_data["occupation_class"].astype(int),
        "policy_type": policy_data["policy_type"],
        "claim_count": claim_count.astype(int),
        "incurred": incurred.astype(float),
        "exposure": exposure.astype(float),
    })

    # Polars infers date columns from Python date objects; cast to Date type
    df = df.with_columns([
        pl.col("inception_date").cast(pl.Date),
        pl.col("expiry_date").cast(pl.Date),
    ])

    column_order = [
        "policy_id", "inception_date", "expiry_date", "accident_year",
        "vehicle_age", "vehicle_group", "driver_age", "driver_experience",
        "ncd_years", "ncd_protected", "conviction_points", "annual_mileage",
        "area", "occupation_class", "policy_type", "claim_count", "incurred",
        "exposure",
    ]
    return df.select(column_order)
