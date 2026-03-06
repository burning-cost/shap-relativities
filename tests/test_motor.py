"""
Tests for the synthetic UK motor dataset generator.

We test:
- Shape and column correctness
- Dtype correctness (the schema must be exact)
- Distributional properties that a pricing actuary would expect
- Claim count distribution relative to exposure
- Reproducibility via seed

We deliberately do NOT test GLM parameter recovery here — that belongs in
integration tests once the models module is built.

load_motor() returns a Polars DataFrame. Tests use Polars expressions
throughout.
"""

from datetime import date

import numpy as np
import polars as pl
import pytest

from shap_relativities.datasets.motor import (
    TRUE_FREQ_PARAMS,
    TRUE_SEV_PARAMS,
    load_motor,
)

EXPECTED_COLUMNS = [
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


@pytest.fixture(scope="module")
def motor_df() -> pl.DataFrame:
    """Standard 50k dataset, loaded once for the module."""
    return load_motor(n_policies=50_000, seed=42)


@pytest.fixture(scope="module")
def motor_df_small() -> pl.DataFrame:
    """Small dataset for fast tests."""
    return load_motor(n_policies=5_000, seed=99)


class TestShape:
    def test_row_count(self, motor_df: pl.DataFrame) -> None:
        assert len(motor_df) == 50_000

    def test_columns_present(self, motor_df: pl.DataFrame) -> None:
        for col in EXPECTED_COLUMNS:
            assert col in motor_df.columns, f"Missing column: {col}"

    def test_no_extra_columns(self, motor_df: pl.DataFrame) -> None:
        extra = set(motor_df.columns) - set(EXPECTED_COLUMNS)
        assert extra == set(), f"Unexpected columns: {extra}"

    def test_no_nulls(self, motor_df: pl.DataFrame) -> None:
        null_counts = motor_df.null_count()
        total_nulls = sum(null_counts.row(0))
        assert total_nulls == 0, f"Null values found in dataset"


class TestDtypes:
    def test_policy_id_int(self, motor_df: pl.DataFrame) -> None:
        assert motor_df["policy_id"].dtype in (pl.Int32, pl.Int64)

    def test_accident_year_int(self, motor_df: pl.DataFrame) -> None:
        assert motor_df["accident_year"].dtype in (pl.Int32, pl.Int64)

    def test_vehicle_age_int(self, motor_df: pl.DataFrame) -> None:
        assert motor_df["vehicle_age"].dtype in (pl.Int32, pl.Int64)

    def test_vehicle_group_int(self, motor_df: pl.DataFrame) -> None:
        assert motor_df["vehicle_group"].dtype in (pl.Int32, pl.Int64)

    def test_driver_age_int(self, motor_df: pl.DataFrame) -> None:
        assert motor_df["driver_age"].dtype in (pl.Int32, pl.Int64)

    def test_ncd_years_int(self, motor_df: pl.DataFrame) -> None:
        assert motor_df["ncd_years"].dtype in (pl.Int32, pl.Int64)

    def test_ncd_protected_bool(self, motor_df: pl.DataFrame) -> None:
        assert motor_df["ncd_protected"].dtype == pl.Boolean

    def test_claim_count_int(self, motor_df: pl.DataFrame) -> None:
        assert motor_df["claim_count"].dtype in (pl.Int32, pl.Int64)

    def test_incurred_float(self, motor_df: pl.DataFrame) -> None:
        assert motor_df["incurred"].dtype in (pl.Float32, pl.Float64)

    def test_exposure_float(self, motor_df: pl.DataFrame) -> None:
        assert motor_df["exposure"].dtype in (pl.Float32, pl.Float64)

    def test_inception_date_is_date(self, motor_df: pl.DataFrame) -> None:
        assert motor_df["inception_date"].dtype == pl.Date

    def test_expiry_date_is_date(self, motor_df: pl.DataFrame) -> None:
        assert motor_df["expiry_date"].dtype == pl.Date


class TestValueRanges:
    def test_policy_id_unique(self, motor_df: pl.DataFrame) -> None:
        assert motor_df["policy_id"].n_unique() == len(motor_df)

    def test_vehicle_group_bounds(self, motor_df: pl.DataFrame) -> None:
        assert motor_df["vehicle_group"].min() >= 1
        assert motor_df["vehicle_group"].max() <= 50

    def test_vehicle_age_bounds(self, motor_df: pl.DataFrame) -> None:
        assert motor_df["vehicle_age"].min() >= 0
        assert motor_df["vehicle_age"].max() <= 20

    def test_driver_age_bounds(self, motor_df: pl.DataFrame) -> None:
        assert motor_df["driver_age"].min() >= 17
        assert motor_df["driver_age"].max() <= 85

    def test_ncd_years_bounds(self, motor_df: pl.DataFrame) -> None:
        assert motor_df["ncd_years"].min() >= 0
        assert motor_df["ncd_years"].max() <= 5

    def test_ncd_protected_only_for_high_ncd(self, motor_df: pl.DataFrame) -> None:
        # NCD protection requires NCD >= 4 in the DGP
        protected = motor_df.filter(pl.col("ncd_protected"))
        assert (protected["ncd_years"] >= 4).all()

    def test_area_valid_bands(self, motor_df: pl.DataFrame) -> None:
        assert set(motor_df["area"].unique().to_list()).issubset({"A", "B", "C", "D", "E", "F"})

    def test_area_all_bands_represented(self, motor_df: pl.DataFrame) -> None:
        # All 6 bands should appear in a 50k dataset
        assert set(motor_df["area"].unique().to_list()) == {"A", "B", "C", "D", "E", "F"}

    def test_policy_type_values(self, motor_df: pl.DataFrame) -> None:
        assert set(motor_df["policy_type"].unique().to_list()).issubset({"Comp", "TPFT"})

    def test_occupation_class_bounds(self, motor_df: pl.DataFrame) -> None:
        assert motor_df["occupation_class"].min() >= 1
        assert motor_df["occupation_class"].max() <= 5

    def test_annual_mileage_bounds(self, motor_df: pl.DataFrame) -> None:
        assert motor_df["annual_mileage"].min() >= 2_000
        assert motor_df["annual_mileage"].max() <= 30_000

    def test_conviction_points_non_negative(self, motor_df: pl.DataFrame) -> None:
        assert (motor_df["conviction_points"] >= 0).all()

    def test_accident_year_range(self, motor_df: pl.DataFrame) -> None:
        years = motor_df["accident_year"].unique().to_list()
        assert all(2019 <= y <= 2023 for y in years)

    def test_all_accident_years_present(self, motor_df: pl.DataFrame) -> None:
        years = set(motor_df["accident_year"].unique().to_list())
        assert {2019, 2020, 2021, 2022, 2023}.issubset(years)


class TestDates:
    def test_expiry_after_inception(self, motor_df: pl.DataFrame) -> None:
        # Allow same-day (zero exposure) but not before
        assert (motor_df["expiry_date"] >= motor_df["inception_date"]).all()

    def test_exposure_non_negative(self, motor_df: pl.DataFrame) -> None:
        assert (motor_df["exposure"] >= 0).all()

    def test_exposure_at_most_one_year(self, motor_df: pl.DataFrame) -> None:
        # All policies are annual or less in the DGP
        assert (motor_df["exposure"] <= 1.01).all()  # small tolerance

    def test_cancellation_proportion(self, motor_df: pl.DataFrame) -> None:
        # ~8% should be cancellations (exposure < 0.95)
        short = (motor_df["exposure"] < 0.95).mean()
        assert 0.04 < short < 0.15, f"Cancellation rate {short:.2%} out of expected range"

    def test_exposure_consistent_with_dates(self, motor_df: pl.DataFrame) -> None:
        # Exposure should be consistent with date difference
        days_series = (
            motor_df["expiry_date"].cast(pl.Date).to_physical()
            - motor_df["inception_date"].cast(pl.Date).to_physical()
        ).cast(pl.Float64)
        expected_exposure = days_series / 365.25
        diff = (motor_df["exposure"] - expected_exposure).abs()
        assert (diff < 0.01).all()


class TestClaimDistribution:
    def test_claim_rate_plausible(self, motor_df: pl.DataFrame) -> None:
        """Claims per earned year should be in realistic UK motor range."""
        total_claims = motor_df["claim_count"].sum()
        total_exposure = motor_df["exposure"].sum()
        claim_freq = total_claims / total_exposure
        # UK motor market: 5-18% depending on risk mix. This DGP targets ~10% average.
        assert 0.04 < claim_freq < 0.18, f"Claim frequency {claim_freq:.3f} out of range"

    def test_zero_claim_majority(self, motor_df: pl.DataFrame) -> None:
        """Most policies should have no claims."""
        zero_claim_rate = (motor_df["claim_count"] == 0).mean()
        assert zero_claim_rate > 0.85

    def test_incurred_zero_when_no_claims(self, motor_df: pl.DataFrame) -> None:
        no_claim_rows = motor_df.filter(pl.col("claim_count") == 0)
        assert (no_claim_rows["incurred"] == 0.0).all()

    def test_incurred_positive_when_claims(self, motor_df: pl.DataFrame) -> None:
        claim_rows = motor_df.filter(pl.col("claim_count") > 0)
        assert (claim_rows["incurred"] > 0).all()

    def test_claim_count_non_negative(self, motor_df: pl.DataFrame) -> None:
        assert (motor_df["claim_count"] >= 0).all()

    def test_severity_plausible(self, motor_df: pl.DataFrame) -> None:
        """Average severity should be in realistic UK motor range (£1k-£5k)."""
        claims_only = motor_df.filter(pl.col("claim_count") > 0)
        avg_sev = claims_only["incurred"].sum() / claims_only["claim_count"].sum()
        assert 1_000 < avg_sev < 10_000, f"Average severity £{avg_sev:.0f} out of range"

    def test_young_driver_higher_frequency(self, motor_df: pl.DataFrame) -> None:
        """Young drivers (<25) should have higher claim frequency than mid-age."""
        young = motor_df.filter(pl.col("driver_age") < 25)
        mid = motor_df.filter((pl.col("driver_age") >= 30) & (pl.col("driver_age") < 50))

        young_freq = young["claim_count"].sum() / young["exposure"].sum()
        mid_freq = mid["claim_count"].sum() / mid["exposure"].sum()

        assert young_freq > mid_freq, (
            f"Young driver frequency {young_freq:.3f} should exceed "
            f"mid-age frequency {mid_freq:.3f}"
        )

    def test_high_ncd_lower_frequency(self, motor_df: pl.DataFrame) -> None:
        """Higher NCD should be associated with lower claim frequency."""
        low_ncd = motor_df.filter(pl.col("ncd_years") <= 1)
        high_ncd = motor_df.filter(pl.col("ncd_years") >= 4)

        low_freq = low_ncd["claim_count"].sum() / low_ncd["exposure"].sum()
        high_freq = high_ncd["claim_count"].sum() / high_ncd["exposure"].sum()

        assert high_freq < low_freq, (
            f"High NCD frequency {high_freq:.3f} should be lower than "
            f"low NCD frequency {low_freq:.3f}"
        )

    def test_conviction_higher_frequency(self, motor_df: pl.DataFrame) -> None:
        """Drivers with convictions should have higher claim frequency."""
        convicted = motor_df.filter(pl.col("conviction_points") > 0)
        clean = motor_df.filter(pl.col("conviction_points") == 0)

        convicted_freq = convicted["claim_count"].sum() / convicted["exposure"].sum()
        clean_freq = clean["claim_count"].sum() / clean["exposure"].sum()

        assert convicted_freq > clean_freq


class TestReproducibility:
    def test_same_seed_same_output(self) -> None:
        df1 = load_motor(n_policies=1_000, seed=7)
        df2 = load_motor(n_policies=1_000, seed=7)
        # frame_equal was renamed to equals in Polars 0.20+; use numpy for compatibility
        assert np.array_equal(df1["claim_count"].to_numpy(), df2["claim_count"].to_numpy())
        assert np.array_equal(df1["vehicle_group"].to_numpy(), df2["vehicle_group"].to_numpy())

    def test_different_seed_different_output(self) -> None:
        df1 = load_motor(n_policies=1_000, seed=1)
        df2 = load_motor(n_policies=1_000, seed=2)
        # Claim counts should differ
        assert not np.array_equal(df1["claim_count"].to_numpy(), df2["claim_count"].to_numpy())

    def test_smaller_n(self) -> None:
        df = load_motor(n_policies=100, seed=0)
        assert len(df) == 100


class TestTrueParams:
    def test_true_freq_params_keys(self) -> None:
        expected_keys = {
            "intercept", "vehicle_group", "driver_age_young",
            "driver_age_old", "ncd_years", "area_B", "area_C",
            "area_D", "area_E", "area_F", "has_convictions",
        }
        assert set(TRUE_FREQ_PARAMS.keys()) == expected_keys

    def test_true_sev_params_keys(self) -> None:
        expected_keys = {"intercept", "vehicle_group", "driver_age_young"}
        assert set(TRUE_SEV_PARAMS.keys()) == expected_keys

    def test_ncd_effect_negative(self) -> None:
        """NCD should reduce frequency — negative coefficient."""
        assert TRUE_FREQ_PARAMS["ncd_years"] < 0

    def test_young_driver_effect_positive(self) -> None:
        assert TRUE_FREQ_PARAMS["driver_age_young"] > 0
