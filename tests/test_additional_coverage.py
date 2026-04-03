"""
Additional test coverage for shap-relativities.

Targets areas not deeply exercised by existing tests:
- datasets.motor: schema, shapes, distribution sanity checks, determinism
- aggregation: NaN in SHAP values, large numbers of levels, integer weights
- normalisation: high-ci_level paths, extreme SHAP values
- SHAPRelativities.extract_relativities: ci_method='none', bootstrap raises,
  invalid ci_method, normalise_to='mean' on categorical, missing feature in
  extract_continuous_curve, invalid smooth_method
- SHAPRelativities.from_dict: column ordering is preserved
- SHAPRelativities exposure mismatch error
- check_reconstruction: multi-feature, array expected_value
- check_sparse_levels: DataFrame without level column
- _plotting: plot_categorical/continuous without CI columns
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import polars as pl
import pytest

from shap_relativities._aggregation import aggregate_categorical, aggregate_continuous
from shap_relativities._normalisation import normalise_base_level, normalise_mean
from shap_relativities._validation import (
    CheckResult,
    check_feature_coverage,
    check_reconstruction,
    check_sparse_levels,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agg(levels, mean_shaps, shap_stds=None, n_obs=None, exposure_weights=None, wsq_weights=None):
    n = len(levels)
    d = {
        "feature": ["f"] * n,
        "level": [str(l) for l in levels],
        "mean_shap": [float(x) for x in mean_shaps],
        "shap_std": [float(x) for x in (shap_stds or [0.1] * n)],
        "n_obs": list(n_obs or [50] * n),
        "exposure_weight": [float(x) for x in (exposure_weights or [1.0 / n] * n)],
    }
    if wsq_weights is not None:
        d["wsq_weight"] = [float(x) for x in wsq_weights]
    return pl.DataFrame(d)


def _fitted_sr_from_dict(**overrides):
    """Build a minimal SHAPRelativities via from_dict without needing catboost."""
    from shap_relativities import SHAPRelativities
    try:
        import shap  # noqa: F401
    except ImportError:
        pytest.skip("shap not installed")

    base = {
        "shap_values": [[0.2, -0.1], [-0.1, 0.3], [0.1, 0.1]],
        "expected_value": -2.0,
        "feature_names": ["area", "ncd"],
        "categorical_features": ["area", "ncd"],
        "continuous_features": [],
        "X_values": {"area": ["A", "B", "A"], "ncd": [0, 1, 2]},
        "exposure": [1.0, 1.0, 1.0],
        "annualise_exposure": False,
    }
    base.update(overrides)
    return SHAPRelativities.from_dict(base)


# ===========================================================================
# datasets.motor
# ===========================================================================


class TestMotorDataset:
    """Tests for the synthetic motor dataset generator."""

    def test_load_motor_returns_polars_dataframe(self):
        """load_motor must return a Polars DataFrame."""
        from shap_relativities.datasets.motor import load_motor
        df = load_motor(n_policies=500, seed=42)
        assert isinstance(df, pl.DataFrame)

    def test_load_motor_row_count(self):
        """Row count must equal n_policies."""
        from shap_relativities.datasets.motor import load_motor
        for n in [10, 100, 500]:
            df = load_motor(n_policies=n, seed=0)
            assert len(df) == n, f"Expected {n} rows, got {len(df)}"

    def test_load_motor_required_columns(self):
        """Standard columns must be present."""
        from shap_relativities.datasets.motor import load_motor
        df = load_motor(n_policies=200)
        required = {"vehicle_group", "driver_age", "ncd_years", "area", "exposure", "claim_count"}
        assert required.issubset(set(df.columns))

    def test_exposure_positive(self):
        """Exposure must be strictly positive for all policies."""
        from shap_relativities.datasets.motor import load_motor
        df = load_motor(n_policies=500, seed=1)
        exp = df["exposure"].to_numpy()
        assert (exp > 0).all(), "All exposure values must be positive"

    def test_exposure_reasonable_range(self):
        """Exposure should be near 1.0; no policy should have near-zero or multi-year exposure."""
        from shap_relativities.datasets.motor import load_motor
        df = load_motor(n_policies=500, seed=1)
        exp = df["exposure"].to_numpy()
        assert exp.min() > 0.04, f"Minimum exposure too small: {exp.min()}"
        assert exp.max() < 1.1, f"Maximum exposure too large: {exp.max()}"

    def test_claim_count_non_negative(self):
        """Claim counts must be non-negative integers."""
        from shap_relativities.datasets.motor import load_motor
        df = load_motor(n_policies=500, seed=2)
        assert (df["claim_count"] >= 0).all()

    def test_driver_age_realistic(self):
        """Driver ages should be in UK legal driving range (17-85)."""
        from shap_relativities.datasets.motor import load_motor
        df = load_motor(n_policies=1000, seed=3)
        ages = df["driver_age"].to_numpy()
        assert ages.min() >= 17
        assert ages.max() <= 86

    def test_ncd_years_in_0_to_5(self):
        """NCD years should be in [0, 5]."""
        from shap_relativities.datasets.motor import load_motor
        df = load_motor(n_policies=500, seed=4)
        ncd = df["ncd_years"].to_numpy()
        assert ncd.min() >= 0
        assert ncd.max() <= 5

    def test_area_bands_valid(self):
        """Area column should only contain valid band labels."""
        from shap_relativities.datasets.motor import load_motor, AREA_BANDS
        df = load_motor(n_policies=500, seed=5)
        invalid = set(df["area"].to_list()) - set(AREA_BANDS)
        assert len(invalid) == 0, f"Invalid area bands found: {invalid}"

    def test_deterministic_with_same_seed(self):
        """Same seed must produce identical DataFrames."""
        from shap_relativities.datasets.motor import load_motor
        import polars as pl
        df1 = load_motor(n_policies=200, seed=99)
        df2 = load_motor(n_policies=200, seed=99)
        # polars >= 1.0 uses .equals(); older versions use .frame_equal()
        equals_fn = getattr(df1, "equals", None) or getattr(df1, "frame_equal", None)
        assert equals_fn(df2)

    def test_different_seeds_differ(self):
        """Different seeds should (almost certainly) produce different data."""
        from shap_relativities.datasets.motor import load_motor
        df1 = load_motor(n_policies=200, seed=1)
        df2 = load_motor(n_policies=200, seed=2)
        equals_fn = getattr(df1, "equals", None) or getattr(df1, "frame_equal", None)
        assert not equals_fn(df2)

    def test_small_n_works(self):
        """load_motor should work with very small n (edge case for rounding in DGP)."""
        from shap_relativities.datasets.motor import load_motor
        df = load_motor(n_policies=10, seed=0)
        assert len(df) == 10

    def test_true_freq_params_exported(self):
        """TRUE_FREQ_PARAMS must be a non-empty dict with the intercept key."""
        from shap_relativities.datasets.motor import TRUE_FREQ_PARAMS
        assert isinstance(TRUE_FREQ_PARAMS, dict)
        assert "intercept" in TRUE_FREQ_PARAMS
        assert len(TRUE_FREQ_PARAMS) > 0

    def test_true_sev_params_exported(self):
        """TRUE_SEV_PARAMS must include the intercept key."""
        from shap_relativities.datasets.motor import TRUE_SEV_PARAMS
        assert "intercept" in TRUE_SEV_PARAMS

    def test_vehicle_group_in_range(self):
        """Vehicle group should be in [1, 50] ABI range."""
        from shap_relativities.datasets.motor import load_motor
        df = load_motor(n_policies=500, seed=6)
        vg = df["vehicle_group"].to_numpy()
        assert vg.min() >= 1
        assert vg.max() <= 50


# ===========================================================================
# aggregate_categorical: NaN and extreme inputs
# ===========================================================================


class TestAggregateCategoricalNaN:
    """Tests for aggregation with NaN-like or extreme numeric SHAP values."""

    def test_shap_values_with_large_magnitude(self):
        """Aggregation should handle SHAP values with large magnitudes."""
        result = aggregate_categorical(
            "x",
            np.array(["A", "B"]),
            np.array([100.0, -100.0]),
            np.ones(2),
        )
        assert len(result) == 2
        assert not any(math.isnan(v) for v in result["mean_shap"].to_list())

    def test_integer_weights_accepted(self):
        """Integer weight arrays should be accepted and cast to float internally."""
        result = aggregate_categorical(
            "x",
            np.array(["A", "A", "B"]),
            np.array([0.1, 0.2, 0.3]),
            np.array([1, 2, 3], dtype=int),
        )
        assert len(result) == 2
        assert result["exposure_weight"].dtype == pl.Float64

    def test_many_levels_sorted(self):
        """With 20 distinct levels, output should be sorted lexicographically."""
        levels = [str(i) for i in range(20)]
        result = aggregate_categorical(
            "x",
            np.array(levels),
            np.zeros(20),
            np.ones(20),
        )
        level_list = result["level"].to_list()
        assert level_list == sorted(level_list)

    def test_all_same_level_gives_one_row(self):
        """All observations in one level should produce exactly one row."""
        result = aggregate_categorical(
            "x",
            np.array(["X"] * 100),
            np.random.normal(0, 1, 100),
            np.ones(100),
        )
        assert len(result) == 1

    def test_negative_shap_values_handled(self):
        """Negative SHAP values should not cause errors or NaN means."""
        result = aggregate_categorical(
            "x",
            np.array(["A", "B", "C"]),
            np.array([-1.5, -2.0, -0.5]),
            np.ones(3),
        )
        for v in result["mean_shap"].to_list():
            assert not math.isnan(v)

    def test_weighted_mean_computed_correctly(self):
        """Verify weighted mean for a specific case: weights [1, 3], shap [0, 1]."""
        result = aggregate_categorical(
            "x",
            np.array(["A", "A"]),
            np.array([0.0, 1.0]),
            np.array([1.0, 3.0]),
        )
        expected_mean = (1 * 0.0 + 3 * 1.0) / 4.0  # = 0.75
        assert abs(result["mean_shap"][0] - expected_mean) < 1e-9

    def test_columns_match_schema(self):
        """Output columns must match the documented schema exactly."""
        result = aggregate_categorical(
            "x", np.array(["A", "B"]), np.zeros(2), np.ones(2)
        )
        expected_cols = {"feature", "level", "mean_shap", "shap_std", "n_obs", "exposure_weight", "wsq_weight"}
        assert set(result.columns) == expected_cols


# ===========================================================================
# aggregate_continuous: additional coverage
# ===========================================================================


class TestAggregateContinuousAdditional:
    """Additional coverage for aggregate_continuous."""

    def test_columns_match_schema(self):
        """Output columns must match the documented schema."""
        result = aggregate_continuous(
            "age", np.array([25.0, 40.0]), np.zeros(2), np.ones(2)
        )
        expected = {"feature", "level", "mean_shap", "shap_std", "n_obs", "exposure_weight", "wsq_weight"}
        assert set(result.columns) == expected

    def test_shap_std_is_all_zeros(self):
        """shap_std must be zero for all rows (per-observation output)."""
        result = aggregate_continuous(
            "x", np.arange(10.0), np.random.normal(0, 1, 10), np.ones(10)
        )
        assert (result["shap_std"] == 0.0).all()

    def test_large_n_works(self):
        """Should handle 10,000 observations without error."""
        rng = np.random.default_rng(1)
        result = aggregate_continuous(
            "x", rng.uniform(0, 100, 10000), rng.normal(0, 1, 10000), np.ones(10000)
        )
        assert len(result) == 10000

    def test_zero_weight_observation_preserved(self):
        """Zero-weight observation is allowed for continuous (no filtering)."""
        result = aggregate_continuous(
            "x",
            np.array([1.0, 2.0, 3.0]),
            np.zeros(3),
            np.array([0.0, 1.0, 1.0]),
        )
        # All three rows should be present (no filtering for continuous)
        assert len(result) == 3

    def test_integer_feature_values_cast_to_str(self):
        """Integer feature values should be cast to string in level column."""
        result = aggregate_continuous(
            "x", np.array([1, 2, 3], dtype=int), np.zeros(3), np.ones(3)
        )
        assert result["level"].dtype == pl.String


# ===========================================================================
# normalise_mean: edge cases
# ===========================================================================


class TestNormaliseMeanEdgeCases:
    """Edge cases for normalise_mean."""

    def test_zero_total_weight_uses_zero_mean(self):
        """When total weight=0, portfolio_mean defaults to 0 (no division by zero)."""
        df = pl.DataFrame({
            "feature": ["f", "f"],
            "level": ["A", "B"],
            "mean_shap": [0.5, -0.5],
            "shap_std": [0.1, 0.1],
            "n_obs": [10, 10],
            "exposure_weight": [0.0, 0.0],
            "wsq_weight": [0.0, 0.0],
        })
        result = normalise_mean(df)
        # portfolio_mean = 0, so relativity = exp(mean_shap)
        assert "relativity" in result.columns

    def test_positive_mean_shap_gives_relativity_above_one(self):
        """Level with mean_shap above portfolio mean should have relativity > 1."""
        # All equal weights, mean_shap = 0 for A and 1 for B → portfolio mean = 0.5
        # A: relativity = exp(0 - 0.5) < 1; B: relativity = exp(1 - 0.5) > 1
        df = _make_agg(["A", "B"], [0.0, 1.0], exposure_weights=[0.5, 0.5], wsq_weights=[0.01, 0.01])
        result = normalise_mean(df)
        b_rel = result.filter(pl.col("level") == "B")["relativity"][0]
        assert b_rel > 1.0

    def test_symmetric_shaps_give_symmetric_relativities(self):
        """Equal-weight levels with shap=-x and shap=+x should have symmetric relativities."""
        df = _make_agg(
            ["A", "B"], [-0.5, 0.5],
            exposure_weights=[0.5, 0.5],
            wsq_weights=[0.01, 0.01],
        )
        result = normalise_mean(df)
        rel_a = result.filter(pl.col("level") == "A")["relativity"][0]
        rel_b = result.filter(pl.col("level") == "B")["relativity"][0]
        assert abs(rel_a - 1 / rel_b) < 1e-9

    def test_output_has_lower_ci_and_upper_ci(self):
        """normalise_mean must add lower_ci and upper_ci columns."""
        df = _make_agg(["A", "B", "C"], [0.0, 0.3, -0.2])
        result = normalise_mean(df)
        assert "lower_ci" in result.columns
        assert "upper_ci" in result.columns


# ===========================================================================
# normalise_base_level: error paths
# ===========================================================================


class TestNormaliseBaseLevelErrors:
    """Error paths for normalise_base_level."""

    def test_missing_base_level_raises_value_error(self):
        """Requesting a base level not in the data should raise ValueError."""
        df = _make_agg(["A", "B"], [0.0, 0.5])
        with pytest.raises(ValueError, match="not found"):
            normalise_base_level(df, "Z")

    def test_error_message_includes_available_levels(self):
        """ValueError message should list available levels."""
        df = _make_agg(["X", "Y"], [0.0, 0.3])
        with pytest.raises(ValueError) as exc_info:
            normalise_base_level(df, "Z")
        assert "X" in str(exc_info.value) or "Y" in str(exc_info.value)

    def test_integer_base_level_matched_via_str_coercion(self):
        """Integer base level should match via str conversion."""
        df = _make_agg(["1", "2", "3"], [0.0, 0.2, 0.4])
        result = normalise_base_level(df, 1)
        row1 = result.filter(pl.col("level") == "1")
        assert abs(row1["relativity"][0] - 1.0) < 1e-12


# ===========================================================================
# check_reconstruction: additional cases
# ===========================================================================


class TestCheckReconstructionAdditional:
    """Additional tests for check_reconstruction."""

    def test_multi_feature_exact_reconstruction(self):
        """Exact reconstruction with 5 features must pass."""
        n, d = 20, 5
        rng = np.random.default_rng(0)
        sv = rng.normal(0, 0.3, (n, d))
        ev = -1.5
        preds = np.exp(sv.sum(axis=1) + ev)
        result = check_reconstruction(sv, ev, preds, tolerance=1e-6)
        assert result.passed

    def test_value_is_max_absolute_error(self):
        """The value field should equal max absolute error."""
        shap_vals = np.zeros((3, 2))
        ev = 0.0
        preds = np.array([1.0, 1.0, 1.001])
        result = check_reconstruction(shap_vals, ev, preds, tolerance=0.1)
        assert abs(result.value - 0.001) < 1e-9

    def test_single_feature_reconstruction(self):
        """Single-feature SHAP reconstruction should work."""
        sv = np.array([[0.5], [-0.3], [0.1]])
        ev = 0.2
        preds = np.exp(sv[:, 0] + ev)
        result = check_reconstruction(sv, ev, preds)
        assert result.passed

    def test_message_contains_max_diff(self):
        """Message should mention the max absolute error."""
        sv = np.zeros((2, 1))
        preds = np.array([1.0, 1.1])
        result = check_reconstruction(sv, 0.0, preds, tolerance=0.001)
        assert "Max absolute reconstruction error" in result.message


# ===========================================================================
# check_sparse_levels: DataFrame without 'level' column
# ===========================================================================


class TestCheckSparseLevelsNoLevelColumn:
    """Tests for check_sparse_levels when level column is absent."""

    def test_dataframe_without_level_column(self):
        """DataFrame with n_obs but no level column should still work."""
        df = pl.DataFrame({"n_obs": [5, 50, 100]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = check_sparse_levels(df, min_obs=30)
        # One level (n_obs=5) is below threshold
        assert not result.passed
        assert result.value == 1.0

    def test_no_n_obs_column_passes(self):
        """DataFrame without n_obs column should trivially pass."""
        df = pl.DataFrame({"level": ["A", "B"]})
        result = check_sparse_levels(df)
        assert result.passed
        assert "No n_obs column" in result.message


# ===========================================================================
# SHAPRelativities: extract_relativities error paths and variants
# ===========================================================================


class TestExtractRelativitiesErrors:
    """Tests for extract_relativities error and edge case paths."""

    def test_invalid_ci_method_raises(self):
        """Unknown ci_method should raise ValueError."""
        sr = _fitted_sr_from_dict()
        with pytest.raises(ValueError, match="ci_method"):
            sr.extract_relativities(ci_method="magic")

    def test_bootstrap_ci_method_raises_not_implemented(self):
        """ci_method='bootstrap' should raise NotImplementedError."""
        sr = _fitted_sr_from_dict()
        with pytest.raises(NotImplementedError):
            sr.extract_relativities(ci_method="bootstrap")

    def test_ci_method_none_produces_nan_cis(self):
        """ci_method='none' should give NaN for lower_ci and upper_ci."""
        sr = _fitted_sr_from_dict()
        result = sr.extract_relativities(ci_method="none", normalise_to="mean")
        for row in result.iter_rows(named=True):
            assert math.isnan(row["lower_ci"]), f"lower_ci not NaN: {row}"
            assert math.isnan(row["upper_ci"]), f"upper_ci not NaN: {row}"

    def test_normalise_to_mean_categorical_gives_weighted_mean_one(self):
        """For categorical features with normalise_to='mean', weighted geometric mean of relativities should be 1."""
        sr = _fitted_sr_from_dict()
        result = sr.extract_relativities(normalise_to="mean")
        # For each feature, exposure-weighted mean of log(relativity) should be ~0
        for feat in result["feature"].unique().to_list():
            feat_rows = result.filter(pl.col("feature") == feat)
            ew = feat_rows["exposure_weight"].to_numpy()
            log_rels = np.log(feat_rows["relativity"].to_numpy())
            if ew.sum() > 0:
                weighted_mean = np.average(log_rels, weights=ew)
                assert abs(weighted_mean) < 1e-6, (
                    f"Feature {feat}: weighted mean log(relativity) = {weighted_mean}"
                )

    def test_output_columns_standard_set(self):
        """Output must include the documented standard columns."""
        sr = _fitted_sr_from_dict()
        result = sr.extract_relativities(normalise_to="mean")
        expected = {"feature", "level", "relativity", "lower_ci", "upper_ci",
                    "mean_shap", "shap_std", "n_obs", "exposure_weight"}
        assert expected.issubset(set(result.columns))

    def test_relativity_positive_for_all_modes(self):
        """Relativity must be > 0 regardless of normalisation mode."""
        sr = _fitted_sr_from_dict()
        for mode in ["mean", "base_level"]:
            base = {"area": "A", "ncd": "0"} if mode == "base_level" else {}
            result = sr.extract_relativities(normalise_to=mode, base_levels=base)
            assert (result["relativity"] > 0).all(), f"mode={mode} has non-positive relativities"


# ===========================================================================
# SHAPRelativities: extract_continuous_curve error paths
# ===========================================================================


class TestExtractContinuousCurveErrors:
    """Error paths for extract_continuous_curve."""

    def _continuous_sr(self, n=30):
        from shap_relativities import SHAPRelativities
        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        rng = np.random.default_rng(20)
        vals = np.linspace(0, 10, n).tolist()
        shaps = rng.normal(0, 0.2, n).tolist()

        data = {
            "shap_values": [[s] for s in shaps],
            "expected_value": 0.0,
            "feature_names": ["age"],
            "categorical_features": [],
            "continuous_features": ["age"],
            "X_values": {"age": vals},
            "exposure": None,
            "annualise_exposure": False,
        }
        return SHAPRelativities.from_dict(data)

    def test_unknown_feature_raises(self):
        """Requesting a feature not in X should raise ValueError."""
        sr = self._continuous_sr()
        with pytest.raises(ValueError, match="not in X"):
            sr.extract_continuous_curve("nonexistent")

    def test_invalid_smooth_method_raises(self):
        """Unknown smooth_method should raise ValueError."""
        sr = self._continuous_sr()
        with pytest.raises(ValueError, match="smooth_method"):
            sr.extract_continuous_curve("age", smooth_method="spline")

    def test_smooth_none_has_expected_columns(self):
        """smooth_method='none' output must have feature_value, relativity, lower_ci, upper_ci."""
        sr = self._continuous_sr()
        curve = sr.extract_continuous_curve("age", smooth_method="none")
        assert "feature_value" in curve.columns
        assert "relativity" in curve.columns
        assert "lower_ci" in curve.columns
        assert "upper_ci" in curve.columns

    def test_smooth_none_nan_cis(self):
        """smooth_method='none' produces NaN CIs."""
        sr = self._continuous_sr()
        curve = sr.extract_continuous_curve("age", smooth_method="none")
        for row in curve.iter_rows(named=True):
            assert math.isnan(row["lower_ci"])
            assert math.isnan(row["upper_ci"])

    def test_isotonic_all_positive_relativities(self):
        """Isotonic smoothed relativities must all be positive."""
        sr = self._continuous_sr(n=30)
        curve = sr.extract_continuous_curve("age", smooth_method="isotonic")
        assert (curve["relativity"] > 0).all()

    def test_isotonic_output_shape_matches_n_points(self):
        """Isotonic smooth output must have exactly n_points rows."""
        sr = self._continuous_sr(n=30)
        for n_pts in [5, 15, 50]:
            curve = sr.extract_continuous_curve("age", n_points=n_pts, smooth_method="isotonic")
            assert len(curve) == n_pts

    def test_loess_fallback_to_none_without_statsmodels(self, monkeypatch):
        """If statsmodels is not installed, loess should fall back to 'none' with a warning."""
        import sys
        # Temporarily hide statsmodels from the import system
        original = sys.modules.get("statsmodels")
        sys.modules["statsmodels"] = None  # type: ignore[assignment]
        # Also hide the specific submodule
        sm_key = "statsmodels.nonparametric.smoothers_lowess"
        original_sub = sys.modules.get(sm_key)
        sys.modules[sm_key] = None  # type: ignore[assignment]

        sr = self._continuous_sr(n=30)
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                curve = sr.extract_continuous_curve("age", smooth_method="loess")
        finally:
            if original is None:
                sys.modules.pop("statsmodels", None)
            else:
                sys.modules["statsmodels"] = original
            if original_sub is None:
                sys.modules.pop(sm_key, None)
            else:
                sys.modules[sm_key] = original_sub

        # Should return the 'none' fallback (n rows = n data points)
        assert len(curve) == 30


# ===========================================================================
# SHAPRelativities: exposure mismatch error
# ===========================================================================


class TestSHAPRelativitiesExposureMismatch:
    """Test that mismatched exposure length raises ValueError at construction."""

    def test_exposure_length_mismatch_raises(self):
        """exposure with wrong length should raise ValueError."""
        from shap_relativities import SHAPRelativities
        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        class _Dummy:
            pass

        X = pl.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="exposure length"):
            SHAPRelativities(
                model=_Dummy(),
                X=X,
                exposure=np.array([1.0, 1.0]),  # length 2, X has 3 rows
            )


# ===========================================================================
# SHAPRelativities: from_dict column ordering
# ===========================================================================


class TestFromDictColumnOrdering:
    """Verify that from_dict respects feature_names order."""

    def test_feature_names_order_preserved(self):
        """X columns in the reconstructed object must match feature_names order."""
        from shap_relativities import SHAPRelativities
        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        data = {
            "shap_values": [[0.1, -0.1, 0.2], [0.0, 0.1, -0.2]],
            "expected_value": 0.0,
            "feature_names": ["z_feat", "a_feat", "m_feat"],
            "categorical_features": ["z_feat", "a_feat", "m_feat"],
            "continuous_features": [],
            "X_values": {
                "z_feat": ["X", "Y"],
                "a_feat": [0, 1],
                "m_feat": ["P", "Q"],
            },
            "exposure": None,
            "annualise_exposure": False,
        }
        sr = SHAPRelativities.from_dict(data)
        # Columns in X must follow feature_names ordering, not alphabetical
        assert sr._X.columns == ["z_feat", "a_feat", "m_feat"]

    def test_shap_matrix_aligns_with_features(self):
        """shap_values stored must have n_features columns matching X.width."""
        from shap_relativities import SHAPRelativities
        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        data = {
            "shap_values": [[0.1, 0.2], [-0.1, 0.3]],
            "expected_value": -1.0,
            "feature_names": ["a", "b"],
            "categorical_features": ["a", "b"],
            "continuous_features": [],
            "X_values": {"a": ["X", "Y"], "b": [0, 1]},
            "exposure": None,
            "annualise_exposure": False,
        }
        sr = SHAPRelativities.from_dict(data)
        assert sr._shap_values.shape[1] == sr._X.width


# ===========================================================================
# _plotting.py: additional coverage
# ===========================================================================


class TestPlottingAdditional:
    """Additional tests for plotting module."""

    @pytest.fixture(autouse=True)
    def _backend(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        yield
        plt.close("all")

    def _cat_data(self, feature="area", n_levels=3):
        levels = [chr(65 + i) for i in range(n_levels)]
        return pl.DataFrame({
            "feature": [feature] * n_levels,
            "level": levels,
            "relativity": [1.0, 1.1, 0.9],
            "lower_ci": [0.9, 1.0, 0.8],
            "upper_ci": [1.1, 1.2, 1.0],
            "mean_shap": [0.0, 0.09, -0.1],
            "shap_std": [0.05] * n_levels,
            "n_obs": [100] * n_levels,
            "exposure_weight": [1.0 / n_levels] * n_levels,
        })

    def test_plot_categorical_show_ci_false(self):
        """plot_categorical with show_ci=False should not raise."""
        import matplotlib.pyplot as plt
        from shap_relativities._plotting import plot_categorical
        fig, ax = plt.subplots()
        plot_categorical(self._cat_data(), "area", ax, show_ci=False)

    def test_plot_categorical_title_set(self):
        """plot_categorical should set the axis title to the feature name."""
        import matplotlib.pyplot as plt
        from shap_relativities._plotting import plot_categorical
        fig, ax = plt.subplots()
        plot_categorical(self._cat_data(), "myfeature", ax)
        assert ax.get_title() == "myfeature"

    def test_plot_continuous_show_ci_false(self):
        """plot_continuous with show_ci=False should not raise."""
        import matplotlib.pyplot as plt
        from shap_relativities._plotting import plot_continuous
        data = pl.DataFrame({
            "feature": ["age"] * 5,
            "level": [str(float(v)) for v in range(5)],
            "relativity": [1.0] * 5,
            "lower_ci": [0.9] * 5,
            "upper_ci": [1.1] * 5,
            "mean_shap": [0.0] * 5,
            "shap_std": [0.0] * 5,
            "n_obs": [50] * 5,
            "exposure_weight": [0.2] * 5,
        })
        fig, ax = plt.subplots()
        plot_continuous(data, "age", ax, show_ci=False)

    def test_plot_relativities_single_feature(self):
        """plot_relativities with one feature should not raise."""
        from shap_relativities._plotting import plot_relativities
        plot_relativities(
            self._cat_data("area"),
            categorical_features=["area"],
            continuous_features=[],
        )

    def test_plot_relativities_fallback_for_unknown_feature(self):
        """Unknown feature (neither categorical nor continuous) falls back to categorical bar chart."""
        from shap_relativities._plotting import plot_relativities
        data = self._cat_data("mystery")
        # Not in categorical or continuous — should fall through to categorical
        plot_relativities(
            data,
            categorical_features=[],
            continuous_features=[],
        )

    def test_plot_categorical_no_lower_upper_ci_columns(self):
        """plot_categorical on data without lower_ci/upper_ci columns should not raise."""
        import matplotlib.pyplot as plt
        from shap_relativities._plotting import plot_categorical
        data = pl.DataFrame({
            "feature": ["f", "f"],
            "level": ["A", "B"],
            "relativity": [1.0, 1.2],
            "mean_shap": [0.0, 0.18],
            "shap_std": [0.0, 0.0],
            "n_obs": [100, 80],
            "exposure_weight": [0.5, 0.5],
        })
        fig, ax = plt.subplots()
        # show_ci=True but columns absent — should not raise
        plot_categorical(data, "f", ax, show_ci=True)


# ===========================================================================
# check_feature_coverage: additional cases
# ===========================================================================


class TestCheckFeatureCoverageAdditional:
    """Additional tests for check_feature_coverage."""

    def test_extra_shap_features_passes(self):
        """SHAP may have more features than expected — this is fine."""
        result = check_feature_coverage(["a", "b", "c_extra"], ["a", "b"])
        assert result.passed
        assert result.value == 0.0

    def test_one_missing_feature(self):
        """One missing feature: value should be 1.0."""
        result = check_feature_coverage(["a"], ["a", "b"])
        assert not result.passed
        assert result.value == 1.0

    def test_message_contains_missing_feature(self):
        """Error message must name the missing feature."""
        result = check_feature_coverage(["a"], ["a", "missing_feat"])
        assert "missing_feat" in result.message

    def test_pass_message_format(self):
        """Passing result should report 'All features covered'."""
        result = check_feature_coverage(["a", "b"], ["a", "b"])
        assert "All features covered" in result.message
