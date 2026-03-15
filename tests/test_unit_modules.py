"""
Unit tests for the lower-level modules in shap-relativities.

These tests exercise the individual building blocks directly — aggregation,
normalisation, validation, and plotting — rather than going through the full
SHAPRelativities pipeline. This keeps the feedback loop fast and makes
failure attribution obvious.

All tests are deliberately self-contained: they construct their own small
DataFrames and numpy arrays rather than relying on expensive catboost fixtures.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Import the modules under test
# ---------------------------------------------------------------------------

from shap_relativities._aggregation import aggregate_categorical, aggregate_continuous
from shap_relativities._normalisation import normalise_base_level, normalise_mean
from shap_relativities._validation import (
    CheckResult,
    check_feature_coverage,
    check_reconstruction,
    check_sparse_levels,
)


# ---------------------------------------------------------------------------
# Helpers — tiny synthetic SHAP aggregation DataFrames
# ---------------------------------------------------------------------------


def _make_agg_df(
    levels: list[str],
    mean_shaps: list[float],
    shap_stds: list[float] | None = None,
    n_obs: list[int] | None = None,
    exposure_weights: list[float] | None = None,
) -> pl.DataFrame:
    """Build a minimal aggregated DataFrame matching aggregate_categorical output."""
    n = len(levels)
    return pl.DataFrame({
        "feature": ["feat"] * n,
        "level": levels,
        "mean_shap": mean_shaps,
        "shap_std": shap_stds or [0.0] * n,
        "n_obs": n_obs or [100] * n,
        "exposure_weight": exposure_weights or [1.0 / n] * n,
    })


# ===========================================================================
# _aggregation.py — aggregate_categorical
# ===========================================================================


class TestAggregateCategorical:
    """Tests for aggregate_categorical."""

    def test_output_schema(self) -> None:
        """Output must have the expected six columns."""
        result = aggregate_categorical(
            "area",
            np.array(["A", "B", "A", "C"]),
            np.array([0.1, 0.2, 0.15, 0.3]),
            np.array([1.0, 1.0, 1.0, 1.0]),
        )
        expected = {"feature", "level", "mean_shap", "shap_std", "n_obs", "exposure_weight"}
        assert expected == set(result.columns)

    def test_correct_level_count(self) -> None:
        """One row per unique level."""
        result = aggregate_categorical(
            "area",
            np.array(["A", "B", "A", "B", "A"]),
            np.array([0.1, 0.2, 0.1, 0.2, 0.1]),
            np.ones(5),
        )
        assert len(result) == 2

    def test_weighted_mean_shap(self) -> None:
        """Weighted mean SHAP is exposure-weighted, not a simple average."""
        # Level "A": obs with w=3 (shap=0.1) and w=1 (shap=0.5) → mean = (0.3+0.5)/4 = 0.2
        result = aggregate_categorical(
            "x",
            np.array(["A", "A"]),
            np.array([0.1, 0.5]),
            np.array([3.0, 1.0]),
        )
        row = result.filter(pl.col("level") == "A")
        assert abs(row["mean_shap"][0] - 0.2) < 1e-9

    def test_n_obs_count(self) -> None:
        """n_obs counts raw observations, not exposure weight."""
        result = aggregate_categorical(
            "x",
            np.array(["A", "A", "B"]),
            np.zeros(3),
            np.array([0.5, 1.5, 1.0]),
        )
        a_row = result.filter(pl.col("level") == "A")
        assert a_row["n_obs"][0] == 2

    def test_exposure_weight_sum(self) -> None:
        """exposure_weight is the sum of observation weights for the level."""
        result = aggregate_categorical(
            "x",
            np.array(["A", "A", "A"]),
            np.zeros(3),
            np.array([0.3, 0.5, 0.2]),
        )
        a_row = result.filter(pl.col("level") == "A")
        assert abs(a_row["exposure_weight"][0] - 1.0) < 1e-9

    def test_integer_levels_cast_to_str(self) -> None:
        """Integer feature values should appear as strings in the output."""
        result = aggregate_categorical(
            "ncd",
            np.array([0, 1, 2, 0]),
            np.zeros(4),
            np.ones(4),
        )
        # All levels should be strings
        assert result["level"].dtype == pl.Utf8

    def test_shap_std_non_negative(self) -> None:
        """Standard deviation of SHAP values must be >= 0 (clipped if needed)."""
        result = aggregate_categorical(
            "x",
            np.array(["A", "A", "B", "B"]),
            np.array([0.1, 0.9, 0.5, 0.5]),
            np.ones(4),
        )
        assert (result["shap_std"] >= 0.0).all()

    def test_feature_name_in_output(self) -> None:
        """The feature column must contain the supplied feature name."""
        result = aggregate_categorical(
            "my_feature",
            np.array(["X", "Y"]),
            np.zeros(2),
            np.ones(2),
        )
        assert (result["feature"] == "my_feature").all()

    def test_sorted_by_level(self) -> None:
        """Output must be sorted by level (ascending string order)."""
        result = aggregate_categorical(
            "x",
            np.array(["C", "A", "B"]),
            np.zeros(3),
            np.ones(3),
        )
        assert result["level"].to_list() == ["A", "B", "C"]


# ===========================================================================
# _aggregation.py — aggregate_continuous
# ===========================================================================


class TestAggregateContinuous:
    """Tests for aggregate_continuous."""

    def test_output_schema(self) -> None:
        """Output must match the aggregate_categorical schema for concat compatibility."""
        result = aggregate_continuous(
            "age",
            np.array([17.0, 25.0, 40.0]),
            np.array([0.1, 0.2, 0.3]),
            np.ones(3),
        )
        expected = {"feature", "level", "mean_shap", "shap_std", "n_obs", "exposure_weight"}
        assert expected == set(result.columns)

    def test_row_count_equals_obs(self) -> None:
        """One row per observation (no aggregation for continuous)."""
        n = 50
        result = aggregate_continuous(
            "age",
            np.linspace(17, 80, n),
            np.random.default_rng(0).normal(size=n),
            np.ones(n),
        )
        assert len(result) == n

    def test_shap_std_is_zero(self) -> None:
        """shap_std is zero for continuous features (no within-level variance)."""
        result = aggregate_continuous(
            "x",
            np.array([1.0, 2.0, 3.0]),
            np.array([0.1, 0.2, 0.3]),
            np.ones(3),
        )
        assert (result["shap_std"] == 0.0).all()

    def test_n_obs_is_one_per_row(self) -> None:
        """n_obs is 1 for every row in the continuous case."""
        result = aggregate_continuous(
            "x",
            np.array([1.0, 2.0]),
            np.array([0.0, 0.0]),
            np.ones(2),
        )
        assert (result["n_obs"] == 1).all()

    def test_mean_shap_passes_through(self) -> None:
        """The SHAP values should appear unmodified in mean_shap."""
        shap_vals = np.array([0.1, -0.2, 0.3, -0.4])
        result = aggregate_continuous("x", np.arange(4.0), shap_vals, np.ones(4))
        np.testing.assert_allclose(result["mean_shap"].to_numpy(), shap_vals)

    def test_feature_name_in_output(self) -> None:
        """The feature column must contain the supplied name."""
        result = aggregate_continuous(
            "vehicle_age", np.array([1.0]), np.array([0.0]), np.ones(1),
        )
        assert result["feature"][0] == "vehicle_age"


# ===========================================================================
# _normalisation.py — normalise_base_level
# ===========================================================================


class TestNormaliseBaseLevel:
    """Tests for normalise_base_level."""

    def test_base_level_relativity_is_one(self) -> None:
        """The nominated base level must have relativity exactly 1.0."""
        df = _make_agg_df(["A", "B", "C"], [0.0, 0.3, -0.2])
        result = normalise_base_level(df, "A")
        base_row = result.filter(pl.col("level") == "A")
        assert abs(base_row["relativity"][0] - 1.0) < 1e-12

    def test_relativities_correct(self) -> None:
        """exp(mean_shap - base_shap) should give the correct ratios."""
        df = _make_agg_df(["A", "B"], [0.0, 0.5])
        result = normalise_base_level(df, "A")
        expected = math.exp(0.5)
        assert abs(result.filter(pl.col("level") == "B")["relativity"][0] - expected) < 1e-9

    def test_missing_base_level_raises(self) -> None:
        """ValueError must be raised when the base level is not in the data."""
        df = _make_agg_df(["A", "B"], [0.0, 0.3])
        with pytest.raises(ValueError, match="not found in levels"):
            normalise_base_level(df, "Z")

    def test_ci_columns_present(self) -> None:
        """Output must include lower_ci and upper_ci columns."""
        df = _make_agg_df(["A", "B"], [0.0, 0.3], shap_stds=[0.1, 0.2])
        result = normalise_base_level(df, "A")
        assert "lower_ci" in result.columns
        assert "upper_ci" in result.columns

    def test_lower_ci_leq_relativity_leq_upper_ci(self) -> None:
        """Confidence bands must straddle the point estimate."""
        df = _make_agg_df(["A", "B", "C"], [0.0, 0.3, -0.2], shap_stds=[0.1, 0.2, 0.15])
        result = normalise_base_level(df, "A")
        assert (result["lower_ci"] <= result["relativity"]).all()
        assert (result["relativity"] <= result["upper_ci"]).all()

    def test_ci_level_90(self) -> None:
        """A 90% CI should be narrower than the default 95% CI."""
        df = _make_agg_df(["A", "B"], [0.0, 0.3], shap_stds=[0.2, 0.2])
        res_95 = normalise_base_level(df, "A", ci_level=0.95)
        res_90 = normalise_base_level(df, "A", ci_level=0.90)
        b_95 = res_95.filter(pl.col("level") == "B")
        b_90 = res_90.filter(pl.col("level") == "B")
        width_95 = float(b_95["upper_ci"][0]) - float(b_95["lower_ci"][0])
        width_90 = float(b_90["upper_ci"][0]) - float(b_90["lower_ci"][0])
        assert width_90 < width_95

    def test_integer_base_level_coerced(self) -> None:
        """Integer base_level should be compared after str() coercion."""
        df = _make_agg_df(["0", "1", "2"], [0.0, 0.2, 0.4])
        # Pass int 0 — should match level "0"
        result = normalise_base_level(df, 0)
        base_row = result.filter(pl.col("level") == "0")
        assert abs(base_row["relativity"][0] - 1.0) < 1e-12


# ===========================================================================
# _normalisation.py — normalise_mean
# ===========================================================================


class TestNormaliseMean:
    """Tests for normalise_mean."""

    def test_weighted_geometric_mean_is_one(self) -> None:
        """Exposure-weighted geometric mean of relativities must equal 1.0."""
        df = _make_agg_df(
            ["A", "B", "C"],
            [0.1, -0.2, 0.3],
            exposure_weights=[0.5, 0.3, 0.2],
        )
        result = normalise_mean(df)
        log_rels = np.log(result["relativity"].to_numpy())
        weights = result["exposure_weight"].to_numpy()
        wm = np.average(log_rels, weights=weights)
        assert abs(wm) < 1e-9

    def test_zero_total_weight_no_crash(self) -> None:
        """When all exposure weights are zero, the function should not raise."""
        df = _make_agg_df(
            ["A", "B"],
            [0.1, -0.1],
            exposure_weights=[0.0, 0.0],
        )
        result = normalise_mean(df)
        assert "relativity" in result.columns

    def test_ci_columns_present(self) -> None:
        """Both CI columns must appear in the output."""
        df = _make_agg_df(["A", "B"], [0.0, 0.3])
        result = normalise_mean(df)
        assert "lower_ci" in result.columns
        assert "upper_ci" in result.columns

    def test_lower_leq_upper(self) -> None:
        """lower_ci must never exceed upper_ci."""
        df = _make_agg_df(
            ["A", "B", "C"],
            [0.0, 0.5, -0.3],
            shap_stds=[0.1, 0.2, 0.15],
        )
        result = normalise_mean(df)
        assert (result["lower_ci"] <= result["upper_ci"]).all()

    def test_relativities_positive(self) -> None:
        """All relativities must be strictly positive."""
        df = _make_agg_df(["A", "B", "C"], [0.1, -5.0, 5.0])
        result = normalise_mean(df)
        assert (result["relativity"] > 0).all()


# ===========================================================================
# _validation.py
# ===========================================================================


class TestCheckReconstruction:
    """Tests for check_reconstruction."""

    def test_passes_when_within_tolerance(self) -> None:
        """Perfect reconstruction should pass."""
        n, p = 10, 3
        shap_vals = np.zeros((n, p))
        expected = 0.5
        preds = np.full(n, math.exp(expected))
        result = check_reconstruction(shap_vals, expected, preds)
        assert isinstance(result, CheckResult)
        assert result.passed

    def test_fails_when_outside_tolerance(self) -> None:
        """Large reconstruction error should fail."""
        n, p = 5, 2
        shap_vals = np.zeros((n, p))
        expected = 0.0
        preds = np.ones(n) * 2.0  # exp(0) = 1, so error = 1.0
        result = check_reconstruction(shap_vals, expected, preds, tolerance=0.1)
        assert not result.passed

    def test_value_is_max_abs_error(self) -> None:
        """The value field should be the maximum absolute error across obs."""
        shap_vals = np.array([[0.0, 0.0], [0.0, 0.0]])
        expected = 0.0
        preds = np.array([1.0, 2.0])  # errors: 0, 1
        result = check_reconstruction(shap_vals, expected, preds)
        assert abs(result.value - 1.0) < 1e-9

    def test_message_contains_error(self) -> None:
        """The message should always contain the reconstruction error value."""
        shap_vals = np.zeros((3, 2))
        result = check_reconstruction(shap_vals, 0.0, np.ones(3))
        assert "Max absolute" in result.message


class TestCheckFeatureCoverage:
    """Tests for check_feature_coverage."""

    def test_passes_when_all_present(self) -> None:
        """Coverage passes when SHAP features match expected features."""
        result = check_feature_coverage(["a", "b", "c"], ["a", "b", "c"])
        assert result.passed

    def test_fails_when_features_missing(self) -> None:
        """Coverage fails when expected features are absent from SHAP output."""
        result = check_feature_coverage(["a"], ["a", "b"])
        assert not result.passed

    def test_value_is_count_of_missing(self) -> None:
        """Value should be the number of missing features."""
        result = check_feature_coverage(["a"], ["a", "b", "c"])
        assert result.value == 2.0

    def test_message_names_missing_features(self) -> None:
        """The message should list the missing feature names."""
        result = check_feature_coverage(["a"], ["a", "b"])
        assert "b" in result.message

    def test_extra_shap_features_still_pass(self) -> None:
        """Extra SHAP features beyond what is expected should not cause failure."""
        result = check_feature_coverage(["a", "b", "extra"], ["a", "b"])
        assert result.passed


class TestCheckSparseLevels:
    """Tests for check_sparse_levels."""

    def test_passes_when_all_levels_dense(self) -> None:
        """No sparse levels — should pass cleanly."""
        df = pl.DataFrame({"level": ["A", "B"], "n_obs": [100, 200]})
        result = check_sparse_levels(df, min_obs=30)
        assert result.passed

    def test_fails_with_warning_when_sparse(self) -> None:
        """Sparse level should fail and emit a UserWarning."""
        df = pl.DataFrame({"level": ["A", "B"], "n_obs": [100, 5]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = check_sparse_levels(df, min_obs=30)
        assert not result.passed
        assert any(issubclass(ww.category, UserWarning) for ww in w)

    def test_value_is_sparse_count(self) -> None:
        """Value should equal the number of sparse levels."""
        df = pl.DataFrame({"level": ["A", "B", "C"], "n_obs": [1, 2, 200]})
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = check_sparse_levels(df, min_obs=30)
        assert result.value == 2.0

    def test_no_n_obs_column_passes(self) -> None:
        """When n_obs is absent, the check should pass gracefully."""
        df = pl.DataFrame({"level": ["A", "B"]})
        result = check_sparse_levels(df)
        assert result.passed

    def test_all_above_custom_threshold(self) -> None:
        """Custom min_obs threshold should be respected."""
        df = pl.DataFrame({"level": ["A", "B"], "n_obs": [50, 75]})
        # min_obs=100 — both are below
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = check_sparse_levels(df, min_obs=100)
        assert not result.passed
        assert result.value == 2.0


# ===========================================================================
# _core.py — _to_polars type guard
# ===========================================================================


class TestToPolarsTypeGuard:
    """Tests for the _to_polars helper in _core."""

    def test_unsupported_type_raises_type_error(self) -> None:
        """Passing a dict should raise TypeError, not a cryptic crash."""
        from shap_relativities._core import _to_polars
        with pytest.raises(TypeError, match="Polars or pandas DataFrame"):
            _to_polars({"a": [1, 2, 3]})

    def test_polars_frame_passes_through(self) -> None:
        """A Polars DataFrame should be returned as-is."""
        from shap_relativities._core import _to_polars
        df = pl.DataFrame({"x": [1, 2, 3]})
        result = _to_polars(df)
        assert result is df

    def test_pandas_frame_converts(self) -> None:
        """A pandas DataFrame should be converted to Polars."""
        import pandas as pd
        from shap_relativities._core import _to_polars
        pdf = pd.DataFrame({"x": [1, 2, 3]})
        result = _to_polars(pdf)
        assert isinstance(result, pl.DataFrame)


# ===========================================================================
# _core.py — feature inference
# ===========================================================================


class TestFeatureInference:
    """Tests for _infer_categorical and _infer_continuous on SHAPRelativities."""

    def _minimal_sr(self, X: pl.DataFrame):
        """Create a SHAPRelativities with no explicit feature lists."""
        from shap_relativities import SHAPRelativities
        # We need a model stub — use a real CatBoost if available
        try:
            import catboost  # noqa: F401
        except ImportError:
            pytest.skip("catboost not installed")
        import shap  # noqa: F401
        # We only want to test __init__, not fit(), so pass a dummy model object
        # (the model is not called during __init__)
        class _DummyModel:
            pass
        return SHAPRelativities(model=_DummyModel(), X=X)

    def test_infers_string_col_as_categorical(self) -> None:
        """String columns should be inferred as categorical."""
        X = pl.DataFrame({"area": ["A", "B", "C"], "ncd": [0, 1, 2]})
        sr = self._minimal_sr(X)
        assert "area" in sr._categorical_features

    def test_infers_numeric_col_as_continuous(self) -> None:
        """Numeric columns should be inferred as continuous."""
        X = pl.DataFrame({"area": ["A", "B", "C"], "ncd": [0, 1, 2]})
        sr = self._minimal_sr(X)
        assert "ncd" in sr._continuous_features

    def test_explicit_categorical_overrides_inference(self) -> None:
        """Explicitly listed categorical features must be respected."""
        from shap_relativities import SHAPRelativities

        try:
            import catboost  # noqa: F401
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("catboost not installed")

        class _DummyModel:
            pass

        X = pl.DataFrame({"ncd": [0, 1, 2], "age": [17, 25, 40]})
        sr = SHAPRelativities(
            model=_DummyModel(), X=X, categorical_features=["ncd", "age"]
        )
        assert "ncd" in sr._categorical_features
        assert "age" in sr._categorical_features


# ===========================================================================
# _core.py — baseline() without exposure
# ===========================================================================


class TestBaselineNoExposure:
    """Tests for baseline() when no exposure is provided."""

    def test_baseline_from_dict_no_exposure(self) -> None:
        """baseline() on a from_dict instance without exposure should return exp(ev)."""
        from shap_relativities import SHAPRelativities

        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        data = {
            "shap_values": [[0.1, -0.2], [0.3, 0.4]],
            "expected_value": 0.5,
            "feature_names": ["a", "b"],
            "categorical_features": ["a"],
            "continuous_features": ["b"],
            "X_values": {"a": [1, 2], "b": [10.0, 20.0]},
            "exposure": None,
            "annualise_exposure": True,
        }
        sr = SHAPRelativities.from_dict(data)
        b = sr.baseline()
        assert abs(b - math.exp(0.5)) < 1e-9


# ===========================================================================
# _core.py — validate() without a model (from_dict path)
# ===========================================================================


class TestValidateNoModel:
    """Tests for validate() when no model is attached (from_dict path)."""

    def _sr_from_dict(self):
        from shap_relativities import SHAPRelativities

        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        data = {
            "shap_values": [[0.1, -0.2], [0.3, 0.4], [-0.1, 0.1]],
            "expected_value": -2.5,
            "feature_names": ["ncd", "area"],
            "categorical_features": ["ncd", "area"],
            "continuous_features": [],
            "X_values": {"ncd": [0, 1, 2], "area": ["A", "B", "C"]},
            "exposure": None,
            "annualise_exposure": True,
        }
        return SHAPRelativities.from_dict(data)

    def test_validate_no_model_reconstruction_fails(self) -> None:
        """When model=None, reconstruction check must return passed=False."""
        sr = self._sr_from_dict()
        checks = sr.validate()
        assert "reconstruction" in checks
        assert not checks["reconstruction"].passed

    def test_validate_no_model_feature_coverage_passes(self) -> None:
        """Feature coverage should still pass even without a model."""
        sr = self._sr_from_dict()
        checks = sr.validate()
        assert checks["feature_coverage"].passed


# ===========================================================================
# _core.py — extract_relativities: continuous features go through mean path
# ===========================================================================


class TestExtractRelativitiesContinuous:
    """Tests for extract_relativities() when continuous features are present."""

    def _sr_from_dict_with_continuous(self):
        from shap_relativities import SHAPRelativities

        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        data = {
            "shap_values": [[0.1, -0.2], [0.3, 0.4], [-0.1, 0.05]],
            "expected_value": -2.5,
            "feature_names": ["ncd", "driver_age"],
            "categorical_features": ["ncd"],
            "continuous_features": ["driver_age"],
            "X_values": {"ncd": [0, 1, 2], "driver_age": [17.0, 35.0, 55.0]},
            "exposure": [1.0, 1.0, 1.0],
            "annualise_exposure": True,
        }
        return SHAPRelativities.from_dict(data)

    def test_continuous_feature_in_output(self) -> None:
        """Continuous features must appear in the extract_relativities output."""
        sr = self._sr_from_dict_with_continuous()
        rels = sr.extract_relativities(normalise_to="mean")
        assert "driver_age" in rels["feature"].to_list()

    def test_continuous_relativities_positive(self) -> None:
        """All relativities (including continuous) must be positive."""
        sr = self._sr_from_dict_with_continuous()
        rels = sr.extract_relativities(normalise_to="mean")
        assert (rels["relativity"] > 0).all()

    def test_continuous_ci_method_none(self) -> None:
        """ci_method='none' on a mixed feature set should produce NaN CIs."""
        sr = self._sr_from_dict_with_continuous()
        rels = sr.extract_relativities(normalise_to="mean", ci_method="none")
        for row in rels.iter_rows(named=True):
            assert math.isnan(row["lower_ci"])
            assert math.isnan(row["upper_ci"])


# ===========================================================================
# _plotting.py — smoke tests (no assertions on pixel values)
# ===========================================================================


class TestPlotting:
    """Smoke tests for _plotting.py.

    These verify that the plotting functions run without error. We do not
    assert on visual output, only that no exception is raised and the
    Axes objects are modified as expected.
    """

    @pytest.fixture(autouse=True)
    def _use_non_interactive_backend(self):
        """Force a non-interactive matplotlib backend for headless testing."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        yield
        plt.close("all")

    def _categorical_data(self) -> pl.DataFrame:
        return pl.DataFrame({
            "feature": ["area"] * 3,
            "level": ["A", "B", "C"],
            "relativity": [1.0, 1.3, 0.8],
            "lower_ci": [0.9, 1.1, 0.65],
            "upper_ci": [1.1, 1.5, 0.95],
            "mean_shap": [0.0, 0.26, -0.22],
            "shap_std": [0.1, 0.15, 0.12],
            "n_obs": [1000, 800, 500],
            "exposure_weight": [0.4, 0.35, 0.25],
        })

    def _continuous_data(self) -> pl.DataFrame:
        n = 20
        vals = np.linspace(17, 80, n)
        return pl.DataFrame({
            "feature": ["driver_age"] * n,
            "level": vals.astype(str).tolist(),
            "relativity": np.exp(np.linspace(-0.3, 0.3, n)).tolist(),
            "lower_ci": np.exp(np.linspace(-0.5, 0.1, n)).tolist(),
            "upper_ci": np.exp(np.linspace(-0.1, 0.5, n)).tolist(),
            "mean_shap": np.linspace(-0.3, 0.3, n).tolist(),
            "shap_std": np.full(n, 0.1).tolist(),
            "n_obs": [50] * n,
            "exposure_weight": np.full(n, 1.0 / n).tolist(),
        })

    def test_plot_categorical_no_exception(self) -> None:
        """plot_categorical should run without raising."""
        import matplotlib.pyplot as plt
        from shap_relativities._plotting import plot_categorical

        fig, ax = plt.subplots()
        plot_categorical(self._categorical_data(), "area", ax, show_ci=True)

    def test_plot_categorical_no_ci(self) -> None:
        """plot_categorical with show_ci=False should also run cleanly."""
        import matplotlib.pyplot as plt
        from shap_relativities._plotting import plot_categorical

        fig, ax = plt.subplots()
        plot_categorical(self._categorical_data(), "area", ax, show_ci=False)

    def test_plot_continuous_no_exception(self) -> None:
        """plot_continuous should run without raising."""
        import matplotlib.pyplot as plt
        from shap_relativities._plotting import plot_continuous

        fig, ax = plt.subplots()
        plot_continuous(self._continuous_data(), "driver_age", ax, show_ci=True)

    def test_plot_continuous_no_ci(self) -> None:
        """plot_continuous with show_ci=False should run cleanly."""
        import matplotlib.pyplot as plt
        from shap_relativities._plotting import plot_continuous

        fig, ax = plt.subplots()
        plot_continuous(self._continuous_data(), "driver_age", ax, show_ci=False)

    def test_plot_relativities_grid_no_exception(self) -> None:
        """The grid plotter should render all features without raising."""
        import matplotlib.pyplot as plt
        from shap_relativities._plotting import plot_relativities

        combined = pl.concat([
            self._categorical_data(),
            self._continuous_data(),
        ])
        plot_relativities(
            combined,
            categorical_features=["area"],
            continuous_features=["driver_age"],
            show_ci=True,
        )

    def test_plot_relativities_single_feature(self) -> None:
        """Single-feature grid (n=1) should not crash on axis flattening."""
        import matplotlib.pyplot as plt
        from shap_relativities._plotting import plot_relativities

        plot_relativities(
            self._categorical_data(),
            categorical_features=["area"],
            continuous_features=[],
        )

    def test_plot_relativities_unknown_feature_uses_categorical(self) -> None:
        """Features not in categorical or continuous lists use the fallback bar chart."""
        import matplotlib.pyplot as plt
        from shap_relativities._plotting import plot_relativities

        # 'area' is not in categorical_features or continuous_features
        plot_relativities(
            self._categorical_data(),
            categorical_features=[],
            continuous_features=[],
        )

    def test_plot_relativities_subset_features(self) -> None:
        """The features parameter should restrict what is plotted."""
        import matplotlib.pyplot as plt
        from shap_relativities._plotting import plot_relativities

        combined = pl.concat([
            self._categorical_data(),
            self._continuous_data(),
        ])
        # Only plot area, not driver_age
        plot_relativities(
            combined,
            categorical_features=["area"],
            continuous_features=["driver_age"],
            features=["area"],
        )
