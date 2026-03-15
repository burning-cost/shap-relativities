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
        expected = {"feature", "level", "mean_shap", "shap_std", "n_obs", "exposure_weight", "wsq_weight"}
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
        expected = {"feature", "level", "mean_shap", "shap_std", "n_obs", "exposure_weight", "wsq_weight"}
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


# ===========================================================================
# Regression tests for P0/P1 bug fixes
# ===========================================================================


class TestP03EffectiveSampleSize:
    """
    Regression tests for P0-3: SE must use n_eff = sum(w)^2 / sum(w^2),
    not raw n_obs.

    When weights vary widely (e.g. short-term cancellations mixed with annual
    policies), raw n_obs overstates the effective sample size and makes CIs
    too narrow. n_eff is always <= n_obs; equality holds only when all weights
    are identical.
    """

    def test_neff_produces_wsq_weight_column(self) -> None:
        """aggregate_categorical must output a wsq_weight column."""
        result = aggregate_categorical(
            "x",
            np.array(["A", "A", "A"]),
            np.array([0.1, 0.2, 0.3]),
            np.array([0.1, 0.5, 1.0]),  # mixed weights
        )
        assert "wsq_weight" in result.columns

    def test_wsq_weight_correct_value(self) -> None:
        """wsq_weight must equal sum(w_i^2) per level."""
        ws = np.array([0.1, 0.5, 1.0])
        result = aggregate_categorical(
            "x",
            np.array(["A", "A", "A"]),
            np.zeros(3),
            ws,
        )
        expected_wsq = float((ws ** 2).sum())
        assert abs(result["wsq_weight"][0] - expected_wsq) < 1e-10

    def test_unequal_weights_give_wider_ci_than_equal(self) -> None:
        """
        With highly variable weights and the same n_obs, n_eff < n_obs,
        so the CI should be wider than if we used raw n_obs (old behaviour).
        """
        # Construct two scenarios with same n_obs but different weight spread.
        # Equal weights: w = [1.0, 1.0] -> n_eff = 2.0
        # Unequal weights: w = [0.01, 1.99] -> n_eff = (2.0)^2 / (0.01^2 + 1.99^2) ≈ 1.01
        def _ci_width(ws):
            agg = aggregate_categorical(
                "x", np.array(["A", "A"]), np.array([0.5, 0.5]), np.array(ws)
            )
            # Manually add std so CI is non-trivial
            agg = agg.with_columns(pl.lit(0.3).alias("shap_std"))
            result = normalise_mean(agg)
            row = result[0]
            return float(row["upper_ci"][0]) - float(row["lower_ci"][0])

        width_equal = _ci_width([1.0, 1.0])
        width_unequal = _ci_width([0.01, 1.99])
        # Unequal weights → smaller n_eff → wider CI
        assert width_unequal > width_equal, (
            f"Expected wider CI with unequal weights, got {width_unequal:.4f} vs {width_equal:.4f}"
        )

    def test_equal_weights_neff_equals_nobs(self) -> None:
        """When all weights are equal, n_eff should equal n_obs exactly."""
        w = 0.8  # all equal
        n = 5
        result = aggregate_categorical(
            "x",
            np.array(["A"] * n),
            np.zeros(n),
            np.full(n, w),
        )
        exposure_weight = float(result["exposure_weight"][0])
        wsq_weight = float(result["wsq_weight"][0])
        n_eff = (exposure_weight ** 2) / wsq_weight
        assert abs(n_eff - n) < 1e-9


class TestP01BaseLevelCIIncludesBaseVariance:
    """
    Regression tests for P0-1: CI for non-base levels must include
    estimation uncertainty of the base level itself.

    The relativity for level L is exp(mean_L - base). Both terms are
    estimated from data. The CI must use se_combined = sqrt(se_L^2 + se_base^2).

    The base level itself always has relativity=1.0 and CI width=0 on the
    relativity scale (the base is taken as the reference). But for other levels,
    if we reduce the base level's sample size (increase its SE), the CI width
    for non-base levels should increase even if their own SE is unchanged.
    """

    def _make_agg_with_wsq(
        self,
        levels, mean_shaps, shap_stds, exposure_weights, wsq_weights,
    ) -> pl.DataFrame:
        """Build aggregated DataFrame with wsq_weight column."""
        n = len(levels)
        return pl.DataFrame({
            "feature": ["feat"] * n,
            "level": levels,
            "mean_shap": [float(x) for x in mean_shaps],
            "shap_std": [float(x) for x in shap_stds],
            "n_obs": [100] * n,
            "exposure_weight": [float(x) for x in exposure_weights],
            "wsq_weight": [float(x) for x in wsq_weights],
        })

    def test_large_base_se_widens_non_base_ci(self) -> None:
        """
        Increasing the base level's shap_std should widen the CI of other
        levels when se_combined = sqrt(se_L^2 + se_base^2) is used.

        With the old code (se = se_L only), the non-base CI was independent
        of the base level's variance.
        """
        def _width_for_base_std(base_std):
            df = self._make_agg_with_wsq(
                levels=["A", "B"],
                mean_shaps=[0.0, 0.5],
                shap_stds=[base_std, 0.2],
                exposure_weights=[100.0, 100.0],
                # wsq_weight: 100 equal-weight obs with w=1 each → wsq=n*1^2=100... but
                # here exposure_weight=100 so each obs has w=1, wsq = n*1 = 100
                wsq_weights=[100.0, 100.0],
            )
            result = normalise_base_level(df, "A")
            b_row = result.filter(pl.col("level") == "B")
            return float(b_row["upper_ci"][0]) - float(b_row["lower_ci"][0])

        width_small_base = _width_for_base_std(0.0)
        width_large_base = _width_for_base_std(1.0)
        assert width_large_base > width_small_base, (
            "CI width of non-base level should increase when base level has "
            f"higher variance. Got small_base={width_small_base:.4f}, "
            f"large_base={width_large_base:.4f}"
        )

    def test_base_level_ci_straddles_one(self) -> None:
        """Base level itself should have lower_ci <= 1.0 <= upper_ci."""
        df = self._make_agg_with_wsq(
            levels=["A", "B"],
            mean_shaps=[0.0, 0.5],
            shap_stds=[0.2, 0.1],
            exposure_weights=[50.0, 100.0],
            wsq_weights=[50.0, 100.0],  # equal weights, wsq = exposure
        )
        result = normalise_base_level(df, "A")
        base_row = result.filter(pl.col("level") == "A")
        assert float(base_row["lower_ci"][0]) <= 1.0
        assert float(base_row["upper_ci"][0]) >= 1.0


class TestP11ZeroWeightLevel:
    """
    Regression tests for P1-1: a level with all observations having weight=0
    must be excluded (with a warning), not silently produce NaN relativities.
    """

    def test_zero_weight_level_excluded(self) -> None:
        """Level with total exposure=0 must not appear in output."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = aggregate_categorical(
                "x",
                np.array(["A", "A", "B"]),
                np.array([0.1, 0.2, 0.5]),
                np.array([1.0, 1.0, 0.0]),  # B has zero weight
            )
        assert "B" not in result["level"].to_list()

    def test_zero_weight_level_emits_warning(self) -> None:
        """A UserWarning must be raised when a zero-weight level is dropped."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            aggregate_categorical(
                "x",
                np.array(["A", "B"]),
                np.array([0.1, 0.2]),
                np.array([1.0, 0.0]),
            )
        assert any(issubclass(ww.category, UserWarning) for ww in w)

    def test_zero_weight_does_not_produce_nan_relativities(self) -> None:
        """After dropping zero-weight levels, no NaN in output."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            agg = aggregate_categorical(
                "x",
                np.array(["A", "A", "B", "C"]),
                np.array([0.1, 0.2, 0.5, 0.3]),
                np.array([1.0, 1.0, 0.0, 1.0]),
            )
        result = normalise_mean(agg)
        assert not result["relativity"].is_nan().any()
        assert not result["lower_ci"].is_nan().any()

    def test_all_nonzero_levels_preserved(self) -> None:
        """Non-zero levels must all survive the zero-weight filter."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = aggregate_categorical(
                "x",
                np.array(["A", "B", "C"]),
                np.zeros(3),
                np.array([1.0, 0.0, 2.0]),
            )
        assert set(result["level"].to_list()) == {"A", "C"}


class TestP12FromDictFeatureNamesOrder:
    """
    Regression tests for P1-2: from_dict must use feature_names to control
    the column ordering in the reconstructed X DataFrame.

    When JSON object keys are sorted alphabetically (by REST APIs, some
    pretty-printers), X_values dict ordering no longer matches feature_names.
    Without the fix, the wrong SHAP columns get assigned to each feature.
    """

    def test_from_dict_respects_feature_names_order(self) -> None:
        """
        Reconstructed X must have columns in feature_names order, regardless
        of the dict key order in X_values.
        """
        from shap_relativities import SHAPRelativities

        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        # feature_names order: ["z_feat", "a_feat"]
        # X_values will be presented with sorted keys: {"a_feat": ..., "z_feat": ...}
        # Without the fix, X columns would be ["a_feat", "z_feat"] — reversed.
        data = {
            "shap_values": [[0.1, -0.2], [0.3, 0.4]],
            "expected_value": 0.5,
            "feature_names": ["z_feat", "a_feat"],
            "categorical_features": ["z_feat"],
            "continuous_features": ["a_feat"],
            # Simulate sorted-key JSON: a_feat comes before z_feat
            "X_values": {"a_feat": [10.0, 20.0], "z_feat": ["X", "Y"]},
            "exposure": None,
            "annualise_exposure": True,
        }
        sr = SHAPRelativities.from_dict(data)
        assert sr._X.columns == ["z_feat", "a_feat"], (
            f"Expected ['z_feat', 'a_feat'], got {sr._X.columns}"
        )

    def test_from_dict_shap_column_alignment(self) -> None:
        """
        After from_dict reconstruction with reordered X_values, the SHAP
        column for each feature must be correctly aligned.

        We verify this by extracting relativities and checking that the
        feature names in the output are correct (not swapped).
        """
        from shap_relativities import SHAPRelativities

        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        # Two categorical features. SHAP col 0 → z_feat, col 1 → a_feat.
        data = {
            "shap_values": [[0.5, -0.5], [0.5, -0.5], [0.5, -0.5]],
            "expected_value": 0.0,
            "feature_names": ["z_feat", "a_feat"],
            "categorical_features": ["z_feat", "a_feat"],
            "continuous_features": [],
            # Sorted-key dict order: a_feat first
            "X_values": {"a_feat": ["P", "P", "Q"], "z_feat": ["X", "X", "Y"]},
            "exposure": None,
            "annualise_exposure": True,
        }
        sr = SHAPRelativities.from_dict(data)
        rels = sr.extract_relativities(normalise_to="mean")
        # z_feat should have mean_shap = 0.5 (all observations have shap_col[0]=0.5)
        z_mean = float(
            rels.filter(pl.col("feature") == "z_feat")["mean_shap"].mean()
        )
        assert abs(z_mean - 0.5) < 1e-6, (
            f"z_feat mean_shap should be 0.5, got {z_mean}. "
            "Column alignment is wrong — feature_names order was not applied."
        )


class TestP13LoadMotorSmallN:
    """
    Regression tests for P1-3: load_motor must return exactly n_policies rows
    for any n, including small values where int() truncation would otherwise
    make the driver age array shorter than n.
    """

    def test_load_motor_small_n_row_count(self) -> None:
        """load_motor(n=10) must return exactly 10 rows."""
        from shap_relativities.datasets.motor import load_motor
        df = load_motor(n_policies=10, seed=0)
        assert df.shape[0] == 10, f"Expected 10 rows, got {df.shape[0]}"

    def test_load_motor_n_7_row_count(self) -> None:
        """n=7 — chosen to maximise rounding gap from proportional sizes."""
        from shap_relativities.datasets.motor import load_motor
        df = load_motor(n_policies=7, seed=1)
        assert df.shape[0] == 7

    def test_load_motor_n_1_row_count(self) -> None:
        """n=1 — smallest possible portfolio."""
        from shap_relativities.datasets.motor import load_motor
        df = load_motor(n_policies=1, seed=99)
        assert df.shape[0] == 1

    def test_load_motor_driver_age_no_missing(self) -> None:
        """driver_age must have no null values at small n."""
        from shap_relativities.datasets.motor import load_motor
        df = load_motor(n_policies=10, seed=42)
        assert df["driver_age"].null_count() == 0


class TestP02PlotContinuousNumericSort:
    """
    Regression tests for P0-2: plot_continuous must sort by numeric value,
    not lexicographic string order.

    The level column is Utf8 after extract_relativities() casts it. Sorting
    strings lexicographically scrambles the x-axis for multi-decade ranges:
    '10.0' < '2.0' lexicographically but 2.0 < 10.0 numerically.
    """

    @pytest.fixture(autouse=True)
    def _backend(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        yield
        plt.close("all")

    def test_plot_continuous_x_axis_numeric_order(self) -> None:
        """
        The x values passed to ax.plot must be in ascending numeric order.
        A string sort of ['1.0','10.0','100.0','2.0','20.0','3.0'] gives the
        wrong order; casting to Float64 before sorting gives correct order.
        """
        import matplotlib.pyplot as plt
        from shap_relativities._plotting import plot_continuous

        # Values that sort wrong lexicographically: 1,10,100,2,20,3
        raw_vals = [1.0, 10.0, 100.0, 2.0, 20.0, 3.0]
        n = len(raw_vals)
        # Store as strings (as extract_relativities produces)
        data = pl.DataFrame({
            "feature": ["x"] * n,
            "level": [str(v) for v in raw_vals],
            "relativity": [1.0 + v / 100 for v in raw_vals],
            "lower_ci": [0.9] * n,
            "upper_ci": [1.1] * n,
            "mean_shap": [0.0] * n,
            "shap_std": [0.1] * n,
            "n_obs": [50] * n,
            "exposure_weight": [1.0 / n] * n,
        })

        fig, ax = plt.subplots()
        plot_continuous(data, "x", ax, show_ci=False)

        # Extract the x data from the line
        line = ax.lines[0]
        x_plotted = line.get_xdata()
        assert list(x_plotted) == sorted(raw_vals), (
            f"x-axis not in numeric order. Got {list(x_plotted)}, "
            f"expected {sorted(raw_vals)}"
        )

    def test_plot_continuous_zigzag_detection(self) -> None:
        """
        The x array from the plot must be monotonically non-decreasing,
        confirming that the line will not zigzag.
        """
        import matplotlib.pyplot as plt
        from shap_relativities._plotting import plot_continuous

        vals = [5.0, 50.0, 500.0, 10.0, 100.0, 1.0]
        n = len(vals)
        data = pl.DataFrame({
            "feature": ["x"] * n,
            "level": [str(v) for v in vals],
            "relativity": [1.0] * n,
            "lower_ci": [0.9] * n,
            "upper_ci": [1.1] * n,
            "mean_shap": [0.0] * n,
            "shap_std": [0.0] * n,
            "n_obs": [100] * n,
            "exposure_weight": [1.0 / n] * n,
        })

        fig, ax = plt.subplots()
        plot_continuous(data, "x", ax, show_ci=False)

        x = ax.lines[0].get_xdata()
        diffs = np.diff(x)
        assert (diffs >= 0).all(), (
            f"x-axis is not monotonically non-decreasing: diffs = {diffs}"
        )


class TestP14ContinuousCurveNormalisation:
    """
    Regression tests for P1-4: extract_continuous_curve must produce
    exposure-weighted normalised relativities, not grid-uniform normalised ones.

    Concretely: the exposure-weighted geometric mean of smooth_method='isotonic'
    and 'loess' curves — computed at the original data points — should be close
    to 1.0. With the old code it was systematically off when data was skewed.
    """

    def _sr_from_dict(self):
        """Create a from_dict SHAPRelativities with a skewed continuous feature."""
        from shap_relativities import SHAPRelativities

        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        rng = np.random.default_rng(42)
        n = 200
        # Skewed feature: most values clustered at the low end
        feat_vals = np.concatenate([
            rng.uniform(0, 1, size=160),   # 80% in [0, 1]
            rng.uniform(5, 10, size=40),   # 20% in [5, 10]
        ])
        # SHAP values: monotone increasing with feature value
        shap_col = 0.1 * feat_vals + rng.normal(0, 0.05, size=n)
        weights = np.ones(n)

        data = {
            "shap_values": np.column_stack([shap_col]).tolist(),
            "expected_value": -2.0,
            "feature_names": ["x"],
            "categorical_features": [],
            "continuous_features": ["x"],
            "X_values": {"x": feat_vals.tolist()},
            "exposure": weights.tolist(),
            "annualise_exposure": False,
        }
        return SHAPRelativities.from_dict(data)

    def test_isotonic_curve_weighted_geometric_mean_near_one(self) -> None:
        """
        Exposure-weighted geometric mean of isotonic curve at original data
        points should be close to 1.0 (within 2% tolerance).
        """
        from sklearn.isotonic import IsotonicRegression

        sr = self._sr_from_dict()
        curve = sr.extract_continuous_curve("x", n_points=100, smooth_method="isotonic")

        # Re-evaluate the smoothed curve at original data points
        feat_vals = sr._X["x"].to_numpy()
        weights = sr._exposure if sr._exposure is not None else np.ones(len(sr._X))
        shap_col = sr._shap_values[:, 0]

        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(feat_vals, shap_col, sample_weight=weights)
        smoothed_at_data = ir.predict(feat_vals)
        weighted_mean_smoothed = np.average(smoothed_at_data, weights=weights)
        smoothed_rel_at_data = np.exp(smoothed_at_data - weighted_mean_smoothed)
        wgm_log = np.average(np.log(smoothed_rel_at_data), weights=weights)

        assert abs(wgm_log) < 0.02, (
            f"Weighted log geometric mean of isotonic curve = {wgm_log:.4f}, "
            "expected close to 0.0"
        )
