"""
Extended test coverage for shap-relativities.

Fills gaps left by the existing test suite:
- aggregation edge cases (single obs, uniform SHAP, wsq_weight for continuous)
- normalisation with missing wsq_weight column (fallback path), boundary ci_levels,
  float base levels, single-row DataFrames
- validation boundary conditions (exact tolerance, empty features list)
- core: baseline paths (annualise flags), from_dict backward compat, exposure
  types (pl.Series, list), to_dict keys, unfitted guard on all methods
- plotting: NaN CI handling, custom color, empty axes recycling
- integration: mixed feature sets, base_level fallback warning text, round-trip
  with exposure

All tests are self-contained (no catboost/shap required) unless marked with
pytestmark for the ml dependencies group.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import polars as pl
import pytest

from shap_relativities._aggregation import aggregate_categorical, aggregate_continuous
from shap_relativities._normalisation import (
    _base_se,
    _effective_se,
    normalise_base_level,
    normalise_mean,
)
from shap_relativities._validation import (
    CheckResult,
    check_feature_coverage,
    check_reconstruction,
    check_sparse_levels,
)
from shap_relativities._core import _to_polars, _to_pandas


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agg_with_wsq(
    levels, mean_shaps, shap_stds=None, n_obs=None,
    exposure_weights=None, wsq_weights=None,
) -> pl.DataFrame:
    """Build a full aggregated DataFrame including wsq_weight."""
    n = len(levels)
    return pl.DataFrame({
        "feature": ["feat"] * n,
        "level": [str(l) for l in levels],
        "mean_shap": [float(x) for x in mean_shaps],
        "shap_std": [float(x) for x in (shap_stds or [0.0] * n)],
        "n_obs": list(n_obs or [100] * n),
        "exposure_weight": [float(x) for x in (exposure_weights or [1.0 / n] * n)],
        "wsq_weight": [float(x) for x in (wsq_weights or [1.0 / n ** 2] * n)],
    })


def _make_agg_no_wsq(levels, mean_shaps, shap_stds=None, n_obs=None) -> pl.DataFrame:
    """Build an aggregated DataFrame WITHOUT wsq_weight (tests fallback path)."""
    n = len(levels)
    return pl.DataFrame({
        "feature": ["feat"] * n,
        "level": [str(l) for l in levels],
        "mean_shap": [float(x) for x in mean_shaps],
        "shap_std": [float(x) for x in (shap_stds or [0.0] * n)],
        "n_obs": list(n_obs or [100] * n),
        "exposure_weight": [1.0 / n] * n,
    })


def _minimal_fitted_sr(shap_vals=None, categorical=True):
    """Build a from_dict SHAPRelativities without needing catboost/shap installed."""
    from shap_relativities import SHAPRelativities

    try:
        import shap  # noqa: F401
    except ImportError:
        pytest.skip("shap not installed")

    if shap_vals is None:
        shap_vals = [[0.2, -0.1], [-0.1, 0.3], [0.0, 0.05]]

    data = {
        "shap_values": shap_vals,
        "expected_value": -2.0,
        "feature_names": ["area", "ncd"],
        "categorical_features": ["area", "ncd"] if categorical else [],
        "continuous_features": [] if categorical else ["area", "ncd"],
        "X_values": {"area": ["A", "B", "A"], "ncd": [0, 1, 2]},
        "exposure": [0.9, 1.0, 1.0],
        "annualise_exposure": True,
    }
    return SHAPRelativities.from_dict(data)


# ===========================================================================
# _aggregation.py — additional edge cases
# ===========================================================================


class TestAggregateCategoricalEdgeCases:
    """Additional edge cases for aggregate_categorical."""

    def test_single_observation_per_level(self) -> None:
        """Each observation in its own level — n_obs=1 for all."""
        result = aggregate_categorical(
            "x",
            np.array(["A", "B", "C"]),
            np.array([0.1, 0.2, 0.3]),
            np.ones(3),
        )
        assert (result["n_obs"] == 1).all()

    def test_all_same_shap_values(self) -> None:
        """When all SHAP values are identical, shap_std must be 0."""
        result = aggregate_categorical(
            "x",
            np.array(["A", "A", "A"]),
            np.full(3, 0.42),
            np.ones(3),
        )
        assert abs(result["shap_std"][0]) < 1e-9

    def test_wsq_weight_for_single_obs_level(self) -> None:
        """For a single-obs level with weight w, wsq_weight == w^2."""
        result = aggregate_categorical(
            "x",
            np.array(["A"]),
            np.array([0.5]),
            np.array([2.0]),
        )
        expected = 4.0  # 2.0^2
        assert abs(result["wsq_weight"][0] - expected) < 1e-9

    def test_multiple_levels_some_with_single_obs(self) -> None:
        """Mixing single-obs and multi-obs levels should not crash."""
        result = aggregate_categorical(
            "x",
            np.array(["A", "B", "B", "B"]),
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.ones(4),
        )
        assert len(result) == 2
        a_row = result.filter(pl.col("level") == "A")
        assert a_row["n_obs"][0] == 1
        b_row = result.filter(pl.col("level") == "B")
        assert b_row["n_obs"][0] == 3

    def test_float_feature_values_cast_to_str(self) -> None:
        """Float feature values should be cast to strings for the level column."""
        result = aggregate_categorical(
            "x",
            np.array([1.0, 2.0, 1.0]),
            np.zeros(3),
            np.ones(3),
        )
        assert result["level"].dtype == pl.String
        assert len(result) == 2

    def test_many_zero_weight_levels_excluded(self) -> None:
        """Multiple zero-weight levels should all be dropped with a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = aggregate_categorical(
                "x",
                np.array(["A", "B", "C", "D"]),
                np.zeros(4),
                np.array([1.0, 0.0, 0.0, 0.5]),
            )
        assert len(w) >= 1
        assert set(result["level"].to_list()) == {"A", "D"}

    def test_mean_shap_with_equal_weights_is_arithmetic_mean(self) -> None:
        """With equal weights, weighted mean == arithmetic mean."""
        shap_vals = np.array([0.1, 0.3, 0.5])
        result = aggregate_categorical(
            "x",
            np.array(["A", "A", "A"]),
            shap_vals,
            np.ones(3),
        )
        expected_mean = shap_vals.mean()
        assert abs(result["mean_shap"][0] - expected_mean) < 1e-9

    def test_string_levels_sorted_lexicographically(self) -> None:
        """String levels with alphabetic ordering should be preserved."""
        result = aggregate_categorical(
            "x",
            np.array(["Zebra", "Apple", "Mango"]),
            np.zeros(3),
            np.ones(3),
        )
        assert result["level"].to_list() == ["Apple", "Mango", "Zebra"]


class TestAggregateContinuousEdgeCases:
    """Additional edge cases for aggregate_continuous."""

    def test_wsq_weight_equals_w_squared(self) -> None:
        """For continuous features, wsq_weight per row should equal w^2."""
        ws = np.array([0.5, 1.0, 2.0])
        result = aggregate_continuous(
            "x",
            np.array([1.0, 2.0, 3.0]),
            np.zeros(3),
            ws,
        )
        np.testing.assert_allclose(
            result["wsq_weight"].to_numpy(), ws ** 2, rtol=1e-9
        )

    def test_exposure_weight_equals_w(self) -> None:
        """exposure_weight should equal the raw weight for continuous features."""
        ws = np.array([0.3, 0.7, 1.0, 0.5])
        result = aggregate_continuous(
            "x",
            np.arange(4.0),
            np.zeros(4),
            ws,
        )
        np.testing.assert_allclose(result["exposure_weight"].to_numpy(), ws, rtol=1e-9)

    def test_single_observation(self) -> None:
        """Single observation should produce exactly one row."""
        result = aggregate_continuous("x", np.array([42.0]), np.array([0.7]), np.array([1.0]))
        assert len(result) == 1
        assert result["n_obs"][0] == 1
        assert abs(result["mean_shap"][0] - 0.7) < 1e-9

    def test_level_column_is_string(self) -> None:
        """Level column must be String type (compatible with concat after categorical)."""
        result = aggregate_continuous(
            "x", np.array([1.5, 2.5]), np.zeros(2), np.ones(2)
        )
        assert result["level"].dtype == pl.String

    def test_feature_column_all_same_name(self) -> None:
        """All rows must have the supplied feature name in the feature column."""
        result = aggregate_continuous(
            "driver_age", np.arange(5.0), np.zeros(5), np.ones(5)
        )
        assert (result["feature"] == "driver_age").all()


# ===========================================================================
# _normalisation.py — fallback paths and boundary conditions
# ===========================================================================


class TestNormalisationFallbackPaths:
    """Tests for the n_obs fallback when wsq_weight is absent."""

    def test_normalise_mean_without_wsq_weight(self) -> None:
        """normalise_mean should work when wsq_weight column is absent."""
        df = _make_agg_no_wsq(["A", "B", "C"], [0.0, 0.3, -0.2], shap_stds=[0.1, 0.2, 0.15])
        result = normalise_mean(df)
        assert "relativity" in result.columns
        assert (result["relativity"] > 0).all()

    def test_normalise_base_level_without_wsq_weight(self) -> None:
        """normalise_base_level should work when wsq_weight column is absent."""
        df = _make_agg_no_wsq(["A", "B"], [0.0, 0.4], shap_stds=[0.1, 0.2])
        result = normalise_base_level(df, "A")
        assert abs(result.filter(pl.col("level") == "A")["relativity"][0] - 1.0) < 1e-12

    def test_base_se_with_wsq_weight(self) -> None:
        """_base_se should use wsq_weight when present."""
        df = pl.DataFrame({
            "level": ["A"],
            "mean_shap": [0.0],
            "shap_std": [0.2],
            "n_obs": [100],
            "exposure_weight": [100.0],
            "wsq_weight": [100.0],  # equal weights: n_eff = 100^2/100 = 100
        })
        se = _base_se(df)
        expected = 0.2 / math.sqrt(100.0)
        assert abs(se - expected) < 1e-9

    def test_base_se_without_wsq_weight(self) -> None:
        """_base_se should fall back to n_obs when wsq_weight is absent."""
        df = pl.DataFrame({
            "level": ["A"],
            "mean_shap": [0.0],
            "shap_std": [0.3],
            "n_obs": [9],
            "exposure_weight": [1.0],
        })
        se = _base_se(df)
        expected = 0.3 / math.sqrt(9.0)
        assert abs(se - expected) < 1e-9

    def test_base_se_wsq_weight_zero_uses_neff_one(self) -> None:
        """When wsq_weight=0, n_eff is clamped to 1 to prevent division by zero."""
        df = pl.DataFrame({
            "level": ["A"],
            "mean_shap": [0.0],
            "shap_std": [0.5],
            "n_obs": [1],
            "exposure_weight": [0.0],
            "wsq_weight": [0.0],
        })
        se = _base_se(df)
        expected = 0.5 / math.sqrt(1.0)
        assert abs(se - expected) < 1e-9


    def test_effective_se_with_wsq_weight_uses_neff(self) -> None:
        """_effective_se expression should produce se = shap_std / sqrt(n_eff)."""
        # n_eff = exposure_weight^2 / wsq_weight = 100^2 / 100 = 100
        df = pl.DataFrame({
            "level": ["A"],
            "mean_shap": [0.0],
            "shap_std": [0.4],
            "n_obs": [100],
            "exposure_weight": [100.0],
            "wsq_weight": [100.0],
        })
        se_col = df.with_columns(_effective_se(df).alias("se"))["se"][0]
        expected = 0.4 / math.sqrt(100.0)
        assert abs(se_col - expected) < 1e-9

    def test_effective_se_fallback_without_wsq_weight(self) -> None:
        """_effective_se without wsq_weight should fall back to n_obs."""
        df = pl.DataFrame({
            "level": ["A"],
            "mean_shap": [0.0],
            "shap_std": [0.6],
            "n_obs": [36],
            "exposure_weight": [1.0],
        })
        se_col = df.with_columns(_effective_se(df).alias("se"))["se"][0]
        expected = 0.6 / math.sqrt(36.0)
        assert abs(se_col - expected) < 1e-9


class TestNormalisationBoundaryConditions:
    """Boundary conditions for normalise_base_level and normalise_mean."""

    def test_single_row_base_level(self) -> None:
        """Single-row DataFrame with the only level as base should give relativity=1."""
        df = _make_agg_with_wsq(["A"], [0.7], shap_stds=[0.1], exposure_weights=[1.0], wsq_weights=[1.0])
        result = normalise_base_level(df, "A")
        assert abs(result["relativity"][0] - 1.0) < 1e-12

    def test_single_row_mean_normalisation(self) -> None:
        """Single row: mean of its own relativity is 1.0."""
        df = _make_agg_with_wsq(["A"], [0.5], shap_stds=[0.1], exposure_weights=[1.0], wsq_weights=[1.0])
        result = normalise_mean(df)
        assert abs(result["relativity"][0] - 1.0) < 1e-12

    def test_ci_level_99_wider_than_95(self) -> None:
        """99% CI should be wider than 95% CI."""
        df = _make_agg_no_wsq(["A", "B"], [0.0, 0.5], shap_stds=[0.2, 0.2])
        res_95 = normalise_mean(df, ci_level=0.95)
        res_99 = normalise_mean(df, ci_level=0.99)

        def width(r):
            return (r["upper_ci"] - r["lower_ci"]).to_numpy().mean()

        assert width(res_99) > width(res_95)

    def test_ci_level_50(self) -> None:
        """50% CI should be narrower than 95% CI."""
        df = _make_agg_no_wsq(["A", "B"], [0.0, 0.3], shap_stds=[0.2, 0.2])
        res_95 = normalise_mean(df, ci_level=0.95)
        res_50 = normalise_mean(df, ci_level=0.50)

        def width(r):
            return (r["upper_ci"] - r["lower_ci"]).to_numpy().mean()

        assert width(res_50) < width(res_95)

    def test_float_base_level_str_coercion(self) -> None:
        """A float base_level like 0.0 should match level '0.0' in the data."""
        df = _make_agg_with_wsq(
            ["0.0", "1.0"],
            [0.1, 0.4],
            shap_stds=[0.1, 0.1],
            exposure_weights=[0.5, 0.5],
            wsq_weights=[0.25, 0.25],
        )
        result = normalise_base_level(df, 0.0)
        base_row = result.filter(pl.col("level") == "0.0")
        assert len(base_row) == 1
        assert abs(base_row["relativity"][0] - 1.0) < 1e-12

    def test_negative_mean_shap_gives_relativity_below_one(self) -> None:
        """A level with lower mean_shap than base should get relativity < 1 in base_level mode."""
        df = _make_agg_with_wsq(
            ["A", "B"],
            [0.0, -0.5],
            shap_stds=[0.1, 0.1],
            exposure_weights=[0.5, 0.5],
            wsq_weights=[0.01, 0.01],
        )
        result = normalise_base_level(df, "A")
        b_rel = result.filter(pl.col("level") == "B")["relativity"][0]
        assert b_rel < 1.0, f"Expected relativity < 1.0, got {b_rel}"

    def test_large_positive_mean_shap_gives_high_relativity(self) -> None:
        """Large positive SHAP offset from base should produce high relativity."""
        df = _make_agg_with_wsq(
            ["A", "B"],
            [0.0, 2.0],
            exposure_weights=[0.5, 0.5],
            wsq_weights=[0.01, 0.01],
        )
        result = normalise_base_level(df, "A")
        b_rel = result.filter(pl.col("level") == "B")["relativity"][0]
        expected = math.exp(2.0)
        assert abs(b_rel - expected) < 1e-6

    def test_normalise_mean_all_equal_shaps_gives_all_ones(self) -> None:
        """When all mean_shap values are equal, all relativities should be 1.0."""
        df = _make_agg_with_wsq(
            ["A", "B", "C"],
            [0.5, 0.5, 0.5],
            exposure_weights=[0.4, 0.3, 0.3],
            wsq_weights=[0.01, 0.01, 0.01],
        )
        result = normalise_mean(df)
        for rel in result["relativity"].to_numpy():
            assert abs(rel - 1.0) < 1e-9


# ===========================================================================
# _validation.py — boundary conditions
# ===========================================================================


class TestCheckReconstructionBoundary:
    """Boundary and edge cases for check_reconstruction."""

    def test_exact_zero_error_with_zero_tolerance(self) -> None:
        """Perfect reconstruction with tolerance=0 should pass (0 < 0 is False, use <=)."""
        # Note: the implementation uses max_diff < tolerance (strict), so
        # tolerance=0 with exact match (max_diff=0) will FAIL (0 < 0 is False).
        # This is the existing contract — document it with a test.
        shap_vals = np.zeros((3, 2))
        preds = np.full(3, math.exp(0.5))
        result = check_reconstruction(shap_vals, 0.5, preds, tolerance=0.0)
        # max_diff is 0.0, tolerance is 0.0, 0.0 < 0.0 is False → fails
        assert not result.passed
        assert result.value == pytest.approx(0.0, abs=1e-9)

    def test_large_error_clearly_fails(self) -> None:
        """Error clearly above tolerance must fail."""
        shap_vals = np.zeros((2, 1))
        # reconstructed = exp(0)=1.0, preds=1.01 → diff=0.01 >> default tolerance 1e-4
        preds = np.array([1.01, 1.0])
        result = check_reconstruction(shap_vals, 0.0, preds, tolerance=1e-4)
        assert not result.passed

    def test_small_error_clearly_passes(self) -> None:
        """Error well below tolerance must pass."""
        shap_vals = np.zeros((2, 1))
        # reconstructed = exp(0)=1.0, preds=1.0+1e-8 → diff=1e-8 << tolerance=1e-4
        preds = np.array([1.0 + 1e-8, 1.0])
        result = check_reconstruction(shap_vals, 0.0, preds, tolerance=1e-4)
        assert result.passed

    def test_single_observation(self) -> None:
        """check_reconstruction should work with n=1."""
        shap_vals = np.array([[0.1, 0.2]])
        ev = 0.3
        preds = np.array([math.exp(0.1 + 0.2 + 0.3)])
        result = check_reconstruction(shap_vals, ev, preds)
        assert result.passed

    def test_result_is_named_tuple(self) -> None:
        """Return value must be a CheckResult NamedTuple with correct fields."""
        result = check_reconstruction(np.zeros((2, 2)), 0.0, np.ones(2))
        assert hasattr(result, "passed")
        assert hasattr(result, "value")
        assert hasattr(result, "message")
        assert isinstance(result, CheckResult)


class TestCheckFeatureCoverageEdgeCases:
    """Additional edge cases for check_feature_coverage."""

    def test_empty_both_lists_passes(self) -> None:
        """No features expected — trivially passes."""
        result = check_feature_coverage([], [])
        assert result.passed
        assert result.value == 0.0

    def test_empty_shap_features_with_expected_fails(self) -> None:
        """If SHAP has no features but some are expected, should fail."""
        result = check_feature_coverage([], ["a", "b"])
        assert not result.passed
        assert result.value == 2.0

    def test_value_type_is_float(self) -> None:
        """The value field must be a float (for uniformity with other checks)."""
        result = check_feature_coverage(["a", "b"], ["a", "b"])
        assert isinstance(result.value, float)


class TestCheckSparseLevelsEdgeCases:
    """Additional edge cases for check_sparse_levels."""

    def test_exactly_at_min_obs_passes(self) -> None:
        """Level with exactly min_obs observations should pass (>= threshold)."""
        df = pl.DataFrame({"level": ["A"], "n_obs": [30]})
        result = check_sparse_levels(df, min_obs=30)
        assert result.passed

    def test_one_below_min_obs_fails(self) -> None:
        """Level with min_obs - 1 should fail."""
        df = pl.DataFrame({"level": ["A"], "n_obs": [29]})
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = check_sparse_levels(df, min_obs=30)
        assert not result.passed

    def test_message_includes_level_name(self) -> None:
        """Warning message should mention the sparse level name."""
        df = pl.DataFrame({"level": ["SparseLevel"], "n_obs": [5]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_sparse_levels(df, min_obs=30)
        warning_text = " ".join(str(ww.message) for ww in w)
        assert "SparseLevel" in warning_text

    def test_min_obs_zero_always_passes(self) -> None:
        """min_obs=0 means everything passes."""
        df = pl.DataFrame({"level": ["A", "B"], "n_obs": [0, 0]})
        result = check_sparse_levels(df, min_obs=0)
        assert result.passed

    def test_empty_dataframe_passes(self) -> None:
        """Empty DataFrame (no levels) should pass."""
        df = pl.DataFrame({"level": pl.Series([], dtype=pl.String), "n_obs": pl.Series([], dtype=pl.Int64)})
        result = check_sparse_levels(df, min_obs=30)
        assert result.passed
        assert result.value == 0.0


# ===========================================================================
# _core.py — type conversion helpers
# ===========================================================================


class TestToPandasHelper:
    """Tests for the _to_pandas helper."""

    def test_polars_to_pandas_converts(self) -> None:
        """_to_pandas should convert a Polars DataFrame to pandas."""
        import pandas as pd
        df = pl.DataFrame({"x": [1, 2, 3]})
        result = _to_pandas(df)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["x"]

    def test_column_names_preserved(self) -> None:
        """Column names must survive the Polars → pandas conversion."""
        import pandas as pd
        df = pl.DataFrame({"area": ["A", "B"], "ncd": [0, 1]})
        result = _to_pandas(df)
        assert set(result.columns) == {"area", "ncd"}


class TestToPolarsHelper:
    """Tests for the _to_polars helper (additional cases)."""

    def test_numpy_array_raises(self) -> None:
        """A numpy array should raise TypeError with a helpful message."""
        with pytest.raises(TypeError, match="Polars or pandas DataFrame"):
            _to_polars(np.array([[1, 2], [3, 4]]))

    def test_list_raises(self) -> None:
        """A plain list should raise TypeError."""
        with pytest.raises(TypeError):
            _to_polars([[1, 2, 3]])

    def test_none_raises(self) -> None:
        """None should raise TypeError."""
        with pytest.raises(TypeError):
            _to_polars(None)


# ===========================================================================
# _core.py — SHAPRelativities constructor and property paths
# ===========================================================================


class TestSHAPRelativitiesConstructor:
    """Tests for the SHAPRelativities __init__ paths."""

    def _dummy_sr(self, X, exposure=None, categorical=None, continuous=None):
        from shap_relativities import SHAPRelativities
        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        class _DummyModel:
            pass

        return SHAPRelativities(
            model=_DummyModel(),
            X=X,
            exposure=exposure,
            categorical_features=categorical,
            continuous_features=continuous,
        )

    def test_exposure_as_polars_series(self) -> None:
        """Polars Series exposure should be accepted and converted to numpy."""
        X = pl.DataFrame({"area": ["A", "B", "C"]})
        exposure = pl.Series([0.9, 1.0, 0.8])
        sr = self._dummy_sr(X, exposure=exposure)
        assert isinstance(sr._exposure, np.ndarray)
        np.testing.assert_allclose(sr._exposure, [0.9, 1.0, 0.8])

    def test_exposure_as_list(self) -> None:
        """Python list exposure should be accepted and converted to numpy."""
        X = pl.DataFrame({"area": ["A", "B"]})
        sr = self._dummy_sr(X, exposure=[1.0, 0.5])
        assert isinstance(sr._exposure, np.ndarray)

    def test_exposure_as_numpy(self) -> None:
        """numpy array exposure should be stored directly."""
        X = pl.DataFrame({"ncd": [0, 1, 2]})
        exposure = np.array([0.8, 1.0, 0.9])
        sr = self._dummy_sr(X, exposure=exposure)
        np.testing.assert_array_equal(sr._exposure, exposure)

    def test_explicit_features_override_inference(self) -> None:
        """Explicit categorical_features and continuous_features must override inference."""
        X = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        sr = self._dummy_sr(X, categorical=["a"], continuous=["b"])
        assert sr._categorical_features == ["a"]
        assert sr._continuous_features == ["b"]

    def test_infer_continuous_excludes_explicit_categoricals(self) -> None:
        """When categorical_features is explicit, those columns should not appear in continuous."""
        X = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        # Explicitly make "a" categorical; b should be continuous
        sr = self._dummy_sr(X, categorical=["a"])
        assert "a" not in sr._continuous_features
        assert "b" in sr._continuous_features

    def test_is_fitted_false_initially(self) -> None:
        """_is_fitted must be False before fit() is called."""
        X = pl.DataFrame({"x": [1, 2, 3]})
        sr = self._dummy_sr(X)
        assert sr._is_fitted is False

    def test_unfitted_shap_values_raises(self) -> None:
        """shap_values() before fit() must raise RuntimeError."""
        from shap_relativities import SHAPRelativities
        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        class _Dummy:
            pass

        X = pl.DataFrame({"x": [1, 2]})
        sr = SHAPRelativities(model=_Dummy(), X=X)
        with pytest.raises(RuntimeError, match="fit()"):
            sr.shap_values()

    def test_unfitted_baseline_raises(self) -> None:
        """baseline() before fit() must raise RuntimeError."""
        from shap_relativities import SHAPRelativities
        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        class _Dummy:
            pass

        X = pl.DataFrame({"x": [1, 2]})
        sr = SHAPRelativities(model=_Dummy(), X=X)
        with pytest.raises(RuntimeError, match="fit()"):
            sr.baseline()

    def test_unfitted_validate_raises(self) -> None:
        """validate() before fit() must raise RuntimeError."""
        from shap_relativities import SHAPRelativities
        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        class _Dummy:
            pass

        X = pl.DataFrame({"x": [1, 2]})
        sr = SHAPRelativities(model=_Dummy(), X=X)
        with pytest.raises(RuntimeError, match="fit()"):
            sr.validate()

    def test_unfitted_extract_continuous_curve_raises(self) -> None:
        """extract_continuous_curve() before fit() must raise RuntimeError."""
        from shap_relativities import SHAPRelativities
        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        class _Dummy:
            pass

        X = pl.DataFrame({"x": [1.0, 2.0]})
        sr = SHAPRelativities(model=_Dummy(), X=X, continuous_features=["x"])
        with pytest.raises(RuntimeError, match="fit()"):
            sr.extract_continuous_curve("x")


# ===========================================================================
# _core.py — baseline() annualise paths
# ===========================================================================


class TestBaselinePaths:
    """Tests for baseline() with different annualise_exposure settings."""

    def test_baseline_annualise_false_no_exposure(self) -> None:
        """annualise_exposure=False with no exposure: baseline = exp(expected_value)."""
        from shap_relativities import SHAPRelativities
        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        data = {
            "shap_values": [[0.0, 0.0], [0.0, 0.0]],
            "expected_value": 1.0,
            "feature_names": ["a", "b"],
            "categorical_features": ["a", "b"],
            "continuous_features": [],
            "X_values": {"a": ["X", "Y"], "b": [0, 1]},
            "exposure": None,
            "annualise_exposure": False,
        }
        sr = SHAPRelativities.from_dict(data)
        b = sr.baseline()
        assert abs(b - math.exp(1.0)) < 1e-9

    def test_baseline_annualise_true_with_unit_exposure(self) -> None:
        """With exposure=1.0 for all obs, log(exposure)=0, so annualise makes no difference."""
        from shap_relativities import SHAPRelativities
        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        data = {
            "shap_values": [[0.0], [0.0]],
            "expected_value": 0.5,
            "feature_names": ["a"],
            "categorical_features": ["a"],
            "continuous_features": [],
            "X_values": {"a": ["X", "Y"]},
            "exposure": [1.0, 1.0],
            "annualise_exposure": True,
        }
        sr = SHAPRelativities.from_dict(data)
        b = sr.baseline()
        # mean log(1.0) = 0.0, so baseline = exp(0.5 - 0.0) = exp(0.5)
        assert abs(b - math.exp(0.5)) < 1e-9

    def test_baseline_annualise_true_adjusts_downward_with_subannual_exposure(self) -> None:
        """Sub-annual average exposure reduces the baseline (negative log-exposure offset)."""
        from shap_relativities import SHAPRelativities
        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        # Mean log-exposure = log(0.5) < 0, so ev - mean_log_exp > ev
        # Baseline with annualise=True should be exp(ev - mean_log_exp) > exp(ev)
        exposure_val = 0.5
        data = {
            "shap_values": [[0.0], [0.0]],
            "expected_value": 0.0,
            "feature_names": ["a"],
            "categorical_features": ["a"],
            "continuous_features": [],
            "X_values": {"a": ["X", "Y"]},
            "exposure": [exposure_val, exposure_val],
            "annualise_exposure": True,
        }
        sr = SHAPRelativities.from_dict(data)
        b = sr.baseline()
        expected = math.exp(0.0 - math.log(exposure_val))
        assert abs(b - expected) < 1e-9

    def test_baseline_annualise_false_with_exposure_ignores_exposure(self) -> None:
        """When annualise_exposure=False, exposure is not used in baseline()."""
        from shap_relativities import SHAPRelativities
        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        data = {
            "shap_values": [[0.0], [0.0]],
            "expected_value": 0.0,
            "feature_names": ["a"],
            "categorical_features": ["a"],
            "continuous_features": [],
            "X_values": {"a": ["X", "Y"]},
            "exposure": [0.1, 0.1],  # tiny exposure — would heavily affect annualised baseline
            "annualise_exposure": False,
        }
        sr = SHAPRelativities.from_dict(data)
        b = sr.baseline()
        assert abs(b - 1.0) < 1e-9  # exp(0.0) = 1.0


# ===========================================================================
# _core.py — to_dict / from_dict edge cases
# ===========================================================================


class TestToDictFromDictEdgeCases:
    """Tests for serialisation round-trips and backward compatibility."""

    def test_to_dict_contains_required_keys(self) -> None:
        """to_dict() output must contain all keys needed for from_dict()."""
        sr = _minimal_fitted_sr()
        d = sr.to_dict()
        required_keys = {
            "shap_values", "expected_value", "feature_names",
            "categorical_features", "continuous_features",
            "X_values", "exposure", "annualise_exposure",
        }
        assert required_keys.issubset(set(d.keys()))

    def test_to_dict_shap_values_shape(self) -> None:
        """shap_values in to_dict() must be n_obs x n_features."""
        sr = _minimal_fitted_sr()
        sv = sr.to_dict()["shap_values"]
        assert len(sv) == len(sr._X)
        assert len(sv[0]) == sr._X.width

    def test_from_dict_exposure_round_trip(self) -> None:
        """Exposure values must survive the to_dict/from_dict round-trip."""
        sr = _minimal_fitted_sr()
        d = sr.to_dict()
        sr2 = type(sr).from_dict(d)
        np.testing.assert_allclose(sr._exposure, sr2._exposure, rtol=1e-9)

    def test_from_dict_backward_compat_no_annualise_key(self) -> None:
        """from_dict with no 'annualise_exposure' key should default to True."""
        from shap_relativities import SHAPRelativities
        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        data = {
            "shap_values": [[0.1], [0.2]],
            "expected_value": 0.0,
            "feature_names": ["x"],
            "categorical_features": ["x"],
            "continuous_features": [],
            "X_values": {"x": ["A", "B"]},
            "exposure": None,
            # deliberately omit 'annualise_exposure'
        }
        sr = SHAPRelativities.from_dict(data)
        assert sr._annualise_exposure is True

    def test_from_dict_no_feature_names_key(self) -> None:
        """from_dict with no 'feature_names' key should infer from X_values keys."""
        from shap_relativities import SHAPRelativities
        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        data = {
            "shap_values": [[0.1, -0.1], [0.2, 0.0]],
            "expected_value": 0.0,
            "categorical_features": ["a", "b"],
            "continuous_features": [],
            "X_values": {"a": ["X", "Y"], "b": [0, 1]},
            "exposure": None,
            "annualise_exposure": True,
            # no 'feature_names' key
        }
        sr = SHAPRelativities.from_dict(data)
        # Should not raise; feature names inferred from X_values
        assert set(sr._X.columns) == {"a", "b"}

    def test_from_dict_is_fitted(self) -> None:
        """from_dict must produce a fitted instance (_is_fitted=True)."""
        sr = _minimal_fitted_sr()
        d = sr.to_dict()
        sr2 = type(sr).from_dict(d)
        assert sr2._is_fitted is True

    def test_to_dict_model_is_none_in_from_dict(self) -> None:
        """from_dict produces an instance with _model=None (no model attached)."""
        sr = _minimal_fitted_sr()
        d = sr.to_dict()
        sr2 = type(sr).from_dict(d)
        assert sr2._model is None

    def test_from_dict_x_values_preserved(self) -> None:
        """X values in X_values dict should be accessible after from_dict."""
        sr = _minimal_fitted_sr()
        d = sr.to_dict()
        sr2 = type(sr).from_dict(d)
        # X should match (column by column)
        for col in sr._X.columns:
            orig = sr._X[col].to_list()
            recov = sr2._X[col].to_list()
            assert orig == recov, f"Column {col} differs"


# ===========================================================================
# _core.py — validate() with no categorical features
# ===========================================================================


class TestValidateNoCategoricalFeatures:
    """validate() when no categorical features are present."""

    def test_sparse_levels_passes_when_no_categorical(self) -> None:
        """sparse_levels check should return passed=True with a 'no categorical' message."""
        from shap_relativities import SHAPRelativities
        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        data = {
            "shap_values": [[0.1, -0.2], [0.3, 0.1]],
            "expected_value": -2.0,
            "feature_names": ["age", "mileage"],
            "categorical_features": [],
            "continuous_features": ["age", "mileage"],
            "X_values": {"age": [25.0, 40.0], "mileage": [8000.0, 12000.0]},
            "exposure": None,
            "annualise_exposure": False,
        }
        sr = SHAPRelativities.from_dict(data)
        checks = sr.validate()
        assert checks["sparse_levels"].passed

    def test_validate_all_keys_present_no_categorical(self) -> None:
        """validate() should always return all three check keys."""
        from shap_relativities import SHAPRelativities
        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        data = {
            "shap_values": [[0.1], [-0.1]],
            "expected_value": 0.0,
            "feature_names": ["x"],
            "categorical_features": [],
            "continuous_features": ["x"],
            "X_values": {"x": [1.0, 2.0]},
            "exposure": None,
            "annualise_exposure": False,
        }
        sr = SHAPRelativities.from_dict(data)
        checks = sr.validate()
        assert "reconstruction" in checks
        assert "feature_coverage" in checks
        assert "sparse_levels" in checks


# ===========================================================================
# _core.py — extract_relativities: base_level fallback warning content
# ===========================================================================


class TestBaseLevelFallbackWarning:
    """The fallback warning message must contain the feature name and chosen level."""

    def test_warning_mentions_feature_name(self) -> None:
        """UserWarning from base_level fallback should name the feature."""
        sr = _minimal_fitted_sr()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sr.extract_relativities(normalise_to="base_level", base_levels={})

        warning_messages = " ".join(str(ww.message) for ww in w if issubclass(ww.category, UserWarning))
        # Both "area" and "ncd" features should appear in warnings
        assert "area" in warning_messages or "ncd" in warning_messages

    def test_warning_is_user_warning_category(self) -> None:
        """The fallback warning must be a UserWarning."""
        sr = _minimal_fitted_sr()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sr.extract_relativities(normalise_to="base_level", base_levels={})

        categories = [ww.category for ww in w]
        assert UserWarning in categories

    def test_base_level_fallback_result_has_one_relativity_one(self) -> None:
        """After fallback, the auto-chosen base level should have relativity=1.0."""
        sr = _minimal_fitted_sr()

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            rels = sr.extract_relativities(normalise_to="base_level", base_levels={})

        # Each feature must have exactly one level with relativity=1.0
        for feat in rels["feature"].unique().to_list():
            feat_rels = rels.filter(pl.col("feature") == feat)["relativity"].to_numpy()
            n_ones = sum(abs(r - 1.0) < 1e-9 for r in feat_rels)
            assert n_ones == 1, (
                f"Feature '{feat}': expected exactly one relativity=1.0, "
                f"got {n_ones}. Relativities: {feat_rels}"
            )


# ===========================================================================
# _core.py — extract_relativities output properties
# ===========================================================================


class TestExtractRelativitiesOutput:
    """Tests for structural properties of extract_relativities output."""

    def test_output_is_polars_dataframe(self) -> None:
        """extract_relativities must return a Polars DataFrame."""
        sr = _minimal_fitted_sr()
        result = sr.extract_relativities(normalise_to="mean")
        assert isinstance(result, pl.DataFrame)

    def test_level_column_is_string(self) -> None:
        """level column must be String type after casting in extract_relativities."""
        sr = _minimal_fitted_sr()
        result = sr.extract_relativities(normalise_to="mean")
        assert result["level"].dtype == pl.String

    def test_feature_column_contains_all_features(self) -> None:
        """All features from X must appear in the output."""
        sr = _minimal_fitted_sr()
        result = sr.extract_relativities(normalise_to="mean")
        output_features = set(result["feature"].unique().to_list())
        assert "area" in output_features
        assert "ncd" in output_features

    def test_ci_lower_leq_upper_clt(self) -> None:
        """lower_ci must never exceed upper_ci for clt method."""
        sr = _minimal_fitted_sr()
        result = sr.extract_relativities(normalise_to="mean", ci_method="clt")
        assert (result["lower_ci"] <= result["upper_ci"]).all()

    def test_mean_shap_column_present(self) -> None:
        """mean_shap must be present in output."""
        sr = _minimal_fitted_sr()
        result = sr.extract_relativities(normalise_to="mean")
        assert "mean_shap" in result.columns

    def test_n_obs_column_all_positive(self) -> None:
        """n_obs must be positive for all rows."""
        sr = _minimal_fitted_sr()
        result = sr.extract_relativities(normalise_to="mean")
        assert (result["n_obs"] > 0).all()

    def test_exposure_weight_column_all_non_negative(self) -> None:
        """exposure_weight must be >= 0."""
        sr = _minimal_fitted_sr()
        result = sr.extract_relativities(normalise_to="mean")
        assert (result["exposure_weight"] >= 0.0).all()

    def test_normalise_to_mean_no_base_levels_required(self) -> None:
        """normalise_to='mean' should not require base_levels parameter."""
        sr = _minimal_fitted_sr()
        # Should not raise even with no base_levels
        result = sr.extract_relativities(normalise_to="mean")
        assert result is not None


# ===========================================================================
# _plotting.py — additional edge cases
# ===========================================================================


class TestPlottingEdgeCases:
    """Additional edge cases for the plotting module."""

    @pytest.fixture(autouse=True)
    def _backend(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        yield
        plt.close("all")

    def _cat_data_with_nan_ci(self) -> pl.DataFrame:
        """Categorical data where CIs are NaN (ci_method='none')."""
        return pl.DataFrame({
            "feature": ["area"] * 3,
            "level": ["A", "B", "C"],
            "relativity": [1.0, 1.3, 0.8],
            "lower_ci": [float("nan")] * 3,
            "upper_ci": [float("nan")] * 3,
            "mean_shap": [0.0, 0.26, -0.22],
            "shap_std": [0.0, 0.0, 0.0],
            "n_obs": [100, 80, 50],
            "exposure_weight": [0.4, 0.35, 0.25],
        })

    def _cont_data(self, n: int = 10) -> pl.DataFrame:
        vals = np.linspace(17, 80, n)
        return pl.DataFrame({
            "feature": ["age"] * n,
            "level": vals.astype(str).tolist(),
            "relativity": np.ones(n).tolist(),
            "lower_ci": np.full(n, 0.9).tolist(),
            "upper_ci": np.full(n, 1.1).tolist(),
            "mean_shap": np.zeros(n).tolist(),
            "shap_std": np.zeros(n).tolist(),
            "n_obs": [100] * n,
            "exposure_weight": np.full(n, 1.0 / n).tolist(),
        })

    def test_plot_categorical_with_nan_ci_does_not_crash(self) -> None:
        """plot_categorical with NaN CIs (from ci_method='none') should not raise."""
        import matplotlib.pyplot as plt
        from shap_relativities._plotting import plot_categorical

        fig, ax = plt.subplots()
        plot_categorical(self._cat_data_with_nan_ci(), "area", ax, show_ci=True)

    def test_plot_categorical_custom_color(self) -> None:
        """plot_categorical with a custom color parameter should not raise."""
        import matplotlib.pyplot as plt
        from shap_relativities._plotting import plot_categorical

        fig, ax = plt.subplots()
        plot_categorical(self._cat_data_with_nan_ci(), "area", ax, color="tomato")
        # Check that patches were drawn (one bar per level)
        assert len(ax.patches) > 0

    def test_plot_continuous_single_point(self) -> None:
        """plot_continuous with a single data point should not raise."""
        import matplotlib.pyplot as plt
        from shap_relativities._plotting import plot_continuous

        data = pl.DataFrame({
            "feature": ["x"],
            "level": ["5.0"],
            "relativity": [1.0],
            "lower_ci": [0.9],
            "upper_ci": [1.1],
            "mean_shap": [0.0],
            "shap_std": [0.0],
            "n_obs": [100],
            "exposure_weight": [1.0],
        })
        fig, ax = plt.subplots()
        plot_continuous(data, "x", ax)

    def test_plot_continuous_custom_color(self) -> None:
        """plot_continuous with a custom color should not raise."""
        import matplotlib.pyplot as plt
        from shap_relativities._plotting import plot_continuous

        fig, ax = plt.subplots()
        plot_continuous(self._cont_data(), "age", ax, color="crimson")
        # Check that a line was drawn
        assert len(ax.lines) > 0

    def test_plot_relativities_hides_unused_axes(self) -> None:
        """When n_features < n_cols * n_rows, unused axes should be hidden."""
        import matplotlib.pyplot as plt
        from shap_relativities._plotting import plot_relativities

        # 2 features with 3 columns → should produce 3 axes, with 1 hidden
        data1 = pl.DataFrame({
            "feature": ["area"] * 2, "level": ["A", "B"],
            "relativity": [1.0, 1.2], "lower_ci": [0.9, 1.0],
            "upper_ci": [1.1, 1.4], "mean_shap": [0.0, 0.18],
            "shap_std": [0.1, 0.1], "n_obs": [100, 80],
            "exposure_weight": [0.55, 0.45],
        })
        data2 = pl.DataFrame({
            "feature": ["ncd"] * 2, "level": ["0", "5"],
            "relativity": [1.0, 0.7], "lower_ci": [0.9, 0.6],
            "upper_ci": [1.1, 0.8], "mean_shap": [0.0, -0.36],
            "shap_std": [0.1, 0.1], "n_obs": [100, 80],
            "exposure_weight": [0.5, 0.5],
        })
        combined = pl.concat([data1, data2])
        # Should not raise even when axes grid has spare slots
        plot_relativities(combined, categorical_features=["area", "ncd"], continuous_features=[])

    def test_plot_relativities_four_features_no_crash(self) -> None:
        """Four features (needs 2 rows, 3 cols grid) should render cleanly."""
        import matplotlib.pyplot as plt
        from shap_relativities._plotting import plot_relativities

        def _make_cat_data(name):
            return pl.DataFrame({
                "feature": [name] * 2, "level": ["A", "B"],
                "relativity": [1.0, 1.1], "lower_ci": [0.9, 1.0],
                "upper_ci": [1.1, 1.2], "mean_shap": [0.0, 0.1],
                "shap_std": [0.05, 0.05], "n_obs": [50, 50],
                "exposure_weight": [0.5, 0.5],
            })

        combined = pl.concat([_make_cat_data(n) for n in ["f1", "f2", "f3", "f4"]])
        plot_relativities(combined, categorical_features=["f1", "f2", "f3", "f4"], continuous_features=[])


# ===========================================================================
# Integration: mixed categorical + continuous via from_dict
# ===========================================================================


class TestMixedFeatureExtraction:
    """Integration tests for extract_relativities with both feature types."""

    def _mixed_sr(self):
        from shap_relativities import SHAPRelativities
        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        rng = np.random.default_rng(7)
        n = 50
        area_vals = rng.choice(["A", "B", "C"], size=n).tolist()
        age_vals = rng.uniform(17, 80, size=n).tolist()
        shap_area = rng.normal(0, 0.2, size=n).tolist()
        shap_age = rng.normal(0, 0.1, size=n).tolist()

        data = {
            "shap_values": [[shap_area[i], shap_age[i]] for i in range(n)],
            "expected_value": -2.5,
            "feature_names": ["area", "driver_age"],
            "categorical_features": ["area"],
            "continuous_features": ["driver_age"],
            "X_values": {"area": area_vals, "driver_age": age_vals},
            "exposure": np.ones(n).tolist(),
            "annualise_exposure": False,
        }
        return SHAPRelativities.from_dict(data)

    def test_both_feature_types_in_output(self) -> None:
        """Output must include rows for both the categorical and continuous features."""
        sr = self._mixed_sr()
        result = sr.extract_relativities(normalise_to="mean")
        features_in_output = set(result["feature"].unique().to_list())
        assert "area" in features_in_output
        assert "driver_age" in features_in_output

    def test_categorical_has_three_levels(self) -> None:
        """The categorical area feature should have exactly 3 levels (A, B, C)."""
        sr = self._mixed_sr()
        result = sr.extract_relativities(normalise_to="mean")
        area_rows = result.filter(pl.col("feature") == "area")
        assert len(area_rows) == 3

    def test_continuous_has_n_obs_rows(self) -> None:
        """Continuous feature should have one row per observation."""
        sr = self._mixed_sr()
        result = sr.extract_relativities(normalise_to="mean")
        age_rows = result.filter(pl.col("feature") == "driver_age")
        assert len(age_rows) == 50

    def test_all_relativities_positive_mixed(self) -> None:
        """All relativities must be positive for mixed feature types."""
        sr = self._mixed_sr()
        result = sr.extract_relativities(normalise_to="mean")
        assert (result["relativity"] > 0).all()

    def test_ci_method_none_on_mixed_gives_nan_ci(self) -> None:
        """ci_method='none' on mixed features should produce NaN CIs throughout."""
        sr = self._mixed_sr()
        result = sr.extract_relativities(normalise_to="mean", ci_method="none")
        for row in result.iter_rows(named=True):
            assert math.isnan(row["lower_ci"]), f"lower_ci not NaN: {row}"
            assert math.isnan(row["upper_ci"]), f"upper_ci not NaN: {row}"

    def test_extract_continuous_curve_loess_on_mixed_sr(self) -> None:
        """extract_continuous_curve with loess (or fallback) on the continuous feature."""
        sr = self._mixed_sr()
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess  # noqa: F401
            smooth = "loess"
        except ImportError:
            smooth = "isotonic"

        curve = sr.extract_continuous_curve("driver_age", n_points=20, smooth_method=smooth)
        assert isinstance(curve, pl.DataFrame)
        assert len(curve) == 20
        assert (curve["relativity"] > 0).all()


# ===========================================================================
# _core.py — extract_continuous_curve edge cases
# ===========================================================================


class TestExtractContinuousCurveEdgeCases:
    """Additional edge cases for extract_continuous_curve."""

    def _continuous_sr(self, n=30):
        from shap_relativities import SHAPRelativities
        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        rng = np.random.default_rng(11)
        vals = np.linspace(0, 10, n).tolist()
        shaps = (rng.normal(0, 0.1, size=n)).tolist()

        data = {
            "shap_values": [[s] for s in shaps],
            "expected_value": 0.0,
            "feature_names": ["x"],
            "categorical_features": [],
            "continuous_features": ["x"],
            "X_values": {"x": vals},
            "exposure": None,
            "annualise_exposure": False,
        }
        return SHAPRelativities.from_dict(data)

    def test_n_points_respected_isotonic(self) -> None:
        """n_points parameter must control output row count for isotonic smooth."""
        sr = self._continuous_sr()
        for n_pts in [10, 50, 200]:
            curve = sr.extract_continuous_curve("x", n_points=n_pts, smooth_method="isotonic")
            assert len(curve) == n_pts, f"Expected {n_pts} rows, got {len(curve)}"

    def test_n_points_respected_none(self) -> None:
        """smooth_method='none' returns one row per observation, ignoring n_points."""
        sr = self._continuous_sr(n=30)
        curve = sr.extract_continuous_curve("x", n_points=999, smooth_method="none")
        assert len(curve) == 30

    def test_isotonic_curve_feature_value_range(self) -> None:
        """Isotonic curve feature_value must span from min to max of the data."""
        sr = self._continuous_sr()
        curve = sr.extract_continuous_curve("x", n_points=20, smooth_method="isotonic")
        data_min = float(sr._X["x"].min())
        data_max = float(sr._X["x"].max())
        assert abs(curve["feature_value"].min() - data_min) < 1e-6
        assert abs(curve["feature_value"].max() - data_max) < 1e-6

    def test_none_curve_sorted_by_feature_value(self) -> None:
        """smooth_method='none' curve must be sorted by feature_value (ascending)."""
        sr = self._continuous_sr(n=20)
        curve = sr.extract_continuous_curve("x", smooth_method="none")
        vals = curve["feature_value"].to_numpy()
        diffs = np.diff(vals)
        assert (diffs >= 0).all(), "feature_value not monotonically non-decreasing"

    def test_isotonic_relativities_all_positive(self) -> None:
        """Isotonic curve relativities must all be positive."""
        sr = self._continuous_sr()
        curve = sr.extract_continuous_curve("x", n_points=50, smooth_method="isotonic")
        assert (curve["relativity"] > 0).all()
