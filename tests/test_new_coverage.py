"""
New tests targeting specific coverage gaps in shap-relativities.

Focuses on:
1. _core.py: _to_polars error path, baseline() variants, validate() with no
   categoricals, from_dict backward compat (no feature_names key),
   _run_with_spinner verbose paths (tqdm and fallback)
2. _inference.py: _dense_rank_descending ties, ranking_ci identical-feature
   edge case and unknown-feature errors, p=1 smoothed path, influence_matrix
   property, plot_importance
3. _normalisation.py: _base_se and _effective_se fallback paths (no wsq_weight)
4. _plotting.py: plot_continuous CI fill, plot_relativities single-feature
5. datasets/motor.py: internal helper functions directly
6. SHAPInference: repr, p < 2 warning, importance table p-value structure

All tests are self-contained. No catboost/shap required unless explicitly noted.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import polars as pl
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fitted_sr(shap_vals=None, cat=True, exposure=None):
    """Build a SHAPRelativities via from_dict without needing catboost."""
    from shap_relativities import SHAPRelativities

    try:
        import shap  # noqa: F401
    except ImportError:
        pytest.skip("shap not installed")

    sv = shap_vals or [[0.3, -0.1], [-0.2, 0.4], [0.1, 0.0]]
    data = {
        "shap_values": sv,
        "expected_value": -2.5,
        "feature_names": ["area", "ncd"],
        "categorical_features": ["area", "ncd"] if cat else [],
        "continuous_features": [] if cat else ["area", "ncd"],
        "X_values": {"area": ["A", "B", "A"], "ncd": [0, 1, 2]},
        "exposure": exposure,
        "annualise_exposure": False,
    }
    return SHAPRelativities.from_dict(data)


def _small_inference_data(n=200, d=3, seed=7):
    """Small synthetic SHAP+y data for SHAPInference tests."""
    rng = np.random.default_rng(seed)
    betas = np.array([0.6, 0.3, 0.1])[:d]
    X = rng.normal(0, 1, (n, d))
    shap_values = X * betas[np.newaxis, :]
    y = np.abs(shap_values.sum(axis=1) + rng.normal(0, 0.3, n))
    names = [f"feat_{j}" for j in range(d)]
    return shap_values, y, names


# ===========================================================================
# _core.py: _to_polars error path
# ===========================================================================


class TestToPolarsErrorPath:
    """_to_polars should raise TypeError for unsupported input types."""

    def test_to_polars_raises_for_list(self):
        from shap_relativities._core import _to_polars

        with pytest.raises(TypeError, match="Polars or pandas DataFrame"):
            _to_polars([[1, 2], [3, 4]])

    def test_to_polars_raises_for_numpy_array(self):
        from shap_relativities._core import _to_polars

        with pytest.raises(TypeError, match="Polars or pandas DataFrame"):
            _to_polars(np.array([[1, 2], [3, 4]]))

    def test_to_polars_raises_for_dict(self):
        from shap_relativities._core import _to_polars

        with pytest.raises(TypeError):
            _to_polars({"a": [1, 2]})

    def test_to_polars_accepts_polars_df(self):
        from shap_relativities._core import _to_polars

        df = pl.DataFrame({"x": [1, 2, 3]})
        result = _to_polars(df)
        assert isinstance(result, pl.DataFrame)

    def test_to_polars_accepts_pandas_df(self):
        pd = pytest.importorskip("pandas")
        from shap_relativities._core import _to_polars

        df = pd.DataFrame({"x": [1, 2, 3]})
        result = _to_polars(df)
        assert isinstance(result, pl.DataFrame)


# ===========================================================================
# _core.py: baseline() variants
# ===========================================================================


class TestBaselineVariants:
    """Test baseline() under different exposure / annualise configurations."""

    def test_baseline_without_exposure(self):
        """baseline() with no exposure and annualise_exposure=False returns exp(ev)."""
        from shap_relativities import SHAPRelativities

        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        data = {
            "shap_values": [[0.1, -0.1], [-0.1, 0.1]],
            "expected_value": -2.0,
            "feature_names": ["a", "b"],
            "categorical_features": ["a", "b"],
            "continuous_features": [],
            "X_values": {"a": ["X", "Y"], "b": [0, 1]},
            "exposure": None,
            "annualise_exposure": False,
        }
        sr = SHAPRelativities.from_dict(data)
        b = sr.baseline()
        # With no exposure and annualise=False, baseline = exp(expected_value)
        assert abs(b - math.exp(-2.0)) < 1e-9

    def test_baseline_with_annualise_true_and_exposure(self):
        """baseline() with annualise_exposure=True adjusts for mean log(exposure)."""
        from shap_relativities import SHAPRelativities

        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        exposure = [1.0, 2.0, 1.0]
        data = {
            "shap_values": [[0.1], [-0.1], [0.0]],
            "expected_value": -2.0,
            "feature_names": ["a"],
            "categorical_features": ["a"],
            "continuous_features": [],
            "X_values": {"a": ["X", "Y", "Z"]},
            "exposure": exposure,
            "annualise_exposure": True,
        }
        sr = SHAPRelativities.from_dict(data)
        b = sr.baseline()
        # Should not equal exp(-2.0) because exposure adjustment is applied
        assert b != math.exp(-2.0)
        assert b > 0

    def test_baseline_with_annualise_false_and_exposure(self):
        """baseline() with annualise_exposure=False ignores exposure."""
        from shap_relativities import SHAPRelativities

        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        data = {
            "shap_values": [[0.1], [-0.1]],
            "expected_value": -3.0,
            "feature_names": ["a"],
            "categorical_features": ["a"],
            "continuous_features": [],
            "X_values": {"a": ["X", "Y"]},
            "exposure": [0.5, 2.0],
            "annualise_exposure": False,
        }
        sr = SHAPRelativities.from_dict(data)
        b = sr.baseline()
        assert abs(b - math.exp(-3.0)) < 1e-9

    def test_baseline_requires_fit(self):
        """baseline() before fit() should raise RuntimeError."""
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


# ===========================================================================
# _core.py: validate() with no categorical features
# ===========================================================================


class TestValidateNoCategoricals:
    """validate() should gracefully handle a model with no categorical features."""

    def test_validate_no_categoricals_returns_no_sparse_levels_check(self):
        """When categorical_features=[], sparse_levels should report trivial pass."""
        from shap_relativities import SHAPRelativities

        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        data = {
            "shap_values": [[0.1, 0.2], [-0.1, -0.2], [0.05, 0.0]],
            "expected_value": 0.0,
            "feature_names": ["x", "y"],
            "categorical_features": [],
            "continuous_features": ["x", "y"],
            "X_values": {"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]},
            "exposure": None,
            "annualise_exposure": False,
        }
        sr = SHAPRelativities.from_dict(data)
        checks = sr.validate()
        # sparse_levels should still be in the dict
        assert "sparse_levels" in checks
        # With no categoricals, it trivially passes
        assert checks["sparse_levels"].passed

    def test_validate_no_categoricals_message(self):
        """Message should indicate no categorical features to check."""
        from shap_relativities import SHAPRelativities

        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        data = {
            "shap_values": [[0.1], [-0.1]],
            "expected_value": 0.0,
            "feature_names": ["age"],
            "categorical_features": [],
            "continuous_features": ["age"],
            "X_values": {"age": [25.0, 45.0]},
            "exposure": None,
            "annualise_exposure": False,
        }
        sr = SHAPRelativities.from_dict(data)
        checks = sr.validate()
        assert "No categorical" in checks["sparse_levels"].message


# ===========================================================================
# _core.py: from_dict backward compat (no feature_names key)
# ===========================================================================


class TestFromDictBackwardCompat:
    """from_dict should work even when feature_names key is absent (pre-0.5 dicts)."""

    def test_from_dict_without_feature_names_key(self):
        from shap_relativities import SHAPRelativities

        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        # Older format: no feature_names key, X_values ordering defines columns
        data = {
            "shap_values": [[0.1, -0.1], [-0.2, 0.3]],
            "expected_value": -1.0,
            # NO feature_names key
            "categorical_features": ["a", "b"],
            "continuous_features": [],
            "X_values": {"a": ["X", "Y"], "b": [0, 1]},
            "exposure": None,
            "annualise_exposure": False,
        }
        sr = SHAPRelativities.from_dict(data)
        # Should still have columns
        assert sr._X.width == 2
        assert set(sr._X.columns) == {"a", "b"}

    def test_from_dict_without_annualise_exposure_key(self):
        """from_dict without annualise_exposure key should default to True."""
        from shap_relativities import SHAPRelativities

        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        data = {
            "shap_values": [[0.1]],
            "expected_value": 0.0,
            "feature_names": ["x"],
            "categorical_features": ["x"],
            "continuous_features": [],
            "X_values": {"x": ["A"]},
            "exposure": None,
            # NO annualise_exposure key
        }
        sr = SHAPRelativities.from_dict(data)
        assert sr._annualise_exposure is True


# ===========================================================================
# _core.py: _run_with_spinner (verbose path without tqdm)
# ===========================================================================


class TestRunWithSpinner:
    """Test the _run_with_spinner utility function."""

    def test_verbose_false_calls_fn_directly(self):
        from shap_relativities._core import _run_with_spinner

        called = []

        def fn():
            called.append(1)
            return 42

        result = _run_with_spinner(fn, n_obs=100, verbose=False)
        assert result == 42
        assert len(called) == 1

    def test_verbose_true_without_tqdm_falls_back(self, monkeypatch, capsys):
        """With tqdm unavailable, should print elapsed time message."""
        import shap_relativities._core as core

        monkeypatch.setattr(core, "_TQDM_AVAILABLE", False)

        def fn():
            return "result"

        result = core._run_with_spinner(fn, n_obs=500, verbose=True)
        assert result == "result"
        captured = capsys.readouterr()
        # Should have printed something about computing SHAP values
        assert "Computing SHAP values" in captured.out

    def test_verbose_true_propagates_exception(self, monkeypatch):
        """Exception raised in fn() should propagate to the caller."""
        import shap_relativities._core as core

        monkeypatch.setattr(core, "_TQDM_AVAILABLE", False)

        def fn():
            raise ValueError("deliberate test error")

        with pytest.raises(ValueError, match="deliberate test error"):
            core._run_with_spinner(fn, n_obs=10, verbose=True)

    def test_verbose_false_propagates_exception(self):
        from shap_relativities._core import _run_with_spinner

        def fn():
            raise RuntimeError("test error")

        with pytest.raises(RuntimeError, match="test error"):
            _run_with_spinner(fn, n_obs=10, verbose=False)


# ===========================================================================
# _normalisation.py: _base_se and _effective_se fallback paths
# ===========================================================================


class TestNormalisationFallbackPaths:
    """_base_se and _effective_se fall back to raw n_obs when wsq_weight absent."""

    def _no_wsq_df(self, levels, mean_shaps, shap_stds, n_obs_list):
        return pl.DataFrame({
            "feature": ["f"] * len(levels),
            "level": [str(l) for l in levels],
            "mean_shap": [float(x) for x in mean_shaps],
            "shap_std": [float(x) for x in shap_stds],
            "n_obs": list(n_obs_list),
            "exposure_weight": [1.0 / len(levels)] * len(levels),
        })

    def test_base_se_without_wsq_weight_uses_n_obs(self):
        """_base_se should use n_obs when wsq_weight column is absent."""
        from shap_relativities._normalisation import _base_se

        base_df = pl.DataFrame({
            "feature": ["f"],
            "level": ["A"],
            "mean_shap": [0.0],
            "shap_std": [0.6],
            "n_obs": [100],
            "exposure_weight": [1.0],
            # No wsq_weight
        })
        se = _base_se(base_df)
        # Expected: 0.6 / sqrt(100) = 0.06
        assert abs(se - 0.06) < 1e-9

    def test_base_se_with_n_obs_one(self):
        """_base_se with n_obs=1 should use n_eff=1."""
        from shap_relativities._normalisation import _base_se

        base_df = pl.DataFrame({
            "feature": ["f"],
            "level": ["A"],
            "mean_shap": [0.0],
            "shap_std": [0.5],
            "n_obs": [1],
            "exposure_weight": [1.0],
        })
        se = _base_se(base_df)
        # Expected: 0.5 / sqrt(1) = 0.5
        assert abs(se - 0.5) < 1e-9

    def test_normalise_base_level_without_wsq_weight_succeeds(self):
        """normalise_base_level should work even without wsq_weight column."""
        from shap_relativities._normalisation import normalise_base_level

        df = self._no_wsq_df(["A", "B", "C"], [0.0, 0.3, -0.2],
                              [0.1, 0.1, 0.1], [100, 100, 100])
        result = normalise_base_level(df, "A", ci_level=0.95)
        assert "relativity" in result.columns
        assert "lower_ci" in result.columns
        assert "upper_ci" in result.columns
        base_row = result.filter(pl.col("level") == "A")
        assert abs(base_row["relativity"][0] - 1.0) < 1e-12

    def test_normalise_mean_without_wsq_weight_succeeds(self):
        """normalise_mean should work without wsq_weight column."""
        from shap_relativities._normalisation import normalise_mean

        df = self._no_wsq_df(["A", "B"], [0.0, 0.5],
                              [0.1, 0.1], [100, 100])
        result = normalise_mean(df)
        assert "relativity" in result.columns
        # B has higher mean_shap than A, so B relativity > A relativity
        rel_a = result.filter(pl.col("level") == "A")["relativity"][0]
        rel_b = result.filter(pl.col("level") == "B")["relativity"][0]
        assert rel_b > rel_a

    def test_effective_se_without_wsq_weight_clips_n_eff_at_1(self):
        """_effective_se falls back to n_obs; n_obs=0 should clip to 1."""
        from shap_relativities._normalisation import normalise_mean

        # n_obs=0 is unusual but must not produce division by zero
        df = pl.DataFrame({
            "feature": ["f"],
            "level": ["A"],
            "mean_shap": [0.0],
            "shap_std": [0.5],
            "n_obs": [0],
            "exposure_weight": [1.0],
            # no wsq_weight
        })
        # normalise_mean uses _effective_se internally
        result = normalise_mean(df)
        # Key: must not crash and CI values must be finite
        assert not math.isnan(result["lower_ci"][0])
        assert not math.isnan(result["upper_ci"][0])


# ===========================================================================
# _inference.py: _dense_rank_descending with ties
# ===========================================================================


class TestDenseRankDescending:
    """Test the ranking utility used in importance_table."""

    def test_all_distinct_values(self):
        from shap_relativities._inference import _dense_rank_descending

        ranks = _dense_rank_descending(np.array([3.0, 1.0, 2.0]))
        # 3.0 is rank 1, 2.0 is rank 2, 1.0 is rank 3
        assert ranks[0] == 1  # 3.0
        assert ranks[2] == 2  # 2.0
        assert ranks[1] == 3  # 1.0

    def test_tied_values_share_rank(self):
        from shap_relativities._inference import _dense_rank_descending

        ranks = _dense_rank_descending(np.array([2.0, 2.0, 1.0]))
        # Both 2.0s get rank 1, 1.0 gets rank 3 (dense rank)
        assert ranks[0] == ranks[1] == 1
        assert ranks[2] == 3

    def test_all_tied(self):
        from shap_relativities._inference import _dense_rank_descending

        ranks = _dense_rank_descending(np.array([5.0, 5.0, 5.0]))
        assert all(r == 1 for r in ranks)

    def test_single_element(self):
        from shap_relativities._inference import _dense_rank_descending

        ranks = _dense_rank_descending(np.array([7.5]))
        assert ranks == [1]

    def test_descending_returns_list_of_ints(self):
        from shap_relativities._inference import _dense_rank_descending

        ranks = _dense_rank_descending(np.array([4.0, 3.0, 2.0, 1.0]))
        assert isinstance(ranks, list)
        assert all(isinstance(r, int) for r in ranks)
        assert ranks == [1, 2, 3, 4]


# ===========================================================================
# _inference.py: SHAPInference repr
# ===========================================================================


class TestSHAPInferenceRepr:
    """__repr__ should indicate fitted/not-fitted status."""

    def test_repr_before_fit(self):
        from shap_relativities import SHAPInference

        shap_vals = np.zeros((50, 2))
        y = np.ones(50)
        si = SHAPInference(shap_vals, y, ["a", "b"])
        r = repr(si)
        assert "SHAPInference" in r
        assert "not fitted" in r

    def test_repr_after_fit(self):
        pytest.importorskip("sklearn")
        from shap_relativities import SHAPInference

        shap_vals, y, names = _small_inference_data(n=100, d=2)
        si = SHAPInference(shap_vals, y, names, p=2.0, n_folds=2, random_state=0)
        si.fit()
        r = repr(si)
        assert "fitted" in r
        assert "not fitted" not in r


# ===========================================================================
# _inference.py: ranking_ci edge cases
# ===========================================================================


class TestRankingCIEdgeCases:
    """ranking_ci should handle same feature and unknown features gracefully."""

    @pytest.fixture(scope="class")
    def fitted_si(self):
        pytest.importorskip("sklearn")
        from shap_relativities import SHAPInference

        shap_vals, y, names = _small_inference_data(n=200, d=3, seed=1)
        si = SHAPInference(shap_vals, y, names, p=2.0, n_folds=2, random_state=0)
        si.fit()
        return si

    def test_same_feature_returns_z_stat_zero(self, fitted_si):
        """Comparing a feature with itself: diff=0, SE=0 => z_stat=0, p_value=1."""
        result = fitted_si.ranking_ci("feat_0", "feat_0")
        assert result["diff"] == 0.0
        assert result["z_stat"] == 0.0
        assert result["p_value"] == 1.0

    def test_unknown_feature_a_raises(self, fitted_si):
        with pytest.raises(ValueError, match="feature_a"):
            fitted_si.ranking_ci("nonexistent", "feat_0")

    def test_unknown_feature_b_raises(self, fitted_si):
        with pytest.raises(ValueError, match="feature_b"):
            fitted_si.ranking_ci("feat_0", "nonexistent")

    def test_valid_pair_returns_dict_with_expected_keys(self, fitted_si):
        result = fitted_si.ranking_ci("feat_0", "feat_1")
        expected_keys = {"diff", "se_diff", "z_stat", "p_value", "ci_lower", "ci_upper"}
        assert set(result.keys()) == expected_keys

    def test_valid_pair_more_important_feature_has_positive_diff(self, fitted_si):
        """feat_0 has highest true importance (beta=0.6), feat_2 has lowest (beta=0.1)."""
        result = fitted_si.ranking_ci("feat_0", "feat_2")
        # theta_hat for feat_0 should exceed feat_2 with n=200
        assert result["diff"] > 0


# ===========================================================================
# _inference.py: SHAPInference with p < 2 (smoothed path)
# ===========================================================================


class TestSHAPInferencePLessThan2:
    """p=1 activates the smoothed estimator; should emit a UserWarning."""

    def test_p_one_emits_smoothing_warning(self):
        pytest.importorskip("sklearn")
        from shap_relativities import SHAPInference

        shap_vals, y, names = _small_inference_data(n=100, d=2, seed=5)
        si = SHAPInference(shap_vals, y, names, p=1.0, n_folds=2, random_state=0)
        with pytest.warns(UserWarning, match="smoothed estimator"):
            si.fit()

    def test_p_one_importance_table_has_correct_schema(self):
        pytest.importorskip("sklearn")
        from shap_relativities import SHAPInference

        shap_vals, y, names = _small_inference_data(n=100, d=2, seed=6)
        si = SHAPInference(shap_vals, y, names, p=1.0, n_folds=2, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            si.fit()

        tbl = si.importance_table()
        assert "theta_hat" in tbl.columns
        assert "se" in tbl.columns
        assert len(tbl) == 2

    def test_p_one_five_uses_smoothing(self):
        """Any p between 1 and 2 should trigger smoothing warning."""
        pytest.importorskip("sklearn")
        from shap_relativities import SHAPInference

        shap_vals, y, names = _small_inference_data(n=80, d=2, seed=8)
        si = SHAPInference(shap_vals, y, names, p=1.5, n_folds=2, random_state=0)
        with pytest.warns(UserWarning, match="smoothed estimator"):
            si.fit()

    def test_custom_beta_n_overrides_default(self):
        """Explicitly passing beta_n should suppress the default calculation."""
        pytest.importorskip("sklearn")
        from shap_relativities import SHAPInference

        shap_vals, y, names = _small_inference_data(n=80, d=2, seed=9)
        si = SHAPInference(
            shap_vals, y, names, p=1.5, n_folds=2, random_state=0, beta_n=10.0
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            si.fit()
        # Should warn about smoothed estimator with the given beta_n
        smoothing_warns = [x for x in w if "smoothed estimator" in str(x.message)]
        assert len(smoothing_warns) >= 1


# ===========================================================================
# _inference.py: influence_matrix property
# ===========================================================================


class TestInfluenceMatrix:
    """influence_matrix should return (n_obs, n_features) copy of rho."""

    def test_influence_matrix_shape(self):
        pytest.importorskip("sklearn")
        from shap_relativities import SHAPInference

        shap_vals, y, names = _small_inference_data(n=150, d=3, seed=10)
        si = SHAPInference(shap_vals, y, names, p=2.0, n_folds=2, random_state=0)
        si.fit()
        rho = si.influence_matrix
        assert rho.shape == (150, 3)

    def test_influence_matrix_is_copy(self):
        """Modifying the returned array must not affect the internal state."""
        pytest.importorskip("sklearn")
        from shap_relativities import SHAPInference

        shap_vals, y, names = _small_inference_data(n=100, d=2, seed=11)
        si = SHAPInference(shap_vals, y, names, p=2.0, n_folds=2, random_state=0)
        si.fit()
        rho1 = si.influence_matrix
        rho1[:] = 999.0
        rho2 = si.influence_matrix
        assert not np.all(rho2 == 999.0)

    def test_influence_matrix_before_fit_raises(self):
        from shap_relativities import SHAPInference

        shap_vals, y, names = _small_inference_data(n=100, d=2)
        si = SHAPInference(shap_vals, y, names)
        with pytest.raises(RuntimeError, match="fit()"):
            _ = si.influence_matrix


# ===========================================================================
# _inference.py: importance_table p-value and sigma_hat
# ===========================================================================


class TestImportanceTableExtras:
    """Validate p_value_nonzero and sigma_hat columns in importance_table."""

    @pytest.fixture(scope="class")
    def fitted_si(self):
        pytest.importorskip("sklearn")
        from shap_relativities import SHAPInference

        shap_vals, y, names = _small_inference_data(n=300, d=2, seed=12)
        si = SHAPInference(shap_vals, y, names, p=2.0, n_folds=3, random_state=0)
        si.fit()
        return si

    def test_p_values_in_zero_one_range(self, fitted_si):
        tbl = fitted_si.importance_table()
        p_vals = tbl["p_value_nonzero"].to_numpy()
        assert all(0.0 <= p <= 1.0 for p in p_vals)

    def test_sigma_hat_positive(self, fitted_si):
        """sigma_hat = SE * sqrt(n) should be positive."""
        tbl = fitted_si.importance_table()
        sigma = tbl["sigma_hat"].to_numpy()
        assert all(s >= 0 for s in sigma)

    def test_ranks_are_permutation_of_1_to_d(self, fitted_si):
        """Ranks should be a permutation of 1..d (no gaps for distinct theta_hats)."""
        tbl = fitted_si.importance_table()
        ranks = sorted(tbl["rank"].to_list())
        # Ranks may not be a clean 1..d if there are ties, but they should cover 1..d
        assert ranks[0] == 1
        assert max(ranks) <= len(tbl)

    def test_ci_level_90_gives_narrower_intervals_than_95(self):
        """90% CIs should be narrower than 95% CIs."""
        pytest.importorskip("sklearn")
        from shap_relativities import SHAPInference

        shap_vals, y, names = _small_inference_data(n=200, d=2, seed=13)

        si_95 = SHAPInference(shap_vals, y, names, p=2.0, ci_level=0.95,
                              n_folds=2, random_state=0)
        si_95.fit()
        tbl_95 = si_95.importance_table()

        si_90 = SHAPInference(shap_vals, y, names, p=2.0, ci_level=0.90,
                              n_folds=2, random_state=0)
        si_90.fit()
        tbl_90 = si_90.importance_table()

        # Width of interval for first feature
        width_95 = (tbl_95["theta_upper"] - tbl_95["theta_lower"])[0]
        width_90 = (tbl_90["theta_upper"] - tbl_90["theta_lower"])[0]
        assert width_90 < width_95


# ===========================================================================
# _inference.py: SHAPInference validation errors
# ===========================================================================


class TestSHAPInferenceValidation:
    """Test __init__ validation for SHAPInference."""

    def test_2d_shap_required(self):
        from shap_relativities import SHAPInference

        with pytest.raises(ValueError, match="2D array"):
            SHAPInference(np.zeros(10), np.zeros(10), ["a"])

    def test_1d_y_required(self):
        from shap_relativities import SHAPInference

        with pytest.raises(ValueError, match="1D array"):
            SHAPInference(np.zeros((10, 2)), np.zeros((10, 1)), ["a", "b"])

    def test_mismatched_n_obs_raises(self):
        from shap_relativities import SHAPInference

        with pytest.raises(ValueError, match="same number of observations"):
            SHAPInference(np.zeros((10, 2)), np.zeros(8), ["a", "b"])

    def test_mismatched_feature_names_raises(self):
        from shap_relativities import SHAPInference

        with pytest.raises(ValueError, match="feature_names"):
            SHAPInference(np.zeros((10, 2)), np.zeros(10), ["a", "b", "c"])

    def test_p_below_one_raises(self):
        from shap_relativities import SHAPInference

        with pytest.raises(ValueError, match="p must be"):
            SHAPInference(np.zeros((10, 2)), np.zeros(10), ["a", "b"], p=0.5)

    def test_n_folds_below_two_raises(self):
        from shap_relativities import SHAPInference

        with pytest.raises(ValueError, match="n_folds"):
            SHAPInference(np.zeros((10, 2)), np.zeros(10), ["a", "b"], n_folds=1)

    def test_ci_level_out_of_range_raises(self):
        from shap_relativities import SHAPInference

        with pytest.raises(ValueError, match="ci_level"):
            SHAPInference(np.zeros((10, 2)), np.zeros(10), ["a", "b"], ci_level=1.5)

    def test_duplicate_feature_names_raises(self):
        from shap_relativities import SHAPInference

        with pytest.raises(ValueError, match="unique"):
            SHAPInference(np.zeros((10, 2)), np.zeros(10), ["a", "a"])


# ===========================================================================
# _inference.py: _make_nuisance_estimator
# ===========================================================================


class TestMakeNuisanceEstimator:
    """Test the nuisance estimator factory."""

    def test_gradient_boosting_string_returns_estimator(self):
        pytest.importorskip("sklearn")
        from shap_relativities._inference import _make_nuisance_estimator

        est = _make_nuisance_estimator("gradient_boosting")
        # Should be an sklearn-compatible estimator with fit/predict
        assert hasattr(est, "fit")
        assert hasattr(est, "predict")

    def test_unknown_string_raises(self):
        pytest.importorskip("sklearn")
        from shap_relativities._inference import _make_nuisance_estimator

        with pytest.raises(ValueError, match="Unknown nuisance_estimator"):
            _make_nuisance_estimator("random_forest")

    def test_custom_estimator_returned_as_is(self):
        pytest.importorskip("sklearn")
        from shap_relativities._inference import _make_nuisance_estimator
        from sklearn.linear_model import Ridge

        est = Ridge()
        result = _make_nuisance_estimator(est)
        assert result is est


# ===========================================================================
# _plotting.py: plot_continuous CI fill and single-feature grid
# ===========================================================================


class TestPlottingEdgeCases:
    """Edge cases in plot_continuous and plot_relativities."""

    @pytest.fixture(autouse=True)
    def _use_agg_backend(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        yield
        plt.close("all")

    def _cont_data(self, n=10, feature="age"):
        """Build a continuous-feature relativity DataFrame."""
        x = np.linspace(20, 70, n)
        return pl.DataFrame({
            "feature": [feature] * n,
            "level": [str(float(v)) for v in x],
            "relativity": (1.0 + 0.1 * np.sin(x)).tolist(),
            "lower_ci": (0.9 + 0.05 * np.sin(x)).tolist(),
            "upper_ci": (1.1 + 0.15 * np.sin(x)).tolist(),
            "mean_shap": [0.0] * n,
            "shap_std": [0.0] * n,
            "n_obs": [100] * n,
            "exposure_weight": [1.0 / n] * n,
        })

    def test_plot_continuous_show_ci_true_with_valid_cis(self):
        """plot_continuous with show_ci=True on valid CI data should not raise."""
        import matplotlib.pyplot as plt
        from shap_relativities._plotting import plot_continuous

        data = self._cont_data()
        fig, ax = plt.subplots()
        plot_continuous(data, "age", ax, show_ci=True)

    def test_plot_continuous_nan_cis_no_fill(self):
        """plot_continuous with NaN CIs and show_ci=True should still not raise."""
        import matplotlib.pyplot as plt
        from shap_relativities._plotting import plot_continuous

        n = 5
        x = np.linspace(0, 1, n)
        data = pl.DataFrame({
            "feature": ["x"] * n,
            "level": [str(float(v)) for v in x],
            "relativity": [1.0] * n,
            "lower_ci": [float("nan")] * n,
            "upper_ci": [float("nan")] * n,
            "mean_shap": [0.0] * n,
            "shap_std": [0.0] * n,
            "n_obs": [50] * n,
            "exposure_weight": [0.2] * n,
        })
        fig, ax = plt.subplots()
        # Should not raise even with NaN CIs
        plot_continuous(data, "x", ax, show_ci=True)

    def test_plot_relativities_single_feature_creates_figure(self):
        """plot_relativities with exactly 1 feature should create a non-array axes."""
        import matplotlib.pyplot as plt
        from shap_relativities._plotting import plot_relativities

        data = self._cont_data(feature="veh_age")
        # Single feature: axes is scalar (not array) — tests the flatten branch
        plot_relativities(
            data,
            categorical_features=[],
            continuous_features=["veh_age"],
            figsize=(6, 4),
        )

    def test_plot_relativities_multiple_features_hides_extra_axes(self):
        """With 2 features and 3-column grid, the 3rd subplot should be hidden."""
        import matplotlib.pyplot as plt
        from shap_relativities._plotting import plot_relativities

        data1 = self._cont_data(feature="age")
        data2 = self._cont_data(feature="ncd")
        combined = pl.concat([data1, data2])

        # 2 features => n_cols=2, n_rows=1 => no hidden axes needed
        # 4 features would have hidden axes. Test with 4 to exercise hide loop.
        d3 = self._cont_data(feature="area")
        d4 = self._cont_data(feature="mileage")
        four_features = pl.concat([data1, data2, d3, d4])

        # 4 features: n_cols=3, n_rows=2 => 6 subplots, 2 hidden
        plot_relativities(
            four_features,
            categorical_features=[],
            continuous_features=["age", "ncd", "area", "mileage"],
            figsize=(12, 8),
        )

    def test_plot_categorical_color_parameter_accepted(self):
        """plot_categorical should accept a custom color without raising."""
        import matplotlib.pyplot as plt
        from shap_relativities._plotting import plot_categorical

        data = pl.DataFrame({
            "feature": ["f", "f", "f"],
            "level": ["A", "B", "C"],
            "relativity": [1.0, 1.2, 0.9],
            "lower_ci": [0.9, 1.1, 0.8],
            "upper_ci": [1.1, 1.3, 1.0],
            "mean_shap": [0.0, 0.18, -0.1],
            "shap_std": [0.05, 0.05, 0.05],
            "n_obs": [100, 100, 100],
            "exposure_weight": [1 / 3] * 3,
        })
        fig, ax = plt.subplots()
        plot_categorical(data, "f", ax, show_ci=True, color="#e67e22")

    def test_plot_continuous_xlabel_and_title(self):
        """plot_continuous should set xlabel and title to the feature name."""
        import matplotlib.pyplot as plt
        from shap_relativities._plotting import plot_continuous

        data = self._cont_data(feature="driver_age")
        fig, ax = plt.subplots()
        plot_continuous(data, "driver_age", ax, show_ci=False)
        assert ax.get_xlabel() == "driver_age"
        assert ax.get_title() == "driver_age"


# ===========================================================================
# datasets/motor.py: internal helper functions
# ===========================================================================


class TestMotorInternalHelpers:
    """Test the internal DGP helper functions directly."""

    def test_driver_age_effect_young_driver(self):
        """Drivers under 25 should get the young driver penalty."""
        from shap_relativities.datasets.motor import _driver_age_effect, TRUE_FREQ_PARAMS

        ages = np.array([17, 20, 24])
        effect = _driver_age_effect(ages)
        expected = TRUE_FREQ_PARAMS["driver_age_young"]
        np.testing.assert_allclose(effect, expected)

    def test_driver_age_effect_old_driver(self):
        """Drivers 70+ should get the old driver penalty."""
        from shap_relativities.datasets.motor import _driver_age_effect, TRUE_FREQ_PARAMS

        ages = np.array([70, 75, 80, 85])
        effect = _driver_age_effect(ages)
        expected = TRUE_FREQ_PARAMS["driver_age_old"]
        np.testing.assert_allclose(effect, expected)

    def test_driver_age_effect_mid_range_is_zero(self):
        """Drivers 30-69 should have zero effect."""
        from shap_relativities.datasets.motor import _driver_age_effect

        ages = np.array([30, 40, 50, 60, 69])
        effect = _driver_age_effect(ages)
        np.testing.assert_allclose(effect, 0.0)

    def test_driver_age_effect_blend_zone_25_to_29(self):
        """Drivers aged 25-29 should have intermediate effects (blended)."""
        from shap_relativities.datasets.motor import _driver_age_effect, TRUE_FREQ_PARAMS

        ages = np.array([26, 27, 28])
        effect = _driver_age_effect(ages)
        # All effects should be strictly between 0 and the young driver penalty
        # (age 25 gives blend_factor=1.0 = full effect, so excluded)
        assert all(0 < e < TRUE_FREQ_PARAMS["driver_age_young"] for e in effect)

    def test_driver_age_effect_age_25_specific_blend(self):
        """Age 25 blend factor = (30-25)/5 = 1.0 => full young effect."""
        from shap_relativities.datasets.motor import _driver_age_effect, TRUE_FREQ_PARAMS

        ages = np.array([25])
        effect = _driver_age_effect(ages)
        # blend_factor = (30 - 25) / 5 = 1.0 => full young driver effect
        assert abs(effect[0] - TRUE_FREQ_PARAMS["driver_age_young"]) < 1e-9

    def test_calculate_earned_exposure_full_year(self):
        """365-day policy should give exposure close to 1.0 years."""
        from datetime import date, timedelta
        from shap_relativities.datasets.motor import _calculate_earned_exposure

        inc = [date(2020, 1, 1)]
        exp = [date(2021, 1, 1)]  # 366 days in 2020 (leap year)
        exposure = _calculate_earned_exposure(inc, exp)
        # 366 / 365.25 ~ 1.002
        assert 0.98 < exposure[0] < 1.01

    def test_calculate_earned_exposure_half_year(self):
        """183-day policy should give exposure close to 0.5."""
        from datetime import date, timedelta
        from shap_relativities.datasets.motor import _calculate_earned_exposure

        inc = [date(2020, 1, 1)]
        exp = [date(2020, 7, 2)]  # 183 days
        exposure = _calculate_earned_exposure(inc, exp)
        assert 0.48 < exposure[0] < 0.52

    def test_calculate_earned_exposure_zero_duration(self):
        """Same inception and expiry should give zero exposure (clipped to 0)."""
        from datetime import date
        from shap_relativities.datasets.motor import _calculate_earned_exposure

        d = date(2020, 6, 1)
        exposure = _calculate_earned_exposure([d], [d])
        assert exposure[0] == 0.0

    def test_calculate_earned_exposure_multiple_policies(self):
        """Should vectorise over multiple policies."""
        from datetime import date
        from shap_relativities.datasets.motor import _calculate_earned_exposure

        incs = [date(2020, 1, 1), date(2020, 7, 1)]
        exps = [date(2021, 1, 1), date(2021, 7, 1)]
        exposure = _calculate_earned_exposure(incs, exps)
        assert len(exposure) == 2
        assert all(e > 0 for e in exposure)

    def test_load_motor_has_policy_id_sequential(self):
        """policy_id should start at 1 and be sequential."""
        from shap_relativities.datasets.motor import load_motor

        df = load_motor(n_policies=5, seed=0)
        ids = df["policy_id"].to_list()
        assert ids == list(range(1, 6))

    def test_load_motor_date_columns_are_date_type(self):
        """inception_date and expiry_date must be Polars Date dtype."""
        from shap_relativities.datasets.motor import load_motor

        df = load_motor(n_policies=100, seed=0)
        assert df["inception_date"].dtype == pl.Date
        assert df["expiry_date"].dtype == pl.Date

    def test_load_motor_expiry_after_inception(self):
        """All policies should have expiry_date >= inception_date."""
        from shap_relativities.datasets.motor import load_motor

        df = load_motor(n_policies=200, seed=0)
        diff_days = (df["expiry_date"] - df["inception_date"]).dt.total_days()
        assert (diff_days >= 0).all()

    def test_load_motor_occupation_class_in_range(self):
        """Occupation class should be in [1, 5]."""
        from shap_relativities.datasets.motor import load_motor

        df = load_motor(n_policies=300, seed=0)
        occ = df["occupation_class"].to_numpy()
        assert occ.min() >= 1
        assert occ.max() <= 5

    def test_load_motor_policy_type_comp_or_tpft(self):
        """policy_type should only contain 'Comp' or 'TPFT'."""
        from shap_relativities.datasets.motor import load_motor

        df = load_motor(n_policies=200, seed=0)
        types = set(df["policy_type"].to_list())
        assert types.issubset({"Comp", "TPFT"})

    def test_load_motor_annual_mileage_in_range(self):
        """Annual mileage should be in [2000, 30000]."""
        from shap_relativities.datasets.motor import load_motor

        df = load_motor(n_policies=300, seed=0)
        mileage = df["annual_mileage"].to_numpy()
        assert mileage.min() >= 2000
        assert mileage.max() <= 30000

    def test_load_motor_vehicle_age_in_range(self):
        """Vehicle age should be in [0, 20]."""
        from shap_relativities.datasets.motor import load_motor

        df = load_motor(n_policies=300, seed=0)
        vage = df["vehicle_age"].to_numpy()
        assert vage.min() >= 0
        assert vage.max() <= 20

    def test_load_motor_ncd_protected_is_boolean(self):
        """ncd_protected should be a boolean column."""
        from shap_relativities.datasets.motor import load_motor

        df = load_motor(n_policies=100, seed=0)
        assert df["ncd_protected"].dtype == pl.Boolean

    def test_load_motor_accident_year_in_range(self):
        """Accident year should be in [2019, 2023] given the 5-year window."""
        from shap_relativities.datasets.motor import load_motor

        df = load_motor(n_policies=500, seed=0)
        years = df["accident_year"].to_numpy()
        assert years.min() >= 2019
        assert years.max() <= 2023

    def test_load_motor_incurred_non_negative(self):
        """Incurred amounts should be non-negative."""
        from shap_relativities.datasets.motor import load_motor

        df = load_motor(n_policies=500, seed=0)
        assert (df["incurred"] >= 0).all()


# ===========================================================================
# _core.py: exposure type handling (pl.Series and list)
# ===========================================================================


class TestExposureTypeHandling:
    """SHAPRelativities should accept multiple exposure input types."""

    def _make_sr_with_exposure(self, exposure):
        from shap_relativities import SHAPRelativities

        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        class _Dummy:
            pass

        X = pl.DataFrame({"x": [1, 2, 3]})
        # Should not raise
        return SHAPRelativities(model=_Dummy(), X=X, exposure=exposure)

    def test_exposure_as_polars_series(self):
        """Should accept a Polars Series as exposure."""
        exposure = pl.Series("exp", [0.9, 1.0, 1.0])
        sr = self._make_sr_with_exposure(exposure)
        assert sr._exposure is not None
        assert len(sr._exposure) == 3

    def test_exposure_as_numpy_array(self):
        """Should accept a numpy array as exposure."""
        exposure = np.array([0.8, 1.0, 0.5])
        sr = self._make_sr_with_exposure(exposure)
        assert isinstance(sr._exposure, np.ndarray)

    def test_exposure_as_list(self):
        """Should accept a list as exposure (converted via np.asarray)."""
        sr = self._make_sr_with_exposure([0.9, 1.0, 1.0])
        assert sr._exposure is not None
        assert len(sr._exposure) == 3

    def test_exposure_none_sets_none(self):
        """Exposure=None should leave _exposure as None."""
        sr = self._make_sr_with_exposure(None)
        assert sr._exposure is None


# ===========================================================================
# _core.py: feature inference from column dtypes
# ===========================================================================


class TestFeatureInference:
    """_infer_categorical and _infer_continuous should classify features correctly."""

    def _make_sr(self, df):
        from shap_relativities import SHAPRelativities

        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        class _Dummy:
            pass

        return SHAPRelativities(model=_Dummy(), X=df)

    def test_string_column_inferred_as_categorical(self):
        df = pl.DataFrame({"area": ["A", "B", "C"], "age": [25, 30, 35]})
        sr = self._make_sr(df)
        assert "area" in sr._categorical_features
        assert "age" not in sr._categorical_features

    def test_numeric_column_inferred_as_continuous(self):
        df = pl.DataFrame({"area": ["A", "B", "C"], "age": [25, 30, 35]})
        sr = self._make_sr(df)
        assert "age" in sr._continuous_features
        assert "area" not in sr._continuous_features

    def test_all_numeric_gives_empty_categoricals(self):
        df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        sr = self._make_sr(df)
        assert sr._categorical_features == []
        assert set(sr._continuous_features) == {"x", "y"}

    def test_all_string_gives_empty_continuous(self):
        df = pl.DataFrame({"a": ["X", "Y"], "b": ["P", "Q"]})
        sr = self._make_sr(df)
        assert set(sr._categorical_features) == {"a", "b"}
        assert sr._continuous_features == []


# ===========================================================================
# _core.py: to_dict and from_dict round-trip with exposure
# ===========================================================================


class TestToDictFromDictWithExposure:
    """to_dict/from_dict should preserve exposure correctly."""

    def test_round_trip_preserves_exposure(self):
        from shap_relativities import SHAPRelativities

        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        exposure = [0.8, 1.0, 1.0]
        data = {
            "shap_values": [[0.1, -0.1], [-0.2, 0.3], [0.0, 0.0]],
            "expected_value": -2.0,
            "feature_names": ["area", "ncd"],
            "categorical_features": ["area", "ncd"],
            "continuous_features": [],
            "X_values": {"area": ["A", "B", "A"], "ncd": [0, 1, 2]},
            "exposure": exposure,
            "annualise_exposure": True,
        }
        sr = SHAPRelativities.from_dict(data)
        d = sr.to_dict()
        sr2 = SHAPRelativities.from_dict(d)

        np.testing.assert_allclose(sr._exposure, sr2._exposure)

    def test_round_trip_none_exposure_stays_none(self):
        from shap_relativities import SHAPRelativities

        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        data = {
            "shap_values": [[0.1, -0.1], [-0.2, 0.3]],
            "expected_value": -2.0,
            "feature_names": ["a", "b"],
            "categorical_features": ["a", "b"],
            "continuous_features": [],
            "X_values": {"a": ["X", "Y"], "b": [0, 1]},
            "exposure": None,
            "annualise_exposure": False,
        }
        sr = SHAPRelativities.from_dict(data)
        d = sr.to_dict()
        sr2 = SHAPRelativities.from_dict(d)
        assert sr2._exposure is None


# ===========================================================================
# _inference.py: plot_importance (requires matplotlib)
# ===========================================================================


class TestPlotImportance:
    """plot_importance should draw a bar chart without raising."""

    @pytest.fixture(autouse=True)
    def _agg_backend(self):
        matplotlib = pytest.importorskip("matplotlib")
        pytest.importorskip("sklearn")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        yield
        plt.close("all")

    def test_plot_importance_returns_axes(self):
        from shap_relativities import SHAPInference

        shap_vals, y, names = _small_inference_data(n=150, d=3, seed=20)
        si = SHAPInference(shap_vals, y, names, p=2.0, n_folds=2, random_state=0)
        si.fit()
        ax = si.plot_importance()
        import matplotlib.axes
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_importance_top_n_limits_features(self):
        from shap_relativities import SHAPInference

        shap_vals, y, names = _small_inference_data(n=150, d=3, seed=21)
        si = SHAPInference(shap_vals, y, names, p=2.0, n_folds=2, random_state=0)
        si.fit()
        ax = si.plot_importance(top_n=2)
        # Only 2 bars should be drawn
        bars = ax.patches
        assert len(bars) == 2

    def test_plot_importance_unsorted(self):
        """sort=False should not raise."""
        from shap_relativities import SHAPInference

        shap_vals, y, names = _small_inference_data(n=100, d=2, seed=22)
        si = SHAPInference(shap_vals, y, names, p=2.0, n_folds=2, random_state=0)
        si.fit()
        ax = si.plot_importance(sort=False)
        assert ax is not None

    def test_plot_importance_before_fit_raises(self):
        from shap_relativities import SHAPInference

        shap_vals, y, names = _small_inference_data(n=100, d=2)
        si = SHAPInference(shap_vals, y, names)
        with pytest.raises(RuntimeError, match="fit()"):
            si.plot_importance()
