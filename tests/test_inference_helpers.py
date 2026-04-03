"""
Tests for the internal helper functions in shap_relativities._inference.

Covers the mathematical building blocks that the SHAPInference class relies on:
- _smoothed_phi and _smoothed_gamma_deriv (smoothing for p < 2)
- _default_beta_n (rate calibration)
- _unsmoothed_gamma (derivative of |phi|^p for p >= 2)
- _dense_rank_descending (ranking utility)
- _make_nuisance_estimator (string shorthand dispatch)
- SHAPInference validation errors (p<1, n_folds<2, shape mismatches, etc.)
- SHAPInference.influence_matrix property
- SHAPInference.ranking_ci error paths
- SHAPInference.__repr__ before and after fit
- SHAPInference.importance_table p-value and sigma_hat properties
- SHAPInference.fit with custom estimator objects
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from shap_relativities._inference import (
    _default_beta_n,
    _dense_rank_descending,
    _make_nuisance_estimator,
    _smoothed_gamma_deriv,
    _smoothed_phi,
    _unsmoothed_gamma,
    SHAPInference,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_linear_data(n: int = 300, d: int = 2, seed: int = 0):
    """Return (shap_values, y, feature_names) for a tiny linear DGP."""
    rng = np.random.default_rng(seed)
    betas = np.array([0.5, 0.2][:d])
    X = rng.normal(0, 1, size=(n, d))
    shap_values = X * betas[np.newaxis, :]
    y = np.abs(shap_values.sum(axis=1) + rng.normal(0, 0.3, n))
    names = [f"f{j}" for j in range(d)]
    return shap_values, y, names


# ===========================================================================
# _smoothed_phi
# ===========================================================================


class TestSmoothedPhi:
    """Tests for the smoothed approximation of |phi|^p."""

    def test_zero_input_gives_zero(self):
        """smoothed_phi at phi=0 must be zero (tanh(0)=0)."""
        result = _smoothed_phi(np.array([0.0]), p=1.5, beta=10.0)
        assert abs(result[0]) < 1e-9

    def test_positive_and_negative_symmetric(self):
        """smoothed_phi should be symmetric: f(-x) == f(x)."""
        x = np.array([0.1, 0.3, 0.7, 1.5])
        assert np.allclose(_smoothed_phi(x, p=1.5, beta=5.0), _smoothed_phi(-x, p=1.5, beta=5.0))

    def test_large_beta_approaches_abs_pow_p(self):
        """For very large beta, smoothed_phi should closely approximate |phi|^p."""
        phi = np.array([0.5, 1.0, 2.0])
        p = 1.5
        smoothed = _smoothed_phi(phi, p=p, beta=1e6)
        expected = np.abs(phi) ** p
        np.testing.assert_allclose(smoothed, expected, rtol=1e-4)

    def test_p2_large_beta_matches_phi_squared(self):
        """p=2 with large beta: smoothed_phi ~ phi^2."""
        phi = np.array([0.3, 0.8, 1.4])
        result = _smoothed_phi(phi, p=2.0, beta=1e5)
        np.testing.assert_allclose(result, phi ** 2, rtol=1e-4)

    def test_output_shape_matches_input(self):
        """Output shape must match input shape."""
        phi = np.arange(10, dtype=float)
        result = _smoothed_phi(phi, p=1.2, beta=3.0)
        assert result.shape == phi.shape

    def test_small_beta_gives_small_values_near_zero(self):
        """Very small beta means tanh(beta * ...) ≈ 0 near zero, giving near-zero output."""
        phi = np.array([0.01, 0.02])
        result = _smoothed_phi(phi, p=1.5, beta=0.001)
        # tanh argument is tiny, so result should be much less than |phi|^p
        expected_unsmoothed = np.abs(phi) ** 1.5
        assert (result < expected_unsmoothed).all()

    def test_all_positive_output(self):
        """Smoothed phi must be non-negative for non-zero inputs."""
        phi = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
        result = _smoothed_phi(phi, p=1.3, beta=5.0)
        assert (result >= 0).all()

    def test_vectorised_over_large_array(self):
        """Should handle large arrays without error."""
        rng = np.random.default_rng(99)
        phi = rng.normal(0, 1, size=10000)
        result = _smoothed_phi(phi, p=1.5, beta=10.0)
        assert result.shape == (10000,)
        assert not np.any(np.isnan(result))


# ===========================================================================
# _smoothed_gamma_deriv
# ===========================================================================


class TestSmoothedGammaDeriv:
    """Tests for the derivative of smoothed_phi."""

    def test_zero_phi_gives_zero_derivative(self):
        """At phi=0, the derivative should be zero (sign(0)=0)."""
        result = _smoothed_gamma_deriv(np.array([0.0]), p=1.5, beta=10.0)
        assert abs(result[0]) < 1e-9

    def test_antisymmetric_in_phi(self):
        """Derivative must be antisymmetric: gamma(-x) == -gamma(x)."""
        x = np.array([0.2, 0.7, 1.5])
        pos = _smoothed_gamma_deriv(x, p=1.5, beta=5.0)
        neg = _smoothed_gamma_deriv(-x, p=1.5, beta=5.0)
        np.testing.assert_allclose(neg, -pos, rtol=1e-9)

    def test_output_shape_matches_input(self):
        """Output shape must match input shape."""
        phi = np.ones(7)
        result = _smoothed_gamma_deriv(phi, p=1.2, beta=2.0)
        assert result.shape == phi.shape

    def test_large_positive_phi_positive_derivative(self):
        """For large positive phi, derivative should be positive."""
        phi = np.array([5.0, 10.0])
        result = _smoothed_gamma_deriv(phi, p=1.5, beta=10.0)
        assert (result > 0).all()

    def test_no_nans_in_large_array(self):
        """Should not produce NaN for arbitrary inputs."""
        rng = np.random.default_rng(42)
        phi = rng.normal(0, 2, size=5000)
        result = _smoothed_gamma_deriv(phi, p=1.3, beta=8.0)
        assert not np.any(np.isnan(result))

    def test_numerical_gradient_matches_analytic(self):
        """Finite-difference check of derivative against _smoothed_phi."""
        phi0 = np.array([0.5, 1.2, -0.8])
        p, beta = 1.5, 5.0
        eps = 1e-5
        analytic = _smoothed_gamma_deriv(phi0, p=p, beta=beta)
        numerical = (
            _smoothed_phi(phi0 + eps, p=p, beta=beta)
            - _smoothed_phi(phi0 - eps, p=p, beta=beta)
        ) / (2 * eps)
        np.testing.assert_allclose(analytic, numerical, rtol=1e-3)


# ===========================================================================
# _default_beta_n
# ===========================================================================


class TestDefaultBetaN:
    """Tests for the beta_n rate calibration."""

    def test_increases_with_n(self):
        """Beta_n should increase as n grows (p < 2 implies a positive rate)."""
        b_small = _default_beta_n(100, p=1.5)
        b_large = _default_beta_n(10000, p=1.5)
        assert b_large > b_small

    def test_positive_output(self):
        """Beta_n must always be positive."""
        for n in [50, 500, 5000]:
            for p in [1.0, 1.3, 1.7, 1.9]:
                result = _default_beta_n(n, p)
                assert result > 0, f"beta_n({n}, {p}) = {result}"

    def test_p_equal_2_gives_zero_rate(self):
        """At p=2, (2-p)=0, so beta_n = n^0 = 1 regardless of n."""
        # rate = (2-p) / (2*(p+delta)) = 0/(2*3) = 0
        b = _default_beta_n(1000, p=2.0)
        assert abs(b - 1.0) < 1e-9

    def test_float_output(self):
        """Return value must be a Python float."""
        result = _default_beta_n(500, p=1.5)
        assert isinstance(result, float)

    def test_p1_rate_is_one_quarter(self):
        """For p=1, delta=1: rate = (2-1)/(2*(1+1)) = 1/4."""
        n = 10000
        result = _default_beta_n(n, p=1.0)
        expected = float(n ** (1 / 4))
        assert abs(result - expected) < 1e-9


# ===========================================================================
# _unsmoothed_gamma
# ===========================================================================


class TestUnsmoothedGamma:
    """Tests for the unsmoothed gamma derivative (p >= 2)."""

    def test_zero_phi_gives_zero(self):
        """At phi=0, gamma_p(0) = p * sign(0) * 0^{p-1} = 0."""
        result = _unsmoothed_gamma(np.array([0.0]), p=2.0)
        assert abs(result[0]) < 1e-9

    def test_positive_phi_positive_result(self):
        """For positive phi, gamma should be positive."""
        result = _unsmoothed_gamma(np.array([0.5, 1.0, 2.0]), p=2.0)
        assert (result > 0).all()

    def test_negative_phi_negative_result(self):
        """For negative phi, gamma should be negative."""
        result = _unsmoothed_gamma(np.array([-0.5, -1.0, -2.0]), p=2.0)
        assert (result < 0).all()

    def test_p2_matches_derivative_of_phi_squared(self):
        """For p=2: gamma_2(phi) = 2*phi."""
        phi = np.array([0.3, 0.7, 1.5, -1.0])
        result = _unsmoothed_gamma(phi, p=2.0)
        expected = 2 * phi
        np.testing.assert_allclose(result, expected, rtol=1e-9)

    def test_antisymmetric(self):
        """gamma_p must be antisymmetric: gamma_p(-x) == -gamma_p(x)."""
        phi = np.array([0.2, 0.8, 1.6])
        pos = _unsmoothed_gamma(phi, p=3.0)
        neg = _unsmoothed_gamma(-phi, p=3.0)
        np.testing.assert_allclose(neg, -pos, rtol=1e-9)

    def test_output_shape(self):
        """Output shape must match input shape."""
        phi = np.ones(8)
        assert _unsmoothed_gamma(phi, p=2.5).shape == (8,)


# ===========================================================================
# _dense_rank_descending
# ===========================================================================


class TestDenseRankDescending:
    """Tests for the dense-rank-descending utility."""

    def test_basic_ranking(self):
        """Largest value gets rank 1."""
        values = np.array([0.1, 0.5, 0.3])
        ranks = _dense_rank_descending(values)
        assert ranks[1] == 1
        assert ranks[2] == 2
        assert ranks[0] == 3

    def test_ties_share_same_rank(self):
        """Tied values must share the same rank."""
        values = np.array([0.5, 0.5, 0.2])
        ranks = _dense_rank_descending(values)
        assert ranks[0] == ranks[1]
        assert ranks[2] > ranks[0]

    def test_single_element(self):
        """Single-element array should receive rank 1."""
        ranks = _dense_rank_descending(np.array([3.14]))
        assert ranks == [1]

    def test_all_equal(self):
        """All equal values should all receive rank 1."""
        ranks = _dense_rank_descending(np.array([2.0, 2.0, 2.0]))
        assert all(r == 1 for r in ranks)

    def test_sorted_ascending_gives_reverse_ranks(self):
        """Ascending values [a, b, c] where a<b<c give ranks [3, 2, 1]."""
        values = np.array([0.1, 0.5, 0.9])
        ranks = _dense_rank_descending(values)
        assert ranks == [3, 2, 1]

    def test_returns_list_of_ints(self):
        """Return type must be list of Python ints."""
        ranks = _dense_rank_descending(np.array([0.3, 0.7]))
        assert isinstance(ranks, list)
        assert all(isinstance(r, int) for r in ranks)

    def test_five_elements_no_ties(self):
        """Distinct values: check full rank assignment."""
        values = np.array([5.0, 1.0, 3.0, 4.0, 2.0])
        ranks = _dense_rank_descending(values)
        # Sorted descending: 5,4,3,2,1 → indices 0,3,2,4,1
        assert ranks[0] == 1  # 5.0 → rank 1
        assert ranks[3] == 2  # 4.0 → rank 2
        assert ranks[2] == 3  # 3.0 → rank 3
        assert ranks[4] == 4  # 2.0 → rank 4
        assert ranks[1] == 5  # 1.0 → rank 5


# ===========================================================================
# _make_nuisance_estimator
# ===========================================================================


class TestMakeNuisanceEstimator:
    """Tests for nuisance estimator factory."""

    def test_gradient_boosting_string(self):
        """'gradient_boosting' string should return a HistGBT instance."""
        from sklearn.ensemble import HistGradientBoostingRegressor
        est = _make_nuisance_estimator("gradient_boosting")
        assert isinstance(est, HistGradientBoostingRegressor)

    def test_unknown_string_raises(self):
        """Unknown string should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown nuisance_estimator"):
            _make_nuisance_estimator("random_forest")

    def test_sklearn_estimator_returned_as_is(self):
        """Passing an sklearn estimator directly should return it unchanged."""
        from sklearn.ensemble import HistGradientBoostingRegressor
        est = HistGradientBoostingRegressor()
        result = _make_nuisance_estimator(est)
        assert result is est

    def test_custom_object_returned_as_is(self):
        """Any non-string object is returned without modification."""
        class _Dummy:
            pass
        dummy = _Dummy()
        result = _make_nuisance_estimator(dummy)
        assert result is dummy


# ===========================================================================
# SHAPInference: validation errors
# ===========================================================================


class TestSHAPInferenceValidation:
    """Tests for input validation in SHAPInference.__init__."""

    def _base_arrays(self, n=200, d=2):
        rng = np.random.default_rng(0)
        sv = rng.normal(0, 1, (n, d))
        y = np.abs(rng.normal(0, 1, n))
        names = [f"f{j}" for j in range(d)]
        return sv, y, names

    def test_1d_shap_values_raises(self):
        """1D shap_values should raise ValueError."""
        sv, y, names = self._base_arrays()
        with pytest.raises(ValueError, match="2D array"):
            SHAPInference(sv[:, 0], y, names[:1])

    def test_2d_y_raises(self):
        """2D y should raise ValueError."""
        sv, y, names = self._base_arrays()
        with pytest.raises(ValueError, match="1D array"):
            SHAPInference(sv, y.reshape(-1, 1), names)

    def test_mismatched_n_obs_raises(self):
        """shap_values and y with different n should raise ValueError."""
        sv, y, names = self._base_arrays(n=100)
        with pytest.raises(ValueError, match="same number of observations"):
            SHAPInference(sv, y[:50], names)

    def test_mismatched_feature_names_raises(self):
        """len(feature_names) != n_features should raise ValueError."""
        sv, y, names = self._base_arrays(n=100, d=2)
        with pytest.raises(ValueError, match="len\\(feature_names\\)"):
            SHAPInference(sv, y, names + ["extra"])

    def test_p_below_one_raises(self):
        """p < 1 should raise ValueError."""
        sv, y, names = self._base_arrays()
        with pytest.raises(ValueError, match="p must be >= 1"):
            SHAPInference(sv, y, names, p=0.5)

    def test_p_exactly_one_is_valid(self):
        """p=1.0 is valid (with warning on fit)."""
        sv, y, names = self._base_arrays()
        # Should not raise during construction
        si = SHAPInference(sv, y, names, p=1.0)
        assert si.p == 1.0

    def test_n_folds_below_2_raises(self):
        """n_folds < 2 should raise ValueError."""
        sv, y, names = self._base_arrays()
        with pytest.raises(ValueError, match="n_folds must be >= 2"):
            SHAPInference(sv, y, names, n_folds=1)

    def test_ci_level_zero_raises(self):
        """ci_level=0 should raise ValueError."""
        sv, y, names = self._base_arrays()
        with pytest.raises(ValueError, match="ci_level must be in"):
            SHAPInference(sv, y, names, ci_level=0.0)

    def test_ci_level_one_raises(self):
        """ci_level=1 should raise ValueError."""
        sv, y, names = self._base_arrays()
        with pytest.raises(ValueError, match="ci_level must be in"):
            SHAPInference(sv, y, names, ci_level=1.0)

    def test_duplicate_feature_names_raises(self):
        """Duplicate feature_names should raise ValueError."""
        sv, y, names = self._base_arrays(d=2)
        with pytest.raises(ValueError, match="unique"):
            SHAPInference(sv, y, ["x", "x"])

    def test_valid_construction_stores_attributes(self):
        """Valid construction should store all provided attributes."""
        sv, y, names = self._base_arrays(n=200, d=2)
        si = SHAPInference(sv, y, names, p=2.0, n_folds=4, ci_level=0.90, random_state=7)
        assert si.p == 2.0
        assert si.n_folds == 4
        assert si.ci_level == 0.90
        assert si.random_state == 7
        assert si.feature_names == names


# ===========================================================================
# SHAPInference: unfitted guard
# ===========================================================================


class TestSHAPInferenceUnfittedGuard:
    """Methods that require fit() should raise RuntimeError if called before it."""

    def _unfitted_si(self):
        sv, y, names = _small_linear_data(n=100, d=2)
        return SHAPInference(sv, y, names, p=2.0, n_folds=2)

    def test_importance_table_before_fit_raises(self):
        """importance_table() before fit() must raise RuntimeError."""
        si = self._unfitted_si()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            si.importance_table()

    def test_ranking_ci_before_fit_raises(self):
        """ranking_ci() before fit() must raise RuntimeError."""
        si = self._unfitted_si()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            si.ranking_ci("f0", "f1")

    def test_influence_matrix_before_fit_raises(self):
        """influence_matrix before fit() must raise RuntimeError."""
        si = self._unfitted_si()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            _ = si.influence_matrix


# ===========================================================================
# SHAPInference: repr
# ===========================================================================


class TestSHAPInferenceRepr:
    """Tests for SHAPInference __repr__."""

    def test_repr_before_fit(self):
        """repr should show 'not fitted' before fit()."""
        sv, y, names = _small_linear_data(n=100, d=2)
        si = SHAPInference(sv, y, names, p=2.0)
        r = repr(si)
        assert "not fitted" in r
        assert "n_obs=100" in r
        assert "n_features=2" in r

    def test_repr_after_fit(self):
        """repr should show 'fitted' after fit()."""
        sv, y, names = _small_linear_data(n=200, d=2)
        si = SHAPInference(sv, y, names, p=2.0, n_folds=2, random_state=0)
        si.fit()
        r = repr(si)
        assert "fitted" in r

    def test_repr_shows_p(self):
        """repr should include the p value."""
        sv, y, names = _small_linear_data(n=100, d=2)
        si = SHAPInference(sv, y, names, p=2.0)
        assert "p=2.0" in repr(si)


# ===========================================================================
# SHAPInference: influence_matrix property
# ===========================================================================


class TestInfluenceMatrix:
    """Tests for the influence_matrix property."""

    def test_shape_correct(self):
        """influence_matrix should be (n_obs, n_features)."""
        sv, y, names = _small_linear_data(n=200, d=2)
        si = SHAPInference(sv, y, names, p=2.0, n_folds=3, random_state=0)
        si.fit()
        im = si.influence_matrix
        assert im.shape == (200, 2)

    def test_returns_copy_not_reference(self):
        """influence_matrix must return a copy, not the internal array."""
        sv, y, names = _small_linear_data(n=200, d=2)
        si = SHAPInference(sv, y, names, p=2.0, n_folds=3, random_state=0)
        si.fit()
        im = si.influence_matrix
        im[:] = 999.0
        # Internal rho should not be modified
        assert si._rho[0, 0] != 999.0

    def test_mean_equals_theta_hat(self):
        """mean(rho[:, j]) should equal theta_hat[j] by construction."""
        sv, y, names = _small_linear_data(n=300, d=2)
        si = SHAPInference(sv, y, names, p=2.0, n_folds=3, random_state=0)
        si.fit()
        im = si.influence_matrix
        theta = si._theta_hat
        for j in range(len(names)):
            computed_mean = float(np.mean(im[:, j]))
            assert abs(computed_mean - theta[j]) < 1e-9, (
                f"Feature {j}: mean(rho) = {computed_mean:.6f} != theta_hat = {theta[j]:.6f}"
            )

    def test_no_nans_in_influence_matrix(self):
        """influence_matrix must not contain NaN values after a successful fit."""
        sv, y, names = _small_linear_data(n=200, d=2)
        si = SHAPInference(sv, y, names, p=2.0, n_folds=3, random_state=0)
        si.fit()
        assert not np.any(np.isnan(si.influence_matrix))


# ===========================================================================
# SHAPInference: ranking_ci error paths
# ===========================================================================


class TestRankingCIErrors:
    """Tests for ranking_ci error paths."""

    @pytest.fixture(scope="class")
    def fitted_si(self):
        sv, y, names = _small_linear_data(n=300, d=3, seed=5)
        si = SHAPInference(sv, y, names, p=2.0, n_folds=3, random_state=0)
        si.fit()
        return si

    def test_unknown_feature_a_raises(self, fitted_si):
        """Unknown feature_a should raise ValueError."""
        with pytest.raises(ValueError, match="feature_a"):
            fitted_si.ranking_ci("nonexistent", "f0")

    def test_unknown_feature_b_raises(self, fitted_si):
        """Unknown feature_b should raise ValueError."""
        with pytest.raises(ValueError, match="feature_b"):
            fitted_si.ranking_ci("f0", "nonexistent")

    def test_same_feature_gives_zero_diff(self, fitted_si):
        """Comparing a feature with itself: diff must be zero.
        
        When se_diff=0 (diff_rho is all zeros), z_stat is set to float('inf')
        by the implementation (zero / zero case). The diff itself is always 0.
        """
        result = fitted_si.ranking_ci("f0", "f0")
        assert result["diff"] == pytest.approx(0.0)
        # z_stat is inf when se_diff=0 (implementation choice for zero/zero)
        assert result["se_diff"] == pytest.approx(0.0)

    def test_return_dict_has_expected_keys(self, fitted_si):
        """ranking_ci result must contain all documented keys."""
        result = fitted_si.ranking_ci("f0", "f1")
        expected_keys = {"diff", "se_diff", "z_stat", "p_value", "ci_lower", "ci_upper"}
        assert expected_keys == set(result.keys())

    def test_p_value_in_unit_interval(self, fitted_si):
        """p_value must be in [0, 1]."""
        result = fitted_si.ranking_ci("f0", "f1")
        assert 0.0 <= result["p_value"] <= 1.0

    def test_ci_lower_leq_ci_upper(self, fitted_si):
        """ci_lower must not exceed ci_upper."""
        result = fitted_si.ranking_ci("f0", "f1")
        assert result["ci_lower"] <= result["ci_upper"]

    def test_antisymmetry_of_diff(self, fitted_si):
        """diff(a, b) should equal -diff(b, a)."""
        ab = fitted_si.ranking_ci("f0", "f1")
        ba = fitted_si.ranking_ci("f1", "f0")
        assert ab["diff"] == pytest.approx(-ba["diff"], rel=1e-9)

    def test_se_diff_non_negative(self, fitted_si):
        """se_diff must be non-negative."""
        result = fitted_si.ranking_ci("f0", "f2")
        assert result["se_diff"] >= 0.0


# ===========================================================================
# SHAPInference: importance_table additional properties
# ===========================================================================


class TestImportanceTableAdditional:
    """Additional properties of importance_table()."""

    @pytest.fixture(scope="class")
    def fitted_si(self):
        sv, y, names = _small_linear_data(n=400, d=3, seed=7)
        si = SHAPInference(sv, y, names, p=2.0, n_folds=3, random_state=0)
        si.fit()
        return si

    def test_lower_leq_theta_hat(self, fitted_si):
        """theta_lower must be <= theta_hat for all rows."""
        tbl = fitted_si.importance_table()
        assert (tbl["theta_lower"] <= tbl["theta_hat"]).all()

    def test_upper_geq_theta_hat(self, fitted_si):
        """theta_upper must be >= theta_hat for all rows."""
        tbl = fitted_si.importance_table()
        assert (tbl["theta_upper"] >= tbl["theta_hat"]).all()

    def test_sigma_hat_equals_se_times_sqrt_n(self, fitted_si):
        """sigma_hat == se * sqrt(n) should hold exactly."""
        n = fitted_si.shap_values.shape[0]
        tbl = fitted_si.importance_table()
        computed_sigma = tbl["se"].to_numpy() * math.sqrt(n)
        np.testing.assert_allclose(
            tbl["sigma_hat"].to_numpy(), computed_sigma, rtol=1e-9
        )

    def test_p_values_in_unit_interval(self, fitted_si):
        """p_value_nonzero must all be in [0, 1]."""
        tbl = fitted_si.importance_table()
        pvals = tbl["p_value_nonzero"].to_numpy()
        assert (pvals >= 0.0).all()
        assert (pvals <= 1.0).all()

    def test_n_rows_equals_n_features(self, fitted_si):
        """importance_table should have one row per feature."""
        tbl = fitted_si.importance_table()
        assert len(tbl) == 3

    def test_sorted_by_rank(self, fitted_si):
        """importance_table() is sorted by rank ascending (rank 1 first)."""
        tbl = fitted_si.importance_table()
        ranks = tbl["rank"].to_list()
        assert ranks == sorted(ranks)

    def test_all_features_present(self, fitted_si):
        """All feature names should appear exactly once in the table."""
        tbl = fitted_si.importance_table()
        in_table = set(tbl["feature"].to_list())
        assert in_table == set(fitted_si.feature_names)

    def test_se_positive(self, fitted_si):
        """SE must be positive (non-degenerate data)."""
        tbl = fitted_si.importance_table()
        assert (tbl["se"] > 0).all()


# ===========================================================================
# SHAPInference: p=1 warning and fit
# ===========================================================================


class TestSHAPInferencePOne:
    """Tests for the p=1 smoothed estimator path."""

    def test_p1_emits_warning_on_fit(self):
        """p=1 should emit a UserWarning about smoothed estimator."""
        sv, y, names = _small_linear_data(n=200, d=2)
        si = SHAPInference(sv, y, names, p=1.0, n_folds=2, random_state=0)
        with pytest.warns(UserWarning, match="smoothed estimator"):
            si.fit()

    def test_p1_fit_produces_importance_table(self):
        """p=1 fit should still produce a valid importance_table."""
        sv, y, names = _small_linear_data(n=200, d=2)
        si = SHAPInference(sv, y, names, p=1.0, n_folds=2, random_state=0)
        with pytest.warns(UserWarning):
            si.fit()
        tbl = si.importance_table()
        assert len(tbl) == 2
        assert "theta_hat" in tbl.columns

    def test_p1_beta_n_auto_set(self):
        """When beta_n=None and p=1, fit should auto-compute beta_n."""
        sv, y, names = _small_linear_data(n=200, d=2)
        si = SHAPInference(sv, y, names, p=1.0, n_folds=2, random_state=0, beta_n=None)
        with pytest.warns(UserWarning):
            si.fit()
        # Should not raise; beta_n is computed internally
        assert si._is_fitted

    def test_p1_custom_beta_n(self):
        """Custom beta_n should suppress the auto-computed value."""
        sv, y, names = _small_linear_data(n=200, d=2)
        si = SHAPInference(sv, y, names, p=1.0, n_folds=2, random_state=0, beta_n=100.0)
        with pytest.warns(UserWarning):
            si.fit()
        assert si._is_fitted


# ===========================================================================
# SHAPInference: fit with custom estimator
# ===========================================================================


class TestSHAPInferenceFitCustomEstimator:
    """Tests for fit() with custom sklearn-compatible estimators."""

    def test_fit_with_custom_estimator(self):
        """A custom sklearn estimator should be accepted without error."""
        from sklearn.linear_model import Ridge
        sv, y, names = _small_linear_data(n=200, d=2)
        si = SHAPInference(
            sv, y, names, p=2.0, n_folds=2,
            nuisance_estimator=Ridge(alpha=1.0),
            alpha_estimator=Ridge(alpha=1.0),
            random_state=0,
        )
        si.fit()
        assert si._is_fitted

    def test_fit_n_folds_2(self):
        """n_folds=2 (minimum) should not raise."""
        sv, y, names = _small_linear_data(n=100, d=2)
        si = SHAPInference(sv, y, names, p=2.0, n_folds=2, random_state=0)
        si.fit()
        assert si._is_fitted

    def test_fit_single_feature(self):
        """Fitting with a single feature should produce a one-row table."""
        rng = np.random.default_rng(10)
        sv = rng.normal(0, 1, (200, 1))
        y = np.abs(sv[:, 0] + rng.normal(0, 0.2, 200))
        si = SHAPInference(sv, y, ["x"], p=2.0, n_folds=2, random_state=0)
        si.fit()
        tbl = si.importance_table()
        assert len(tbl) == 1
        assert tbl["feature"][0] == "x"


# ===========================================================================
# Bug fix regression test: importance_table rank.tolist() on list
# ===========================================================================


class TestImportanceTableRankBugFix:
    """
    Regression tests for the importance_table() bug where rank.tolist() was
    called on a list returned by _dense_rank_descending().

    _dense_rank_descending() returns list[int], not np.ndarray. Calling
    .tolist() on a list raises AttributeError. Fixed by removing the .tolist()
    calls for the rank columns.
    """

    def test_importance_table_does_not_raise_with_two_features(self):
        """importance_table() with 2 features must not raise AttributeError."""
        sv, y, names = _small_linear_data(n=200, d=2)
        si = SHAPInference(sv, y, names, p=2.0, n_folds=2, random_state=0)
        si.fit()
        # This would raise AttributeError: 'list' object has no attribute 'tolist'
        # before the bug fix.
        tbl = si.importance_table()
        assert len(tbl) == 2

    def test_importance_table_does_not_raise_with_one_feature(self):
        """importance_table() with 1 feature must not raise AttributeError."""
        rng = np.random.default_rng(42)
        sv = rng.normal(0, 1, (200, 1))
        y = np.abs(sv[:, 0] + rng.normal(0, 0.2, 200))
        si = SHAPInference(sv, y, ["x"], p=2.0, n_folds=2, random_state=0)
        si.fit()
        tbl = si.importance_table()
        assert len(tbl) == 1

    def test_rank_column_type_is_integer(self):
        """rank, rank_lower, rank_upper columns should have integer dtype."""
        sv, y, names = _small_linear_data(n=200, d=2)
        si = SHAPInference(sv, y, names, p=2.0, n_folds=2, random_state=0)
        si.fit()
        tbl = si.importance_table()
        import polars as pl
        assert tbl["rank"].dtype in (pl.Int32, pl.Int64)
        assert tbl["rank_lower"].dtype in (pl.Int32, pl.Int64)
        assert tbl["rank_upper"].dtype in (pl.Int32, pl.Int64)
