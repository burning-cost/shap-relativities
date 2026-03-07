"""
Tests for SHAP relativities module.

Six test groups:

1. Reconstruction - SHAP values must reconstruct model predictions exactly.
2. Relativity recovery - extracted relativities should approximate the true DGP
   parameters from the synthetic motor dataset.
3. Normalisation - base_level mode gives base=1.0; mean mode gives geometric
   exposure-weighted mean=1.0 (log-space mean=0).
4. Validation diagnostics - validate() returns sensible results.
5. Edge cases - single-level categorical, serialisation round-trip.

Test data: synthetic motor dataset at 10,000 policies. Features are chosen
to be integers 0-5 (area_code, ncd_years) or binary (has_convictions), so all
are treated as categorical for clean level-by-level relativity comparison.

Model: CatBoost with a Poisson objective. CatBoost is the default GBM in
this library - it handles categoricals natively without label encoding, and
its SHAP implementation is compatible with shap's TreeExplainer.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

try:
    import catboost
    import shap  # noqa: F401
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False

from shap_relativities import SHAPRelativities, extract_relativities
from shap_relativities.datasets.motor import TRUE_FREQ_PARAMS, load_motor

pytestmark = pytest.mark.skipif(
    not _DEPS_AVAILABLE,
    reason="catboost and shap required for shap_relativities tests",
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def motor_data() -> pl.DataFrame:
    """10k policy synthetic dataset - fast to generate, stable enough for GLM tests."""
    return load_motor(n_policies=10_000, seed=42)


@pytest.fixture(scope="module")
def catboost_model_and_X(motor_data: pl.DataFrame):
    """
    Train a CatBoost Poisson model on a subset of motor features.

    Features: area_code, ncd_years, has_convictions - all treated as categorical
    (discrete integer levels) for clean relativity comparison.

    CatBoost is trained via its Pool API. The feature matrix is converted to
    pandas for the Pool constructor (CatBoost's requirement), but SHAPRelativities
    receives the Polars DataFrame directly.
    """
    df = motor_data.with_columns([
        ((pl.col("conviction_points") > 0).cast(pl.Int32)).alias("has_convictions"),
        # area_code: 0=A (lowest risk) .. 5=F (highest risk)
        pl.col("area").replace(
            {"A": "0", "B": "1", "C": "2", "D": "3", "E": "4", "F": "5"}
        ).cast(pl.Int32).alias("area_code"),
    ])

    features = ["area_code", "ncd_years", "has_convictions"]
    X = df.select(features)
    y = df["claim_count"].to_numpy()
    w = df["exposure"].to_numpy()

    # CatBoost requires pandas for Pool construction
    X_pd = X.to_pandas()

    pool = catboost.Pool(
        data=X_pd,
        label=y,
        weight=w,
    )

    model = catboost.CatBoostRegressor(
        loss_function="Poisson",
        iterations=200,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0,
    )
    model.fit(pool)

    return model, X, df


# ---------------------------------------------------------------------------
# 1. Reconstruction test
# ---------------------------------------------------------------------------


class TestReconstruction:
    def test_shap_reconstruction_matches_predictions(
        self, catboost_model_and_X
    ) -> None:
        """
        exp(shap_values.sum(axis=1) + expected_value) must match
        model.predict() to within 1e-4.

        This verifies that model_output="raw" was used correctly and that
        TreeExplainer's efficiency axiom holds for this model.
        """
        model, X, _ = catboost_model_and_X

        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()

        shap_vals = sr.shap_values()
        ev = sr._expected_value

        reconstructed = np.exp(shap_vals.sum(axis=1) + ev)
        predictions = model.predict(X.to_pandas())

        max_diff = np.abs(reconstructed - predictions).max()
        assert max_diff < 1e-4, (
            f"SHAP reconstruction error {max_diff:.2e} exceeds 1e-4. "
            "Check that model_output='raw' was used."
        )

    def test_validate_reconstruction_passes(self, catboost_model_and_X) -> None:
        """validate() reconstruction check should pass for a well-formed model."""
        model, X, _ = catboost_model_and_X

        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()

        checks = sr.validate()
        assert "reconstruction" in checks
        assert checks["reconstruction"].passed, checks["reconstruction"].message


# ---------------------------------------------------------------------------
# 2. Relativity recovery test
# ---------------------------------------------------------------------------


class TestRelativityRecovery:
    def test_ncd_relativity_direction(self, catboost_model_and_X) -> None:
        """
        NCD=5 should have lower relativity than NCD=0. The true DGP has
        ncd_years coefficient = -0.12, so NCD=5 vs NCD=0 gives exp(-0.6) ≈ 0.549.
        """
        model, X, _ = catboost_model_and_X

        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()

        rels = sr.extract_relativities(
            normalise_to="base_level",
            base_levels={"area_code": 0, "ncd_years": 0, "has_convictions": 0},
        )

        ncd_rels = (
            rels.filter(pl.col("feature") == "ncd_years")
            .with_columns(pl.col("level").cast(pl.Int32).alias("level_int"))
            .sort("level_int")
        )

        rel_ncd0 = ncd_rels.filter(pl.col("level_int") == 0)["relativity"][0]
        rel_ncd5 = ncd_rels.filter(pl.col("level_int") == 5)["relativity"][0]

        assert rel_ncd5 < rel_ncd0, (
            f"NCD=5 relativity {rel_ncd5:.3f} should be less than "
            f"NCD=0 relativity {rel_ncd0:.3f}"
        )

    def test_conviction_relativity_above_one(self, catboost_model_and_X) -> None:
        """
        has_convictions=1 should have a relativity > 1 relative to
        has_convictions=0. True coefficient is +0.45 → exp(0.45) ≈ 1.57.
        """
        model, X, _ = catboost_model_and_X

        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()

        rels = sr.extract_relativities(
            normalise_to="base_level",
            base_levels={"area_code": 0, "ncd_years": 0, "has_convictions": 0},
        )

        conv_rels = rels.filter(pl.col("feature") == "has_convictions").with_columns(
            pl.col("level").cast(pl.Int32).alias("level_int")
        )
        rel_1 = conv_rels.filter(pl.col("level_int") == 1)["relativity"][0]
        assert rel_1 > 1.0, f"Conviction relativity {rel_1:.3f} should be > 1.0"

    def test_area_relativities_ordered(self, catboost_model_and_X) -> None:
        """
        Area bands A-F have increasing true frequency effects (codes 0-5).
        The extracted relativities should be generally increasing from 0 to 5.
        """
        model, X, _ = catboost_model_and_X

        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()

        rels = sr.extract_relativities(
            normalise_to="base_level",
            base_levels={"area_code": 0, "ncd_years": 0, "has_convictions": 0},
        )

        area_rels = (
            rels.filter(pl.col("feature") == "area_code")
            .with_columns(pl.col("level").cast(pl.Int32).alias("level_int"))
            .sort("level_int")
        )

        rel_low = area_rels["relativity"][0]
        rel_high = area_rels["relativity"][-1]

        assert rel_high > rel_low, (
            f"Highest area code relativity ({rel_high:.3f}) should exceed "
            f"lowest ({rel_low:.3f})"
        )

    def test_ncd_relativity_magnitude(self, catboost_model_and_X) -> None:
        """
        The ratio of NCD=5 relativity to NCD=0 should be roughly exp(-0.12*5)=0.549.
        Allow a wide tolerance (0.25 to 0.90) since the GBM won't recover the
        exact GLM coefficient on only 10k policies.
        """
        model, X, _ = catboost_model_and_X

        true_ratio = math.exp(TRUE_FREQ_PARAMS["ncd_years"] * 5)  # exp(-0.6) ≈ 0.549

        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()

        rels = sr.extract_relativities(normalise_to="mean")

        ncd_rels = rels.filter(pl.col("feature") == "ncd_years").with_columns(
            pl.col("level").cast(pl.Int32).alias("level_int")
        )

        rel_ncd0 = ncd_rels.filter(pl.col("level_int") == 0)["relativity"][0]
        rel_ncd5 = ncd_rels.filter(pl.col("level_int") == 5)["relativity"][0]

        extracted_ratio = rel_ncd5 / rel_ncd0
        assert 0.25 < extracted_ratio < 0.90, (
            f"NCD ratio (5 vs 0) = {extracted_ratio:.3f}, true = {true_ratio:.3f}. "
            "Outside expected range 0.25-0.90."
        )


# ---------------------------------------------------------------------------
# 3. Normalisation test
# ---------------------------------------------------------------------------


class TestNormalisation:
    def test_base_level_gets_relativity_one(self, catboost_model_and_X) -> None:
        """In base_level mode, the chosen base level must have relativity = 1.0."""
        model, X, _ = catboost_model_and_X

        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()

        rels = sr.extract_relativities(
            normalise_to="base_level",
            base_levels={"area_code": 0, "ncd_years": 0, "has_convictions": 0},
        )

        for feat, base_val in [("area_code", 0), ("ncd_years", 0), ("has_convictions", 0)]:
            feat_rels = rels.filter(pl.col("feature") == feat)
            base_row = feat_rels.filter(pl.col("level") == str(base_val))
            assert len(base_row) == 1, f"Base level {base_val} not found for {feat}"
            assert abs(base_row["relativity"][0] - 1.0) < 1e-10, (
                f"{feat} base level relativity = {base_row['relativity'][0]:.6f}"
            )

    def test_mean_mode_geometric_mean_is_one(self, catboost_model_and_X) -> None:
        """
        In mean mode, the exposure-weighted geometric mean relativity = 1.0.

        The log-space normalisation subtracts the exposure-weighted mean of
        mean_shap values. This means the geometric weighted mean of relativities
        equals 1.0 (i.e. the weighted mean of log(relativity) = 0).
        """
        model, X, df = catboost_model_and_X

        exposure = df["exposure"]
        sr = SHAPRelativities(
            model=model,
            X=X,
            exposure=exposure,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()

        rels = sr.extract_relativities(normalise_to="mean")

        for feat in ["area_code", "ncd_years", "has_convictions"]:
            feat_rels = rels.filter(pl.col("feature") == feat)
            log_rels = np.log(feat_rels["relativity"].to_numpy())
            exp_weights = feat_rels["exposure_weight"].to_numpy()
            wm_log = np.average(log_rels, weights=exp_weights)
            assert abs(wm_log) < 1e-6, (
                f"Feature '{feat}': weighted mean of log(relativity) = "
                f"{wm_log:.8f}, expected 0.0 (geometric mean = 1.0)"
            )

    def test_base_level_ci_includes_one(self, catboost_model_and_X) -> None:
        """Base level CI in base_level mode should straddle 1.0."""
        model, X, _ = catboost_model_and_X

        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()

        rels = sr.extract_relativities(
            normalise_to="base_level",
            base_levels={"area_code": 0, "ncd_years": 0, "has_convictions": 0},
        )

        base_row = rels.filter(
            (pl.col("feature") == "area_code") & (pl.col("level") == "0")
        ).row(0, named=True)
        assert base_row["lower_ci"] <= 1.0 <= base_row["upper_ci"]


# ---------------------------------------------------------------------------
# 4. Validation diagnostics test
# ---------------------------------------------------------------------------


class TestValidation:
    def test_validate_returns_expected_keys(self, catboost_model_and_X) -> None:
        """validate() must return at least reconstruction and feature_coverage."""
        model, X, _ = catboost_model_and_X

        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()

        checks = sr.validate()
        assert "reconstruction" in checks
        assert "feature_coverage" in checks

    def test_feature_coverage_always_passes(self, catboost_model_and_X) -> None:
        """All X columns should appear in SHAP output - coverage always passes."""
        model, X, _ = catboost_model_and_X

        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()

        checks = sr.validate()
        assert checks["feature_coverage"].passed

    def test_reconstruction_check_value(self, catboost_model_and_X) -> None:
        """Reconstruction error value should be a non-negative float."""
        model, X, _ = catboost_model_and_X

        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()

        checks = sr.validate()
        assert checks["reconstruction"].value >= 0.0

    def test_sparse_levels_key_present(self, catboost_model_and_X) -> None:
        """sparse_levels check should be present when categorical features exist."""
        model, X, _ = catboost_model_and_X

        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()

        checks = sr.validate()
        assert "sparse_levels" in checks


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_level_categorical_relativity_is_one(
        self, catboost_model_and_X
    ) -> None:
        """
        A categorical feature with only one unique value should produce
        relativity = 1.0 (it contributes nothing to discrimination).
        """
        model, X, _ = catboost_model_and_X

        # Create a version of X where area_code is always 0
        X_const = X.with_columns(pl.lit(0).alias("area_code"))

        sr = SHAPRelativities(
            model=model,
            X=X_const,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()

        rels = sr.extract_relativities(
            normalise_to="base_level",
            base_levels={"area_code": 0, "ncd_years": 0, "has_convictions": 0},
        )

        area_rels = rels.filter(pl.col("feature") == "area_code")
        assert len(area_rels) == 1
        assert abs(area_rels["relativity"][0] - 1.0) < 1e-10

    def test_to_dict_round_trip(self, catboost_model_and_X) -> None:
        """Serialise and deserialise via to_dict/from_dict."""
        model, X, _ = catboost_model_and_X

        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()

        original_rels = sr.extract_relativities(normalise_to="mean")

        d = sr.to_dict()
        sr2 = SHAPRelativities.from_dict(d)

        recovered_rels = sr2.extract_relativities(normalise_to="mean")

        # Compare as numpy arrays for tolerance check
        for col in ["relativity", "lower_ci", "upper_ci", "mean_shap"]:
            orig = original_rels[col].to_numpy()
            recov = recovered_rels[col].to_numpy()
            assert np.allclose(orig, recov, rtol=1e-6), f"Column {col} differs after round-trip"

    def test_extract_relativities_convenience_function(
        self, catboost_model_and_X
    ) -> None:
        """extract_relativities() convenience wrapper returns a Polars DataFrame."""
        model, X, _ = catboost_model_and_X

        result = extract_relativities(
            model, X,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
            base_levels={"area_code": 0, "ncd_years": 0, "has_convictions": 0},
        )

        assert isinstance(result, pl.DataFrame)
        assert "feature" in result.columns
        assert "relativity" in result.columns
        assert len(result) > 0

    def test_output_columns_present(self, catboost_model_and_X) -> None:
        """extract_relativities() must return all expected columns."""
        model, X, _ = catboost_model_and_X

        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()
        rels = sr.extract_relativities(
            base_levels={"area_code": 0, "ncd_years": 0, "has_convictions": 0},
        )

        expected_cols = {
            "feature", "level", "relativity", "lower_ci", "upper_ci",
            "mean_shap", "shap_std", "n_obs", "exposure_weight",
        }
        assert expected_cols.issubset(set(rels.columns)), (
            f"Missing columns: {expected_cols - set(rels.columns)}"
        )

    def test_relativities_positive(self, catboost_model_and_X) -> None:
        """All relativities should be strictly positive."""
        model, X, _ = catboost_model_and_X

        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()
        rels = sr.extract_relativities(normalise_to="mean")

        assert (rels["relativity"] > 0).all(), "Negative relativities found"

    def test_baseline_positive(self, catboost_model_and_X) -> None:
        """baseline() must return a positive float."""
        model, X, df = catboost_model_and_X

        sr = SHAPRelativities(
            model=model,
            X=X,
            exposure=df["exposure"],
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()

        b = sr.baseline()
        assert b > 0
        assert isinstance(b, float)

    def test_shap_values_shape(self, catboost_model_and_X) -> None:
        """shap_values() must return array of shape (n_obs, n_features)."""
        model, X, _ = catboost_model_and_X

        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()

        sv = sr.shap_values()
        assert sv.shape == (len(X), X.width)

    def test_unfitted_raises(self, catboost_model_and_X) -> None:
        """Calling extract_relativities before fit() should raise RuntimeError."""
        model, X, _ = catboost_model_and_X

        sr = SHAPRelativities(model=model, X=X)
        with pytest.raises(RuntimeError, match="fit()"):
            sr.extract_relativities()

    def test_accepts_pandas_input(self, catboost_model_and_X) -> None:
        """SHAPRelativities should accept a pandas DataFrame as X."""
        model, X, _ = catboost_model_and_X

        X_pd = X.to_pandas()
        sr = SHAPRelativities(
            model=model,
            X=X_pd,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()
        rels = sr.extract_relativities(normalise_to="mean")
        assert isinstance(rels, pl.DataFrame)
        assert len(rels) > 0


# ---------------------------------------------------------------------------
# 6. ci_method, continuous curves, and error handling
# ---------------------------------------------------------------------------


class TestCiMethodAndErrors:
    def test_ci_method_none_returns_nan_ci(self, catboost_model_and_X) -> None:
        """ci_method='none' should produce NaN lower_ci and upper_ci."""
        model, X, _ = catboost_model_and_X

        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()

        rels = sr.extract_relativities(
            normalise_to="base_level",
            base_levels={"area_code": 0, "ncd_years": 0, "has_convictions": 0},
            ci_method="none",
        )

        import math
        for row in rels.iter_rows(named=True):
            assert math.isnan(row["lower_ci"]), f"lower_ci not NaN for {row}"
            assert math.isnan(row["upper_ci"]), f"upper_ci not NaN for {row}"

    def test_ci_method_none_mean_normalisation(self, catboost_model_and_X) -> None:
        """ci_method='none' with normalise_to='mean' should return valid relativities."""
        model, X, _ = catboost_model_and_X

        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()

        rels = sr.extract_relativities(normalise_to="mean", ci_method="none")

        assert isinstance(rels, pl.DataFrame)
        assert (rels["relativity"] > 0).all()

    def test_bootstrap_not_implemented(self, catboost_model_and_X) -> None:
        """ci_method='bootstrap' should raise NotImplementedError."""
        model, X, _ = catboost_model_and_X

        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()

        with pytest.raises(NotImplementedError):
            sr.extract_relativities(ci_method="bootstrap")

    def test_missing_base_level_raises_or_warns(self, catboost_model_and_X) -> None:
        """
        If base_level is omitted for a feature, a warning is issued and
        the lowest mean SHAP level is used as fallback.
        """
        model, X, _ = catboost_model_and_X

        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()

        with pytest.warns(UserWarning, match="No base level specified"):
            rels = sr.extract_relativities(
                normalise_to="base_level",
                base_levels={},  # no base levels specified
            )

        # Should still return a valid DataFrame
        assert isinstance(rels, pl.DataFrame)
        assert len(rels) > 0

    def test_invalid_base_level_raises(self, catboost_model_and_X) -> None:
        """Passing a base level that doesn't exist in the data should raise ValueError."""
        model, X, _ = catboost_model_and_X

        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()

        with pytest.raises(ValueError, match="not found in levels"):
            sr.extract_relativities(
                normalise_to="base_level",
                base_levels={"area_code": 99, "ncd_years": 0, "has_convictions": 0},
            )

    def test_extract_continuous_curve_isotonic(self, catboost_model_and_X) -> None:
        """extract_continuous_curve with smooth_method='isotonic' returns a valid DataFrame."""
        model, X, _ = catboost_model_and_X

        # Add a continuous feature by including ncd_years but as continuous
        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "has_convictions"],
            continuous_features=["ncd_years"],
        )
        sr.fit()

        curve = sr.extract_continuous_curve("ncd_years", n_points=50, smooth_method="isotonic")

        assert isinstance(curve, pl.DataFrame)
        assert "feature_value" in curve.columns
        assert "relativity" in curve.columns
        assert len(curve) == 50
        assert (curve["relativity"] > 0).all()

    def test_extract_continuous_curve_none(self, catboost_model_and_X) -> None:
        """extract_continuous_curve with smooth_method='none' returns per-observation rows."""
        model, X, _ = catboost_model_and_X

        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "has_convictions"],
            continuous_features=["ncd_years"],
        )
        sr.fit()

        curve = sr.extract_continuous_curve("ncd_years", smooth_method="none")

        assert isinstance(curve, pl.DataFrame)
        assert len(curve) == len(X)

    def test_extract_continuous_curve_invalid_feature(self, catboost_model_and_X) -> None:
        """extract_continuous_curve for unknown feature should raise ValueError."""
        model, X, _ = catboost_model_and_X

        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()

        with pytest.raises(ValueError, match="not in X"):
            sr.extract_continuous_curve("nonexistent_feature")

    def test_extract_continuous_curve_invalid_smooth_method(
        self, catboost_model_and_X
    ) -> None:
        """Unknown smooth_method should raise ValueError."""
        model, X, _ = catboost_model_and_X

        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "has_convictions"],
            continuous_features=["ncd_years"],
        )
        sr.fit()

        with pytest.raises(ValueError, match="Unknown smooth_method"):
            sr.extract_continuous_curve("ncd_years", smooth_method="spline")
