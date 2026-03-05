"""
Tests for SHAP relativities module.

Five test groups:

1. Reconstruction — SHAP values must reconstruct model predictions exactly.
2. Relativity recovery — extracted relativities should approximate the true DGP
   parameters from the synthetic motor dataset.
3. Normalisation — base_level mode gives base=1.0; mean mode gives geometric
   exposure-weighted mean=1.0 (log-space mean=0).
4. Validation diagnostics — validate() returns sensible results.
5. Edge cases — single-level categorical, serialisation round-trip.

Test data: synthetic motor dataset at 10,000 policies. Features are chosen
to be integers 0-5 (area_code, ncd_years) or binary (has_convictions), so all
are treated as categorical for clean level-by-level relativity comparison.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

try:
    import lightgbm as lgb
    import shap  # noqa: F401
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False

from shap_relativities import SHAPRelativities, extract_relativities
from shap_relativities.datasets.motor import TRUE_FREQ_PARAMS, load_motor

pytestmark = pytest.mark.skipif(
    not _DEPS_AVAILABLE,
    reason="lightgbm and shap required for shap_relativities tests",
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def motor_data() -> pd.DataFrame:
    """10k policy synthetic dataset — fast to generate, stable enough for GLM tests."""
    return load_motor(n_policies=10_000, seed=42)


@pytest.fixture(scope="module")
def lgb_model_and_X(motor_data: pd.DataFrame):
    """
    Train a LightGBM Poisson model on a subset of motor features.

    Features: area_code, ncd_years, has_convictions — all treated as categorical
    (discrete integer levels) for clean relativity comparison.
    """
    df = motor_data.copy()
    df["has_convictions"] = (df["conviction_points"] > 0).astype(int)
    # area_code: 0=A (lowest risk) .. 5=F (highest risk)
    df["area_code"] = pd.Categorical(df["area"]).codes

    features = ["area_code", "ncd_years", "has_convictions"]
    X = df[features].copy()
    y = df["claim_count"].values
    w = df["exposure"].values

    dtrain = lgb.Dataset(X, label=y, weight=w)

    params = {
        "objective": "poisson",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "verbose": -1,
        "seed": 42,
    }
    model = lgb.train(
        params, dtrain, num_boost_round=200,
        valid_sets=[dtrain],
        callbacks=[lgb.log_evaluation(period=-1)],
    )

    return model, X, df


# ---------------------------------------------------------------------------
# 1. Reconstruction test
# ---------------------------------------------------------------------------


class TestReconstruction:
    def test_shap_reconstruction_matches_predictions(
        self, lgb_model_and_X
    ) -> None:
        """
        exp(shap_values.sum(axis=1) + expected_value) must match
        model.predict() to within 1e-4.

        This verifies that model_output="raw" was used correctly and that
        TreeExplainer's efficiency axiom holds for this model.
        """
        model, X, _ = lgb_model_and_X

        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()

        shap_vals = sr.shap_values()
        ev = sr._expected_value

        reconstructed = np.exp(shap_vals.sum(axis=1) + ev)
        predictions = model.predict(X)

        max_diff = np.abs(reconstructed - predictions).max()
        assert max_diff < 1e-4, (
            f"SHAP reconstruction error {max_diff:.2e} exceeds 1e-4. "
            "Check that model_output='raw' was used."
        )

    def test_validate_reconstruction_passes(self, lgb_model_and_X) -> None:
        """validate() reconstruction check should pass for a well-formed model."""
        model, X, _ = lgb_model_and_X

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
    def test_ncd_relativity_direction(self, lgb_model_and_X) -> None:
        """
        NCD=5 should have lower relativity than NCD=0. The true DGP has
        ncd_years coefficient = -0.12, so NCD=5 vs NCD=0 gives exp(-0.6) ≈ 0.549.
        """
        model, X, _ = lgb_model_and_X

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

        ncd_rels = rels[rels["feature"] == "ncd_years"].copy()
        ncd_rels["level_int"] = ncd_rels["level"].apply(lambda v: int(str(v)))
        ncd_rels = ncd_rels.sort_values("level_int")

        rel_ncd0 = ncd_rels.loc[ncd_rels["level_int"] == 0, "relativity"].values[0]
        rel_ncd5 = ncd_rels.loc[ncd_rels["level_int"] == 5, "relativity"].values[0]

        assert rel_ncd5 < rel_ncd0, (
            f"NCD=5 relativity {rel_ncd5:.3f} should be less than "
            f"NCD=0 relativity {rel_ncd0:.3f}"
        )

    def test_conviction_relativity_above_one(self, lgb_model_and_X) -> None:
        """
        has_convictions=1 should have a relativity > 1 relative to
        has_convictions=0. True coefficient is +0.45 → exp(0.45) ≈ 1.57.
        """
        model, X, _ = lgb_model_and_X

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

        conv_rels = rels[rels["feature"] == "has_convictions"].copy()
        conv_rels["level_int"] = conv_rels["level"].apply(lambda v: int(str(v)))

        rel_1 = conv_rels.loc[conv_rels["level_int"] == 1, "relativity"].values[0]
        assert rel_1 > 1.0, f"Conviction relativity {rel_1:.3f} should be > 1.0"

    def test_area_relativities_ordered(self, lgb_model_and_X) -> None:
        """
        Area bands A-F have increasing true frequency effects (codes 0-5).
        The extracted relativities should be generally increasing from 0 to 5.
        """
        model, X, _ = lgb_model_and_X

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

        area_rels = rels[rels["feature"] == "area_code"].copy()
        area_rels["level_int"] = area_rels["level"].apply(lambda v: int(str(v)))
        area_rels = area_rels.sort_values("level_int")

        rel_low = area_rels["relativity"].iloc[0]
        rel_high = area_rels["relativity"].iloc[-1]

        assert rel_high > rel_low, (
            f"Highest area code relativity ({rel_high:.3f}) should exceed "
            f"lowest ({rel_low:.3f})"
        )

    def test_ncd_relativity_magnitude(self, lgb_model_and_X) -> None:
        """
        The ratio of NCD=5 relativity to NCD=0 should be roughly exp(-0.12*5)=0.549.
        Allow a wide tolerance (0.25 to 0.90) since the GBM won't recover the
        exact GLM coefficient on only 10k policies.
        """
        model, X, _ = lgb_model_and_X

        true_ratio = math.exp(TRUE_FREQ_PARAMS["ncd_years"] * 5)  # exp(-0.6) ≈ 0.549

        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()

        rels = sr.extract_relativities(normalise_to="mean")

        ncd_rels = rels[rels["feature"] == "ncd_years"].copy()
        ncd_rels["level_int"] = ncd_rels["level"].apply(lambda v: int(str(v)))

        rel_ncd0 = ncd_rels.loc[ncd_rels["level_int"] == 0, "relativity"].values[0]
        rel_ncd5 = ncd_rels.loc[ncd_rels["level_int"] == 5, "relativity"].values[0]

        extracted_ratio = rel_ncd5 / rel_ncd0
        assert 0.25 < extracted_ratio < 0.90, (
            f"NCD ratio (5 vs 0) = {extracted_ratio:.3f}, true = {true_ratio:.3f}. "
            "Outside expected range 0.25-0.90."
        )


# ---------------------------------------------------------------------------
# 3. Normalisation test
# ---------------------------------------------------------------------------


class TestNormalisation:
    def test_base_level_gets_relativity_one(self, lgb_model_and_X) -> None:
        """In base_level mode, the chosen base level must have relativity = 1.0."""
        model, X, _ = lgb_model_and_X

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
            feat_rels = rels[rels["feature"] == feat]
            base_row = feat_rels[feat_rels["level"].astype(str) == str(base_val)]
            assert len(base_row) == 1, f"Base level {base_val} not found for {feat}"
            assert abs(base_row["relativity"].values[0] - 1.0) < 1e-10, (
                f"{feat} base level relativity = {base_row['relativity'].values[0]:.6f}"
            )

    def test_mean_mode_geometric_mean_is_one(self, lgb_model_and_X) -> None:
        """
        In mean mode, the exposure-weighted geometric mean relativity = 1.0.

        The log-space normalisation subtracts the exposure-weighted mean of
        mean_shap values. This means the geometric weighted mean of relativities
        equals 1.0 (i.e. the weighted mean of log(relativity) = 0).
        """
        model, X, df = lgb_model_and_X

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
            feat_rels = rels[rels["feature"] == feat]
            # Geometric weighted mean: exp(weighted_mean(log(relativities))) = 1
            # i.e. weighted_mean(log(relativities)) == 0
            wm_log = np.average(
                np.log(feat_rels["relativity"]),
                weights=feat_rels["exposure_weight"],
            )
            assert abs(wm_log) < 1e-6, (
                f"Feature '{feat}': weighted mean of log(relativity) = "
                f"{wm_log:.8f}, expected 0.0 (geometric mean = 1.0)"
            )

    def test_base_level_ci_includes_one(self, lgb_model_and_X) -> None:
        """Base level CI in base_level mode should straddle 1.0."""
        model, X, _ = lgb_model_and_X

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

        base_row = rels[
            (rels["feature"] == "area_code") & (rels["level"].astype(str) == "0")
        ].iloc[0]
        assert base_row["lower_ci"] <= 1.0 <= base_row["upper_ci"]


# ---------------------------------------------------------------------------
# 4. Validation diagnostics test
# ---------------------------------------------------------------------------


class TestValidation:
    def test_validate_returns_expected_keys(self, lgb_model_and_X) -> None:
        """validate() must return at least reconstruction and feature_coverage."""
        model, X, _ = lgb_model_and_X

        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()

        checks = sr.validate()
        assert "reconstruction" in checks
        assert "feature_coverage" in checks

    def test_feature_coverage_always_passes(self, lgb_model_and_X) -> None:
        """All X columns should appear in SHAP output — coverage always passes."""
        model, X, _ = lgb_model_and_X

        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()

        checks = sr.validate()
        assert checks["feature_coverage"].passed

    def test_reconstruction_check_value(self, lgb_model_and_X) -> None:
        """Reconstruction error value should be a non-negative float."""
        model, X, _ = lgb_model_and_X

        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()

        checks = sr.validate()
        assert checks["reconstruction"].value >= 0.0

    def test_sparse_levels_key_present(self, lgb_model_and_X) -> None:
        """sparse_levels check should be present when categorical features exist."""
        model, X, _ = lgb_model_and_X

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
        self, lgb_model_and_X
    ) -> None:
        """
        A categorical feature with only one unique value should produce
        relativity = 1.0 (it contributes nothing to discrimination).
        """
        model, X, _ = lgb_model_and_X

        # Create a version of X where area_code is always 0
        X_const = X.copy()
        X_const["area_code"] = 0

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

        area_rels = rels[rels["feature"] == "area_code"]
        assert len(area_rels) == 1
        assert abs(area_rels["relativity"].values[0] - 1.0) < 1e-10

    def test_to_dict_round_trip(self, lgb_model_and_X) -> None:
        """Serialise and deserialise via to_dict/from_dict."""
        model, X, _ = lgb_model_and_X

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

        pd.testing.assert_frame_equal(
            original_rels.reset_index(drop=True),
            recovered_rels.reset_index(drop=True),
            check_exact=False,
            rtol=1e-6,
        )

    def test_extract_relativities_convenience_function(
        self, lgb_model_and_X
    ) -> None:
        """extract_relativities() convenience wrapper returns a DataFrame."""
        model, X, _ = lgb_model_and_X

        result = extract_relativities(
            model, X,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
            base_levels={"area_code": 0, "ncd_years": 0, "has_convictions": 0},
        )

        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns
        assert "relativity" in result.columns
        assert len(result) > 0

    def test_output_columns_present(self, lgb_model_and_X) -> None:
        """extract_relativities() must return all expected columns."""
        model, X, _ = lgb_model_and_X

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

    def test_relativities_positive(self, lgb_model_and_X) -> None:
        """All relativities should be strictly positive."""
        model, X, _ = lgb_model_and_X

        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()
        rels = sr.extract_relativities(normalise_to="mean")

        assert (rels["relativity"] > 0).all(), "Negative relativities found"

    def test_baseline_positive(self, lgb_model_and_X) -> None:
        """baseline() must return a positive float."""
        model, X, df = lgb_model_and_X

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

    def test_shap_values_shape(self, lgb_model_and_X) -> None:
        """shap_values() must return array of shape (n_obs, n_features)."""
        model, X, _ = lgb_model_and_X

        sr = SHAPRelativities(
            model=model,
            X=X,
            categorical_features=["area_code", "ncd_years", "has_convictions"],
        )
        sr.fit()

        sv = sr.shap_values()
        assert sv.shape == (len(X), X.shape[1])

    def test_unfitted_raises(self, lgb_model_and_X) -> None:
        """Calling extract_relativities before fit() should raise RuntimeError."""
        model, X, _ = lgb_model_and_X

        sr = SHAPRelativities(model=model, X=X)
        with pytest.raises(RuntimeError, match="fit()"):
            sr.extract_relativities()
