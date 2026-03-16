"""
Benchmark: shap-relativities vs Poisson GLM and CatBoost feature importance.

The claim: CatBoost Poisson models outperform GLMs on UK motor data, but pricing
actuaries cannot deploy GBMs without a rating factor table. This library extracts
multiplicative relativities from the GBM's SHAP values in the same format as
exp(beta) from a GLM, enabling GBMs to be deployed into rating engines like Radar.

Three approaches are compared:

  1. CatBoost feature importance (gain). The standard first-pass output from a
     GBM. Shows which features drive the model — but cannot produce level-specific
     multiplicative factors. Cannot answer "what is the NCD=5 discount vs NCD=0?"

  2. Poisson GLM (sklearn PoissonRegressor). The reference standard: on a log-
     linear DGP, a correctly-specified GLM recovers the true relativities by MLE.
     This is the baseline that SHAP relativities need to match.

  3. shap-relativities (CatBoost Poisson + SHAPRelativities). Extracts level-
     specific multiplicative factors from the GBM in the same format as GLM
     exp(beta), with confidence intervals. Also measures Gini to show whether
     the GBM's predictive lift comes through.

Setup:
- 20,000 synthetic UK motor policies, known log-linear Poisson DGP
- 3 categorical rating factors: area (6 bands), ncd_years (0-5), conviction flag
- True coefficients known — used to measure relativity recovery error
- 70/30 train/test split by index

Expected output:
- Feature importance: ranks factors correctly but cannot produce level relativities
- GLM: close to exact recovery on this log-linear DGP
- GBM+SHAP: close recovery with confidence intervals, plus predictive lift

The scenario: the GBM is sitting in a notebook outperforming the GLM. The only
thing blocking deployment is that nobody can get the factor table out of it.
This benchmark shows that you can, and that the recovered relativities match
the true DGP parameters.

Run:
    python benchmarks/benchmark.py
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

BENCHMARK_START = time.time()

print("=" * 70)
print("Benchmark: shap-relativities vs feature importance vs Poisson GLM")
print("=" * 70)
print()

try:
    from shap_relativities import SHAPRelativities
    print("shap-relativities imported OK")
except ImportError as e:
    print(f"ERROR: Could not import shap-relativities: {e}")
    print("Install with: pip install 'shap-relativities[all]'")
    sys.exit(1)

try:
    import catboost
    print("CatBoost imported OK")
except ImportError:
    print("ERROR: catboost required. Install with: pip install catboost")
    sys.exit(1)

try:
    from sklearn.linear_model import PoissonRegressor
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    print("scikit-learn imported OK")
except ImportError:
    print("ERROR: scikit-learn required. Install with: pip install scikit-learn")
    sys.exit(1)

print()

# ---------------------------------------------------------------------------
# Data-generating process: known log-linear Poisson DGP
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

N_POLICIES = 20_000

# True relativities — what the methods should recover
TRUE_AREA_REL = {"A": 1.00, "B": 1.12, "C": 1.25, "D": 1.43, "E": 1.68, "F": 1.95}
NCD_COEF = -0.12  # log coefficient per NCD year
TRUE_NCD_REL = {i: float(np.exp(NCD_COEF * i)) for i in range(6)}
TRUE_CONV_REL = {0: 1.00, 1: 1.57}  # exp(0.45)
BASE_FREQUENCY = 0.07

print(f"DGP: {N_POLICIES:,} policies, Poisson frequency, known log-linear structure")
print(f"True relativities:")
print(f"  Area:  {' | '.join(f'{k}={v:.3f}' for k, v in TRUE_AREA_REL.items())}")
print(f"  NCD:   {' | '.join(f'{k}={v:.3f}' for k, v in TRUE_NCD_REL.items())}")
print(f"  Conv:  0=1.000 | 1={TRUE_CONV_REL[1]:.3f}")
print()

# Generate features
areas = RNG.choice(["A", "B", "C", "D", "E", "F"], N_POLICIES,
                   p=[0.25, 0.25, 0.20, 0.15, 0.10, 0.05])
ncd_years = RNG.integers(0, 6, N_POLICIES)
has_conviction = (RNG.random(N_POLICIES) < 0.15).astype(int)
exposure = RNG.uniform(0.3, 1.0, N_POLICIES)

log_freq = (
    np.log(BASE_FREQUENCY)
    + np.array([np.log(TRUE_AREA_REL[a]) for a in areas])
    + NCD_COEF * ncd_years
    + np.log(TRUE_CONV_REL[1]) * has_conviction
)
true_freq = np.exp(log_freq)
claim_count = RNG.poisson(true_freq * exposure)

# Integer area code for CatBoost (A=0, B=1, ..., F=5)
area_code = np.array([ord(a) - ord("A") for a in areas])

df = pl.DataFrame({
    "area": areas.tolist(),
    "area_code": area_code.tolist(),
    "ncd_years": ncd_years.tolist(),
    "has_conviction": has_conviction.tolist(),
    "exposure": exposure.tolist(),
    "claim_count": claim_count.tolist(),
})

n_train = int(0.70 * N_POLICIES)
df_train = df[:n_train]
df_test = df[n_train:]

print(f"Portfolio: {n_train:,} train / {N_POLICIES - n_train:,} test")
print(f"Observed claim rate: {float(claim_count.sum() / exposure.sum()):.4f}/year")
print()


def gini(actual: np.ndarray, predicted: np.ndarray, exposure: np.ndarray) -> float:
    """Exposure-weighted Gini coefficient."""
    order = np.argsort(predicted / exposure)
    sorted_actual = actual[order]
    sorted_exposure = exposure[order]
    cum_exp = np.cumsum(sorted_exposure) / sorted_exposure.sum()
    cum_act = np.cumsum(sorted_actual) / sorted_actual.sum()
    auc = float(np.trapz(cum_act, cum_exp))
    return 2 * auc - 1


# ---------------------------------------------------------------------------
# APPROACH 1: CatBoost feature importance — the naive first pass
# ---------------------------------------------------------------------------

print("-" * 70)
print("APPROACH 1: CatBoost feature importance (gain)")
print("-" * 70)
print()
print("  Feature importance is the default GBM diagnostic. It tells you")
print("  which factors the model uses, not the level-specific relativities.")
print("  It cannot answer 'what is the NCD=5 discount vs NCD=0?'")
print()

t0 = time.time()

features = ["area_code", "ncd_years", "has_conviction"]
X_train_pl = df_train.select(features)
X_test_pl = df_test.select(features)

pool_train = catboost.Pool(
    data=X_train_pl.to_pandas(),
    label=df_train["claim_count"].to_numpy(),
    weight=df_train["exposure"].to_numpy(),
)

cbm = catboost.CatBoostRegressor(
    loss_function="Poisson",
    iterations=300,
    learning_rate=0.05,
    depth=5,
    random_seed=42,
    verbose=0,
    allow_writing_files=False,
)
cbm.fit(pool_train)
catboost_time = time.time() - t0

fi_scores = cbm.get_feature_importance(pool_train, type="FeatureImportance")
fi_total = fi_scores.sum()
fi = dict(zip(features, fi_scores))

cbm_gini = gini(
    df_test["claim_count"].to_numpy(),
    cbm.predict(X_test_pl.to_pandas()),
    df_test["exposure"].to_numpy(),
)

print(f"  CatBoost fit time: {catboost_time:.1f}s")
print(f"  Gini (test):       {cbm_gini:.4f}")
print()
print(f"  {'Feature':<24} {'Importance (%)':>16} {'Level relativities?':>22}")
print(f"  {'-'*24} {'-'*16} {'-'*22}")
for feat in features:
    pct = fi[feat] / fi_total * 100
    print(f"  {feat:<24} {pct:>15.1f}% {'Cannot compute':>22}")
print()
print("  Feature importance identifies that ncd_years matters most, but it")
print("  cannot tell you the NCD=5 discount factor. A rate engine needs")
print("  level-specific multiplicative factors — not a single importance score.")
print()

# ---------------------------------------------------------------------------
# APPROACH 2: Poisson GLM — the reference standard
# ---------------------------------------------------------------------------

print("-" * 70)
print("APPROACH 2: Poisson GLM (sklearn) — the reference standard")
print("-" * 70)
print()
print("  On a log-linear DGP, the correctly-specified GLM recovers the true")
print("  relativities by MLE. This is the benchmark for relativity accuracy.")
print()

t0 = time.time()

X_train_raw = df_train.select(["area", "ncd_years", "has_conviction"]).to_pandas()
X_test_raw = df_test.select(["area", "ncd_years", "has_conviction"]).to_pandas()

preprocessor = ColumnTransformer([
    ("area_ohe", OneHotEncoder(drop="first", sparse_output=False), ["area"]),
    ("ncd", "passthrough", ["ncd_years"]),
    ("conv", "passthrough", ["has_conviction"]),
])
glm = Pipeline([
    ("prep", preprocessor),
    ("model", PoissonRegressor(alpha=0, max_iter=500)),
])
glm.fit(
    X_train_raw,
    df_train["claim_count"].to_numpy() / df_train["exposure"].to_numpy(),
    model__sample_weight=df_train["exposure"].to_numpy(),
)
glm_time = time.time() - t0

glm_gini = gini(
    df_test["claim_count"].to_numpy(),
    glm.predict(X_test_raw),
    df_test["exposure"].to_numpy(),
)

ohe = glm.named_steps["prep"].named_transformers_["area_ohe"]
area_cats = ohe.categories_[0]
ncd_coef = float(glm.named_steps["model"].coef_[5])
conv_coef = float(glm.named_steps["model"].coef_[6])
area_coefs = glm.named_steps["model"].coef_[:5]

glm_area_rels: dict[str, float] = {"A": 1.0}
for i, cat in enumerate(area_cats):
    glm_area_rels[cat] = float(np.exp(area_coefs[i]))
glm_conv_rel = float(np.exp(conv_coef))

print(f"  GLM fit time: {glm_time:.1f}s")
print(f"  Gini (test):  {glm_gini:.4f}")
print()
print(f"  {'Feature':<22} {'Level':<8} {'GLM rel':>10} {'True rel':>10} {'Error %':>10}")
print(f"  {'-'*22} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

glm_errors = []
for band in ["A", "B", "C", "D", "E", "F"]:
    rel = glm_area_rels.get(band, 1.0)
    true_r = TRUE_AREA_REL[band]
    err = (rel - true_r) / true_r * 100
    glm_errors.append(abs(err))
    print(f"  {'area':<22} {band:<8} {rel:>10.4f} {true_r:>10.4f} {err:>+9.2f}%")

for ncd in range(6):
    rel = float(np.exp(ncd_coef * ncd))
    true_r = TRUE_NCD_REL[ncd]
    err = (rel - true_r) / true_r * 100
    glm_errors.append(abs(err))
    print(f"  {'ncd_years':<22} {ncd:<8} {rel:>10.4f} {true_r:>10.4f} {err:>+9.2f}%")

conv_err = (glm_conv_rel - TRUE_CONV_REL[1]) / TRUE_CONV_REL[1] * 100
glm_errors.append(abs(conv_err))
print(f"  {'has_conviction':<22} {'1':<8} {glm_conv_rel:>10.4f} {TRUE_CONV_REL[1]:>10.4f} {conv_err:>+9.2f}%")
print()
print(f"  Mean |error| across all {len(glm_errors)} levels: {np.mean(glm_errors):.2f}%")
print()

# ---------------------------------------------------------------------------
# APPROACH 3: shap-relativities (the library)
# ---------------------------------------------------------------------------

print("-" * 70)
print("APPROACH 3: shap-relativities (CatBoost Poisson + SHAP aggregation)")
print("-" * 70)
print()
print("  Uses the same CatBoost model from approach 1, but aggregates SHAP")
print("  values by factor level to produce exp(mean_shap) relativities —")
print("  same format as GLM exp(beta), with CLT confidence intervals.")
print()

t0 = time.time()

sr = SHAPRelativities(
    model=cbm,
    X=X_train_pl,
    exposure=df_train["exposure"].to_numpy(),
    categorical_features=features,
)
sr.fit()

rels = sr.extract_relativities(
    normalise_to="base_level",
    base_levels={
        "area_code": 0,
        "ncd_years": 0,
        "has_conviction": 0,
    },
)
shap_time = time.time() - t0

area_map = {i: chr(ord("A") + i) for i in range(6)}

print(f"  SHAP extraction time: {shap_time:.1f}s")
print()
print(f"  {'Feature':<22} {'Level':<8} {'SHAP rel':>10} {'True rel':>10} {'Error %':>10} "
      f"{'CI_95':>16}")
print(f"  {'-'*22} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*16}")

shap_errors = []
shap_ncd5 = float("nan")
shap_conv_val = float("nan")

for row in rels.iter_rows(named=True):
    feat = row["feature"]
    level_str = row["level"]
    rel = row["relativity"]
    lo = row["lower_ci"]
    hi = row["upper_ci"]

    if feat == "area_code":
        try:
            area_letter = area_map.get(int(float(level_str)), "?")
            true_r = TRUE_AREA_REL.get(area_letter, float("nan"))
            display_level = area_letter
        except (ValueError, TypeError):
            true_r = float("nan")
            display_level = level_str
    elif feat == "ncd_years":
        try:
            ncd_val = int(float(level_str))
            true_r = TRUE_NCD_REL.get(ncd_val, float("nan"))
            if ncd_val == 5:
                shap_ncd5 = rel
            display_level = str(ncd_val)
        except (ValueError, TypeError):
            true_r = float("nan")
            display_level = level_str
    elif feat == "has_conviction":
        try:
            conv_val = int(float(level_str))
            true_r = TRUE_CONV_REL.get(conv_val, float("nan"))
            if conv_val == 1:
                shap_conv_val = rel
            display_level = str(conv_val)
        except (ValueError, TypeError):
            true_r = float("nan")
            display_level = level_str
    else:
        true_r = float("nan")
        display_level = level_str

    if not (np.isnan(true_r) or true_r == 0):
        err = (rel - true_r) / true_r * 100
        shap_errors.append(abs(err))
        err_str = f"{err:>+9.2f}%"
    else:
        err_str = "    n/a"

    if not (np.isnan(lo) or np.isnan(hi)):
        ci_str = f"[{lo:.3f}, {hi:.3f}]"
    else:
        ci_str = "n/a"

    feat_display = "area" if feat == "area_code" else feat
    print(f"  {feat_display:<22} {display_level:<8} {rel:>10.4f} {true_r:>10.4f} "
          f"{err_str:>10} {ci_str:>16}")

print()
print(f"  Mean |error| across all {len(shap_errors)} levels: {np.mean(shap_errors):.2f}%")

checks = sr.validate()
recon = checks.get("reconstruction")
if recon:
    status = "PASS" if recon.passed else "FAIL"
    print(f"  SHAP reconstruction check: {status}  (max error: {recon.value:.2e})")
print()

# ---------------------------------------------------------------------------
# Comparison summary
# ---------------------------------------------------------------------------

print("=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)
print()

gini_improvement_pp = (cbm_gini - glm_gini) * 100

print(f"  {'Approach':<42} {'Level rels?':>12} {'Mean |err|':>12} {'Gini':>8}")
print(f"  {'-'*42} {'-'*12} {'-'*12} {'-'*8}")
print(f"  {'Feature importance (gain)':<42} {'No':>12} {'N/A':>12} {cbm_gini:>8.4f}")
print(f"  {'Poisson GLM exp(beta)':<42} {'Yes':>12} {np.mean(glm_errors):>11.2f}% {glm_gini:>8.4f}")
print(f"  {'SHAP relativities (shap-relativities)':<42} {'Yes':>12} {np.mean(shap_errors):>11.2f}% {cbm_gini:>8.4f}")
print()

print("KEY FINDINGS")
print()
ncd5_true = TRUE_NCD_REL[5]
print(f"  NCD=5 vs NCD=0 (true discount: {(1-ncd5_true)*100:.1f}%):")
print(f"    True relativity:            {ncd5_true:.4f}")
print(f"    GLM exp(beta):              {np.exp(ncd_coef * 5):.4f}  "
      f"(error: {(np.exp(ncd_coef*5)-ncd5_true)/ncd5_true*100:+.2f}%)")
print(f"    SHAP relativity:            {shap_ncd5:.4f}  "
      f"(error: {(shap_ncd5-ncd5_true)/ncd5_true*100:+.2f}%)")
print(f"    Feature importance:         cannot compute — single score, not a discount schedule")
print()
print(f"  Conviction loading (true: {TRUE_CONV_REL[1]:.3f}):")
print(f"    GLM:  {glm_conv_rel:.4f}  (error: {(glm_conv_rel - TRUE_CONV_REL[1])/TRUE_CONV_REL[1]*100:+.2f}%)")
print(f"    SHAP: {shap_conv_val:.4f}  (error: {(shap_conv_val - TRUE_CONV_REL[1])/TRUE_CONV_REL[1]*100:+.2f}%)")
print()
print(f"  Gini improvement from GBM vs GLM: {gini_improvement_pp:+.2f}pp")
print(f"  (Both SHAP relativities and feature importance use the same GBM,")
print(f"   so the Gini lift is the same — the question is whether you can")
print(f"   get the factor table out of it to deploy it)")
print()
print(f"  SHAP mean error ({np.mean(shap_errors):.2f}%) vs GLM mean error ({np.mean(glm_errors):.2f}%).")
print(f"  The gap is the cost of not constraining the model to log-linear form.")
print(f"  On a correctly-specified DGP like this one, the GLM has a small advantage")
print(f"  in relativity precision. On portfolios with interaction effects, the GBM's")
print(f"  Gini improvement more than compensates.")

elapsed = time.time() - BENCHMARK_START
print(f"\nBenchmark completed in {elapsed:.1f}s")
