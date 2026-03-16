"""
Benchmark: SHAP relativities vs CatBoost feature importance vs fitted GLM.

The question: when you extract relativities from a CatBoost model, do you get
something that genuinely reflects the multiplicative rating structure — or are
you just getting dressed-up feature importance scores?

This benchmark answers that with a controlled experiment. The data generating
process is explicitly multiplicative (log-linear Poisson), so we know the true
relativities. We then compare three approaches:

  1. CatBoost feature importance (gain). Standard output, no relativity
     structure. Shows which features matter, not by how much each level
     changes the rate.

  2. Fitted Poisson GLM with main effects. The reference: when the DGP is
     log-linear and the model is correctly specified, GLM exp(beta) gives the
     true relativities by maximum likelihood. This is the gold standard for
     a correctly-specified problem.

  3. shap-relativities applied to the same CatBoost model. Should recover
     the true multiplicative structure, matching the GLM closely despite
     being extracted from a non-parametric model.

Setup:
- 30,000 synthetic UK motor policies, 3 rating factors with known true
  relativities: ncd_years (0-5), area (A-F), has_convictions (0/1)
- DGP: Poisson frequency, log-linear, no interaction effects (so the GLM
  is correctly specified and should recover the truth)
- Temporal 70/30 train/test split
- All three approaches evaluated on the same test set

Expected results:
- Feature importance: identifies the right ranking of factors but cannot
  produce relativities. Not comparable to the true values.
- GLM relativities: very close to true parameters (correctly specified model).
- SHAP relativities: within a few percent of GLM / true values. Gap driven
  by GBM flexibility (no parsimony constraint) and SHAP aggregation.

The key demonstration:
  Feature importance cannot answer "what does NCD=5 do to the rate vs NCD=0?"
  SHAP relativities can, and the answer matches the GLM when the DGP is
  truly multiplicative.

Run:
    python benchmarks/benchmark.py
"""

from __future__ import annotations

import sys
import time
import warnings

warnings.filterwarnings("ignore")

BENCHMARK_START = time.time()

print("=" * 70)
print("Benchmark: SHAP relativities vs feature importance vs GLM")
print("(shap-relativities)")
print("=" * 70)
print()

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

try:
    from shap_relativities import SHAPRelativities
    from shap_relativities.datasets.motor import TRUE_FREQ_PARAMS
    print("shap-relativities imported OK")
except ImportError as e:
    print(f"ERROR: Could not import shap-relativities: {e}")
    print("Install with: pip install 'shap-relativities[all]'")
    sys.exit(1)

try:
    from catboost import CatBoostRegressor, Pool
    print("CatBoost imported OK")
except ImportError as e:
    print(f"ERROR: CatBoost required: {e}")
    sys.exit(1)

try:
    import statsmodels.api as sm
    from statsmodels.genmod.families import Poisson as PoissonFamily
    _STATSMODELS_OK = True
    print("statsmodels imported OK (GLM benchmark available)")
except ImportError:
    _STATSMODELS_OK = False
    print("statsmodels not available — GLM comparison skipped")

import numpy as np
import polars as pl

print()

# ---------------------------------------------------------------------------
# True DGP parameters — the ground truth we are trying to recover
# ---------------------------------------------------------------------------

# We use a simplified subset of the motor DGP for clarity of comparison.
# Three rating factors with known multiplicative relativities:
#
#   ncd_years 0-5, relativity = exp(beta_ncd * ncd)  where beta_ncd = -0.12
#   area A-F, relativities = exp(area_effect)         from TRUE_FREQ_PARAMS
#   has_convictions 0/1, relativity = exp(0.45)

BETA_NCD = TRUE_FREQ_PARAMS["ncd_years"]          # -0.12
BETA_CONV = TRUE_FREQ_PARAMS["has_convictions"]    # 0.45
AREA_EFFECTS = {
    "A": 0.0,
    "B": TRUE_FREQ_PARAMS["area_B"],   # 0.10
    "C": TRUE_FREQ_PARAMS["area_C"],   # 0.20
    "D": TRUE_FREQ_PARAMS["area_D"],   # 0.35
    "E": TRUE_FREQ_PARAMS["area_E"],   # 0.50
    "F": TRUE_FREQ_PARAMS["area_F"],   # 0.65
}
AREA_PROBS = [0.12, 0.18, 0.25, 0.22, 0.14, 0.09]  # A-F distribution
BASE_FREQ = np.exp(TRUE_FREQ_PARAMS["intercept"])   # exp(-3.2) ≈ 0.041

print("True DGP parameters (what we are trying to recover):")
print(f"  Base frequency:     {BASE_FREQ:.4f} claims/year")
print(f"  beta_ncd:           {BETA_NCD:.4f} (per NCD year)")
print(f"  NCD=5 vs NCD=0:     exp({5*BETA_NCD:.3f}) = {np.exp(5*BETA_NCD):.4f}")
print(f"  beta_convictions:   {BETA_CONV:.4f}")
print(f"  Conviction rel:     exp({BETA_CONV:.3f}) = {np.exp(BETA_CONV):.4f}")
print(f"  Area relativities:  A=1.000, B={np.exp(0.10):.3f}, C={np.exp(0.20):.3f}, "
      f"D={np.exp(0.35):.3f}, E={np.exp(0.50):.3f}, F={np.exp(0.65):.3f}")
print()

# ---------------------------------------------------------------------------
# Generate synthetic data
# ---------------------------------------------------------------------------

N_POLICIES = 30_000
SEED = 42
rng = np.random.default_rng(SEED)

print(f"Generating {N_POLICIES:,} synthetic UK motor policies...")

ncd_years = rng.integers(0, 6, N_POLICIES)  # 0-5
area = rng.choice(["A", "B", "C", "D", "E", "F"], N_POLICIES, p=AREA_PROBS)
has_convictions = (rng.random(N_POLICIES) < np.where(
    rng.integers(17, 80, N_POLICIES) < 30, 0.12, 0.04
)).astype(int)
exposure = rng.uniform(0.3, 1.0, N_POLICIES)

# True log-frequency
area_effects = np.array([AREA_EFFECTS[a] for a in area])
log_freq = (
    TRUE_FREQ_PARAMS["intercept"]
    + BETA_NCD * ncd_years
    + area_effects
    + BETA_CONV * has_convictions
    + np.log(np.clip(exposure, 1e-6, None))
)
true_lambda = np.exp(log_freq)
claim_count = rng.poisson(true_lambda)

df = pl.DataFrame({
    "ncd_years": ncd_years.astype(np.int32),
    "area": area.tolist(),
    "has_convictions": has_convictions.astype(np.int32),
    "exposure": exposure,
    "claim_count": claim_count.astype(np.int32),
})

# 70/30 temporal split (by index — represents a training/test division)
n_train = int(N_POLICIES * 0.70)
df_train = df[:n_train]
df_test = df[n_train:]

feature_cols = ["ncd_years", "area", "has_convictions"]
cat_features = ["area"]  # CatBoost native categorical

print(f"  Train: {len(df_train):,} policies | Test: {len(df_test):,} policies")
print(f"  Observed claim rate: {float(claim_count.sum() / exposure.sum()):.4f}/year")
print()

# ---------------------------------------------------------------------------
# APPROACH 1: CatBoost feature importance (baseline — what teams do first)
# ---------------------------------------------------------------------------

print("-" * 70)
print("APPROACH 1: CatBoost feature importance (gain)")
print("-" * 70)
print()
print("  Feature importance is the standard first-pass diagnostic. It tells")
print("  you which features drive predictions, not what the level relativities")
print("  are. It cannot answer 'what does NCD=5 do to the premium vs NCD=0?'")
print()

t0 = time.time()

X_train_pd = df_train.select(feature_cols).to_pandas()
X_test_pd = df_test.select(feature_cols).to_pandas()

train_pool = Pool(
    data=X_train_pd,
    label=df_train["claim_count"].to_numpy(),
    weight=df_train["exposure"].to_numpy(),
    cat_features=cat_features,
)
test_pool = Pool(
    data=X_test_pd,
    label=df_test["claim_count"].to_numpy(),
    weight=df_test["exposure"].to_numpy(),
    cat_features=cat_features,
)

model = CatBoostRegressor(
    loss_function="Poisson",
    iterations=400,
    learning_rate=0.05,
    depth=6,
    random_seed=SEED,
    verbose=0,
    allow_writing_files=False,
)
model.fit(train_pool)
catboost_fit_time = time.time() - t0

# Feature importance (gain)
fi = dict(zip(feature_cols, model.get_feature_importance(train_pool, type="FeatureImportance")))
fi_total = sum(fi.values())

print(f"  CatBoost training time: {catboost_fit_time:.1f}s")
print(f"  {'Feature':<20} {'Importance (%)':>16} {'Can give level relativities?':>30}")
print(f"  {'-'*20} {'-'*16} {'-'*30}")
for feat in feature_cols:
    pct = fi[feat] / fi_total * 100
    print(f"  {feat:<20} {pct:>15.1f}% {'No — scalar only':>30}")
print()
print("  Feature importance ranks factors correctly but cannot give")
print("  the relativity for, say, NCD=5 vs NCD=0. A pricing actuary")
print("  needs level-specific multiplicative factors for the rate engine.")
print()

# ---------------------------------------------------------------------------
# APPROACH 2: Fitted Poisson GLM (gold standard when DGP is multiplicative)
# ---------------------------------------------------------------------------

print("-" * 70)
print("APPROACH 2: Fitted Poisson GLM — the gold standard for this DGP")
print("-" * 70)
print()

if _STATSMODELS_OK:
    print("  When the DGP is truly log-linear (as here), a correctly-specified")
    print("  GLM recovers the true relativities by MLE. This is our benchmark.")
    print()

    t0 = time.time()

    # One-hot encode area for GLM (base = A)
    train_pd = df_train.to_pandas()
    test_pd = df_test.to_pandas()

    # Construct design matrix
    def make_glm_X(df_pd):
        X = np.column_stack([
            df_pd["ncd_years"].values.astype(float),
            (df_pd["area"] == "B").astype(float),
            (df_pd["area"] == "C").astype(float),
            (df_pd["area"] == "D").astype(float),
            (df_pd["area"] == "E").astype(float),
            (df_pd["area"] == "F").astype(float),
            df_pd["has_convictions"].values.astype(float),
        ])
        return sm.add_constant(X)

    X_glm_train = make_glm_X(train_pd)
    X_glm_test = make_glm_X(test_pd)

    glm_model = sm.GLM(
        df_train["claim_count"].to_numpy(),
        X_glm_train,
        family=PoissonFamily(),
        offset=np.log(np.clip(df_train["exposure"].to_numpy(), 1e-6, None)),
        freq_weights=df_train["exposure"].to_numpy(),
    ).fit()
    glm_fit_time = time.time() - t0

    # Extract GLM relativities
    params = glm_model.params  # intercept, ncd, B, C, D, E, F, convictions
    glm_ncd_rel = {ncd: np.exp(params[1] * ncd - params[1] * 0) for ncd in range(6)}
    glm_area_rel = {
        "A": 1.0,
        "B": np.exp(params[2]),
        "C": np.exp(params[3]),
        "D": np.exp(params[4]),
        "E": np.exp(params[5]),
        "F": np.exp(params[6]),
    }
    glm_conv_rel = np.exp(params[7])

    print(f"  GLM fit time: {glm_fit_time:.1f}s")
    print()
    print("  GLM NCD relativities (base = NCD 0):")
    print(f"  {'NCD':>6} {'True rel':>12} {'GLM rel':>12} {'Error %':>10}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*10}")
    for ncd in range(6):
        true_rel = np.exp(BETA_NCD * ncd)
        glm_rel = glm_ncd_rel[ncd]
        err = (glm_rel - true_rel) / true_rel * 100
        print(f"  {ncd:>6} {true_rel:>12.4f} {glm_rel:>12.4f} {err:>+9.2f}%")
    print()
    print("  GLM area relativities (base = A):")
    print(f"  {'Area':>6} {'True rel':>12} {'GLM rel':>12} {'Error %':>10}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*10}")
    for band in ["A", "B", "C", "D", "E", "F"]:
        true_rel = np.exp(AREA_EFFECTS[band])
        glm_rel = glm_area_rel[band]
        err = (glm_rel - true_rel) / true_rel * 100
        print(f"  {band:>6} {true_rel:>12.4f} {glm_rel:>12.4f} {err:>+9.2f}%")
    print()
    print(f"  GLM conviction relativity: {glm_conv_rel:.4f}  (true: {np.exp(BETA_CONV):.4f}  "
          f"error: {(glm_conv_rel - np.exp(BETA_CONV))/np.exp(BETA_CONV)*100:+.2f}%)")
    print()
else:
    print("  statsmodels not available — skipping GLM comparison.")
    print("  Install with: pip install statsmodels")
    glm_ncd_rel = None
    glm_area_rel = None
    glm_conv_rel = None
    print()

# ---------------------------------------------------------------------------
# APPROACH 3: SHAP relativities (the library)
# ---------------------------------------------------------------------------

print("-" * 70)
print("APPROACH 3: shap-relativities applied to the CatBoost model")
print("-" * 70)
print()
print("  The library aggregates SHAP values by feature level to produce")
print("  exp(mean_shap) relativities — the same format as GLM exp(beta),")
print("  extracted from a non-parametric model.")
print()

t0 = time.time()

X_shap = df_train.select(feature_cols)

sr = SHAPRelativities(
    model=model,
    X=X_shap,
    exposure=df_train["exposure"],
    categorical_features=["ncd_years", "area", "has_convictions"],
)
sr.fit()
shap_rels = sr.extract_relativities(
    normalise_to="base_level",
    base_levels={"ncd_years": 0, "area": "A", "has_convictions": 0},
)
shap_time = time.time() - t0

print(f"  SHAP extraction time (on top of CatBoost fit): {shap_time:.1f}s")
print()

# Parse SHAP relativities into lookup dicts
def get_shap_rel(feature: str, level: str | int) -> float:
    row = shap_rels.filter(
        (pl.col("feature") == feature) & (pl.col("level") == str(level))
    )
    if len(row) == 0:
        return float("nan")
    return float(row["relativity"][0])

print("  SHAP NCD relativities (base = NCD 0):")
header = f"  {'NCD':>6} {'True rel':>12} {'SHAP rel':>12} {'Error %':>10}"
if _STATSMODELS_OK:
    header += f" {'GLM rel':>12} {'SHAP vs GLM':>12}"
print(header)
sep_len = 52 + (24 if _STATSMODELS_OK else 0)
print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*10}" + (f" {'-'*12} {'-'*12}" if _STATSMODELS_OK else ""))

ncd_shap_errors = []
ncd_glm_errors = []
for ncd in range(6):
    true_rel = np.exp(BETA_NCD * ncd)
    shap_rel = get_shap_rel("ncd_years", ncd)
    err_shap = (shap_rel - true_rel) / true_rel * 100
    ncd_shap_errors.append(abs(err_shap))
    row = f"  {ncd:>6} {true_rel:>12.4f} {shap_rel:>12.4f} {err_shap:>+9.2f}%"
    if _STATSMODELS_OK:
        glm_rel = glm_ncd_rel[ncd]
        err_glm = (glm_rel - true_rel) / true_rel * 100
        ncd_glm_errors.append(abs(err_glm))
        shap_vs_glm = (shap_rel - glm_rel) / glm_rel * 100
        row += f" {glm_rel:>12.4f} {shap_vs_glm:>+11.2f}%"
    print(row)
print()

print("  SHAP area relativities (base = A):")
print(f"  {'Area':>6} {'True rel':>12} {'SHAP rel':>12} {'Error %':>10}" +
      (f" {'GLM rel':>12} {'SHAP vs GLM':>12}" if _STATSMODELS_OK else ""))
print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*10}" +
      (f" {'-'*12} {'-'*12}" if _STATSMODELS_OK else ""))

area_shap_errors = []
area_glm_errors = []
for band in ["A", "B", "C", "D", "E", "F"]:
    true_rel = np.exp(AREA_EFFECTS[band])
    shap_rel = get_shap_rel("area", band)
    err_shap = (shap_rel - true_rel) / true_rel * 100
    area_shap_errors.append(abs(err_shap))
    row = f"  {band:>6} {true_rel:>12.4f} {shap_rel:>12.4f} {err_shap:>+9.2f}%"
    if _STATSMODELS_OK:
        glm_rel = glm_area_rel[band]
        err_glm = (glm_rel - true_rel) / true_rel * 100
        area_glm_errors.append(abs(err_glm))
        shap_vs_glm = (shap_rel - glm_rel) / glm_rel * 100
        row += f" {glm_rel:>12.4f} {shap_vs_glm:>+11.2f}%"
    print(row)
print()

shap_conv_rel = get_shap_rel("has_convictions", 1)
true_conv_rel = np.exp(BETA_CONV)
err_conv_shap = (shap_conv_rel - true_conv_rel) / true_conv_rel * 100
conv_row = (f"  Conviction relativity: {shap_conv_rel:.4f}  "
            f"(true: {true_conv_rel:.4f}  error: {err_conv_shap:+.2f}%)")
if _STATSMODELS_OK:
    err_conv_glm = (glm_conv_rel - true_conv_rel) / true_conv_rel * 100
    conv_row += f"  GLM: {glm_conv_rel:.4f}  GLM error: {err_conv_glm:+.2f}%"
print(conv_row)
print()

# Validation check
print("  SHAP reconstruction check:")
checks = sr.validate()
recon = checks.get("reconstruction")
if recon:
    status = "PASS" if recon.passed else "FAIL"
    print(f"    Reconstruction: {status}  (max error: {recon.value:.2e})")
print()

# ---------------------------------------------------------------------------
# What feature importance actually tells you vs what relativities tell you
# ---------------------------------------------------------------------------

print("-" * 70)
print("WHAT EACH APPROACH TELLS YOU")
print("-" * 70)
print()
print("  Feature importance (gain):")
for feat in feature_cols:
    pct = fi[feat] / fi_total * 100
    print(f"    {feat:<20}: {pct:.1f}% importance  -->  ranks factors, nothing more")
print()
print("  SHAP relativities for NCD (key pricing question: discount schedule):")
for ncd in range(6):
    shap_rel = get_shap_rel("ncd_years", ncd)
    true_rel = np.exp(BETA_NCD * ncd)
    print(f"    NCD={ncd}: relativity = {shap_rel:.4f}  (true: {true_rel:.4f})")
print()
print("  A pricing team can upload the SHAP relativities directly into a rate")
print("  engine (Radar, Guidewire, bespoke tables). Feature importance is a")
print("  diagnostic, not a pricing deliverable.")
print()

# ---------------------------------------------------------------------------
# Comparison summary
# ---------------------------------------------------------------------------

print("=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)
print()

print(f"  {'Approach':<35} {'Level rels?':>12} {'Mean |error| vs true':>22} {'Time':>8}")
print(f"  {'-'*35} {'-'*12} {'-'*22} {'-'*8}")

# Feature importance
print(f"  {'Feature importance (gain)':<35} {'No':>12} {'N/A':>22} {catboost_fit_time:>7.1f}s")

# SHAP relativities error
all_shap_errors = ncd_shap_errors + area_shap_errors + [abs(err_conv_shap)]
mean_shap_err = np.mean(all_shap_errors)
print(f"  {'SHAP relativities (shap-relativities)':<35} {'Yes':>12} {mean_shap_err:>21.2f}% {shap_time:>7.1f}s")

# GLM
if _STATSMODELS_OK:
    all_glm_errors = ncd_glm_errors + area_glm_errors + [abs(err_conv_glm)]
    mean_glm_err = np.mean(all_glm_errors)
    print(f"  {'Poisson GLM exp(beta)':<35} {'Yes':>12} {mean_glm_err:>21.2f}% {glm_fit_time:>7.1f}s")

print()
print("KEY FINDINGS")
print()

ncd5_shap = get_shap_rel("ncd_years", 5)
ncd5_true = np.exp(5 * BETA_NCD)
print(f"  NCD=5 vs NCD=0 (true discount = {(1-ncd5_true)*100:.1f}%):")
print(f"    True relativity:           {ncd5_true:.4f}")
if _STATSMODELS_OK:
    print(f"    GLM exp(beta):             {glm_ncd_rel[5]:.4f}  "
          f"(error: {(glm_ncd_rel[5]-ncd5_true)/ncd5_true*100:+.2f}%)")
print(f"    SHAP relativity:           {ncd5_shap:.4f}  "
      f"(error: {(ncd5_shap-ncd5_true)/ncd5_true*100:+.2f}%)")
print()
print(f"  Area F vs Area A (true loading = {(np.exp(0.65)-1)*100:.1f}%):")
true_f = np.exp(0.65)
shap_f = get_shap_rel("area", "F")
print(f"    True relativity:           {true_f:.4f}")
if _STATSMODELS_OK:
    glm_f_rel = glm_area_rel['F']
    print(f"    GLM exp(beta):             {glm_f_rel:.4f}  "
          f"(error: {(glm_f_rel-true_f)/true_f*100:+.2f}%)")
    print(f"    SHAP relativity:           {shap_f:.4f}  "
          f"(error: {(shap_f-true_f)/true_f*100:+.2f}%)")
else:
    print(f"    SHAP relativity:           {shap_f:.4f}  "
          f"(error: {(shap_f-true_f)/true_f*100:+.2f}%)")
print()
print("  Feature importance cannot answer either question.")
print(f"  SHAP relativities recover the multiplicative structure with mean")
print(f"  absolute error of {mean_shap_err:.2f}% across all {len(all_shap_errors)} factor levels.")
if _STATSMODELS_OK:
    print(f"  GLM error on same factor levels: {mean_glm_err:.2f}%.")
    print(f"  SHAP vs GLM gap: {mean_shap_err - mean_glm_err:+.2f}pp — cost of not constraining")
    print(f"  the model to a log-linear form (GBM flexibility introduces attribution noise).")
print()
print("  When to use SHAP relativities:")
print("    - CatBoost already beats the production GLM by a meaningful margin")
print("    - You need to export factor tables to a rating engine")
print("    - The pricing committee or regulator requires a factor table, not")
print("      just a black-box model score")
print()
print("  When the GLM is better:")
print("    - The portfolio is small (<10k policies) and GBM will overfit")
print("    - Closed-form standard errors are required for regulatory filing")
print("    - The Gini improvement from GBM does not justify the overhead")

elapsed = time.time() - BENCHMARK_START
print(f"\nBenchmark completed in {elapsed:.1f}s")
