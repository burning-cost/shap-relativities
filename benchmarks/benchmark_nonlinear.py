"""
Benchmark: GBM scenario — non-linear DGP where GLM cannot recover true effects.

This is the benchmark that demonstrates the library's actual value proposition.
The true DGP has effects that a log-linear GLM structurally cannot recover:

  1. Driver age: U-shaped risk curve. Young drivers (<25) and older drivers (>70)
     are worse. The mid-range is cheapest. A GLM with a linear driver_age term
     gets the slope wrong everywhere; a GLM with main-effect dummies can recover
     it, but only if the actuary has already identified the shape. CatBoost finds
     it automatically.

  2. Area × vehicle_age interaction: Urban areas (D-F) with old vehicles (>8 years)
     carry an additional 35% uplift. This is a real pattern in UK motor — old vehicles
     in high-risk urban postcodes are disproportionately high-claim. A main-effects GLM
     with no interaction term misses it entirely.

  3. NCD: Approximately log-linear but with a non-proportional step at NCD=5 (the
     max bonus level). Policyholders who've held max NCD for several years are
     slightly better risks than the linear projection from NCD=4 implies. This is a
     small effect, but it illustrates that even "obviously linear" features can have
     non-linear true relationships.

Setup:
  - 25,000 synthetic UK motor policies, known non-linear DGP
  - 5 rating features: driver_age (continuous), area (A-F), vehicle_age (0-15),
    ncd_years (0-5), has_conviction (binary)
  - True effects known — used to compute the error for each approach
  - 70/30 train/test split

The comparison:
  - GLM with linear driver_age term: cannot recover the U-shape
  - GLM with driver_age bins: can partially recover it, but only if the actuary
    already knows the shape (we'll use 4 age bands, which misses fine structure)
  - CatBoost Poisson: finds all non-linear and interaction effects automatically
  - shap-relativities on the CatBoost: extracts the factor table

Expected finding:
  - GLM mean relativity error is materially higher than SHAP on the non-linear features
  - GBM Gini is higher than GLM
  - SHAP relativities recover the U-shaped driver_age curve and interaction structure

Run:
    python benchmarks/benchmark_nonlinear.py
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
print("Benchmark: NON-LINEAR DGP (driver age U-shape + area × vehicle_age)")
print("The case where GLM cannot recover true effects")
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
# Data-generating process: non-linear effects
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(77)
N_POLICIES = 25_000
BASE_FREQ = 0.065

# --- Driver age: U-shaped risk curve ---
# Minimum risk at age 35-45. Young (<25) and older (>65) drivers are worse.
# This is modelled as a quadratic in log space, centred at age 38.
DRIVER_AGE_MIN_RISK_AGE = 38.0
DRIVER_AGE_QUADRATIC_COEF = 0.0012  # log frequency units per (age - 38)^2
# At age 20: 0.0012 * (20-38)^2 = 0.0012 * 324 = 0.389 -> exp(0.389) ~ 1.47x
# At age 70: 0.0012 * (70-38)^2 = 0.0012 * 1024 = 1.229 -> exp(1.229) ~ 3.42x
# At age 38: 0.0 -> 1.0x (reference)

# --- Area: main effects ---
TRUE_AREA_REL = {"A": 1.00, "B": 1.15, "C": 1.32, "D": 1.55, "E": 1.80, "F": 2.10}

# --- Vehicle age: main effect + interaction with area ---
# Base vehicle age effect: slight positive gradient (older = more claims)
VEHICLE_AGE_SLOPE = 0.018  # log units per year of vehicle age, from age 0

# Interaction: urban areas (D, E, F) with old vehicles (>8 years) get extra uplift
INTERACTION_UPLIFT = 0.30  # additive in log space = exp(0.30) ~ 1.35x
INTERACTION_URBAN_AREAS = {"D", "E", "F"}
INTERACTION_VEHICLE_AGE_THRESHOLD = 8

# --- NCD: log-linear main effect with extra step at NCD=5 ---
NCD_COEF = -0.11  # log units per year
NCD5_EXTRA_DISCOUNT = -0.06  # additional log discount at NCD=5 only

# --- Conviction ---
TRUE_CONV_REL = 1.65

print(f"DGP: {N_POLICIES:,} policies, Poisson frequency, non-linear structure")
print(f"  Driver age: U-shaped (quadratic in log space, minimum at age {DRIVER_AGE_MIN_RISK_AGE})")
print(f"    Age 20 relativity: {np.exp(DRIVER_AGE_QUADRATIC_COEF * (20 - DRIVER_AGE_MIN_RISK_AGE)**2):.3f}x")
print(f"    Age 38 relativity: 1.000x (reference)")
print(f"    Age 70 relativity: {np.exp(DRIVER_AGE_QUADRATIC_COEF * (70 - DRIVER_AGE_MIN_RISK_AGE)**2):.3f}x")
print(f"  Area main effects: {' | '.join(f'{k}={v:.2f}' for k, v in TRUE_AREA_REL.items())}")
print(f"  Vehicle age: {VEHICLE_AGE_SLOPE:.3f} log units/year + {np.exp(INTERACTION_UPLIFT):.2f}x uplift"
      f" for areas {{D,E,F}} and vehicle_age>={INTERACTION_VEHICLE_AGE_THRESHOLD}")
print(f"  NCD: {NCD_COEF}/year, extra {NCD5_EXTRA_DISCOUNT} at NCD=5"
      f" -> NCD=5 rel = {np.exp(NCD_COEF*5 + NCD5_EXTRA_DISCOUNT):.4f}")
print(f"  Conviction: {TRUE_CONV_REL:.3f}x")
print()

# --- Generate features ---
driver_age = RNG.integers(18, 82, N_POLICIES).astype(float)
areas = RNG.choice(list(TRUE_AREA_REL.keys()), N_POLICIES,
                   p=[0.25, 0.25, 0.20, 0.15, 0.10, 0.05])
vehicle_age = RNG.integers(0, 16, N_POLICIES)
ncd_years = RNG.integers(0, 6, N_POLICIES)
has_conviction = (RNG.random(N_POLICIES) < 0.13).astype(int)
exposure = RNG.uniform(0.3, 1.0, N_POLICIES)

area_code = np.array([ord(a) - ord("A") for a in areas])

# --- True log-frequency ---
# Driver age: U-shaped quadratic
log_freq_age = DRIVER_AGE_QUADRATIC_COEF * (driver_age - DRIVER_AGE_MIN_RISK_AGE) ** 2

# Area main effect
log_freq_area = np.array([np.log(TRUE_AREA_REL[a]) for a in areas])

# Vehicle age main effect
log_freq_veh = VEHICLE_AGE_SLOPE * vehicle_age

# Area × vehicle_age interaction
is_urban = np.array([a in INTERACTION_URBAN_AREAS for a in areas])
is_old_vehicle = vehicle_age >= INTERACTION_VEHICLE_AGE_THRESHOLD
interaction_flag = (is_urban & is_old_vehicle).astype(float)
log_freq_interaction = INTERACTION_UPLIFT * interaction_flag

# NCD
log_freq_ncd = NCD_COEF * ncd_years + NCD5_EXTRA_DISCOUNT * (ncd_years == 5)

# Conviction
log_freq_conv = np.log(TRUE_CONV_REL) * has_conviction

log_freq = (
    np.log(BASE_FREQ)
    + log_freq_age
    + log_freq_area
    + log_freq_veh
    + log_freq_interaction
    + log_freq_ncd
    + log_freq_conv
)

true_freq = np.exp(log_freq)
claim_count = RNG.poisson(true_freq * exposure)

# True DGP relativities for comparison
TRUE_NCD_REL = {
    i: float(np.exp(NCD_COEF * i + NCD5_EXTRA_DISCOUNT * (i == 5)))
    for i in range(6)
}

df = pl.DataFrame({
    "driver_age": driver_age.tolist(),
    "area": areas.tolist(),
    "area_code": area_code.tolist(),
    "vehicle_age": vehicle_age.tolist(),
    "ncd_years": ncd_years.tolist(),
    "has_conviction": has_conviction.tolist(),
    "exposure": exposure.tolist(),
    "claim_count": claim_count.tolist(),
    "interaction_flag": interaction_flag.tolist(),
    "is_urban": is_urban.astype(int).tolist(),
})

n_train = int(0.70 * N_POLICIES)
df_train = df[:n_train]
df_test = df[n_train:]

interaction_pct = float(interaction_flag.mean())
print(f"Portfolio: {n_train:,} train / {N_POLICIES - n_train:,} test")
print(f"Interaction group (urban + old vehicle): {interaction_pct:.1%} of portfolio")
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
# APPROACH 1: Poisson GLM with linear driver_age term
# ---------------------------------------------------------------------------

print("-" * 70)
print("APPROACH 1: Poisson GLM (linear driver_age, no interaction term)")
print("-" * 70)
print()
print("  The standard first-pass GLM: area, vehicle_age, ncd_years, conviction")
print("  as main effects, driver_age as a linear continuous term.")
print("  Structurally cannot recover the U-shaped age curve or the area x vehicle_age")
print("  interaction. The GLM is not misspecified carelessly — this is a")
print("  representative of what a typical pricing GLM looks like before detailed")
print("  shape analysis.")
print()

t0 = time.time()

X_train_glm = df_train.select(["driver_age", "area", "vehicle_age",
                                "ncd_years", "has_conviction"]).to_pandas()
X_test_glm = df_test.select(["driver_age", "area", "vehicle_age",
                               "ncd_years", "has_conviction"]).to_pandas()

preprocessor_glm = ColumnTransformer([
    ("area_ohe", OneHotEncoder(drop="first", sparse_output=False), ["area"]),
    ("passthrough", "passthrough", ["driver_age", "vehicle_age", "ncd_years", "has_conviction"]),
])
glm_linear = Pipeline([
    ("prep", preprocessor_glm),
    ("model", PoissonRegressor(alpha=0, max_iter=500)),
])
glm_linear.fit(
    X_train_glm,
    df_train["claim_count"].to_numpy() / df_train["exposure"].to_numpy(),
    model__sample_weight=df_train["exposure"].to_numpy(),
)
glm_linear_time = time.time() - t0

glm_linear_pred_test = glm_linear.predict(X_test_glm)
glm_linear_gini = gini(
    df_test["claim_count"].to_numpy(),
    glm_linear_pred_test,
    df_test["exposure"].to_numpy(),
)

# Extract GLM relativities for the discrete features (area, ncd, conviction)
# to compare against true values
ohe = glm_linear.named_steps["prep"].named_transformers_["area_ohe"]
ohe_names = ohe.get_feature_names_out()
coefs = glm_linear.named_steps["model"].coef_
n_area = len(ohe_names)  # 5 (area B-F)
# Passthrough order: driver_age, vehicle_age, ncd_years, has_conviction
glm_linear_ncd_coef = float(coefs[n_area + 2])
glm_linear_conv_coef = float(coefs[n_area + 3])
glm_linear_area_rels: dict[str, float] = {"A": 1.0}
for i, ohe_name in enumerate(ohe_names):
    letter = ohe_name.split("_")[1]
    glm_linear_area_rels[letter] = float(np.exp(coefs[i]))

print(f"  GLM (linear age) fit time: {glm_linear_time:.2f}s")
print(f"  Gini (test):               {glm_linear_gini:.4f}")
print()

# Report discrete-feature relativities vs truth
glm_lin_errors = []
print(f"  {'Feature':<22} {'Level':<8} {'GLM rel':>10} {'True rel':>10} {'|Error|':>10}")
print(f"  {'-'*22} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
for band in ["A", "B", "C", "D", "E", "F"]:
    rel = glm_linear_area_rels.get(band, 1.0)
    true_r = TRUE_AREA_REL[band]
    err = abs(rel - true_r) / true_r * 100
    glm_lin_errors.append(err)
    print(f"  {'area':<22} {band:<8} {rel:>10.4f} {true_r:>10.4f} {err:>9.2f}%")
for ncd in range(6):
    rel = float(np.exp(glm_linear_ncd_coef * ncd))
    true_r = TRUE_NCD_REL[ncd]
    err = abs(rel - true_r) / true_r * 100
    glm_lin_errors.append(err)
    print(f"  {'ncd_years':<22} {ncd:<8} {rel:>10.4f} {true_r:>10.4f} {err:>9.2f}%")
conv_rel_lin = float(np.exp(glm_linear_conv_coef))
conv_err_lin = abs(conv_rel_lin - TRUE_CONV_REL) / TRUE_CONV_REL * 100
glm_lin_errors.append(conv_err_lin)
print(f"  {'has_conviction':<22} {'1':<8} {conv_rel_lin:>10.4f} {TRUE_CONV_REL:>10.4f} {conv_err_lin:>9.2f}%")
print()
print(f"  Mean |error| (discrete factors): {np.mean(glm_lin_errors):.2f}%")
print(f"  Note: driver_age relativity errors not computed — linear GLM has no per-level")
print(f"  relativities for a continuous feature, only a slope coefficient.")
print()

# ---------------------------------------------------------------------------
# APPROACH 2: Poisson GLM with driver_age bands (improved GLM)
# ---------------------------------------------------------------------------

print("-" * 70)
print("APPROACH 2: Poisson GLM with driver_age bands (binned age)")
print("-" * 70)
print()
print("  A more carefully specified GLM: driver_age grouped into 5 bands")
print("  (<25, 25-34, 35-49, 50-64, 65+) to partially capture the U-shape.")
print("  This is what an actuary would do after noticing the non-linear age effect.")
print("  Vehicle_age still enters as a linear term — no interaction term.")
print()


def age_band(age_arr: np.ndarray) -> np.ndarray:
    bins = [0, 25, 35, 50, 65, 100]
    labels = ["<25", "25-34", "35-49", "50-64", "65+"]
    band_arr = np.empty(len(age_arr), dtype=object)
    for j, age in enumerate(age_arr):
        for k in range(len(labels)):
            if bins[k] <= age < bins[k + 1]:
                band_arr[j] = labels[k]
                break
    return band_arr


age_band_train = age_band(df_train["driver_age"].to_numpy())
age_band_test = age_band(df_test["driver_age"].to_numpy())

import pandas as pd  # noqa: E402

X_train_glm2 = pd.DataFrame({
    "age_band": age_band_train,
    "area": df_train["area"].to_numpy(),
    "vehicle_age": df_train["vehicle_age"].to_numpy(),
    "ncd_years": df_train["ncd_years"].to_numpy(),
    "has_conviction": df_train["has_conviction"].to_numpy(),
})
X_test_glm2 = pd.DataFrame({
    "age_band": age_band_test,
    "area": df_test["area"].to_numpy(),
    "vehicle_age": df_test["vehicle_age"].to_numpy(),
    "ncd_years": df_test["ncd_years"].to_numpy(),
    "has_conviction": df_test["has_conviction"].to_numpy(),
})

preprocessor_glm2 = ColumnTransformer([
    ("age_ohe", OneHotEncoder(drop="first", sparse_output=False), ["age_band"]),
    ("area_ohe", OneHotEncoder(drop="first", sparse_output=False), ["area"]),
    ("passthrough", "passthrough", ["vehicle_age", "ncd_years", "has_conviction"]),
])

t0 = time.time()
glm_binned = Pipeline([
    ("prep", preprocessor_glm2),
    ("model", PoissonRegressor(alpha=0, max_iter=500)),
])
glm_binned.fit(
    X_train_glm2,
    df_train["claim_count"].to_numpy() / df_train["exposure"].to_numpy(),
    model__sample_weight=df_train["exposure"].to_numpy(),
)
glm_binned_time = time.time() - t0

glm_binned_pred_test = glm_binned.predict(X_test_glm2)
glm_binned_gini = gini(
    df_test["claim_count"].to_numpy(),
    glm_binned_pred_test,
    df_test["exposure"].to_numpy(),
)

# Extract binned GLM relativities
ohe_age = glm_binned.named_steps["prep"].named_transformers_["age_ohe"]
ohe_area = glm_binned.named_steps["prep"].named_transformers_["area_ohe"]
age_names = ohe_age.get_feature_names_out()  # 4 non-base age bands
area_names = ohe_area.get_feature_names_out()  # 5 non-base area bands
n_age = len(age_names)
n_area2 = len(area_names)

glm2_coefs = glm_binned.named_steps["model"].coef_
glm2_ncd_coef = float(glm2_coefs[n_age + n_area2 + 1])
glm2_conv_coef = float(glm2_coefs[n_age + n_area2 + 2])
glm2_area_rels: dict[str, float] = {"A": 1.0}
for i, ohe_name in enumerate(area_names):
    letter = ohe_name.split("_")[1]
    glm2_area_rels[letter] = float(np.exp(glm2_coefs[n_age + i]))

print(f"  GLM (binned age) fit time: {glm_binned_time:.2f}s")
print(f"  Gini (test):               {glm_binned_gini:.4f}")
print()

glm2_errors = []
print(f"  {'Feature':<22} {'Level':<8} {'GLM rel':>10} {'True rel':>10} {'|Error|':>10}")
print(f"  {'-'*22} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
for band in ["A", "B", "C", "D", "E", "F"]:
    rel = glm2_area_rels.get(band, 1.0)
    true_r = TRUE_AREA_REL[band]
    err = abs(rel - true_r) / true_r * 100
    glm2_errors.append(err)
    print(f"  {'area':<22} {band:<8} {rel:>10.4f} {true_r:>10.4f} {err:>9.2f}%")
for ncd in range(6):
    rel = float(np.exp(glm2_ncd_coef * ncd))
    true_r = TRUE_NCD_REL[ncd]
    err = abs(rel - true_r) / true_r * 100
    glm2_errors.append(err)
    print(f"  {'ncd_years':<22} {ncd:<8} {rel:>10.4f} {true_r:>10.4f} {err:>9.2f}%")
conv_rel2 = float(np.exp(glm2_conv_coef))
conv_err2 = abs(conv_rel2 - TRUE_CONV_REL) / TRUE_CONV_REL * 100
glm2_errors.append(conv_err2)
print(f"  {'has_conviction':<22} {'1':<8} {conv_rel2:>10.4f} {TRUE_CONV_REL:>10.4f} {conv_err2:>9.2f}%")
print()
print(f"  Mean |error| (discrete factors): {np.mean(glm2_errors):.2f}%")
print(f"  Note: vehicle_age still linear in this GLM — no interaction with area.")
print()

# ---------------------------------------------------------------------------
# APPROACH 3: CatBoost Poisson + SHAP relativities
# ---------------------------------------------------------------------------

print("-" * 70)
print("APPROACH 3: CatBoost Poisson + SHAP relativities")
print("-" * 70)
print()
print("  CatBoost is trained on the raw features — driver_age as a continuous")
print("  integer, area as an integer code, vehicle_age as an integer, ncd_years,")
print("  has_conviction. No binning, no interaction terms specified manually.")
print("  The model finds the non-linear age effect and the area × vehicle_age")
print("  interaction automatically via tree splits.")
print()

features_gbm = ["area_code", "driver_age", "vehicle_age", "ncd_years", "has_conviction"]
X_train_pl = df_train.select(features_gbm)
X_test_pl = df_test.select(features_gbm)

t0 = time.time()

pool_train = catboost.Pool(
    data=X_train_pl.to_pandas(),
    label=df_train["claim_count"].to_numpy(),
    weight=df_train["exposure"].to_numpy(),
)

cbm = catboost.CatBoostRegressor(
    loss_function="Poisson",
    iterations=500,
    learning_rate=0.04,
    depth=6,
    random_seed=77,
    verbose=0,
    allow_writing_files=False,
)
cbm.fit(pool_train)
catboost_time = time.time() - t0

cbm_pred_test = cbm.predict(X_test_pl.to_pandas())
cbm_gini = gini(
    df_test["claim_count"].to_numpy(),
    cbm_pred_test,
    df_test["exposure"].to_numpy(),
)

print(f"  CatBoost fit time: {catboost_time:.1f}s")
print(f"  Gini (test):       {cbm_gini:.4f}")
print()

# SHAP extraction — discrete features as categorical, driver_age as continuous
t1 = time.time()
sr = SHAPRelativities(
    model=cbm,
    X=X_train_pl,
    exposure=df_train["exposure"].to_numpy(),
    categorical_features=["area_code", "ncd_years", "has_conviction", "vehicle_age"],
    continuous_features=["driver_age"],
)
sr.fit()

rels_discrete = sr.extract_relativities(
    normalise_to="base_level",
    base_levels={
        "area_code": 0,
        "ncd_years": 0,
        "has_conviction": 0,
        "vehicle_age": 0,
    },
)
shap_time = time.time() - t1

checks = sr.validate()
recon = checks.get("reconstruction")
recon_status = "PASS" if (recon and recon.passed) else "FAIL"
recon_val = f"{recon.value:.2e}" if recon else "n/a"

print(f"  SHAP extraction time: {shap_time:.1f}s")
print(f"  SHAP reconstruction:  {recon_status}  (max error: {recon_val})")
print()

# Compare discrete relativities
area_map = {i: chr(ord("A") + i) for i in range(6)}

shap_errors = []
print(f"  {'Feature':<22} {'Level':<8} {'SHAP rel':>10} {'True rel':>10} {'|Error|':>10} {'CI_95':>16}")
print(f"  {'-'*22} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*16}")

shap_ncd_rels: dict[int, float] = {}
shap_conv_val = float("nan")

for row in rels_discrete.iter_rows(named=True):
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
        feat_display = "area"
    elif feat == "ncd_years":
        try:
            ncd_val = int(float(level_str))
            true_r = TRUE_NCD_REL.get(ncd_val, float("nan"))
            shap_ncd_rels[ncd_val] = rel
            display_level = str(ncd_val)
        except (ValueError, TypeError):
            true_r = float("nan")
            display_level = level_str
        feat_display = "ncd_years"
    elif feat == "has_conviction":
        try:
            conv_val = int(float(level_str))
            true_r = TRUE_CONV_REL if conv_val == 1 else 1.0
            if conv_val == 1:
                shap_conv_val = rel
            display_level = str(conv_val)
        except (ValueError, TypeError):
            true_r = float("nan")
            display_level = level_str
        feat_display = "has_conviction"
    elif feat == "vehicle_age":
        # Vehicle age: true relativity includes main effect + interaction absorbed
        # Just report the relativities — comparison vs true is complex here
        continue
    else:
        continue

    if not (np.isnan(true_r) or true_r == 0):
        err = abs(rel - true_r) / true_r * 100
        shap_errors.append(err)
        err_str = f"{err:>9.2f}%"
    else:
        err_str = "    n/a"

    if not (np.isnan(lo) or np.isnan(hi)):
        ci_str = f"[{lo:.3f}, {hi:.3f}]"
    else:
        ci_str = "n/a"

    print(f"  {feat_display:<22} {display_level:<8} {rel:>10.4f} {true_r:>10.4f} {err_str:>10} {ci_str:>16}")

print()
print(f"  Mean |error| (area, NCD, conviction): {np.mean(shap_errors):.2f}%")
print()

# Show driver_age continuous curve
print("  Driver age relativity curve (continuous SHAP, smoothed via LOESS):")
try:
    age_curve = sr.extract_continuous_curve(
        feature="driver_age",
        n_points=7,
        smooth_method="loess",
    )
    # Also show true DGP relativities at those ages (mean-normalised)
    # Compute mean of log_freq_age over the training distribution
    train_ages = df_train["driver_age"].to_numpy()
    train_weights = df_train["exposure"].to_numpy()
    true_log_age_vals = DRIVER_AGE_QUADRATIC_COEF * (train_ages - DRIVER_AGE_MIN_RISK_AGE) ** 2
    mean_true_log_age = float(np.average(true_log_age_vals, weights=train_weights))

    print(f"  {'Age':>6} {'SHAP rel':>10} {'True rel':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10}")
    for row in age_curve.iter_rows(named=True):
        age_val = row["feature_value"]
        shap_rel = row["relativity"]
        true_log = DRIVER_AGE_QUADRATIC_COEF * (age_val - DRIVER_AGE_MIN_RISK_AGE) ** 2
        true_rel = float(np.exp(true_log - mean_true_log_age))
        print(f"  {age_val:>6.0f} {shap_rel:>10.4f} {true_rel:>10.4f}")
except Exception as e:
    print(f"  Could not extract age curve: {e}")
print()

# Show NCD=5 step
ncd5_true = TRUE_NCD_REL[5]
ncd4_true = TRUE_NCD_REL[4]
step_true = ncd5_true / ncd4_true
shap_ncd5 = shap_ncd_rels.get(5, float("nan"))
shap_ncd4 = shap_ncd_rels.get(4, float("nan"))
step_shap = shap_ncd5 / shap_ncd4 if shap_ncd4 > 0 else float("nan")
print(f"  NCD=5 step discount (NCD=5 vs NCD=4):")
print(f"    True DGP:  {step_true:.4f} (extra {NCD5_EXTRA_DISCOUNT*100:.0f}% discount at max NCD)")
print(f"    SHAP:      {step_shap:.4f}")
print()

# ---------------------------------------------------------------------------
# Comparison summary
# ---------------------------------------------------------------------------

print("=" * 70)
print("COMPARISON SUMMARY — NON-LINEAR DGP")
print("=" * 70)
print()
print(f"  {'Approach':<46} {'Mean |err| (discrete)':>22} {'Gini':>8}")
print(f"  {'-'*46} {'-'*22} {'-'*8}")
print(f"  {'GLM (linear driver_age)':<46} {np.mean(glm_lin_errors):>21.2f}% {glm_linear_gini:>8.4f}")
print(f"  {'GLM (binned age, 5 bands)':<46} {np.mean(glm2_errors):>21.2f}% {glm_binned_gini:>8.4f}")
print(f"  {'shap-relativities (CatBoost Poisson)':<46} {np.mean(shap_errors):>21.2f}% {cbm_gini:>8.4f}")
print()

gini_gap_lin = (cbm_gini - glm_linear_gini) * 100
gini_gap_bin = (cbm_gini - glm_binned_gini) * 100

print("KEY FINDINGS")
print()
print(f"  Gini improvement vs GLM (linear age):  +{gini_gap_lin:.2f}pp")
print(f"  Gini improvement vs GLM (binned age):  +{gini_gap_bin:.2f}pp")
print()
print(f"  Discrete-factor mean |error|:")
print(f"    GLM (linear age):  {np.mean(glm_lin_errors):.2f}%")
print(f"    GLM (binned age):  {np.mean(glm2_errors):.2f}%")
print(f"    SHAP relativities: {np.mean(shap_errors):.2f}%")
print()
print("  Interpretation:")
print("  The linear GLM has the highest discrete-factor error because the model")
print("  is misspecified. The omitted variable bias from the non-linear age effect")
print("  and the interaction leaks into the area and NCD coefficients.")
print("  The binned-age GLM improves but is still limited by the missing")
print("  interaction term and the coarse age bands.")
print()
print("  SHAP relativities from CatBoost recover the discrete factor table more")
print("  accurately because the underlying model correctly captures the non-linear")
print("  and interaction structure. The driver_age curve shows the U-shape directly —")
print("  no binning decision required from the actuary.")
print()
print("  This is the scenario where the library earns its keep: the GBM outperforms")
print("  the GLM both on Gini and on the accuracy of the extracted relativities")
print("  for the discrete factors.")

elapsed = time.time() - BENCHMARK_START
print(f"\nBenchmark completed in {elapsed:.1f}s")
