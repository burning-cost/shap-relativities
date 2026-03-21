# Databricks notebook source
# MAGIC %md
# MAGIC # shap-relativities: Benchmark Runner
# MAGIC
# MAGIC Runs both benchmark scenarios and captures actual numbers for the README:
# MAGIC
# MAGIC 1. **Scenario 1**: Clean log-linear DGP — GLM is well-specified
# MAGIC 2. **Scenario 2**: Interaction DGP — vehicle_group × NCD interaction
# MAGIC
# MAGIC Run this notebook on serverless compute to reproduce the benchmark numbers.

# COMMAND ----------

# MAGIC %pip install "shap-relativities[all]" --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import sys
import time
import warnings
import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

from shap_relativities import SHAPRelativities

import catboost
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

print("All imports OK")
print(f"shap-relativities version: {__import__('shap_relativities').__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper functions

# COMMAND ----------

def gini(actual, predicted, exposure):
    """Exposure-weighted Gini coefficient."""
    order = np.argsort(predicted / exposure)
    sorted_actual = actual[order]
    sorted_exposure = exposure[order]
    cum_exp = np.cumsum(sorted_exposure) / sorted_exposure.sum()
    cum_act = np.cumsum(sorted_actual) / sorted_actual.sum()
    auc = float(np.trapz(cum_act, cum_exp))
    return 2 * auc - 1


def poisson_deviance(actual, predicted, exposure):
    """Mean Poisson deviance (lower is better)."""
    mu = predicted * exposure
    y = actual
    eps = 1e-10
    d = 2.0 * np.sum(
        np.where(y > 0, y * np.log((y + eps) / (mu + eps)), 0.0) - (y - mu)
    )
    return float(d / len(y))


def worst_decile_ae(actual, predicted, exposure, n_deciles=10):
    """
    Worst-case A/E deviation across predicted-score deciles.
    Returns the maximum |A/E - 1| across decile buckets.
    """
    order = np.argsort(predicted / exposure)
    idx = np.array_split(order, n_deciles)
    ae_devs = []
    for bucket in idx:
        a = actual[bucket].sum()
        e = (predicted[bucket] * exposure[bucket]).sum()
        if e > 0:
            ae_devs.append(abs(a / e - 1.0))
    return float(max(ae_devs))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Scenario 1: Clean log-linear DGP

# COMMAND ----------

print("=" * 70)
print("SCENARIO 1: Clean log-linear Poisson DGP")
print("=" * 70)

RNG1 = np.random.default_rng(42)
N1 = 20_000

TRUE_AREA_REL_1 = {"A": 1.00, "B": 1.12, "C": 1.25, "D": 1.43, "E": 1.68, "F": 1.95}
NCD_COEF_1 = -0.12
TRUE_CONV_REL_1 = 1.57
BASE_FREQ_1 = 0.07

areas_1 = RNG1.choice(["A", "B", "C", "D", "E", "F"], N1, p=[0.25, 0.25, 0.20, 0.15, 0.10, 0.05])
ncd_1 = RNG1.integers(0, 6, N1)
conv_1 = (RNG1.random(N1) < 0.15).astype(int)
exposure_1 = RNG1.uniform(0.3, 1.0, N1)

log_freq_1 = (
    np.log(BASE_FREQ_1)
    + np.array([np.log(TRUE_AREA_REL_1[a]) for a in areas_1])
    + NCD_COEF_1 * ncd_1
    + np.log(TRUE_CONV_REL_1) * conv_1
)
claims_1 = RNG1.poisson(np.exp(log_freq_1) * exposure_1)
area_code_1 = np.array([ord(a) - ord("A") for a in areas_1])

df1 = pl.DataFrame({
    "area": areas_1.tolist(), "area_code": area_code_1.tolist(),
    "ncd_years": ncd_1.tolist(), "has_conviction": conv_1.tolist(),
    "exposure": exposure_1.tolist(), "claim_count": claims_1.tolist(),
})

n_train_1 = int(0.70 * N1)
df1_tr = df1[:n_train_1]
df1_te = df1[n_train_1:]

# GLM
t0 = time.time()
X_tr_raw = df1_tr.select(["area", "ncd_years", "has_conviction"]).to_pandas()
X_te_raw = df1_te.select(["area", "ncd_years", "has_conviction"]).to_pandas()
pre1 = ColumnTransformer([
    ("area", OneHotEncoder(drop="first", sparse_output=False), ["area"]),
    ("ncd", "passthrough", ["ncd_years"]),
    ("conv", "passthrough", ["has_conviction"]),
])
glm1 = Pipeline([("prep", pre1), ("model", PoissonRegressor(alpha=0, max_iter=500))])
glm1.fit(X_tr_raw, df1_tr["claim_count"].to_numpy() / df1_tr["exposure"].to_numpy(),
         model__sample_weight=df1_tr["exposure"].to_numpy())
glm1_time = time.time() - t0

glm1_pred = glm1.predict(X_te_raw)
glm1_gini = gini(df1_te["claim_count"].to_numpy(), glm1_pred, df1_te["exposure"].to_numpy())
glm1_dev = poisson_deviance(df1_te["claim_count"].to_numpy(), glm1_pred, df1_te["exposure"].to_numpy())
glm1_ae = worst_decile_ae(df1_te["claim_count"].to_numpy(), glm1_pred, df1_te["exposure"].to_numpy())

# GLM relativity errors
ncd_coef_glm = float(glm1.named_steps["model"].coef_[5])
conv_coef_glm = float(glm1.named_steps["model"].coef_[6])
area_coefs_glm = glm1.named_steps["model"].coef_[:5]
ohe_names = glm1.named_steps["prep"].named_transformers_["area"].get_feature_names_out()
glm1_area_rels = {"A": 1.0}
for i, nm in enumerate(ohe_names):
    glm1_area_rels[nm.split("_")[1]] = float(np.exp(area_coefs_glm[i]))
glm1_ncd5 = float(np.exp(ncd_coef_glm * 5))
glm1_conv = float(np.exp(conv_coef_glm))

glm1_errors = []
for b in ["A","B","C","D","E","F"]:
    glm1_errors.append(abs(glm1_area_rels.get(b,1.0) - TRUE_AREA_REL_1[b])/TRUE_AREA_REL_1[b]*100)
for n in range(6):
    glm1_errors.append(abs(np.exp(ncd_coef_glm*n) - np.exp(NCD_COEF_1*n))/np.exp(NCD_COEF_1*n)*100)
glm1_errors.append(abs(glm1_conv - TRUE_CONV_REL_1)/TRUE_CONV_REL_1*100)

# CatBoost
t0 = time.time()
feats_1 = ["area_code", "ncd_years", "has_conviction"]
X1_tr_pl = df1_tr.select(feats_1)
X1_te_pl = df1_te.select(feats_1)
pool1 = catboost.Pool(data=X1_tr_pl.to_pandas(), label=df1_tr["claim_count"].to_numpy(),
                      weight=df1_tr["exposure"].to_numpy())
cbm1 = catboost.CatBoostRegressor(loss_function="Poisson", iterations=300, learning_rate=0.05,
                                   depth=5, random_seed=42, verbose=0, allow_writing_files=False)
cbm1.fit(pool1)
cbm1_time = time.time() - t0

cbm1_pred = cbm1.predict(X1_te_pl.to_pandas())
cbm1_gini = gini(df1_te["claim_count"].to_numpy(), cbm1_pred, df1_te["exposure"].to_numpy())
cbm1_dev = poisson_deviance(df1_te["claim_count"].to_numpy(), cbm1_pred, df1_te["exposure"].to_numpy())
cbm1_ae = worst_decile_ae(df1_te["claim_count"].to_numpy(), cbm1_pred, df1_te["exposure"].to_numpy())

# SHAP
t0 = time.time()
sr1 = SHAPRelativities(model=cbm1, X=X1_tr_pl, exposure=df1_tr["exposure"].to_numpy(),
                        categorical_features=feats_1)
sr1.fit()
rels1 = sr1.extract_relativities(normalise_to="base_level",
                                   base_levels={"area_code": 0, "ncd_years": 0, "has_conviction": 0})
shap1_time = time.time() - t0

area_map = {i: chr(ord("A")+i) for i in range(6)}
shap1_errors = []
shap1_ncd5 = None
shap1_conv = None
for row in rels1.iter_rows(named=True):
    feat, lev, rel = row["feature"], row["level"], row["relativity"]
    if feat == "area_code":
        true_r = TRUE_AREA_REL_1.get(area_map.get(int(float(lev)),"?"), None)
    elif feat == "ncd_years":
        n = int(float(lev))
        true_r = float(np.exp(NCD_COEF_1 * n))
        if n == 5: shap1_ncd5 = rel
    elif feat == "has_conviction":
        c = int(float(lev))
        true_r = TRUE_CONV_REL_1 if c == 1 else 1.0
        if c == 1: shap1_conv = rel
    else:
        true_r = None
    if true_r and true_r > 0:
        shap1_errors.append(abs(rel - true_r)/true_r*100)

checks1 = sr1.validate()
recon1 = checks1.get("reconstruction")

print(f"\nScenario 1 results:")
print(f"  Poisson GLM:        Gini={glm1_gini:.4f}  Dev={glm1_dev:.6f}  A/E max={glm1_ae:.3f}  "
      f"Mean|err|={np.mean(glm1_errors):.2f}%  Time={glm1_time:.2f}s")
print(f"  CatBoost+SHAP:      Gini={cbm1_gini:.4f}  Dev={cbm1_dev:.6f}  A/E max={cbm1_ae:.3f}  "
      f"Mean|err|={np.mean(shap1_errors):.2f}%  Time={cbm1_time:.1f}s")
print(f"  Gini improvement: +{(cbm1_gini - glm1_gini)*100:.2f}pp")
print(f"  Deviance reduction: {(glm1_dev - cbm1_dev)/glm1_dev*100:.1f}%")
print(f"  NCD=5 (true={np.exp(NCD_COEF_1*5):.3f}): GLM={glm1_ncd5:.3f}  SHAP={shap1_ncd5:.3f}")
print(f"  Conviction (true={TRUE_CONV_REL_1:.3f}): GLM={glm1_conv:.3f}  SHAP={shap1_conv:.3f}")
print(f"  SHAP reconstruction: {'PASS' if recon1 and recon1.passed else 'FAIL'}  "
      f"(max err: {recon1.value:.2e})" if recon1 else "")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scenario 2: Interaction DGP (vehicle_group × NCD)

# COMMAND ----------

print("=" * 70)
print("SCENARIO 2: Interaction DGP — vehicle_group × NCD")
print("=" * 70)

RNG2 = np.random.default_rng(99)
N2 = 30_000

TRUE_AREA_REL_2 = {"A": 1.00, "B": 1.12, "C": 1.25, "D": 1.43, "E": 1.68, "F": 1.95}
NCD_COEF_2 = -0.12
TRUE_CONV_REL_2 = 1.57
TRUE_VG_REL = {1: 1.00, 2: 1.35, 3: 2.10}
BASE_FREQ_2 = 0.07
INTERACTION_UPLIFT = 1.40

areas_2 = RNG2.choice(["A", "B", "C", "D", "E", "F"], N2, p=[0.25, 0.25, 0.20, 0.15, 0.10, 0.05])
ncd_2 = RNG2.integers(0, 6, N2)
conv_2 = (RNG2.random(N2) < 0.12).astype(int)
vg_2 = RNG2.choice([1, 2, 3], N2, p=[0.50, 0.35, 0.15])
exposure_2 = RNG2.uniform(0.3, 1.0, N2)

log_freq_2 = (
    np.log(BASE_FREQ_2)
    + np.array([np.log(TRUE_AREA_REL_2[a]) for a in areas_2])
    + NCD_COEF_2 * ncd_2
    + np.log(TRUE_CONV_REL_2) * conv_2
    + np.array([np.log(TRUE_VG_REL[v]) for v in vg_2])
)
interaction_flag_2 = ((vg_2 == 3) & (ncd_2 <= 1)).astype(float)
log_freq_2 += np.log(INTERACTION_UPLIFT) * interaction_flag_2
claims_2 = RNG2.poisson(np.exp(log_freq_2) * exposure_2)
area_code_2 = np.array([ord(a) - ord("A") for a in areas_2])

df2 = pl.DataFrame({
    "area": areas_2.tolist(), "area_code": area_code_2.tolist(),
    "ncd_years": ncd_2.tolist(), "has_conviction": conv_2.tolist(),
    "vehicle_group": vg_2.tolist(), "exposure": exposure_2.tolist(),
    "claim_count": claims_2.tolist(),
})

n_train_2 = int(0.70 * N2)
df2_tr = df2[:n_train_2]
df2_te = df2[n_train_2:]

print(f"Interaction group (VG=3, NCD<=1): {float(interaction_flag_2.mean()):.1%} of portfolio")

# GLM (main effects only)
t0 = time.time()
feats_glm2 = ["area", "ncd_years", "has_conviction", "vehicle_group"]
X2_tr_raw = df2_tr.select(feats_glm2).to_pandas()
X2_te_raw = df2_te.select(feats_glm2).to_pandas()
pre2 = ColumnTransformer([
    ("area", OneHotEncoder(drop="first", sparse_output=False), ["area"]),
    ("vg", OneHotEncoder(drop="first", sparse_output=False), ["vehicle_group"]),
    ("ncd", "passthrough", ["ncd_years"]),
    ("conv", "passthrough", ["has_conviction"]),
])
glm2 = Pipeline([("prep", pre2), ("model", PoissonRegressor(alpha=0, max_iter=500))])
glm2.fit(X2_tr_raw, df2_tr["claim_count"].to_numpy() / df2_tr["exposure"].to_numpy(),
         model__sample_weight=df2_tr["exposure"].to_numpy())
glm2_time = time.time() - t0

glm2_pred = glm2.predict(X2_te_raw)
glm2_gini = gini(df2_te["claim_count"].to_numpy(), glm2_pred, df2_te["exposure"].to_numpy())
glm2_dev = poisson_deviance(df2_te["claim_count"].to_numpy(), glm2_pred, df2_te["exposure"].to_numpy())
glm2_ae = worst_decile_ae(df2_te["claim_count"].to_numpy(), glm2_pred, df2_te["exposure"].to_numpy())

# CatBoost
t0 = time.time()
feats_gbm2 = ["area_code", "ncd_years", "has_conviction", "vehicle_group"]
X2_tr_pl = df2_tr.select(feats_gbm2)
X2_te_pl = df2_te.select(feats_gbm2)
pool2 = catboost.Pool(data=X2_tr_pl.to_pandas(), label=df2_tr["claim_count"].to_numpy(),
                      weight=df2_tr["exposure"].to_numpy())
cbm2 = catboost.CatBoostRegressor(loss_function="Poisson", iterations=400, learning_rate=0.05,
                                   depth=5, random_seed=99, verbose=0, allow_writing_files=False)
cbm2.fit(pool2)
cbm2_time = time.time() - t0

cbm2_pred = cbm2.predict(X2_te_pl.to_pandas())
cbm2_gini = gini(df2_te["claim_count"].to_numpy(), cbm2_pred, df2_te["exposure"].to_numpy())
cbm2_dev = poisson_deviance(df2_te["claim_count"].to_numpy(), cbm2_pred, df2_te["exposure"].to_numpy())
cbm2_ae = worst_decile_ae(df2_te["claim_count"].to_numpy(), cbm2_pred, df2_te["exposure"].to_numpy())

# SHAP
t0 = time.time()
sr2 = SHAPRelativities(model=cbm2, X=X2_tr_pl, exposure=df2_tr["exposure"].to_numpy(),
                        categorical_features=feats_gbm2)
sr2.fit()
rels2 = sr2.extract_relativities(normalise_to="base_level",
                                   base_levels={"area_code": 0, "ncd_years": 0,
                                                "has_conviction": 0, "vehicle_group": 1})
shap2_time = time.time() - t0

checks2 = sr2.validate()
recon2 = checks2.get("reconstruction")

print(f"\nScenario 2 results:")
print(f"  Poisson GLM (main effects): Gini={glm2_gini:.4f}  Dev={glm2_dev:.6f}  A/E max={glm2_ae:.3f}")
print(f"  CatBoost + SHAP:            Gini={cbm2_gini:.4f}  Dev={cbm2_dev:.6f}  A/E max={cbm2_ae:.3f}")
print(f"  Gini improvement: +{(cbm2_gini - glm2_gini)*100:.2f}pp")
print(f"  Deviance reduction: {(glm2_dev - cbm2_dev)/glm2_dev*100:.1f}%")

print(f"\n  Absorbed interaction — vehicle_group=3 SHAP relativity vs true main effect:")
vg3_row = rels2.filter((pl.col("feature") == "vehicle_group") & (pl.col("level") == "3")).row(0, named=True)
print(f"  VG=3: SHAP={vg3_row['relativity']:.3f}  True main={TRUE_VG_REL[3]:.3f}  "
      f"True+interaction=~{TRUE_VG_REL[3]*INTERACTION_UPLIFT:.3f} (for NCD<=1 policyholders)")
ncd0_row = rels2.filter((pl.col("feature") == "ncd_years") & (pl.col("level") == "0")).row(0, named=True)
ncd1_row = rels2.filter((pl.col("feature") == "ncd_years") & (pl.col("level") == "1")).row(0, named=True)
print(f"  NCD=0: SHAP={ncd0_row['relativity']:.3f}  NCD=1: SHAP={ncd1_row['relativity']:.3f}  "
      f"(baseline=1.0 by construction)")
print(f"  SHAP reconstruction: {'PASS' if recon2 and recon2.passed else 'FAIL'}  "
      f"(max err: {recon2.value:.2e})" if recon2 else "")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("=" * 70)
print("BENCHMARK SUMMARY — numbers for README")
print("=" * 70)

print(f"""
SCENARIO 1: Clean log-linear DGP (20k policies, 3 features)
  Poisson GLM:   Gini={glm1_gini:.4f}  Dev={glm1_dev:.6f}  A/E max={glm1_ae*100:.1f}%  Mean|err|={np.mean(glm1_errors):.2f}%
  CatBoost+SHAP: Gini={cbm1_gini:.4f}  Dev={cbm1_dev:.6f}  A/E max={cbm1_ae*100:.1f}%  Mean|err|={np.mean(shap1_errors):.2f}%
  Gini improvement: +{(cbm1_gini - glm1_gini)*100:.2f}pp
  Deviance reduction: {(glm1_dev - cbm1_dev)/glm1_dev*100:.1f}%
  A/E improvement: {(glm1_ae - cbm1_ae)/glm1_ae*100:.1f}%
  NCD=5 true={np.exp(NCD_COEF_1*5):.3f}: GLM={glm1_ncd5:.3f} (err {(glm1_ncd5-np.exp(NCD_COEF_1*5))/np.exp(NCD_COEF_1*5)*100:+.1f}%)  SHAP={shap1_ncd5:.3f} (err {(shap1_ncd5-np.exp(NCD_COEF_1*5))/np.exp(NCD_COEF_1*5)*100:+.1f}%)
  Conviction true={TRUE_CONV_REL_1:.3f}: GLM={glm1_conv:.3f}  SHAP={shap1_conv:.3f}

SCENARIO 2: Interaction DGP (30k policies, 4 features, VG×NCD interaction)
  Poisson GLM:   Gini={glm2_gini:.4f}  Dev={glm2_dev:.6f}  A/E max={glm2_ae*100:.1f}%
  CatBoost+SHAP: Gini={cbm2_gini:.4f}  Dev={cbm2_dev:.6f}  A/E max={cbm2_ae*100:.1f}%
  Gini improvement: +{(cbm2_gini - glm2_gini)*100:.2f}pp
  Deviance reduction: {(glm2_dev - cbm2_dev)/glm2_dev*100:.1f}%
  A/E improvement: {(glm2_ae - cbm2_ae)/glm2_ae*100:.1f}%
  Absorbed interaction in VG=3: SHAP={vg3_row['relativity']:.3f} vs true main={TRUE_VG_REL[3]:.3f}

TIMING (serverless):
  GLM (Scenario 1):   {glm1_time:.2f}s
  CatBoost (Sc. 1):   {cbm1_time:.1f}s
  SHAP (Sc. 1):       {shap1_time:.1f}s
  CatBoost (Sc. 2):   {cbm2_time:.1f}s
  SHAP (Sc. 2):       {shap2_time:.1f}s
""")

# COMMAND ----------

# Exit with key numbers for programmatic capture
import json
results = {
    "sc1_glm_gini": round(glm1_gini, 4),
    "sc1_shap_gini": round(cbm1_gini, 4),
    "sc1_glm_dev": round(glm1_dev, 6),
    "sc1_shap_dev": round(cbm1_dev, 6),
    "sc1_glm_ae_max": round(glm1_ae, 4),
    "sc1_shap_ae_max": round(cbm1_ae, 4),
    "sc1_glm_mean_err": round(float(glm1_errors.__class__(glm1_errors).mean() if hasattr(glm1_errors, 'mean') else sum(glm1_errors)/len(glm1_errors)), 2),
    "sc1_shap_mean_err": round(float(sum(shap1_errors)/len(shap1_errors)), 2),
    "sc1_gini_gap_pp": round((cbm1_gini - glm1_gini) * 100, 2),
    "sc1_dev_reduction_pct": round((glm1_dev - cbm1_dev) / glm1_dev * 100, 1),
    "sc1_ae_improvement_pct": round((glm1_ae - cbm1_ae) / glm1_ae * 100, 1),
    "sc1_ncd5_true": round(float(__import__('numpy').exp(-0.12 * 5)), 3),
    "sc1_ncd5_glm": round(glm1_ncd5, 3),
    "sc1_ncd5_shap": round(shap1_ncd5, 3),
    "sc1_conv_glm": round(glm1_conv, 3),
    "sc1_conv_shap": round(shap1_conv, 3),
    "sc1_glm_time_s": round(glm1_time, 2),
    "sc1_cbm_time_s": round(cbm1_time, 1),
    "sc1_shap_time_s": round(shap1_time, 1),
    "sc2_glm_gini": round(glm2_gini, 4),
    "sc2_shap_gini": round(cbm2_gini, 4),
    "sc2_glm_dev": round(glm2_dev, 6),
    "sc2_shap_dev": round(cbm2_dev, 6),
    "sc2_glm_ae_max": round(glm2_ae, 4),
    "sc2_shap_ae_max": round(cbm2_ae, 4),
    "sc2_gini_gap_pp": round((cbm2_gini - glm2_gini) * 100, 2),
    "sc2_dev_reduction_pct": round((glm2_dev - cbm2_dev) / glm2_dev * 100, 1),
    "sc2_ae_improvement_pct": round((glm2_ae - cbm2_ae) / glm2_ae * 100, 1),
    "sc2_vg3_shap": round(vg3_row["relativity"], 3),
    "sc2_vg3_true_main": 2.10,
    "sc2_cbm_time_s": round(cbm2_time, 1),
    "sc2_shap_time_s": round(shap2_time, 1),
}
dbutils.notebook.exit(json.dumps(results))
