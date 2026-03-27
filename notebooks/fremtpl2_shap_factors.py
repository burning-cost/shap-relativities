# Databricks notebook source
# DISCLAIMER: freMTPL2 is French motor data (OpenML dataset 41214).
# It is used here for methodology validation only — to demonstrate that
# shap-relativities extracts interpretable GLM-style factor tables from a
# CatBoost GBM on real insurance data.
# It is NOT UK market data and should not be used to draw conclusions about
# UK motor pricing structure.
#
# Dataset: 678,013 French motor third-party liability policies.
# Source: Noll, Salzmann, Wüthrich (2018) — Case Study: French Motor Third-Party
#         Liability Claims.
# Variables used:
#   ClaimNb    — number of claims (target, Poisson frequency)
#   Exposure   — fraction of year at risk (offset)
#   Area       — categorical area code (A–F)
#   VehPower   — vehicle power (4–15)
#   VehAge     — vehicle age in years (0–100)
#   DrivAge    — driver age in years (18–100)
#   BonusMalus — bonus-malus level (50–350)
#   VehGas     — fuel type (Diesel / Regular)
#   Density    — population density of driver's commune
#
# Library: shap-relativities  |  Date: 2026-03-27

# COMMAND ----------

%pip install "catboost>=1.2" "shap>=0.45" "scikit-learn>=1.3" \
    "polars>=1.0" "pandas>=2.0" "shap-relativities" --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json, time, warnings
import numpy as np
import pandas as pd
import polars as pl
from sklearn.datasets import fetch_openml
from catboost import CatBoostRegressor, Pool
from shap_relativities import SHAPRelativities

warnings.filterwarnings("ignore")
SEED = 42

print("freMTPL2 — CatBoost Poisson + SHAP Relativities")
print("DISCLAIMER: French motor data. Methodology validation only. Not UK market data.")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Load, prepare, subsample
# ---------------------------------------------------------------------------

raw = fetch_openml(data_id=41214, as_frame=True, parser="auto")
df = raw.frame.copy()
df.columns = [c.strip() for c in df.columns]

FEATURES    = ["DrivAge", "VehAge", "BonusMalus", "VehPower", "Area", "Density", "VehGas"]
CAT_FEATURES = ["Area", "VehGas"]

y_all   = df["ClaimNb"].astype(float).values
exp_all = df["Exposure"].astype(float).values.clip(min=1e-6)
X_all   = df[FEATURES].copy()
for col in CAT_FEATURES:
    X_all[col] = X_all[col].astype(str)
for col in [c for c in FEATURES if c not in CAT_FEATURES]:
    X_all[col] = X_all[col].astype(float)

# Random subsample to 75K rows
idx = np.random.default_rng(SEED).choice(len(df), size=75_000, replace=False)
X, y, exp = X_all.iloc[idx].reset_index(drop=True), y_all[idx], exp_all[idx]

N_TRAIN   = int(0.75 * len(X))
X_train, X_test = X.iloc[:N_TRAIN], X.iloc[N_TRAIN:]
y_train, y_test = y[:N_TRAIN], y[N_TRAIN:]
exp_train, exp_test = exp[:N_TRAIN], exp[N_TRAIN:]

print(f"Full dataset: {len(df):,} rows  |  Subsample: 75,000  |  Train: {N_TRAIN:,}  |  Test: {len(y_test):,}")
print(f"Train frequency: {y_train.sum()/exp_train.sum():.4f}  |  Test: {y_test.sum()/exp_test.sum():.4f}")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Fit CatBoost Poisson model
# ---------------------------------------------------------------------------
# loss_function="Poisson" uses a log link — required for SHAP relativities.
# Log(exposure) is passed as baseline (the offset). CatBoost handles Area and
# VehGas as native categoricals; no encoding needed.

cat_idx = [FEATURES.index(f) for f in CAT_FEATURES]

train_pool = Pool(X_train, label=y_train, cat_features=cat_idx,
                  baseline=np.log(exp_train))
test_pool  = Pool(X_test,  cat_features=cat_idx, baseline=np.log(exp_test))

t0 = time.time()
model = CatBoostRegressor(loss_function="Poisson", iterations=500, depth=5,
                          learning_rate=0.05, random_seed=SEED, verbose=100)
model.fit(train_pool)
fit_time = time.time() - t0

pred_test = model.predict(test_pool)

# Unit Poisson deviance
def poisson_dev(yt, yp, w=None):
    yp = np.maximum(yp, 1e-10)
    d = 2 * (np.where(yt > 0, yt * np.log(yt / yp), 0) - (yt - yp))
    return float(np.average(d, weights=w) if w is not None else np.mean(d))

print(f"Fit: {fit_time:.0f}s  |  Test deviance: {poisson_dev(y_test, pred_test):.6f}"
      f"  (wt): {poisson_dev(y_test, pred_test, exp_test):.6f}")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Extract SHAP relativities
# ---------------------------------------------------------------------------
# SHAPRelativities.fit() runs shap.TreeExplainer(model_output="raw").
# Raw output for CatBoost Poisson is log(rate), so SHAP values are in log
# space. extract_relativities() exponentiates to give multiplicative factors.
# exposure=exp_train weights by policy years — standard actuarial convention.

t0 = time.time()
sr = SHAPRelativities(model=model, X=X_train, exposure=exp_train,
                      categorical_features=CAT_FEATURES)
sr.fit()
shap_time = time.time() - t0

area_base   = X_train["Area"].value_counts().idxmax()
vehgas_base = X_train["VehGas"].value_counts().idxmax()

rels = sr.extract_relativities(
    normalise_to="base_level",
    base_levels={"Area": area_base, "VehGas": vehgas_base},
    ci_method="clt",
)

print(f"SHAP time: {shap_time:.0f}s  |  Baseline rate: {sr.baseline():.4f}/yr")
print(f"Relativity table: {rels.shape[0]} rows × {rels.shape[1]} cols")
print(f"Features: {rels['feature'].unique().to_list()}")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Factor tables: DrivAge, Area, VehGas
# ---------------------------------------------------------------------------

def print_table(df: pl.DataFrame, title: str, cast_float: bool = False) -> None:
    print(f"\n--- {title} ---")
    tbl = df
    if cast_float:
        tbl = tbl.with_columns(pl.col("level").cast(pl.Float64)).sort("level")
    print(f"{'level':>10}  {'relativity':>12}  {'lower_ci':>10}  {'upper_ci':>10}  {'n_obs':>7}")
    print("-" * 56)
    rows = tbl[::max(1, len(tbl)//20)] if cast_float else tbl
    for r in rows.iter_rows(named=True):
        print(f"{str(r['level']):>10}  {float(r['relativity']):>12.4f}"
              f"  {float(r['lower_ci']):>10.4f}  {float(r['upper_ci']):>10.4f}"
              f"  {int(r['n_obs']):>7d}")


print_table(rels.filter(pl.col("feature") == "DrivAge"), "DrivAge (sample, sorted)", cast_float=True)
print_table(rels.filter(pl.col("feature") == "Area").sort("relativity"),   "Area")
print_table(rels.filter(pl.col("feature") == "VehGas").sort("relativity"), "VehGas")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Validate SHAP reconstruction
# ---------------------------------------------------------------------------
# efficiency axiom: exp(sum(SHAP) + expected_value) must recover predictions.

checks = sr.validate()
for name, result in checks.items():
    status = "PASS" if result.passed else "FAIL"
    print(f"  [{status}] {name}: {result.message}")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Feature spread summary
# ---------------------------------------------------------------------------

summary = (
    rels.group_by("feature")
    .agg([pl.col("relativity").min().alias("min"),
          pl.col("relativity").max().alias("max"),
          (pl.col("relativity").max() - pl.col("relativity").min()).alias("range"),
          pl.col("level").n_unique().alias("n_levels")])
    .sort("range", descending=True)
)
print("\n--- Feature spread (ordered by relativity range) ---")
print(summary.to_pandas().to_string(index=False))

# COMMAND ----------

results = {
    "dataset": "freMTPL2freq (OpenML 41214)",
    "n_train": int(N_TRAIN), "n_test": int(len(y_test)),
    "fit_seconds": float(fit_time), "shap_seconds": float(shap_time),
    "test_deviance": float(poisson_dev(y_test, pred_test)),
    "test_deviance_wt": float(poisson_dev(y_test, pred_test, exp_test)),
    "baseline_rate": float(sr.baseline()),
    "reconstruction_passed": bool(checks["reconstruction"].passed),
    "reconstruction_max_error": float(checks["reconstruction"].value),
    "disclaimer": "French motor data — methodology validation only, not UK market data",
}
print(json.dumps(results, indent=2))
dbutils.notebook.exit(json.dumps(results))
