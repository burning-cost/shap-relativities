# Databricks notebook source
# MAGIC %md
# MAGIC # SHAPInference: Statistically Valid SHAP Feature Importance
# MAGIC
# MAGIC ## The problem this solves
# MAGIC
# MAGIC Every pricing actuary has produced a SHAP importance bar chart. The chart shows
# MAGIC that postcode is the 3rd most important feature and NCD is 4th. The pricing
# MAGIC committee asks: "Are you sure postcode outranks NCD, or could it be the other
# MAGIC way round if we had more data?"
# MAGIC
# MAGIC Until now, there was no rigorous answer to that question. Standard SHAP gives
# MAGIC point estimates with no uncertainty. Bootstrap on SHAP values is invalid because
# MAGIC it ignores the estimation uncertainty in the underlying model.
# MAGIC
# MAGIC `SHAPInference` implements the de-biased U-statistic estimator from Whitehouse,
# MAGIC Sawarni, Syrgkanis (2026), arXiv:2602.10532. It provides asymptotically valid
# MAGIC confidence intervals for theta_p = E[|phi_a(X)|^p] — the population mean of
# MAGIC the p-th power of SHAP values — for each feature.
# MAGIC
# MAGIC **Why this matters for UK insurance:**
# MAGIC - FCA Consumer Duty (PS22/9) requires evidence that pricing models produce fair
# MAGIC   value. "Postcode is a material driver" is now a testable statistical claim.
# MAGIC - Feature ranking governance: overlapping CIs mean rankings are not confirmed.
# MAGIC - Model refresh decisions: is the drop in feature X's importance statistically
# MAGIC   significant between v1 and v2?
# MAGIC
# MAGIC ## What this notebook demonstrates
# MAGIC
# MAGIC 1. Basic usage: fit SHAPInference on synthetic data, read importance_table()
# MAGIC 2. Visualisation: importance bar chart with CI error bars
# MAGIC 3. Ranking CI: formal test of "is feature A more important than feature B?"
# MAGIC 4. Coverage simulation: verify the 95% CI achieves ~95% empirical coverage
# MAGIC 5. p=1 vs p=2: mean absolute SHAP vs mean squared SHAP
# MAGIC 6. SE scaling: verify the 1/sqrt(n) rate
# MAGIC 7. Integration with SHAPRelativities: full end-to-end workflow

# COMMAND ----------
# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %pip install "shap-relativities[all]" --quiet

# COMMAND ----------

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import warnings

from shap_relativities import SHAPInference

print("shap-relativities imported successfully")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Synthetic Data: Linear DGP with Known True Theta
# MAGIC
# MAGIC We use a linear model where SHAP values are available analytically.
# MAGIC For Y = beta_1*X_1 + beta_2*X_2 + epsilon, the SHAP value for feature j
# MAGIC is phi_j = beta_j * X_j (exact, by additivity).
# MAGIC
# MAGIC The true theta_2 (mean squared SHAP) is:
# MAGIC   theta_2,j = E[phi_j^2] = beta_j^2 * Var(X_j) = beta_j^2 (if X_j ~ N(0,1))
# MAGIC
# MAGIC Feature importances (true theta_2):
# MAGIC   vehicle_age:    0.3^2 = 0.090   (most important)
# MAGIC   ncd_years:      0.25^2 = 0.0625
# MAGIC   driver_age:     0.20^2 = 0.040
# MAGIC   area_score:     0.15^2 = 0.0225
# MAGIC   conviction_ind: 0.05^2 = 0.0025  (least important)

# COMMAND ----------

rng = np.random.default_rng(42)
N = 5000

betas = np.array([0.30, 0.25, 0.20, 0.15, 0.05])
feature_names = ["vehicle_age", "ncd_years", "driver_age", "area_score", "conviction_ind"]

# X ~ N(0,1) for each feature
X = rng.normal(0, 1, size=(N, len(betas)))

# phi_j = beta_j * X_j (exact for linear models)
shap_values = X * betas[np.newaxis, :]

# y = sum of contributions + Poisson noise
mu = np.exp(shap_values.sum(axis=1))  # log-linear mean
y = rng.poisson(mu).astype(float)

# True theta_2 = beta_j^2 (since Var(X_j)=1)
true_theta2 = betas ** 2
print("True theta_2 by feature:")
for name, t in zip(feature_names, true_theta2):
    print(f"  {name}: {t:.4f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Fit SHAPInference

# COMMAND ----------

si = SHAPInference(
    shap_values=shap_values,
    y=y,
    feature_names=feature_names,
    p=2.0,
    n_folds=5,
    random_state=0,
)
si.fit()
print(repr(si))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Importance Table

# COMMAND ----------

tbl = si.importance_table()
print(tbl)

# COMMAND ----------
# MAGIC %md
# MAGIC Checking that true theta_2 is within all CIs:

# COMMAND ----------

print("\nCoverage check (true theta vs 95% CI):")
print(f"{'Feature':<20} {'True':>8} {'hat':>8} {'Lower':>8} {'Upper':>8} {'Covered':>8}")
print("-" * 68)
for i, name in enumerate(feature_names):
    row = tbl.filter(pl.col("feature") == name)
    theta = row["theta_hat"][0]
    lower = row["theta_lower"][0]
    upper = row["theta_upper"][0]
    true_val = true_theta2[i]
    covered = lower <= true_val <= upper
    print(f"{name:<20} {true_val:>8.4f} {theta:>8.4f} {lower:>8.4f} {upper:>8.4f} {'YES' if covered else 'NO':>8}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Importance Bar Chart with CI Error Bars

# COMMAND ----------

fig, ax = plt.subplots(figsize=(8, 5))
si.plot_importance(ax=ax, sort=True)
ax.set_title("Global SHAP Feature Importance (p=2, 95% CI)\nSynthetic UK Motor Data", fontsize=12)
plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Ranking CI: Formal Test of Feature Ordering
# MAGIC
# MAGIC The SHAP importance bar chart shows vehicle_age > ncd_years, but is this
# MAGIC statistically confirmed? ranking_ci() tests H0: theta_a = theta_b against
# MAGIC H1: theta_a > theta_b.

# COMMAND ----------

print("=== Test: vehicle_age vs ncd_years ===")
result = si.ranking_ci("vehicle_age", "ncd_years")
for k, v in result.items():
    print(f"  {k}: {v:.4f}")

print("\n=== Test: conviction_ind vs driver_age (should NOT reject) ===")
result_weak = si.ranking_ci("conviction_ind", "driver_age")
for k, v in result_weak.items():
    print(f"  {k}: {v:.4f}")

# COMMAND ----------
# MAGIC %md
# MAGIC The test for vehicle_age > ncd_years should have p_value << 0.05.
# MAGIC The test for conviction_ind > driver_age should NOT reject (conviction_ind
# MAGIC has much lower true theta).

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Coverage Simulation: 100 Replicates
# MAGIC
# MAGIC This is the gold-standard validation. Run 100 independent datasets and check
# MAGIC that the 95% CI contains the true theta in roughly 95% of cases.
# MAGIC
# MAGIC Note: this takes ~3–5 minutes on serverless compute with n=2000, 5 folds.

# COMMAND ----------

n_reps = 100
n_sim = 2000
betas_sim = np.array([0.3, 0.2])
true_theta2_sim = betas_sim ** 2
covered = np.zeros((n_reps, 2), dtype=bool)

for rep in range(n_reps):
    rng_rep = np.random.default_rng(rep)
    X_rep = rng_rep.normal(0, 1, size=(n_sim, 2))
    sv_rep = X_rep * betas_sim[np.newaxis, :]
    y_rep = rng_rep.poisson(np.exp(sv_rep.sum(axis=1))).astype(float)

    si_rep = SHAPInference(sv_rep, y_rep, ["x1", "x2"], p=2.0, n_folds=5, random_state=rep)
    si_rep.fit()
    t = si_rep.importance_table()

    for j, fname in enumerate(["x1", "x2"]):
        row = t.filter(pl.col("feature") == fname)
        covered[rep, j] = row["theta_lower"][0] <= true_theta2_sim[j] <= row["theta_upper"][0]

print("Coverage simulation results (n=2000, 100 replicates, p=2, nominal 95%):")
for j, fname in enumerate(["x1", "x2"]):
    empirical_cov = covered[:, j].mean()
    print(f"  {fname}: empirical coverage = {empirical_cov:.2%}")

print("\nNote: empirical coverage should be >= 90% for a valid 95% CI.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. p=1 vs p=2: Mean Absolute SHAP vs Mean Squared SHAP
# MAGIC
# MAGIC p=2 (mean squared SHAP) has cleaner asymptotic theory. p=1 (mean absolute
# MAGIC SHAP) is the standard bar chart metric but requires smoothing for valid
# MAGIC inference. Both should rank features the same way.

# COMMAND ----------

print("Fitting p=1 (mean absolute SHAP) with smoothing...")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    si_p1 = SHAPInference(shap_values, y, feature_names, p=1.0, n_folds=5, random_state=0)
    si_p1.fit()
    if w:
        print(f"Warning: {w[0].message}")

tbl_p1 = si_p1.importance_table()

print("\nRanking comparison p=1 vs p=2:")
print(f"{'Feature':<20} {'p=1 rank':>10} {'p=2 rank':>10}")
print("-" * 44)
tbl_p2 = si.importance_table()
rank_map_p2 = {row["feature"]: row["rank"] for row in tbl_p2.iter_rows(named=True)}
for row in tbl_p1.iter_rows(named=True):
    print(f"{row['feature']:<20} {row['rank']:>10} {rank_map_p2[row['feature']]:>10}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. SE Scaling with n
# MAGIC
# MAGIC The SE should decrease at the 1/sqrt(n) rate. We verify this holds.

# COMMAND ----------

ses_by_n = {}
for n_val in [500, 1000, 2000, 5000]:
    rng_n = np.random.default_rng(100)
    X_n = rng_n.normal(0, 1, size=(n_val, 2))
    sv_n = X_n * np.array([0.3, 0.2])
    y_n = rng_n.poisson(np.exp(sv_n.sum(axis=1))).astype(float)

    si_n = SHAPInference(sv_n, y_n, ["x1", "x2"], p=2.0, n_folds=5, random_state=0)
    si_n.fit()
    t_n = si_n.importance_table()
    ses_by_n[n_val] = t_n.filter(pl.col("feature") == "x1")["se"][0]

print("SE for x1 by sample size:")
prev_se = None
for n_val, se in sorted(ses_by_n.items()):
    ratio = f"(ratio to prev: {prev_se/se:.2f}x)" if prev_se else ""
    print(f"  n={n_val:>6}: SE = {se:.5f} {ratio}")
    prev_se = se

print("\nExpected: doubling n should give ~sqrt(2)=1.41x reduction in SE.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9. Influence Matrix: Identifying High-Leverage Policies
# MAGIC
# MAGIC The influence_matrix property gives each observation's contribution to
# MAGIC each feature's importance estimate. High |rho[i, j]| = high leverage for
# MAGIC feature j's estimate — useful for governance reviews.

# COMMAND ----------

rho = si.influence_matrix
print(f"Influence matrix shape: {rho.shape}")

# Top 5 most influential policies for vehicle_age
j_veh = feature_names.index("vehicle_age")
influence_veh = np.abs(rho[:, j_veh])
top_idx = np.argsort(-influence_veh)[:5]

print("\nTop 5 most influential policies for vehicle_age importance estimate:")
print(f"{'Policy idx':>12} {'|rho|':>10} {'SHAP phi':>10} {'y':>8}")
for idx in top_idx:
    print(f"{idx:>12} {influence_veh[idx]:>10.4f} {shap_values[idx, j_veh]:>10.4f} {y[idx]:>8.1f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10. Full End-to-End Workflow with SHAPRelativities
# MAGIC
# MAGIC In production, you fit a CatBoost model, compute interventional SHAP via
# MAGIC SHAPRelativities, then pass the SHAP matrix directly to SHAPInference.
# MAGIC This is the intended workflow.

# COMMAND ----------

from shap_relativities import SHAPRelativities
from catboost import CatBoostRegressor
import pandas as pd

# Synthetic data: 3 features, Poisson frequency model
N_prod = 3000
rng_prod = np.random.default_rng(200)

age = rng_prod.integers(17, 80, size=N_prod).astype(float)
ncd = rng_prod.integers(0, 6, size=N_prod).astype(float)
veh = rng_prod.integers(1, 51, size=N_prod).astype(float)

log_mu = -3.0 + 0.3 * np.log(age / 40) - 0.1 * ncd + 0.02 * veh
y_prod = rng_prod.poisson(np.exp(log_mu)).astype(float)

X_prod = pd.DataFrame({"age": age, "ncd": ncd, "vehicle_group": veh})

# Fit a CatBoost Poisson model
model = CatBoostRegressor(
    loss_function="Poisson",
    iterations=200,
    learning_rate=0.05,
    depth=4,
    verbose=0,
)
model.fit(X_prod, y_prod)

print("CatBoost model fitted.")

# COMMAND ----------

# Extract SHAP values via SHAPRelativities
sr = SHAPRelativities(model, X_prod)
sr.fit()
sv_prod = sr.shap_values()
print(f"SHAP matrix shape: {sv_prod.shape}")
print(f"Feature names from SHAPRelativities: {sr.feature_names_}")

# COMMAND ----------

# Now apply SHAPInference to the SHAP matrix
si_prod = SHAPInference(
    shap_values=sv_prod,
    y=y_prod,
    feature_names=list(X_prod.columns),
    p=2.0,
    n_folds=5,
    random_state=0,
)
si_prod.fit()
print(si_prod.importance_table())

# COMMAND ----------

fig, ax = plt.subplots(figsize=(7, 4))
si_prod.plot_importance(ax=ax)
ax.set_title("Feature Importance with 95% CI\nCatBoost Poisson Motor Model", fontsize=11)
plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | What | How |
# MAGIC |---|---|
# MAGIC | Asymptotically valid CIs on theta_p = E[|phi|^p] | `SHAPInference.fit()` |
# MAGIC | Importance table with CIs, SE, rank, p-value | `si.importance_table()` |
# MAGIC | Test H0: theta_a = theta_b | `si.ranking_ci("a", "b")` |
# MAGIC | Bar chart with CI error bars | `si.plot_importance()` |
# MAGIC | Per-observation leverage scores | `si.influence_matrix` |
# MAGIC
# MAGIC **Coverage:** Empirical 95% CI coverage is typically 93–97% for n >= 2000
# MAGIC with gradient-boosting nuisance models.
# MAGIC
# MAGIC **Valid use:** Requires interventional SHAP (not path-dependent TreeSHAP).
# MAGIC Use SHAPRelativities with a representative background sample.
# MAGIC
# MAGIC **Reference:** Whitehouse, Sawarni, Syrgkanis (2026), arXiv:2602.10532.
