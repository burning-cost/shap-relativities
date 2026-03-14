# shap-relativities
[![Tests](https://github.com/burning-cost/shap-relativities/actions/workflows/tests.yml/badge.svg)](https://github.com/burning-cost/shap-relativities/actions/workflows/tests.yml)

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green) ![PyPI](https://img.shields.io/pypi/v/shap-relativities)

Extract rating relativities from gradient boosting models using SHAP values.

---

Every UK pricing team we've spoken to has the same problem: a GBM sitting on a server outperforming the production GLM, but nobody can get the relativities out of it. The regulator wants a factor table. Radar needs an import file. The head of pricing wants to challenge the model in terms they recognise.

So the GBM sits in a notebook. The GLM goes to production.

`shap-relativities` closes that gap. It extracts multiplicative rating relativities from CatBoost models using SHAP values - the same format as `exp(beta)` from a GLM, with confidence intervals, exposure weighting, and a validation check that the numbers actually reconstruct the model's predictions.

**Output is a Polars DataFrame.** The library accepts either Polars or pandas DataFrames as input, and returns Polars. Pandas is a bridge dependency: shap's TreeExplainer uses it internally, so it is still installed with the `[ml]` extra.

**Blog post:** [Extracting Rating Relativities from GBMs with SHAP](https://burning-cost.github.io/2026/03/05/extracting-rating-relativities-from-gbms-with-shap/) - worked example, the maths, and a discussion of limitations for presenting to regulators and pricing committees.

---

## Installation

```bash
uv add "shap-relativities[all]"
# or
pip install "shap-relativities[all]"
```

Or pick what you need:

```bash
uv add "shap-relativities[ml]"    # shap + catboost + scikit-learn + pandas bridge
uv add "shap-relativities[plot]"  # matplotlib for plots
uv add shap-relativities          # core only (polars, numpy, scipy)
```

---

## Quick start

Train a Poisson CatBoost model on synthetic UK motor data and extract relativities that can be compared to the known true parameters:

```python
import polars as pl
import catboost
from shap_relativities import SHAPRelativities
from shap_relativities.datasets.motor import load_motor, TRUE_FREQ_PARAMS

# Synthetic UK motor portfolio - 50k policies, known DGP
# load_motor() returns a Polars DataFrame
df = load_motor(n_policies=50_000, seed=42)
df = df.with_columns([
    ((pl.col("conviction_points") > 0).cast(pl.Int32)).alias("has_convictions"),
    pl.col("area").replace({"A": "0", "B": "1", "C": "2", "D": "3", "E": "4", "F": "5"})
      .cast(pl.Int32).alias("area_code"),
])

features = ["area_code", "ncd_years", "has_convictions"]
X = df.select(features)

# Train a Poisson frequency model with CatBoost
# CatBoost requires a pandas Pool for training - the bridge conversion is explicit
pool = catboost.Pool(
    data=X.to_pandas(),
    label=df["claim_count"].to_numpy(),
    weight=df["exposure"].to_numpy(),
)
model = catboost.CatBoostRegressor(
    loss_function="Poisson",
    iterations=300,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
    verbose=0,
)
model.fit(pool)

# Extract relativities - pass the Polars DataFrame directly
sr = SHAPRelativities(
    model=model,
    X=X,                           # Polars DataFrame
    exposure=df["exposure"],       # Polars Series
    categorical_features=features,
)
sr.fit()

rels = sr.extract_relativities(
    normalise_to="base_level",
    base_levels={"area_code": 0, "ncd_years": 0, "has_convictions": 0},
)
print(rels.select(["feature", "level", "relativity", "lower_ci", "upper_ci"]))
```

Output (approximately - the GBM recovers the known DGP):

```
              feature level  relativity  lower_ci  upper_ci
0           area_code     0       1.000     1.000     1.000
1           area_code     1       1.108     1.060     1.159
2           area_code     2       1.227     1.178     1.278
3           area_code     3       1.427     1.369     1.487
4           area_code     4       1.667     1.596     1.741
5           area_code     5       1.934     1.841     2.032
6           ncd_years     0       1.000     1.000     1.000
7           ncd_years     1       0.882     0.851     0.913
8           ncd_years     2       0.780     0.750     0.811
9           ncd_years     3       0.683     0.656     0.712
10          ncd_years     4       0.612     0.585     0.641
11          ncd_years     5       0.549     0.521     0.578
12    has_convictions     0       1.000     1.000     1.000
13    has_convictions     1       1.568     1.489     1.651
```

The true DGP NCD coefficient is -0.12, so NCD=5 vs NCD=0 should give `exp(-0.6) ≈ 0.549`. That's exactly what we get. Conviction relativity should be close to `exp(0.45) ≈ 1.57`.

For one-liners, use the convenience function:

```python
from shap_relativities import extract_relativities

rels = extract_relativities(
    model=model,
    X=X,
    exposure=df["exposure"],
    categorical_features=features,
    base_levels={"area_code": 0, "ncd_years": 0, "has_convictions": 0},
)
```

---

## Why CatBoost?

For insurance pricing, CatBoost has two advantages over alternatives:

1. **Native categoricals.** CatBoost handles string and integer categorical features without encoding. You pass area band "A"-"F" directly; no label encoding step that obscures the feature's meaning in the relativity table.

2. **Ordered boosting.** CatBoost's default training algorithm reduces target leakage from high-cardinality categoricals, which is relevant for vehicle group (50 levels) or postcode sector.

---

## The maths

For a Poisson GBM with log link, SHAP values are additive in log space:

```
log(mu_i) = expected_value + SHAP_area_i + SHAP_ncd_i + SHAP_convictions_i + ...
```

Every prediction is fully decomposed into per-feature contributions. To get a multiplicative relativity for `area_code = 3` relative to `area_code = 0`:

1. For each policy with `area_code = 3`, extract its SHAP value for `area_code`. Take the exposure-weighted mean across all such policies.
2. Do the same for `area_code = 0`.
3. Relativity = `exp(mean_shap(3) - mean_shap(0))`.

This is directly analogous to `exp(beta_3 - beta_0)` from a GLM. The base level gets relativity 1.0 by construction.

CLT confidence intervals:

```
SE = shap_std / sqrt(n_obs)
CI = exp(mean_shap ± z * SE - base_shap)
```

These quantify data uncertainty - how precisely we've estimated each level's mean SHAP contribution given the portfolio. They do not capture model uncertainty from the GBM fitting process.

---

## Validation

Before trusting extracted relativities, run `validate()`:

```python
checks = sr.validate()

print(checks["reconstruction"])
# CheckResult(passed=True, value=8.3e-06,
#   message='Max absolute reconstruction error: 8.3e-06.')

print(checks["sparse_levels"])
# CheckResult(passed=False, value=4.0,
#   message='4 factor level(s) have fewer than 30 observations. ...')
```

The reconstruction check verifies that `exp(shap_values.sum(axis=1) + expected_value)` matches the model's predictions to within 1e-4. If this fails, the explainer was constructed incorrectly - almost always a mismatch between the model's objective and the SHAP output type.

The sparse levels check flags categories where CLT CIs will be unreliable. 30 observations is the CLT rule of thumb; treat the intervals for flagged levels with caution.

---

## Continuous features

For continuous features (driver age, vehicle age, annual mileage), aggregation by level produces per-observation SHAP values rather than group means. Use `extract_continuous_curve()` for a smoothed relativity curve:

```python
age_curve = sr.extract_continuous_curve(
    feature="driver_age",
    n_points=100,
    smooth_method="loess",   # or "isotonic" for monotone
)
# Returns a Polars DataFrame: feature_value, relativity, lower_ci, upper_ci
```

`smooth_method="isotonic"` enforces monotonicity via isotonic regression - useful when you have a strong prior that the relativity is one-directional (younger drivers are higher risk, more mileage is more exposure).

---

## API reference

### `SHAPRelativities`

```python
SHAPRelativities(
    model,                                     # CatBoost model
    X: pl.DataFrame | pd.DataFrame,            # feature matrix (Polars preferred)
    exposure: pl.Series | pd.Series | None = None,  # earned policy years
    categorical_features: list[str] | None = None,
    continuous_features: list[str] | None = None,
    feature_perturbation: str = "tree_path_dependent",  # or "interventional"
    background_data: pl.DataFrame | pd.DataFrame | None = None,
    n_background_samples: int = 1000,
    annualise_exposure: bool = True,
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `.fit()` | `self` | Compute SHAP values. Must be called before extraction. |
| `.extract_relativities(normalise_to, base_levels, ci_method, ci_level)` | `pl.DataFrame` | Main output: one row per (feature, level). |
| `.extract_continuous_curve(feature, n_points, smooth_method)` | `pl.DataFrame` | Smoothed relativity curve for a continuous feature. |
| `.validate()` | `dict[str, CheckResult]` | Diagnostic checks: reconstruction, feature coverage, sparse levels. |
| `.baseline()` | `float` | `exp(expected_value)` - the base rate in prediction space. |
| `.shap_values()` | `np.ndarray` | Raw SHAP values, shape `(n_obs, n_features)`. |
| `.plot_relativities(features, show_ci, figsize)` | None | Bar charts (categorical) and line charts (continuous). Requires `[plot]`. |
| `.to_dict()` | `dict` | Serialisable state. Does not include the original model. |
| `.from_dict(data)` | `SHAPRelativities` | Reconstruct from `to_dict()` output. |

`extract_relativities()` output columns: `feature`, `level`, `relativity`, `lower_ci`, `upper_ci`, `mean_shap`, `shap_std`, `n_obs`, `exposure_weight`. All returned as a Polars DataFrame.

### `extract_relativities()` (convenience function)

```python
from shap_relativities import extract_relativities

rels = extract_relativities(
    model=model,
    X=X,
    exposure=df["exposure"],
    categorical_features=["area_code", "ncd_years", "has_convictions"],
    base_levels={"area_code": 0, "ncd_years": 0, "has_convictions": 0},
)
```

Wraps `SHAPRelativities.fit()` and `.extract_relativities()` into one call.

### `load_motor()`

```python
from shap_relativities.datasets.motor import load_motor, TRUE_FREQ_PARAMS, TRUE_SEV_PARAMS

df = load_motor(n_policies=50_000, seed=42)
# Returns a Polars DataFrame
```

Synthetic UK personal lines motor portfolio. 50k policies spanning accident years 2019-2023. Columns: `policy_id`, `inception_date`, `expiry_date`, `accident_year`, `vehicle_age`, `vehicle_group` (ABI 1-50), `driver_age`, `driver_experience`, `ncd_years` (0-5), `ncd_protected`, `conviction_points`, `annual_mileage`, `area` (A-F), `occupation_class`, `policy_type`, `claim_count`, `incurred`, `exposure`.

Frequency is Poisson with log-linear predictor. Severity is Gamma. `TRUE_FREQ_PARAMS` and `TRUE_SEV_PARAMS` export the exact coefficients used to generate the data, so you can validate relativity recovery against the ground truth.

---

## Limitations

**Correlated features.** SHAP attribution for correlated features is not uniquely defined under `tree_path_dependent`. Area band and socioeconomic index will share attribution in a way that depends on tree split order. Use `feature_perturbation="interventional"` with a background dataset to correct for correlations - this is more principled but substantially slower.

**Interaction effects.** TreeSHAP allocates interaction effects back to individual features. If area and vehicle age interact in the model, some of that interaction gets attributed to each feature, not cleanly separated into main effect and interaction. `shap_interaction_values()` gives pure main effects but is O(n * p^2).

**Model uncertainty.** The CLT intervals capture data uncertainty only. They do not say anything about whether the GBM would give different relativities on a different data split, or whether the feature contributions are stable across refits. Bootstrap across model refits for a full uncertainty picture. We haven't implemented this; it is on the roadmap.

**Log-link only.** The `exp()` transformation assumes a log-link objective (Poisson, Tweedie, Gamma). Linear-link models produce SHAP values in response space, not log space. Exponentiating those gives nonsense. Check your objective before using this library.

---

## What's next

**mSHAP for two-part models.** Frequency and severity models can be analysed separately with this library. Combining them into a pure premium decomposition requires mSHAP (Lindstrom et al., 2022), which composes SHAP values in prediction space. This is the next module.

---

---

## Performance

Benchmarked against **Poisson GLM** (statsmodels) on synthetic UK motor data — 50,000 policies, known DGP, temporal 60/20/20 train/calibration/test split. Full notebook: `notebooks/benchmark.py`.

Both models use the same six rating factors. The GLM fits main effects only (the standard first cut). `shap-relativities` uses CatBoost Poisson with SHAP-derived relativities on the calibration set.

| Metric | Poisson GLM | shap-relativities | Notes |
|--------|-------------|-------------------|-------|
| Poisson deviance | baseline | measured at runtime | lower is better |
| Gini coefficient | baseline | measured at runtime | higher is better |
| A/E max deviation (decile) | baseline | measured at runtime | lower is better |
| Fit time | seconds | 5–15x slower | CatBoost training dominates |

The benchmark measures these metrics on the held-out test set and compares Poisson deviance, Gini (discriminatory power), and worst-case A/E by predicted decile. Expected improvement on a portfolio with interaction effects across rating factors: −3% to −8% deviance reduction, +2 to +5 Gini points, −10% to −30% on worst-decile A/E. On homogeneous books where the GLM's log-linear assumptions hold, the gap narrows to under 1 Gini point.

**When to use:** When a CatBoost model already beats the production GLM and you need to get the factor table out of it — for Radar upload, regulatory filing, or a pricing committee review. The value is not just the predictive improvement; it is the ability to present GBM-level accuracy in a format the rating engine already understands.

**When NOT to use:** On small portfolios (under 10,000 policies) where CatBoost will overfit without careful tuning, or when a GLM filing with closed-form standard errors is a regulatory requirement and the Gini improvement does not justify the overhead. Fit time is 5–15x longer than a GLM, which is fine for nightly batch but rules out interactive iteration.



## Databricks Notebook

A ready-to-run Databricks notebook benchmarking this library against standard approaches is available in [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/shap_relativities_demo.py).

## Other Burning Cost libraries

**Model building**

| Library | Description |
|---------|-------------|
| [insurance-interactions](https://github.com/burning-cost/insurance-interactions) | Automated GLM interaction detection via CANN and NID scores |
| [insurance-cv](https://github.com/burning-cost/insurance-cv) | Walk-forward cross-validation respecting IBNR structure |

**Uncertainty quantification**

| Library | Description |
|---------|-------------|
| [insurance-conformal](https://github.com/burning-cost/insurance-conformal) | Distribution-free prediction intervals for Tweedie models |
| [bayesian-pricing](https://github.com/burning-cost/bayesian-pricing) | Hierarchical Bayesian models for thin-data segments |
| [insurance-credibility](https://github.com/burning-cost/insurance-credibility) | Bühlmann-Straub credibility weighting |

**Deployment and optimisation**

| Library | Description |
|---------|-------------|
| [insurance-deploy](https://github.com/burning-cost/insurance-deploy) | Champion/challenger framework with ENBP audit logging |
| [insurance-elasticity](https://github.com/burning-cost/insurance-elasticity) | Causal price elasticity via Double Machine Learning |
| [insurance-optimise](https://github.com/burning-cost/insurance-optimise) | Constrained rate change optimisation with FCA PS21/5 compliance |

**Governance**

| Library | Description |
|---------|-------------|
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | Proxy discrimination auditing for UK insurance models |
| [insurance-governance](https://github.com/burning-cost/insurance-governance) | PRA SS1/23 model validation reports |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model monitoring: PSI, A/E ratios, Gini drift test |

[All libraries and blog posts →](https://burning-cost.github.io)

---


## Related Libraries

| Library | What it does |
|---------|-------------|
| [insurance-cv](https://github.com/burning-cost/insurance-cv) | Temporal cross-validation for insurance models — use walk-forward splits when evaluating GBMs before extracting relativities |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model monitoring with PSI, A/E ratios, and Gini drift — tracks whether SHAP-derived relativities stay valid after deployment |
| [insurance-interactions](https://github.com/burning-cost/insurance-interactions) | Automated GLM interaction detection — use alongside SHAP to identify where the GLM's multiplicative structure breaks down |

## Licence

MIT. Part of the [Burning Cost](https://github.com/burning-cost) insurance pricing toolkit.