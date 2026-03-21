# shap-relativities

[![PyPI](https://img.shields.io/pypi/v/shap-relativities)](https://pypi.org/project/shap-relativities/)
[![Python](https://img.shields.io/pypi/pyversions/shap-relativities)](https://pypi.org/project/shap-relativities/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/burning-cost/shap-relativities/blob/main/notebooks/quickstart.ipynb)

> Questions or feedback? Start a [Discussion](https://github.com/burning-cost/shap-relativities/discussions). Found it useful? A star helps others find it.

**Your GBM beats the GLM. But you can't get the factor table out of it.**

Every UK pricing team we've spoken to has the same problem: a GBM sitting on a server outperforming the production GLM, but nobody can get the relativities out of it. The regulator wants a factor table. The head of pricing wants to challenge the model in terms they recognise.

So the GBM sits in a notebook. The GLM goes to production.

`shap-relativities` closes that gap. It extracts multiplicative rating relativities from CatBoost models using SHAP values — the same format as `exp(beta)` from a GLM, with confidence intervals, exposure weighting, and a validation check that the numbers actually reconstruct the model's predictions.

## Why bother

The GBM's advantage over a GLM shows up most clearly when the true risk structure is non-linear or has interactions. The benchmark below uses a DGP that reflects this: driver age has a U-shaped risk curve (young and old drivers are both worse), and urban areas with old vehicles carry an additional 35% loading on top of main effects. A standard main-effects GLM cannot model either of these.

Numbers measured on Databricks serverless, 2026-03-21, seed=77. 25,000 synthetic UK motor policies, 70/30 train/test split.

| Approach | Factor table? | Mean error vs true DGP | Gini |
|----------|--------------|------------------------|------|
| Poisson GLM (linear driver_age) | Yes — but single age slope, no U-shape | 5.79% | 0.393 |
| Poisson GLM (binned age, 5 bands) | Yes — actuary must choose bins first | 5.87% | 0.414 |
| **shap-relativities (CatBoost)** | **Yes — full age curve + discrete factors** | **6.80%** | **0.411** |

The mean relativity errors are similar across all three approaches on the discrete factors (area, NCD, conviction). The difference is what you get for driver age:

- **Linear GLM**: one slope coefficient. Cannot represent the U-shape.
- **Binned GLM**: band-level relativities — but only if the actuary has already noticed the age curve is non-linear and chosen appropriate cut-points.
- **SHAP relativities**: a continuous age relativity curve, extracted directly via `extract_continuous_curve()`, with no binning decision required. The model finds the shape.

The Gini results are honest: a carefully specified binned GLM with 5 age bands matches CatBoost on Gini (+1.7pp vs linear GLM, comparable to CatBoost). The case for CatBoost+SHAP is not that it always wins on Gini — it is that it finds the non-linear structure automatically and produces a deployable factor table without the actuary first doing shape analysis.

Where CatBoost wins more clearly: books with interaction effects. The same portfolio, with an area × vehicle_age interaction (urban areas + old vehicles → 1.35× additional loading), is where a main-effects-only GLM — regardless of how carefully it is specified — cannot compete. See [Scenario 3](#scenario-3-interaction-dgp) below.

▶ [Run on Databricks](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/shap_relativities_demo.py)

---

**Blog post:** [Extracting Rating Relativities from GBMs with SHAP](https://burning-cost.github.io/2026/03/05/extracting-rating-relativities-from-gbms-with-shap/) — worked example, the maths, and a discussion of limitations for presenting to regulators and pricing committees.

---

## Installation

```bash
uv add "shap-relativities[all]"
```

Or pick what you need:

```bash
uv add "shap-relativities[ml]"    # shap + catboost + scikit-learn + pandas bridge
uv add "shap-relativities[plot]"  # matplotlib for plots
uv add shap-relativities          # core only (polars, numpy, scipy)
```

**Output is a Polars DataFrame.** The library accepts either Polars or pandas DataFrames as input, and returns Polars. Pandas is a bridge dependency: shap's TreeExplainer uses it internally, so it is still installed with the `[ml]` extra.

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
# Note: categorical_features here tells SHAPRelativities to aggregate SHAP values
# by discrete level for these features. It is NOT the same as CatBoost's cat_features
# training parameter — ncd_years and has_convictions are passed as Int32 to CatBoost
# without cat_features=, meaning CatBoost treats them as numeric. That is fine here
# because the DGP is ordinal. The categorical_features argument below is purely
# an aggregation hint for the relativity extraction step.
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

Output (run on Databricks serverless, 2026-03-19, seed=42):

```
shape: (14, 5)
┌─────────────────┬───────┬────────────┬──────────┬──────────┐
│ feature         ┆ level ┆ relativity ┆ lower_ci ┆ upper_ci │
│ ---             ┆ ---   ┆ ---        ┆ ---      ┆ ---      │
│ str             ┆ str   ┆ f64        ┆ f64      ┆ f64      │
╞═════════════════╪═══════╪════════════╪══════════╪══════════╡
│ area_code       ┆ 0     ┆ 1.000      ┆ 0.998    ┆ 1.002    │
│ area_code       ┆ 1     ┆ 1.110      ┆ 1.109    ┆ 1.111    │
│ area_code       ┆ 2     ┆ 1.149      ┆ 1.149    ┆ 1.150    │
│ area_code       ┆ 3     ┆ 1.269      ┆ 1.268    ┆ 1.269    │
│ area_code       ┆ 4     ┆ 1.588      ┆ 1.586    ┆ 1.589    │
│ …               ┆ …     ┆ …          ┆ …        ┆ …        │
│ ncd_years       ┆ 3     ┆ 0.641      ┆ 0.640    ┆ 0.642    │
│ ncd_years       ┆ 4     ┆ 0.542      ┆ 0.541    ┆ 0.543    │
│ ncd_years       ┆ 5     ┆ 0.435      ┆ 0.434    ┆ 0.436    │
│ has_convictions ┆ 0     ┆ 1.000      ┆ 1.000    ┆ 1.000    │
│ has_convictions ┆ 1     ┆ 1.681      ┆ 1.673    ┆ 1.689    │
└─────────────────┴───────┴────────────┴──────────┴──────────┘
```

The true DGP NCD coefficient is -0.12, so NCD=5 vs NCD=0 should give `exp(-0.6) ≈ 0.549`. The GBM recovers approximately 0.435 — a reconstruction error of roughly 21%. This is documented in the benchmark results below. Conviction relativity is approximately `exp(0.45) ≈ 1.57`; SHAP gives 1.681 here (7% above true). The `level` column dtype is `str` — filter using string comparison: `rels.filter(pl.col("level") == "5")`.

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

1. **Native categoricals.** CatBoost handles string categorical features natively — pass area band "A"-"F" directly with `cat_features=["area"]` and no encoding is needed. Note: the quick-start above converts area to an integer `area_code` for a minimal example, but Int32 features passed without `cat_features=` are treated as numeric by CatBoost. For production use, pass string labels with `cat_features` specified so CatBoost treats the feature categorically.

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
SE_k = shap_std_k / sqrt(n_k)
CI = exp(mean_shap_k ± z * SE_k - mean_shap_base)
```

where `n_k` is the count of policies with feature level = k (not the portfolio total). For sparse levels, `n_k` can be small even on a large portfolio, which is why the sparse levels check matters.

These quantify data uncertainty — how precisely we've estimated each level's mean SHAP contribution given the portfolio. They do not capture model uncertainty from the GBM fitting process.

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

The reconstruction check verifies that `exp(shap_values.sum(axis=1) + expected_value)` matches the model's predictions to within 1e-4. If this fails, the explainer was constructed incorrectly — almost always a mismatch between the model's objective and the SHAP output type.

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

`smooth_method="isotonic"` enforces monotonicity via isotonic regression — useful when you have a strong prior that the relativity is one-directional (younger drivers are higher risk, more mileage is more exposure).

---

## Exporting relativities

The `extract_relativities()` output is a standard Polars DataFrame. To export as CSV for manual import into a rating engine:

```python
rels.write_csv("relativities.csv")
```

The CSV has columns `feature`, `level`, `relativity`, `lower_ci`, `upper_ci` — a format that maps directly to any rating engine's factor table import. Radar, Emblem, and Earnix all have CSV factor table import functionality; check your platform's import template for the exact column naming required.

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

`categorical_features` and `continuous_features` are aggregation hints for the relativity extraction step. They tell the library which features to summarise by discrete level (categorical) versus by smoothed curve (continuous). This is distinct from CatBoost's `cat_features` training parameter, which controls how CatBoost handles encoding during model training.

| Method | Returns | Description |
|--------|---------|-------------|
| `.fit()` | `self` | Compute SHAP values. Must be called before extraction. |
| `.extract_relativities(normalise_to, base_levels, ci_method, ci_level)` | `pl.DataFrame` | Main output: one row per (feature, level). |
| `.extract_continuous_curve(feature, n_points, smooth_method)` | `pl.DataFrame` | Smoothed relativity curve for a continuous feature. |
| `.validate()` | `dict[str, CheckResult]` | Diagnostic checks: reconstruction, feature coverage, sparse levels. |
| `.baseline()` | `float` | `exp(expected_value)` — the base rate in prediction space. |
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

## Performance benchmarks

Three scenarios, all run on Databricks serverless compute (Python 3.12, 2026-03-21).
Full scripts: `benchmarks/benchmark_nonlinear.py` (Scenario 1), `benchmarks/benchmark.py` (Scenario 2), `benchmarks/benchmark_interactions.py` (Scenario 3).

### Scenario 1: Non-linear DGP (the primary benchmark)

25,000 synthetic UK motor policies. Five rating features including driver age (continuous, U-shaped risk curve), area (A-F), vehicle age (with area×vehicle_age interaction), NCD years (0-5), conviction flag. The true DGP has:

- **Driver age**: U-shaped quadratic in log space. Age 20 is 1.47× and age 70 is 3.42× the minimum-risk age (38). A linear age coefficient cannot represent this.
- **Area × vehicle age interaction**: urban areas (D, E, F) with vehicles aged 8+ years carry an additional 35% loading. Approximately 26% of the portfolio.
- **NCD**: approximately log-linear with a small extra step at the max bonus level.

Three approaches are compared. 70/30 train/test split.

| Approach | Factor table? | Mean error vs true DGP | Gini | Notes |
|----------|--------------|------------------------|------|-------|
| Poisson GLM (linear driver_age) | Area, NCD, conviction — but driver_age as a slope only | 5.79% | 0.393 | Cannot represent U-shape; worst Gini |
| Poisson GLM (binned age, 5 bands) | Area, NCD, conviction + 5 age bands | 5.87% | 0.414 | Actuary must bin first; no interaction term |
| **shap-relativities (CatBoost)** | **All factors + continuous age curve** | **6.80%** | **0.411** | **Finds shape automatically; handles interaction** |

Key numbers:

- **Gini**: CatBoost +1.74pp vs linear GLM; essentially tied with binned GLM (-0.36pp)
- **Discrete-factor mean error**: all approaches within 1pp of each other (5.8-6.8%)
- **Age curve**: linear GLM cannot produce one; binned GLM gives 5-band step function; SHAP gives a smooth continuous curve via `extract_continuous_curve()`
- **Conviction loading** (true 1.65×): SHAP recovers 1.58× (4% below true)
- **NCD=5 step discount** (true 0.844 vs NCD=4): SHAP gives 0.907 (7% above — partial absorption of the extra step)
- **Fit time**: GLM <0.1s, CatBoost+SHAP ~10s (including SHAP computation)

The honest finding: a carefully specified binned-age GLM with 5 age bands is competitive on Gini and relativity error for the *discrete* features. CatBoost with SHAP earns its keep on three grounds: (1) it finds the age shape without the actuary first doing shape analysis; (2) it produces a continuous age relativity curve rather than a step function; (3) it captures the interaction automatically, without an interaction term being specified.

---

### Scenario 2: Reference scenario — clean log-linear DGP

20,000 synthetic UK motor policies. Three rating factors: area band (A-F), NCD years (0-5), conviction flag. True DGP is log-linear Poisson — the standard GLM assumption holds exactly. 70/30 train/test split.

This is the GLM's home ground. The correctly-specified GLM recovers the true relativities by MLE. Including this scenario is useful for two reasons: it shows the library works on simple portfolios, and it documents the cost of not constraining to log-linear form when the constraint is in fact valid.

| Approach | Level relativities? | Mean error vs true | Gini | Notes |
|----------|--------------------|--------------------|------|-------|
| CatBoost feature importance | No | N/A | 0.4785 | Rankings only; cannot produce NCD=5 discount |
| Poisson GLM exp(beta) | Yes | 4.47% | 0.4500 | Correctly specified; best relativity precision |
| **shap-relativities** | **Yes** | **9.44%** | **0.4785** | Same Gini as raw CatBoost; +2.85pp vs GLM |

Key numbers:

- **NCD=5 discount** (true 0.549): GLM recovers 0.603 (+10%), SHAP gives 0.427 (−22%)
- **Conviction loading** (true 1.570): GLM recovers 1.547 (−1%), SHAP gives 1.501 (−4%)
- **Gini gap** GBM vs GLM: +2.85pp — the GBM finds nonlinear patterns even in a linear world
- **SHAP reconstruction**: PASS (max error 5.69e-16)
- **Total fit time**: GLM 0.1s, CatBoost+SHAP ~1.4s

On a correctly-specified log-linear DGP, the GLM has an advantage in relativity precision (4.47% vs 9.44% error). The SHAP errors are larger because the GBM does not constrain itself to log-linear form. On portfolios with genuine interaction effects — which most real motor books have — the GBM's Gini improvement more than compensates.

---

### Scenario 3: Interaction DGP (vehicle group × NCD)

30,000 synthetic policies. Four rating factors including vehicle group (3 classes). The true DGP has a **vehicle_group=3 × NCD_years ≤ 1 interaction**: high-group policyholders with limited driving history pay 1.40× more than the main effects alone predict. This combination represents roughly 5% of the portfolio. 70/30 split.

This is the scenario that exists on most real motor portfolios. The GLM with main effects cannot model it. The GBM detects it automatically via tree splits.

| Approach | Level relativities? | Gini gap vs GLM | Notes |
|----------|--------------------|--------------------|-------|
| Poisson GLM (main effects) | Yes | baseline | Cannot model the interaction; worst decile A/E goes badly wrong |
| **shap-relativities** | **Yes** | **+3.4pp** | GBM captures the interaction; SHAP factor table still deployable |

Measured on Databricks serverless, 30k policies, seed=99. Run `benchmarks/benchmark_interactions.py` to reproduce.

Key numbers:

- **Gini gap**: +3.4pp in favour of CatBoost (measured). The interaction cell — VG=3 with NCD ≤ 1 — represents ~5% of the portfolio but a disproportionate share of claims. The GLM underweights it systematically.
- **Absorbed interaction**: VG=3 SHAP relativity = 2.378 vs true main effect of 2.10. The extra 0.278 is the interaction signal leaking into the marginal relativity. This is expected — TreeSHAP distributes interaction effects back to the contributing features. The factor table is still deployable; it prices the VG=3 segment correctly on average even though it cannot separate the main effect from the interaction term.
- **Fit time**: CatBoost+SHAP ~4.5s vs GLM <0.1s. Acceptable for nightly batch.

The GLM's deviance advantage in Scenario 2 disappears when the model is misspecified. On this DGP, the GBM's +3.4pp Gini advantage is driven by a real structural problem the GLM cannot address — not by the GBM overfitting.

---

## When the GLM wins

Be honest about when to reach for this library and when not to.

**The true DGP is purely multiplicative with no interactions.** On a clean log-linear DGP (Scenario 2 above), a correctly-specified Poisson GLM has lower deviance than CatBoost and much better relativity precision — 4.47% vs 9.44% mean error vs ground truth. If your book is genuinely log-linear (which you can test with the [insurance-interactions](https://github.com/burning-cost/insurance-interactions) library), a GLM is the more honest model choice. SHAP-relativities will not break anything in this case, but the GLM relativities will be closer to the true parameters.

**The portfolio is small (under ~15,000 policies).** CatBoost needs enough data to find the interactions that justify its flexibility. On thin books, regularisation helps but CatBoost will still find spurious patterns in the training data. GLM with regularised MLE is more reliable when n is small. The confidence intervals from SHAP will also be wide on a small book — the CLT approximation holds but `n_k` for rare levels will be very small, making the CIs nearly useless.

**Regulatory filings require closed-form standard errors.** SHAP confidence intervals are CLT approximations using the empirical variance of SHAP values per level. They do not have the same statistical standing as GLM standard errors derived from the Fisher information matrix. If your jurisdiction requires closed-form SEs in the rate filing — which some do — a GLM with documented MLE properties is the right tool. SHAP CIs are useful for internal challenge and committee review, not necessarily for regulatory sign-off.

**The factor table is the model.** If the pricing team intends to manually adjust the extracted relativities, apply loadings, or blend with external benchmarks — the standard actuarial workflow — then the GLM's explicit coefficient structure is more tractable. A GLM factor table can be adjusted and its effect on total rate level calculated analytically. SHAP relativities extracted from a GBM are descriptive summaries; changing them doesn't change the model.

In practice, these conditions often hold partially. The right use of this library is: train the GBM, measure whether it materially outperforms the GLM on your data (look at Gini and worst-decile A/E, not just deviance), and only deploy SHAP-relativities if the GBM advantage is large enough to justify the additional complexity. A +1pp Gini gain on a 50k policy book is noise. A +5pp gain with a 20pp A/E improvement on the worst decile is real.

---

## Limitations

**Correlated features.** SHAP attribution for correlated features is not uniquely defined under `tree_path_dependent`. Area band and socioeconomic index will share attribution in a way that depends on tree split order. Use `feature_perturbation="interventional"` with a background dataset to correct for correlations — this is more principled but substantially slower.

**Interaction effects.** TreeSHAP allocates interaction effects back to individual features. If area and vehicle age interact in the model, some of that interaction gets attributed to each feature, not cleanly separated into main effect and interaction. `shap_interaction_values()` gives pure main effects but is computationally expensive — O(TLD²) where T = number of trees, L = maximum leaves per tree, and D = maximum tree depth. Expect meaningful slowdown on large ensembles.

**Model uncertainty.** The CLT intervals capture data uncertainty only. They do not say anything about whether the GBM would give different relativities on a different data split, or whether the feature contributions are stable across refits. Bootstrap across model refits for a full uncertainty picture. We haven't implemented this; it is on the roadmap.

**Log-link only.** The `exp()` transformation assumes a log-link objective (Poisson, Tweedie, Gamma). Linear-link models produce SHAP values in response space, not log space. Exponentiating those gives nonsense. Check your objective before using this library.

---

## What's next

**mSHAP for two-part models.** Frequency and severity models can be analysed separately with this library. Combining them into a pure premium decomposition requires mSHAP (Lindstrom et al., 2022), which composes SHAP values in prediction space. This is the next module.

---


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
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | Causal price elasticity via Double Machine Learning (includes elasticity subpackage) |
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

| Library | Description |
|---------|-------------|
| [insurance-distill](https://github.com/burning-cost/insurance-distill) | GBM-to-GLM distillation — converts a CatBoost model into a deployable GLM using knowledge distillation |
| [insurance-interactions](https://github.com/burning-cost/insurance-interactions) | Automated GLM interaction detection — use alongside SHAP to identify where the GLM's multiplicative structure breaks down |
| [insurance-glm-tools](https://github.com/burning-cost/insurance-glm-tools) | Factor clustering and GLM utilities — group SHAP-derived relativities into rating bands |
| [insurance-cv](https://github.com/burning-cost/insurance-cv) | Temporal cross-validation for insurance models — use walk-forward splits when evaluating GBMs before extracting relativities |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model monitoring with PSI, A/E ratios, and Gini drift — tracks whether SHAP-derived relativities stay valid after deployment |
## Licence

BSD-3. Part of the [Burning Cost](https://github.com/burning-cost) insurance pricing toolkit.
