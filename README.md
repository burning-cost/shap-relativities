# shap-relativities

Extract multiplicative rating relativities from GBM models using SHAP values. Built for insurance pricing.

## What this does

Pricing teams train GBMs as technical models, then need to import relativities into rating tools like Radar or Emblem. Those tools expect the same format as a GLM: a table of `(feature, level, relativity)` triples where the base level is 1.0 and relativities multiply together. SHAP provides the bridge.

TreeSHAP decomposes each prediction into per-feature log-space contributions. We aggregate by factor level (exposure-weighted), normalise against a chosen base level, and exponentiate. The output is directly importable into your rating structure.

## Why not use SHAP directly?

You can. But you will end up writing the aggregation, normalisation, exposure weighting, and CI logic yourself — and getting base-level normalisation right in log space is fiddly. This library wraps the full workflow in one class with sensible defaults.

## Installation

```bash
pip install "shap-relativities[ml]"
```

Core only (no ML dependencies):
```bash
pip install shap-relativities
```

## Quick start

```python
import lightgbm as lgb
from shap_relativities import SHAPRelativities
from shap_relativities.datasets.motor import load_motor, TRUE_FREQ_PARAMS

# Synthetic UK motor data with a known data generating process
df = load_motor(n_policies=50_000, seed=42)
df["has_convictions"] = (df["conviction_points"] > 0).astype(int)
df["area_code"] = df["area"].map({"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5})

features = ["area_code", "ncd_years", "has_convictions"]
X = df[features]
y = df["claim_count"]
w = df["exposure"]

# Train a Poisson GBM
params = {"objective": "poisson", "learning_rate": 0.05, "num_leaves": 31, "verbose": -1}
dtrain = lgb.Dataset(X, label=y, weight=w)
model = lgb.train(params, dtrain, num_boost_round=300)

# Extract relativities
sr = SHAPRelativities(
    model=model,
    X=X,
    exposure=w,
    categorical_features=features,
)
sr.fit()

rels = sr.extract_relativities(
    normalise_to="base_level",
    base_levels={"area_code": 0, "ncd_years": 0, "has_convictions": 0},
)
print(rels[["feature", "level", "relativity", "lower_ci", "upper_ci"]])
```

Output (approximately — the GBM recovers the known DGP parameters):
```
            feature level  relativity  lower_ci  upper_ci
0         area_code     0       1.000     1.000     1.000
1         area_code     1       1.108     1.060     1.159
2         area_code     2       1.227     1.178     1.278
3         area_code     3       1.427     1.369     1.487
4         area_code     4       1.667     1.596     1.741
5         area_code     5       1.934     1.841     2.032
6          ncd_years     0       1.000     1.000     1.000
7          ncd_years     1       0.882     0.851     0.913
...
```

True DGP NCD coefficient is -0.12, so the extracted NCD=5 relativity should be close to exp(-0.6) ≈ 0.549.

## Features

- Multiplicative relativities from any LightGBM or XGBoost model with a log-link objective (Poisson, Tweedie, Gamma)
- Exposure weighting throughout — relativities reflect portfolio composition, not raw observation counts
- Base-level normalisation — specify your base level per feature, matching GLM convention
- Portfolio mean normalisation — useful when no natural base level exists
- CLT confidence intervals — quantify estimation uncertainty per level
- Sparse level warnings — flags levels with too few observations for reliable CIs
- Reconstruction validation — verifies SHAP values sum to model predictions before you trust the output
- Serialisation — `to_dict()` / `from_dict()` for storing fitted results without the original model

## Datasets

```python
from shap_relativities.datasets.motor import load_motor, TRUE_FREQ_PARAMS, TRUE_SEV_PARAMS

df = load_motor(n_policies=50_000, seed=42)
```

Synthetic UK personal lines motor portfolio with a known data generating process. Frequency is Poisson with log-linear predictor; severity is Gamma. True parameters are exported so you can validate that your extraction recovers them. Covers accident years 2019-2023, area bands A-F, NCD 0-5, realistic driver age distribution.

## Limitations

Correlated features share SHAP attribution ambiguously. This is fundamental to Shapley values — there is no fix. Interpret relativities for correlated features with caution and document the caveat when presenting results.

CLT confidence intervals capture data sampling uncertainty only, not model uncertainty from the GBM fitting process. For model uncertainty you need bootstrap refits, which is expensive.

Log-link only. The exp() transformation assumes a log-link objective. Linear-link models need different treatment.

## Coming soon

mSHAP for two-part models: frequency-times-severity decomposition using multiplicative SHAP, so you can extract separate freq and sev relativities from a compound model.

## Built by Burning Cost

Part of the [Burning Cost](https://github.com/burning-cost) insurance pricing toolkit.

## Licence

MIT
