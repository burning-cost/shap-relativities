# shap-relativities

Extract actuarial-grade multiplicative rating relativities from gradient boosted models using SHAP.

## What this does

Pricing actuaries fit GBMs (LightGBM, XGBoost) as technical models, then need to translate the output into the same format as GLM relativities — a table of `(feature, level, relativity)` triples where the base level is 1.0 and relativities multiply together. This library does that translation properly.

Under the hood: TreeSHAP decomposes each prediction into per-feature log-space contributions. We aggregate these by factor level (exposure-weighted), normalise against a base level, and exponentiate to get multiplicative relativities with confidence intervals. The output slots directly into your rating structure.

## Why not just use SHAP directly?

You can. But you'll end up writing the aggregation, normalisation, exposure weighting, and CI logic yourself — and getting the base-level normalisation right is surprisingly fiddly. This wraps the full workflow in a single class with sensible defaults.

## Installation

```bash
pip install shap-relativities
```

With LightGBM support:
```bash
pip install "shap-relativities[lightgbm]"
```

## Quick start

```python
import lightgbm as lgb
from shap_relativities import SHAPRelativities

# Train your model as usual
model = lgb.LGBMRegressor(objective="poisson", n_estimators=200)
model.fit(X_train, y_train, sample_weight=exposure_train)

# Extract relativities
sr = SHAPRelativities(
    model, X_train,
    exposure=exposure_train,
    categorical_features=["area", "vehicle_group", "ncd_years"],
)
sr.fit()

rels = sr.extract_relativities(
    normalise_to="base_level",
    base_levels={"area": "A", "vehicle_group": "1", "ncd_years": 0},
)
print(rels[["feature", "level", "relativity", "lower_ci", "upper_ci"]])
```

Output:
```
   feature level  relativity  lower_ci  upper_ci
0     area     A       1.000     1.000     1.000
1     area     B       1.142     1.098     1.188
2     area     C       1.287     1.231     1.346
3     area     D       1.463     1.392     1.538
...
```

## Features

- **Multiplicative relativities** from any LightGBM or XGBoost model with a log-link objective (Poisson, Tweedie, Gamma)
- **Exposure weighting** throughout — relativities reflect portfolio composition
- **Base-level normalisation** — choose your base level per feature, or let it auto-select
- **CLT confidence intervals** — quantify estimation uncertainty per level
- **Sparse level warnings** — flags levels with too few observations for reliable CIs
- **Reconstruction validation** — checks that SHAP values reconstruct model predictions
- **Serialisation** — `to_dict()` / `from_dict()` for storing fitted results without the model

## Validation against known DGP

The test suite generates synthetic motor data with known true relativities, fits a GBM, extracts SHAP relativities, and checks they recover the true parameters. This is the gold standard test: if the extraction is working correctly, recovered relativities should be close to the data-generating process.

## Limitations

- **Correlated features** share SHAP attribution ambiguously. There is no fix — this is fundamental to Shapley values. Document this when presenting results.
- **Log-link only.** The exp() transformation assumes your model uses a log link. Linear-link models need different treatment.
- **CLT CIs** capture data sampling uncertainty, not model uncertainty from the GBM fitting process. For model uncertainty, use repeated fits on bootstrap samples.
- **mSHAP for two-part models** (freq x sev decomposition) is planned but not yet implemented.

## Databricks

Works on Databricks with no special setup. Install into your cluster or notebook-scoped:

```python
%pip install shap-relativities[lightgbm]
```

See the `notebooks/` directory for a complete Databricks workflow.

## Licence

MIT
