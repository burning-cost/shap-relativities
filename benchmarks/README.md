# Benchmarks — shap-relativities

**Headline:** On a non-linear DGP with a U-shaped driver age effect and an area × vehicle age interaction, CatBoost + SHAP relativities reduces mean relativity error by ~35–50% vs a Poisson GLM with main effects, while CatBoost feature importance (gain) cannot produce level-specific factors at all.

---

## Comparison table — clean log-linear DGP (`benchmark.py`)

20,000 synthetic UK motor policies. Known log-linear Poisson DGP with three categorical factors. This is the GLM's home ground — a correctly-specified baseline.

| Metric | CatBoost gain importance | Poisson GLM | CatBoost + SHAP relativities |
|---|---|---|---|
| Level-specific multiplicative factors | No — ranks features only | Yes | Yes |
| Mean relativity error vs true DGP | N/A | ~1–3% | ~2–5% |
| Gini (test set) | — | ~0.28–0.32 | ~0.30–0.35 |
| Confidence intervals per level | No | Yes (Wald) | Yes (bootstrap) |
| Deployable to rating engine (Radar/Emblem) | No | Yes | Yes |

### Non-linear DGP (`benchmark_nonlinear.py`)

25,000 policies. True DGP has: U-shaped driver age (young <25 and old >70 both higher risk), area × vehicle age interaction (urban + old vehicle = 35% uplift), non-proportional NCD step at NCD=5.

| Metric | Poisson GLM (main effects) | Poisson GLM (age bins) | CatBoost + SHAP relativities |
|---|---|---|---|
| Driver age mean relativity error | High (linear misspecifies U-shape) | Moderate (4 bins, coarse) | Low (data-adaptive shape) |
| Area × vehicle age interaction captured | No | No | Partially (via SHAP marginals) |
| Gini (test set) | ~0.25–0.30 | ~0.27–0.32 | ~0.34–0.40 |
| Mean relativity error (all factors) | ~8–15% | ~5–10% | ~3–6% |
| Requires knowing the non-linear shape in advance | N/A | Yes (bin spec) | No |

### Interaction DGP (`benchmark_interactions.py`)

30,000 policies. True DGP has a vehicle_group × NCD interaction: high vehicle group + low NCD = 1.4× uplift beyond main effects.

| Metric | Poisson GLM (main effects only) | CatBoost + SHAP relativities |
|---|---|---|
| Interaction term captured | No | Partially (marginal relativities compress it) |
| Gini advantage of GBM over GLM | ~0.04–0.08 | — |
| Relativity error on interaction segment | High | Lower, but not zero (marginals cannot fully represent a 2-way interaction) |

When the interaction is strong, SHAP marginal relativities are a lossy representation. The library documents this limitation explicitly: if a 2-way interaction is the dominant structure, you need a 2-way relativity table, not marginals.

---

## How to run

```bash
uv run python benchmarks/benchmark.py              # clean log-linear DGP
uv run python benchmarks/benchmark_nonlinear.py   # U-shape + interaction DGP
uv run python benchmarks/benchmark_interactions.py # vehicle_group × NCD interaction
```

### Databricks

```bash
databricks workspace import-dir benchmarks /Workspace/shap-relativities/benchmarks
```

Dependencies: `shap-relativities[all]` (includes `catboost`, `shap`), `numpy`, `polars`, `statsmodels`.
