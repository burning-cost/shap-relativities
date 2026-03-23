# Changelog

## [0.2.5] - 2026-03-23

### Fixed
- Bumped numpy minimum version from >=1.24 to >=1.25 to ensure compatibility with scipy's use of numpy.exceptions (added in numpy 1.25)


## v0.2.3 (2026-03-22) [unreleased]
- Add pyarrow dependency — fixes 2 test failures in validate_no_model tests
- Fix licence footer: BSD-3 was wrong, LICENSE file is MIT
- Add GBM benchmark as primary performance scenario; restructure README
- docs: regenerate API reference [skip ci]
- fix: sync __version__ with pyproject.toml (0.2.0 -> 0.2.3)

## v0.2.3 (2026-03-21)
- fix: remove duplicate install line and update stale insurance-elasticity reference
- Add cross-links to related libraries in README
- docs: replace pip install with uv add in README
- Add community CTA to README
- Add interaction benchmark, fill performance table, add When the GLM Wins section
- Add Google Colab quickstart notebook and Open-in-Colab badge
- Add quickstart notebook
- fix: update quickstart output to match actual execution (level dtype=str, NCD=5=0.435)
- fix: README technical errors from quality review
- fix: update license badge from BSD-3 to MIT
- Add discussions link and star CTA
- Add Benchmark Results section with actual measured numbers
- Fix GLM coefficient extraction: use get_feature_names_out() for OHE areas
- Improve benchmark: add feature importance comparison, tighten structure
- Add benchmark: SHAP relativities vs feature importance vs GLM
- Fix four reviewer-identified issues in README
- docs: regenerate API reference [skip ci]
- Fix P0/P1 bugs: SE, CI, sort order, zero-weight, from_dict, motor small-n, curve normalisation
- docs: regenerate API reference [skip ci]
- docs: add hosted API reference via pdoc and GitHub Pages
- Add benchmark: SHAP relativities vs Poisson GLM for rating factor extraction
- Add unit tests for lower-level modules, raising coverage from 37% to 93%
- fix: use table-form license field to avoid packaging.licenses on older envs
- Fix n_obs dtype mismatch between categorical and continuous aggregation
- Fix level column type mismatch in pl.concat across categorical and continuous features
- Polish flagship README: badges, benchmark table, problem statement
- docs: add Databricks notebook link
- Add Related Libraries section to README
- fix: README quick-start base_levels used wrong feature names
- fix: update cross-references to consolidated repos
- fix: update cross-references to consolidated repos
- Add benchmark notebook: shap-relativities vs Poisson GLM
- Add CITATION.cff for academic and software citation
- fix: remove IndentationError in README quick-start example
- fix: update polars floor to >=1.0 and fix project URLs
- Add Performance section to README
- Add Databricks benchmark notebook: shap-relativities vs Poisson GLM
- Add Databricks demo notebook for shap-relativities

## v0.2.0 (2026-03-09)
- Fix workflow to trigger on master branch
- Add GitHub Actions CI workflow and test badge
- docs: README quality pass — fix URLs, add cross-references
- fix: update URLs to burning-cost org
- Code quality audit: fix colour bug, add 9 tests, clean docs
- Add badges, topics cross-links to README
- Remove lightgbm and xgboost from optional dependency extras
- Replace uv pip install with uv add in error messages
- Remove LightGBM/XGBoost references from docs - CatBoost is the primary model
- docs: switch examples to CatBoost/polars/uv, fix tone
- fix: standardise on CatBoost, uv, clean up style
- fix: uv references
- Migrate to CatBoost as primary GBM and Polars as data layer
- Add blog post link to README
- Update pyproject: tighten version bounds, rename extras, add keywords
- Sharpen README, fix packaging, remove burning-cost references
- Add datasets module and test_motor; update pyproject and README to match spec
- Remove __pycache__ from tracking
- Add .gitignore, remove cached files from tracking
- Initial release: shap-relativities v0.1.0

