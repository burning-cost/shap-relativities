# Databricks notebook source
# MAGIC %md
# MAGIC # Run SHAPInference Tests
# MAGIC
# MAGIC Installs the package from source, runs the test_inference.py suite,
# MAGIC and prints results.

# COMMAND ----------

# MAGIC %pip install "scikit-learn>=1.3" "polars>=1.0" "numpy>=1.25" "scipy>=1.10" "matplotlib>=3.7" --quiet

# COMMAND ----------

import subprocess
import sys

# Install from workspace source
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet",
     "-e", "/Workspace/shap-relativities/src/../"],
    capture_output=True, text=True
)
print("pip install stdout:", result.stdout[-2000:] if result.stdout else "(empty)")
print("pip install stderr:", result.stderr[-2000:] if result.stderr else "(empty)")
print("Return code:", result.returncode)

# COMMAND ----------

# Quick smoke test: can we import?
import sys
sys.path.insert(0, "/Workspace/shap-relativities/src")

from shap_relativities import SHAPInference
import numpy as np

rng = np.random.default_rng(0)
sv = rng.normal(size=(200, 3))
y = np.abs(rng.normal(size=200))
si = SHAPInference(sv, y, ["a", "b", "c"], p=2.0, n_folds=3, random_state=0)
si.fit()
tbl = si.importance_table()
print("Import and basic fit: OK")
print(tbl)

# COMMAND ----------

# Run pytest on the inference tests
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/shap-relativities/tests/test_inference.py",
     "-v", "--tb=short", "--no-header"],
    capture_output=True, text=True,
    env={**__import__("os").environ, "PYTHONPATH": "/Workspace/shap-relativities/src"}
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-3000:])
print("Return code:", result.returncode)
assert result.returncode == 0, "Tests FAILED — see output above"
