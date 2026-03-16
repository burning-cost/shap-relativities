# Databricks notebook source
# Run pytest with coverage for shap-relativities
# MAGIC %pip install -e /Workspace/shap-relativities[dev] pytest-cov

# COMMAND ----------

import subprocess
result = subprocess.run(
    [
        "python", "-m", "pytest",
        "/Workspace/shap-relativities/tests/",
        "--cov=shap_relativities",
        "--cov-report=term-missing",
        "-q",
        "--tb=short",
    ],
    capture_output=True,
    text=True,
    cwd="/Workspace/shap-relativities",
)
print(result.stdout[-8000:] if len(result.stdout) > 8000 else result.stdout)
print(result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr)
print("Return code:", result.returncode)
