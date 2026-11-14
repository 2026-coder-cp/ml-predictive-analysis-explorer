"""Pre-training model suitability screening. Advisory checks are not proof of fit."""
import hashlib
import json
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

VERSION = "0.4.0"
MODELS = ("Random Forest", "XGBoost", "Bayesian Ridge", "Gaussian Process", "Elastic Net", "Quadratic RSM")

def configuration_fingerprint(raw_bytes, features, target, task, models, test_fraction, calibration_fraction, seed):
    payload = {"dataset_sha256": hashlib.sha256(raw_bytes).hexdigest(),
               "features": list(features), "target": target, "task": task,
               "models": list(models), "test_fraction": float(test_fraction),
               "calibration_fraction": float(calibration_fraction), "seed": int(seed),
               "assessment_version": VERSION}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def prepare_data(df, features, target, task):
    if len(set(df.columns)) != len(df.columns):
        raise ValueError("Duplicate column names are not supported.")
    if not features or target in features:
        raise ValueError("Choose predictor columns excluding the target.")
    data = df[list(features) + [target]].dropna(subset=[target]).copy()
    if task == "Regression":
        data[target] = pd.to_numeric(data[target], errors="coerce")
        data = data.dropna(subset=[target])
    if len(data) < 30:
        raise ValueError("At least 30 rows with valid outcomes are required.")
    if data[list(features)].isna().all().any():
        raise ValueError("At least one predictor is entirely missing.")
    if task == "Regression" and (not np.isfinite(data[target].to_numpy(dtype=float)).all()):
        raise ValueError("Regression targets must be finite.")
    if task == "Regression" and data[target].nunique() < 2:
        raise ValueError("Regression requires a varying outcome.")
    return data

def assess(df, features, target, task, models, test_fraction=0.2, calibration_fraction=0.25, seed=42):
    """Inspect only the fit partition; do not inspect held-out test outcomes for model selection."""
    data = prepare_data(df, features, target, task)
    y = data[target]
    strat = y if task == "Classification" else None
    if task == "Classification" and (y.nunique() < 2 or y.value_counts().min() < 3):
        raise ValueError("Classification requires at least two classes with three examples each.")
    try:
        fit_pool, _ = train_test_split(data, test_size=test_fraction, random_state=int(seed), stratify=strat)
        fit, _ = train_test_split(fit_pool, test_size=calibration_fraction, random_state=int(seed)+1,
                                  stratify=fit_pool[target] if task == "Classification" else None)
    except ValueError as exc:
        raise ValueError("Unable to create the configured data partitions: " + str(exc)) from exc
    x = fit[list(features)]
    numeric = x.select_dtypes(include=np.number)
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    if len(numeric.columns) != len(x.select_dtypes(include=np.number).columns):
        raise ValueError("Invalid numeric predictors.")
    if np.isinf(x.select_dtypes(include=np.number).to_numpy(dtype=float)).any():
        raise ValueError("Numeric predictors contain infinity.")
    missing_rate = float(x.isna().mean().max())
    constants = [c for c in features if x[c].nunique(dropna=True) <= 1]
    numeric_count = len(numeric.columns)
    encoded_estimate = numeric_count + sum(max(1, int(x[c].nunique(dropna=True))) for c in features if c not in numeric.columns)
    quadratic_terms = (encoded_estimate * (encoded_estimate + 3)) // 2
    n = len(fit)
    results = []
    for name in models:
        findings = []
        severity = "Suitable"
        def warn(message, not_recommended=False):
            nonlocal severity
            findings.append(message)
            severity = "Not recommended" if not_recommended else ("Warning" if severity == "Suitable" else severity)
        if missing_rate > 0.3:
            warn(f"Maximum predictor missingness is {missing_rate:.1%}; imputation may dominate.")
        if constants:
            warn("Constant/near-empty predictors: " + ", ".join(map(str, constants[:8])))
        if encoded_estimate > n:
            warn(f"Estimated encoded features ({encoded_estimate}) exceed fit observations ({n}).")
        if task == "Classification":
            if name not in ("Random Forest", "XGBoost"):
                severity = "Blocked"
                findings.append("This estimator does not support classification in this application.")
        elif name == "Gaussian Process":
            if n > 1500:
                warn(f"Gaussian Process has {n} fit rows; cubic scaling may be impractical.", True)
            elif n > 500:
                warn(f"Gaussian Process has {n} fit rows; training may be slow.")
            if encoded_estimate > 30:
                warn("High encoded dimensionality may impair the fixed RBF kernel.")
        elif name == "Quadratic RSM":
            if quadratic_terms >= n:
                warn(f"Quadratic expansion estimates {quadratic_terms} terms for {n} fit rows; design is underdetermined.", True)
            elif n < 5 * quadratic_terms:
                warn(f"Only {n} fit rows for {quadratic_terms} quadratic terms; interpret coefficients cautiously.")
        elif name in ("Bayesian Ridge", "Elastic Net"):
            if encoded_estimate > n / 2:
                warn("Feature count is high relative to sample size; regularization helps but does not establish generalization.")
            if numeric_count:
                skew = numeric.skew(numeric_only=True).abs()
                if (skew > 3).any():
                    warn("Highly skewed predictors observed; assess leverage and nonlinear effects after fitting.")
        elif name == "XGBoost":
            if n < 100:
                warn("Small fit sample for boosted trees; assess overfitting and cross-validation stability.")
        elif name == "Random Forest":
            if n < 60:
                warn("Limited fit sample; tree ensemble performance may be unstable.")
        if not findings:
            findings.append("No material pre-fit concern identified; post-fit validation is still required.")
        results.append({"Model": name, "Status": severity, "Fit rows": n,
                        "Encoded feature estimate": encoded_estimate,
                        "Max missing fraction": round(missing_rate, 4),
                        "Quadratic term estimate": quadratic_terms if name == "Quadratic RSM" else None,
                        "Findings": " | ".join(findings)})
    return pd.DataFrame(results)
