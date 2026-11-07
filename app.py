import hashlib
import io
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, ElasticNet, Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

st.set_page_config(page_title="ML Predictive Analysis Explorer", page_icon="📈", layout="wide")
st.title("ML Predictive Analysis Explorer v0.3")
st.caption("Six-model regression comparison | Independent training, blind prediction, and external validation")
st.warning("Research-only prototype. Not qualified for regulated decisions. Do not upload confidential data without authorization.")

def read_csv(upload):
    if upload is None:
        return None
    try:
        return pd.read_csv(io.BytesIO(upload.getvalue()))
    except Exception as exc:
        st.error(f"CSV could not be read: {exc}")
        return None

def scores(actual, predicted, task):
    if task == "Regression":
        return {"MAE": float(mean_absolute_error(actual, predicted)),
                "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
                "R²": float(r2_score(actual, predicted)) if len(actual) > 1 else np.nan,
                "Bias (predicted - actual)": float(np.mean(np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)))}
    return {"Accuracy": float(accuracy_score(actual, predicted)),
            "Macro F1": float(f1_score(actual, predicted, average="macro", zero_division=0))}

def build_pipeline(x, task, name, seed, n_classes):
    numeric = x.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in x.columns if c not in numeric]
    scaled = name in ("Bayesian Ridge", "Gaussian Process", "Elastic Net", "Quadratic RSM")
    transformers = []
    if numeric:
        numeric_steps = [("impute", SimpleImputer(strategy="median"))]
        if scaled:
            numeric_steps.append(("scale", StandardScaler()))
        transformers.append(("num", Pipeline(numeric_steps), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False if scaled else True))]), categorical))
    preprocessor = ColumnTransformer(transformers, sparse_threshold=0 if scaled else 0.3)
    if task == "Regression":
        if name == "Random Forest":
            estimator = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
        elif name == "XGBoost":
            estimator = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=seed, n_jobs=-1)
        elif name == "Bayesian Ridge":
            estimator = BayesianRidge()
        elif name == "Gaussian Process":
            kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 1e1))
            estimator = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=seed, n_restarts_optimizer=1)
        elif name == "Elastic Net":
            estimator = ElasticNet(alpha=0.02, l1_ratio=0.5, max_iter=10000, random_state=seed)
        elif name == "Quadratic RSM":
            estimator = Pipeline([("polynomial", PolynomialFeatures(degree=2, include_bias=False)), ("ridge", Ridge(alpha=0.01))])
        else:
            raise ValueError(f"Unsupported regression model: {name}")
    else:
        estimator = (RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
                     if name == "Random Forest" else
                     XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=seed,
                                   n_jobs=-1, objective="multi:softprob" if n_classes > 2 else "binary:logistic"))
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])

def model_prediction_std(pipeline, x, name):
    if name not in ("Bayesian Ridge", "Gaussian Process"):
        return None
    transformed = pipeline.named_steps["preprocess"].transform(x)
    _, std = pipeline.named_steps["model"].predict(transformed, return_std=True)
    return np.asarray(std)

def clear_state():
    st.session_state.pop("trained", None)
    st.session_state.pop("blind_predictions", None)

tab_train, tab_predict, tab_validate = st.tabs(["1. Train", "2. Predict unseen CSV", "3. External validation"])

with tab_train:
    upload = st.file_uploader("Training CSV with outcome column", type="csv", key="training_upload", on_change=clear_state)
    df = read_csv(upload)
    if df is not None:
        st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")
        st.dataframe(df.head(8), use_container_width=True)
        if len(df) < 30 or len(df.columns) < 2:
            st.error("Provide at least 30 rows and two columns.")
        else:
            target = st.selectbox("Outcome column", df.columns, index=len(df.columns)-1, on_change=clear_state)
            candidate = [c for c in df.columns if c != target]
            features = st.multiselect("Predictor columns (exclude row IDs)", candidate, default=candidate, on_change=clear_state)
            task = st.radio("Task", ["Regression", "Classification"], horizontal=True, on_change=clear_state)
            available_models = ["Random Forest", "XGBoost"] if task == "Classification" else ["Random Forest", "XGBoost", "Bayesian Ridge", "Gaussian Process", "Elastic Net", "Quadratic RSM"]
            model_names = st.multiselect("Models", available_models, default=available_models, on_change=clear_state)
            if task == "Regression":
                st.caption("Bayesian Ridge and Gaussian Process report model-based predictive standard deviations. All six models receive calibration-based prediction intervals.")
            test_fraction = st.slider("Internal test fraction", 0.15, 0.35, 0.20, 0.05, on_change=clear_state)
            calibration_fraction = st.slider("Calibration fraction (from development pool)", 0.15, 0.35, 0.25, 0.05, on_change=clear_state)
            coverage = st.slider("Nominal regression prediction interval coverage", 0.80, 0.95, 0.90, 0.05, on_change=clear_state)
            seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, on_change=clear_state)
            if st.button("Train and retain models", type="primary"):
                if not features or not model_names:
                    st.error("Choose predictors and at least one model.")
                elif len(set(features)) != len(features):
                    st.error("Duplicate feature names are unsupported.")
                else:
                    clean = df[features + [target]].dropna(subset=[target]).copy()
                    X = clean[features]
                    y = clean[target]
                    if task == "Regression":
                        y = pd.to_numeric(y, errors="coerce")
                        X, y = X.loc[y.notna()], y.loc[y.notna()]
                        encoder = None
                    else:
                        encoder = LabelEncoder()
                        y = pd.Series(encoder.fit_transform(y.astype(str)), index=y.index)
                    if len(y) < 30 or (task == "Classification" and (y.nunique() < 2 or y.value_counts().min() < 3)):
                        st.error("Insufficient valid outcomes (classification requires at least 3 rows per class).")
                    elif X.isna().all().any():
                        st.error("One or more predictor columns contain only missing values.")
                    else:
                        try:
                            strat = y if task == "Classification" else None
                            X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=test_fraction, random_state=int(seed), stratify=strat)
                            strat_dev = y_dev if task == "Classification" else None
                            X_fit, X_cal, y_fit, y_cal = train_test_split(X_dev, y_dev, test_size=calibration_fraction, random_state=int(seed)+1, stratify=strat_dev)
                            if task == "Classification" and (y_fit.nunique() != y.nunique() or y_cal.nunique() != y.nunique() or y_test.nunique() != y.nunique()):
                                raise ValueError("All classes must appear in training, calibration, and test splits; use more data.")
                            fitted = {}
                            metrics = []
                            internal = pd.DataFrame({"source_row": X_test.index, "actual": y_test.to_numpy()})
                            duration = {}
                            for name in model_names:
                                pipeline = build_pipeline(X_fit, task, name, int(seed), int(y.nunique()) if task == "Classification" else 0)
                                start = time.perf_counter()
                                pipeline.fit(X_fit, y_fit)
                                duration[name] = round(time.perf_counter()-start, 3)
                                prediction = pipeline.predict(X_test)
                                internal[name + "_prediction"] = prediction
                                if task == "Regression":
                                    std = model_prediction_std(pipeline, X_test, name)
                                    if std is not None:
                                        internal[name + "_predictive_std"] = std
                                row = {"Model": name, **scores(y_test, prediction, task)}
                                if task == "Regression":
                                    cal_pred = pipeline.predict(X_cal)
                                    residuals = np.abs(y_cal.to_numpy() - cal_pred)
                                    k = min(len(residuals), int(np.ceil((len(residuals)+1)*coverage)))
                                    q = float(np.sort(residuals)[k-1])
                                    row["Calibration interval half-width"] = q
                                    row["Internal interval coverage"] = float(np.mean(np.abs(y_test.to_numpy()-prediction) <= q))
                                else:
                                    q = None
                                fitted[name] = {"pipeline": pipeline, "q": q}
                                metrics.append(row)
                            fingerprint = hashlib.sha256(upload.getvalue()).hexdigest()
                            st.session_state["trained"] = {
                                "models": fitted, "features": features, "target": target, "task": task,
                                "encoder": encoder, "metrics": pd.DataFrame(metrics), "internal": internal,
                                "n_fit": len(X_fit), "n_cal": len(X_cal), "n_test": len(X_test),
                                "coverage": coverage, "seed": int(seed), "durations": duration,
                                "fingerprint": fingerprint}
                            st.session_state.pop("blind_predictions", None)
                            st.success("Models trained and retained in this browser session.")
                        except Exception as exc:
                            st.error(f"Training failed: {exc}")
    state = st.session_state.get("trained")
    if state:
        st.subheader("Internal held-out evaluation")
        st.dataframe(state["metrics"], use_container_width=True)
        st.write(f"Fit: {state['n_fit']} | Calibration: {state['n_cal']} | Internal test: {state['n_test']}")
        st.caption(f"Training CSV SHA-256: {state['fingerprint']} | Seed: {state['seed']} | Fit times (seconds): {state['durations']}")
        if state["task"] == "Regression":
            fig, ax = plt.subplots(figsize=(7, 4))
            for name in state["models"]:
                ax.scatter(state["internal"]["actual"], state["internal"][name+"_prediction"], alpha=0.55, label=name)
            lo = float(state["internal"]["actual"].min())
            hi = float(state["internal"]["actual"].max())
            ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray")
            ax.set(xlabel="Actual", ylabel="Predicted", title="Internal held-out predictions")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
        st.download_button("Download internal predictions", state["internal"].to_csv(index=False), file_name="internal_holdout_predictions.csv", mime="text/csv")

with tab_predict:
    state = st.session_state.get("trained")
    if not state:
        st.info("Train models in Tab 1 first. Models remain in this Streamlit session until reset or server restart.")
    else:
        blind_file = st.file_uploader("Upload blind external CSV (predictors only)", type="csv", key="blind_upload", on_change=lambda: st.session_state.pop("blind_predictions", None))
        blind = read_csv(blind_file)
        if blind is not None:
            missing = [c for c in state["features"] if c not in blind.columns]
            if missing:
                st.error("Missing predictor columns: " + ", ".join(missing))
            elif len(blind) == 0:
                st.error("Blind CSV has no observations.")
            else:
                st.write(f"External observations: {len(blind)}")
                if st.button("Predict using retained models", type="primary"):
                    try:
                        out = pd.DataFrame({"source_row": np.arange(len(blind))})
                        if "case_id" in blind.columns:
                            out.insert(0, "case_id", blind["case_id"].to_numpy())
                        for name, item in state["models"].items():
                            pred = item["pipeline"].predict(blind[state["features"]])
                            if state["task"] == "Classification":
                                pred = state["encoder"].inverse_transform(pred.astype(int))
                            out[name+"_prediction"] = pred
                            if state["task"] == "Regression":
                                std = model_prediction_std(item["pipeline"], blind[state["features"]], name)
                                if std is not None:
                                    out[name+"_predictive_std"] = std
                            if state["task"] == "Regression":
                                out[name+"_lower"] = pred-item["q"]
                                out[name+"_upper"] = pred+item["q"]
                        st.session_state["blind_predictions"] = {
                            "data": out, "fingerprint": hashlib.sha256(blind_file.getvalue()).hexdigest()}
                        st.success("Predictions generated without refitting.")
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")
        saved = st.session_state.get("blind_predictions")
        if saved:
            st.dataframe(saved["data"], use_container_width=True)
            st.download_button("Download blind predictions", saved["data"].to_csv(index=False), file_name="external_predictions.csv", mime="text/csv")
            st.caption(f"Blind CSV SHA-256: {saved['fingerprint']}")

with tab_validate:
    state = st.session_state.get("trained")
    saved = st.session_state.get("blind_predictions")
    if not state or not saved:
        st.info("Train in Tab 1 and generate blind predictions in Tab 2 before uploading the answer key.")
    else:
        key_upload = st.file_uploader("Answer-key CSV (upload only after predictions)", type="csv", key="answer_upload")
        key = read_csv(key_upload)
        if key is not None:
            pred_df = saved["data"].copy()
            target = state["target"]
            if target not in key.columns:
                st.error(f"Answer key must contain the target column: {target}")
            else:
                try:
                    if "case_id" in pred_df.columns and "case_id" in key.columns:
                        if pred_df["case_id"].duplicated().any() or key["case_id"].duplicated().any():
                            raise ValueError("Duplicate case_id values are not allowed.")
                        joined = pred_df.merge(key[["case_id", target]], on="case_id", how="left", validate="one_to_one", indicator=True)
                        if len(joined) != len(pred_df) or (joined["_merge"] != "both").any() or len(key) != len(pred_df):
                            raise ValueError("Answer-key case IDs must match predictions exactly.")
                        joined = joined.drop(columns=["_merge"])
                    else:
                        if len(key) != len(pred_df):
                            raise ValueError("Answer key row count must equal prediction row count.")
                        st.warning("No case_id columns found. Matching by row position; confirm the answer key has exactly the same row order as the blind CSV.")
                        joined = pred_df.copy()
                        joined[target] = key[target].to_numpy()
                    if joined[target].isna().any():
                        raise ValueError("Answer key contains missing target values.")
                    actual = joined[target]
                    if state["task"] == "Regression":
                        actual = pd.to_numeric(actual, errors="raise")
                    metrics = []
                    for name, item in state["models"].items():
                        predicted = joined[name+"_prediction"]
                        row = {"Model": name, **scores(actual, predicted, state["task"])}
                        if state["task"] == "Regression":
                            joined[name+"_residual"] = predicted-actual
                            row["Interval coverage"] = float(np.mean((actual >= joined[name+"_lower"]) & (actual <= joined[name+"_upper"])))
                            row["Mean interval width"] = float(np.mean(joined[name+"_upper"]-joined[name+"_lower"]))
                        metrics.append(row)
                    st.subheader("Independent external validation")
                    st.dataframe(pd.DataFrame(metrics), use_container_width=True)
                    st.dataframe(joined, use_container_width=True)
                    st.download_button("Download external validation results", joined.to_csv(index=False), file_name="external_validation_results.csv", mime="text/csv")
                    st.caption("Intervals use split conformal absolute residuals from the training CSV calibration subset. Coverage is marginal under exchangeability, not a per-row probability or regulatory confidence statement.")
                except Exception as exc:
                    st.error(f"External validation failed: {exc}")

st.divider()
st.caption("v0.3 | Six regression models; Bayesian/GPR predictive standard deviations are not calibrated intervals. Internal test and calibration are separate. External outcomes never enter model fitting or interval calibration. Repeated tuning against the external set invalidates its independence.")
