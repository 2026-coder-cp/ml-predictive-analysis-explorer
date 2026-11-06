import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

st.set_page_config(page_title="ML Predictive Analysis Explorer", page_icon="📈", layout="wide")
st.title("ML Predictive Analysis Explorer")
st.caption("XGBoost and Random Forest | CSV upload | Session-only model training")
st.warning("Research prototype—not validated for clinical, quality-release, or regulated decisions. Upload only approved data.")
file = st.file_uploader("Upload CSV dataset", type="csv")
if file is None:
    st.info("CSV format: one observation per row, with predictor columns and one outcome column.")
    st.stop()
try:
    df = pd.read_csv(file)
except Exception as exc:
    st.error(f"Unable to parse CSV: {exc}")
    st.stop()
if len(df) < 15 or len(df.columns) < 2:
    st.error("At least 15 rows and 2 columns are required.")
    st.stop()
st.dataframe(df.head(10), use_container_width=True)
target = st.selectbox("Outcome column", df.columns, index=len(df.columns)-1)
features = st.multiselect("Predictor columns", [c for c in df.columns if c != target], default=[c for c in df.columns if c != target])
task = st.radio("Task", ["Regression", "Classification"], horizontal=True)
models = st.multiselect("Models", ["Random Forest", "XGBoost"], default=["Random Forest", "XGBoost"])
test_fraction = st.slider("Test fraction", 0.15, 0.40, 0.25, 0.05)
seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42)
if not st.button("Train and compare", type="primary"):
    st.stop()
if not features or not models:
    st.error("Select predictors and at least one model.")
    st.stop()
clean = df[features + [target]].dropna(subset=[target]).copy()
X, y = clean[features], clean[target]
if task == "Regression":
    y = pd.to_numeric(y, errors="coerce")
    X, y = X.loc[y.notna()], y.loc[y.notna()]
    if len(y) < 15:
        st.error("At least 15 numeric outcomes are required.")
        st.stop()
else:
    if y.nunique() < 2 or y.value_counts().min() < 2:
        st.error("Classification needs at least two classes and two examples per class.")
        st.stop()
    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(y.astype(str)), index=y.index)
numeric = X.select_dtypes(include=np.number).columns.tolist()
categorical = [c for c in X.columns if c not in numeric]
transformers = []
if numeric:
    transformers.append(("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric))
if categorical:
    transformers.append(("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical))
preprocess = ColumnTransformer(transformers)
stratify = y if task == "Classification" and int(len(y) * test_fraction) >= y.nunique() else None
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, random_state=int(seed), stratify=stratify)
except ValueError as exc:
    st.error(f"Split failed: {exc}")
    st.stop()
results = []
predictions = pd.DataFrame({"source_row": X_test.index, "actual": y_test.to_numpy()})
for name in models:
    if task == "Regression":
        estimator = RandomForestRegressor(n_estimators=200, random_state=int(seed), n_jobs=-1) if name == "Random Forest" else XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=int(seed), n_jobs=-1)
    else:
        n_classes = y.nunique()
        estimator = RandomForestClassifier(n_estimators=200, random_state=int(seed), n_jobs=-1) if name == "Random Forest" else XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=int(seed), n_jobs=-1, objective="multi:softprob" if n_classes > 2 else "binary:logistic")
    pipeline = Pipeline([("preprocess", preprocess), ("model", estimator)])
    try:
        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_test)
        predictions[name + "_prediction"] = predicted
        if task == "Regression":
            results.append({"Model": name, "MAE": mean_absolute_error(y_test, predicted), "RMSE": np.sqrt(mean_squared_error(y_test, predicted)), "R²": r2_score(y_test, predicted)})
        else:
            results.append({"Model": name, "Accuracy": accuracy_score(y_test, predicted), "Macro F1": f1_score(y_test, predicted, average="macro", zero_division=0)})
        try:
            names = pipeline.named_steps["preprocess"].get_feature_names_out()
            importance = pipeline.named_steps["model"].feature_importances_
            top = pd.Series(importance, index=names).nlargest(12).sort_values()
            fig, ax = plt.subplots(figsize=(8, 4))
            top.plot.barh(ax=ax)
            ax.set_title(f"{name}: model-specific feature importance")
            st.pyplot(fig)
            plt.close(fig)
        except Exception:
            st.caption(f"Feature importance unavailable for {name}.")
    except Exception as exc:
        st.warning(f"{name} training failed: {exc}")
if not results:
    st.stop()
st.subheader("Held-out performance")
st.dataframe(pd.DataFrame(results), use_container_width=True)
if task == "Regression":
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in models:
        column = name + "_prediction"
        if column in predictions:
            ax.scatter(predictions["actual"], predictions[column], alpha=0.6, label=name)
    lo = min(predictions["actual"].min(), predictions.filter(like="_prediction").min().min())
    hi = max(predictions["actual"].max(), predictions.filter(like="_prediction").max().max())
    ax.plot([lo, hi], [lo, hi], "--", color="gray")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
else:
    predictions["actual"] = encoder.inverse_transform(predictions["actual"].astype(int))
    for name in models:
        column = name + "_prediction"
        if column in predictions:
            predictions[column] = encoder.inverse_transform(predictions[column].astype(int))
st.download_button("Download held-out predictions CSV", predictions.to_csv(index=False).encode(), file_name="held_out_predictions.csv", mime="text/csv")
st.caption("Single random holdout only; no cross-validation, grouped leakage protection, or external validation. Model agreement is not calibrated confidence.")
