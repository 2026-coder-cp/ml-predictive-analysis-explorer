# ML Predictive Analysis Explorer

CSV-driven Streamlit research prototype comparing Random Forest and XGBoost for regression or classification.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Use Streamlit Community Cloud (https://share.streamlit.io/) and choose this GitHub repository, main branch, and `app.py`. GitHub Pages cannot execute Python.

## Features

- CSV upload, target and feature selection
- Random Forest and XGBoost training
- Holdout performance: MAE, RMSE, R² or accuracy and macro F1
- Feature importance, regression scatter plot, prediction CSV download

## Limitations

Research demonstration only. Single holdout split; no cross-validation, hyperparameter tuning, leakage detection, external validation, or clinical/regulated qualification. Do not upload confidential or employer data to public hosting without authorization. Preprocessing is fitted only on the training partition through scikit-learn pipelines.
