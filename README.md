# ML Predictive Analysis Explorer v0.2

Streamlit research prototype comparing Random Forest and XGBoost with **separate training, blind prediction, and external validation**.

## Run

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Three-tab workflow

1. **Train:** Upload `training_300.csv`; target `seal_strength_N_per_15mm`; four process predictors; Regression; both models; seed 42. The app splits the training CSV into fitting, calibration and internal holdout sets. Models are retained in the current Streamlit session.
2. **Predict:** Upload `external_test_30_blind.csv`, click **Predict using retained models**, and download `external_predictions.csv`. The fitted models are not retrained.
3. **Validate:** Upload `external_test_30_answer_key.csv` only after predictions. Download `external_validation_results.csv`. The provided example files lack `case_id`; the app matches them by **unchanged row order** and warns about this. For future datasets, use a unique `case_id` in both external files for safer matching.

## Interpretation

Regression reports MAE, RMSE, R² and mean prediction bias. The calibration split provides symmetric absolute-residual split-conformal prediction intervals, with empirical coverage reported on internal and external sets. These are marginal coverage estimates under exchangeability, not confidence probabilities for individual observations. Classification reports accuracy and macro F1; regression intervals are not available for classification.

## Limitations

Research prototype only; not validated for clinical or regulated decisions. Model/session state is lost when the Streamlit session or server restarts. No persistent model storage, grouped splitting, automated leakage detection, cross-validation or hyperparameter tuning. Do not use external outcomes to tune models or calibrate intervals. Do not upload proprietary or confidential datasets without approval.

## Deployment

Streamlit Community Cloud deploys the repository's `main` branch and `app.py`. Existing deployment URL should remain unchanged when the connected app follows this repository.
