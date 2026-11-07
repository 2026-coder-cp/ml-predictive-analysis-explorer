# ML Predictive Analysis Explorer v0.3

Streamlit research prototype comparing six regression models:
- Random Forest
- XGBoost
- Bayesian Ridge Regression
- Gaussian Process Regression (RBF + white noise kernel)
- Elastic Net
- Quadratic Response Surface Methodology (degree-2 polynomial with ridge regularization)

Classification continues to support Random Forest and XGBoost only.

## Run

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Workflow

1. Train: Upload labeled CSV; select regression, target, predictors and models. Shared fit/calibration/internal-test splits allow fair comparisons. Bayesian Ridge, Elastic Net and quadratic RSM use standardized numeric features. GPR uses a fitted kernel and is best suited to smaller datasets.
2. Predict: Upload a blind CSV with the same predictors. Fitted models predict without retraining. Bayesian Ridge and GPR additionally export model-based predictive standard deviations.
3. Validate: Upload the answer key. Use a unique `case_id` in both blind and answer-key files whenever possible; otherwise the app warns that it is matching by row position. Compare MAE, RMSE, R², bias, empirical split-conformal coverage and mean interval width.

## Reproducibility and interpretation

The original 30-row external dataset has been inspected and is now a development benchmark, not an untouched final test set for v0.3 model selection. Reserve a new independent dataset for final assessment.

All six models use absolute-residual split-conformal intervals from the same calibration partition. Bayesian Ridge/GPR predictive standard deviations are separate model-based uncertainty estimates; do not interpret them as calibrated confidence probabilities. GPR kernel fitting can be computationally expensive for larger datasets.

The default v0.3 model parameters are illustrative, not optimized. This is a research prototype, not a qualified medical-device or regulatory decision tool. Do not upload proprietary data without authorization.

## v0.2 milestone

The original v0.2 code is preserved at branch `milestone/v0.2`. The deployed app URL should remain unchanged when Streamlit Cloud tracks this repository's main branch.
