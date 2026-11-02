# Methodology

Sections 1–2 cover the churn data and models (phases 1–2). Survival, targeting and NLP follow in later phases.
Results: [reports/eda.md](../reports/eda.md), [reports/churn_models.md](../reports/churn_models.md).

## 1. Data and unit of analysis

- **Data:** IBM's extended Telco sample (7,043 fictional customers, one snapshot each), verified against IBM's
  official file (D-002). The outcome is `churn` (left within the last month); the base rate is 26.5%.
- **Unit:** one customer. There is no time dimension, so no out-of-time test is possible; tenure (months with the
  company) carries the time information and is used for survival analysis in phase 3.
- **Leakage:** churn reason, score, label, value and status exist only after the outcome; they are excluded by name and
  a test asserts they never reach the features (D-003). CLTV and location are kept out of the model (D-004).

## 2. Churn models

- **Validation (D-007):** stratified 20% holdout scored once, with 2,000-resample bootstrap CIs; stratified 5-fold CV
  on the training part for every choice; segments on cross-validated predictions.
- **Baselines:** the base rate (ROC-AUC 0.5) and a rules model (month-to-month and tenure ≤ 12: ROC-AUC 0.67).
- **Logistic regression (D-009):** chosen for interpretability (odds ratios with CIs). Assumptions: log-odds linear in
  the inputs, no perfect collinearity, independent customers. Checked: the design is full rank (tested); tenure enters
  linearly, and the tenure-band calibration table shows where that is too simple (logistic predicts 0.47 against
  0.53 observed for tenure 0–6). The service flags explain most of the variance of monthly charges (D-008), so the
  separate effects of price and services are not identifiable; predictions are unaffected.
- **LightGBM (D-009):** captures non-linear tenure effects and interactions without specification. Fixed settings, no
  search.
- **Metrics:** ROC-AUC (ranking), PR-AUC (the positive class is 27%), Brier score and log loss (probability quality),
  expected calibration error and reliability curves, all with bootstrap CIs on the holdout.
- **Calibration (D-010):** compared none / Platt / isotonic by cross-fitted Brier; both models were already well
  calibrated. Calibration matters because targeting multiplies the probability by value at risk.
- **Explanations:** odds ratios from an unpenalised fit; SHAP for LightGBM compared with standardised logistic
  coefficients (directions agree for every feature; rank correlation 0.91).
- **Where it can fail:** IBM's customers are fictional and the snapshot cannot show seasonality or campaign effects;
  probabilities are calibrated to this base rate and need recalibration if the churn rate shifts; segment PR-AUC is
  low where churn is rare (two-year contracts, 2.8%), so rankings there carry little information.
