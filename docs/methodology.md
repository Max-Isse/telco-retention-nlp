# Methodology

Sections 1–2 cover the churn data and models (phases 1–2), section 3 survival (phase 3), section 4 targeting
(phase 4). NLP follows.
Results: [reports/eda.md](../reports/eda.md), [reports/churn_models.md](../reports/churn_models.md),
[reports/survival.md](../reports/survival.md), [reports/targeting.md](../reports/targeting.md).

## 1. Data and unit of analysis

- **Data:** IBM's extended Telco sample (7,043 fictional customers, one snapshot each), verified against IBM's
  official file (D-002). The outcome is `churn` (IBM: "left within the last month"; treated as churn at the recorded tenure, D-011); the
  base rate is 26.5%.
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

## 3. Survival on tenure

- **Reading of the data (D-011):** IBM's literal description (churn in the last month) would make this a
  left-truncated snapshot; that reading implies a 62% monthly hazard at tenure 1 and almost no customer reaching four
  years, contradicting the 32% with 49+ months. The cohort reading (churners observed at their tenure at churn, stayers
  censored) is coherent and used. Assumption: censoring is independent of churn risk given covariates.
- **Kaplan-Meier by contract:** 72-month survival 13% (month-to-month), 57% (one year), 94% (two year); log-rank test
  rejects equality decisively.
- **Cox proportional hazards (D-012):** hazard ratios at the same tenure (two-year contract 0.08, one-year 0.24, having
  dependents 0.37, no internet 0.39, fibre 1.36). Assumption checked with Schoenfeld residuals: 11 of 21 covariates
  violate proportionality, so their ratios are tenure averages; a model stratified by contract removes the assumption
  for the strongest violator.
- **Remaining lifetime (D-013):** expected months within 24, conditional on tenure, from contract-specific monthly
  hazards; used as a cross-check on CLTV and for the fallback CLV.
- **Where it can fail:** stationarity of hazards across cohorts is assumed; extrapolation beyond 72 months; the sample is
  fictional and its churn rate is far above real monthly churn, so absolute survival is illustrative.

## 4. Retention targeting

- **Probabilities (D-014):** calibrated out-of-fold predictions for every customer; the targeted group's predicted churn
  (74.0%) matches its actual churn (76.0%).
- **Expected value (D-015):** EV = p·V·s − offer − contact − (1 − p)·b·V, with value at risk from IBM's CLTV or a
  transparent CLV estimate (margin × charges × expected remaining life, discounted).
- **Policies (D-016):** random, rules, top churn probability and top expected value at a fixed budget, ranked on
  predictions and scored on outcomes with bootstrap CIs; profit curves and one-at-a-time sensitivity.
- **Assumptions:** save rate, offer and contact cost, backfire rate and budget are assumptions (no treatment data); a
  constant save rate across customers is the strongest one, since uplift varies by customer.
- **Uplift (D-017):** true uplift needs randomisation: a treatment/control split of the customers the policy would
  contact, sized with a two-proportion power calculation, plus a permanent holdout and an uplift model afterwards.
- **Where it can fail:** if the save rate is low or concentrated in customers who would not churn anyway, ranking by EV
  overstates profit; CLTV's formula is unknown; offers may also change the behaviour of stayers (backfire), which is
  only assumed here.

