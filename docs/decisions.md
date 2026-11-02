# Decisions

Judgement calls, with the alternatives considered.

## D-001 Two unlinked datasets; the NLP connects to churn through themes, never through customers
**Choice:** churn modelling on IBM Telco and NLP on Twitter support text are separate pipelines. They meet only at
the level of themes and service types (for example billing complaints ↔ payment method and charges; internet
reliability ↔ fibre and tech support). Any customer-level linkage would be invented; if built for illustration it
is labelled SYNTHETIC and excluded from headline results.
**Why:** the datasets share no customer key, and pretending otherwise would fabricate evidence.

## D-002 Data: IBM's extended sample from Kaggle, verified against IBM's official file
**Choice:** the model data is IBM's extended Telco sample (7,043 × 33: adds CLTV, Churn Score, Churn Reason and
location), downloaded with the Kaggle CLI (`yeanzc/telco-customer-churn-ibm-dataset`) and pinned by SHA-256. It is
verified against IBM's own basic file on IBM's GitHub (also pinned): all 7,043 customer IDs match and every shared
field is identical except `Dependents`, which differs for 551 customers (517 No → Yes, 34 Yes → No). The extended
values are used; the extended file is IBM's later release of the sample.
**Why:** the brief asks for the extended version; CLTV gives value at risk from the data instead of an assumed CLV
formula, and the outcome-derived fields make the leakage audit real. The verification guards against a modified
re-upload. Without the extended file the loader falls back to IBM's basic CSV.
**Alternative:** the basic file only, with a CLV estimate (kept as the fallback path).

## D-003 Outcome-derived fields are excluded by name and the exclusion is tested
**Choice:** `leakage.forbidden_features` lists fields that exist only because the outcome is known (churn reason,
score, category, label, value, status). `features()` drops them with the id and the target; a test builds an
extended-layout frame containing them and asserts none reaches the features, and `telco data` fails if the audit
finds one. **Why:** on the extended file these fields would make the model trivially perfect.

## D-004 Location and CLTV are not model features
**Choice:** the extended file's location columns and CLTV are kept for analysis and value at risk, not fed to the
churn model. **Why:** both layouts then train the same model; location has no counterpart in the basic file, and
CLTV is a value estimate made by the company, used on the value side of the expected-value formula.

## D-005 Blank TotalCharges are customers in their first month
**Choice:** all 11 blank `TotalCharges` belong to customers with tenure 0; they are set to 0 (nothing billed yet).
**Alternative:** drop the rows (loses the newest customers, the most churn-prone group) or impute monthly charges.

## D-006 One uv environment, no Docker for the analysis
**Choice:** the whole project runs in a uv-managed `.venv` (`uv sync`, `uv run telco <step>`); Docker appears only
for the API image in phase 7. **Why:** no JVM or Spark is needed; one environment keeps runs fast and reproducible.

## D-007 Validation without a time dimension: stratified holdout, stratified CV, bootstrap CIs
**Choice:** a stratified 20% holdout (1,409 customers) scored once; stratified 5-fold CV on the remaining 5,634 for
every choice (calibration method, model used for targeting); 2,000-resample bootstrap CIs on holdout metrics, and a
paired bootstrap for the LightGBM − logistic difference. Segment results use the cross-validated predictions of the
training part (more customers per segment than the holdout).
**Why:** Telco is one snapshot with no dates, so out-of-time testing is impossible (stated in the report); survival
analysis on tenure (phase 3) is the substitute for the time dimension.

## D-008 One design matrix; add-on services as flags; total charges left out
**Choice:** add-on services become 0/1 flags (their "No internet service" level duplicates `internet_service = No`),
multi-level categoricals are one-hot with a reference level (month-to-month, DSL, electronic check), and
`total_charges` is excluded (≈ tenure × monthly charges). Both models use the same 22 columns.
**Consequence:** the service flags explain almost all of the variance of monthly charges, so the logistic model cannot
separate the effect of price from the services bought; individual odds ratios for fibre and streaming have wide
intervals, while their joint effect and the predictions are stable.

## D-009 Models: baselines, logistic regression, LightGBM with fixed settings
**Choice:** base rate and a rules baseline (month-to-month and tenure ≤ 12, as a probability learned on training
rows); logistic regression (L2, C = 1, for scoring; unpenalised statsmodels fit for odds ratios and CIs); LightGBM
with fixed, conservative settings (300 trees, 15 leaves, at least 40 rows per leaf). No hyperparameter search: on
5.6k rows the gain would be small and need nested CV to report honestly.

## D-010 Calibration chosen by cross-fitted Brier; the targeting model chosen by cross-validated Brier
**Choice:** for each model, `none`, Platt (`sigmoid`) and `isotonic` are compared by Brier score on out-of-fold
predictions calibrated by the other folds; the best is refitted on all OOF predictions. The model saved for targeting
is the one with the lowest cross-validated Brier score (LightGBM, by 0.002). Targeting multiplies probabilities by
value at risk, so calibration matters more than ranking.
**Result:** both models were already well calibrated (logistic: no recalibration; LightGBM: Platt, Brier change
< 0.001), and LightGBM and logistic regression are statistically indistinguishable on the holdout (ROC-AUC difference
+0.004, 95% CI −0.004 to +0.012). The logistic model remains the explanation of record: SHAP directions agree on all
22 features and importance ranks correlate at 0.91.

## D-011 Survival: the cohort reading of the churn flag, after testing IBM's literal wording
**Choice:** churners are treated as observed at the tenure at which they left and stayers as censored at their current
tenure (standard right-censored survival on tenure).
**Why:** IBM describes the flag as "left within the last month". Read literally (a monthly snapshot, each customer at
risk only in their last month, left-truncated), the monthly hazard at tenure 1 would be 62% and the probability of
reaching 48 months essentially 0, while 31.8% of customers have 49+ months of tenure and a 26.5% one-month churn rate
is far beyond any real telco. The literal reading is inconsistent with the data; the check is computed in
`survival.consistency_check` and reported. The sample is fictional, so absolute survival describes this sample only.

## D-012 Cox model: average hazard ratios plus a model stratified by contract
**Choice:** report the Cox model on all covariates and a model stratified by contract (own baseline hazard per
contract), with a Schoenfeld-residual test against rank of time for every covariate.
**Why:** 11 of 21 covariates violate proportional hazards, including the one-year contract; their hazard ratios are
tenure-averaged effects. Stratifying removes the assumption for contract, the largest violator among strong effects.
Time-varying coefficients for every covariate would fit better but be harder to use; the churn models (phase 2)
already capture non-linear tenure effects for scoring.

## D-013 Remaining lifetime from the contract-specific monthly hazard, conditional on tenure
**Choice:** expected months still a customer within the 24-month horizon, from the discrete monthly hazard of the
customer's contract at their current tenure; beyond 72 months, each contract's pooled hazard over its last 12 months.
**Use:** a cross-check on IBM's CLTV and the basis of the fallback CLV estimate in phase 4.

## D-014 Out-of-fold probabilities for every customer drive targeting
**Choice:** targeting uses calibrated, out-of-fold churn probabilities for all 7,043 customers (the phase 2 model type
and calibration rule, refitted per fold), so policies are compared on customers no model saw in training.
**Why:** in-sample probabilities would overstate how well any policy finds churners. Predicted churn among the top-EV
customers (74.0%) matches their actual churn rate (76.0%).

## D-015 Expected value formula and value at risk
**Choice:** EV = p·V·s − offer cost − contact cost − (1 − p)·b·V (the brief's formula, with the backfire cost of
contacting a stayer). V is IBM's CLTV (primary). The alternative V is a transparent CLV estimate: margin × monthly
charges × expected remaining months (phase 3), discounted. Both are run.
**Why / caveat:** CLTV is supplied by IBM from an undisclosed formula, correlates only 0.09 with the CLV estimate and is
about ten times larger; it changes the scale of profit and partly who is targeted. Top-EV targeting wins under both.

## D-016 Policies ranked on predictions, scored on outcomes
**Choice:** each policy orders customers (random; rules: flagged customers first in random order; top probability;
top EV) and is scored on actual churn, with the save and backfire rates applied in expectation; 95% bootstrap CIs
resample customers and keep each policy's ranking.
**Why:** scoring on outcomes rewards a policy that actually finds churners and value, not one that only believes it does.

## D-017 Messages from SHAP drivers; uplift needs a randomised test
**Choice:** each targeted customer's message comes from their top positive SHAP drivers through ordered rules in
config (contract upgrade, first-year welcome, fibre check, security bundle, auto-pay credit, price review). The save
rate is an assumption; the report sizes a randomised treatment/control test (two-proportion z-test) and recommends a
permanent holdout and an uplift model once test data exist.
**Why:** without treatment data, uplift cannot be estimated; offer–driver matching is a hypothesis to test, not a result.
