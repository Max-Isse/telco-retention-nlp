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
