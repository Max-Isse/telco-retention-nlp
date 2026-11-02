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
