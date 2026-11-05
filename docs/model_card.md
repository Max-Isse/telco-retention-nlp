# Model card

Two served model bundles, both produced by the CLI and loaded by the API from `artifacts/`.

## 1. Churn model (`artifacts/churn_model.joblib`, `telco train`)

| | |
|---|---|
| Type | LightGBM classifier (300 trees, 15 leaves, learning rate 0.03; fixed settings, no search; D-009) with sigmoid calibration fitted on out-of-fold predictions (D-010) |
| Output | calibrated probability of churn; combined with value at risk and the business assumptions into the expected value of an offer (`scoring.py`, D-015) |
| Training data | IBM Telco extended sample (fictional, 7,043 customers, one snapshot; SHA-256 pinned): stratified 80% (5,634 customers, 26.5% churn) |
| Inputs | 22 design features from contract, tenure, charges, services, payment and demographics (`models/design.py`, D-008); outcome-derived fields (churn reason, score, label, value, status) excluded and tested (D-003) |
| Use | ranking customers for a retention offer under a budget; decision support for a retention manager |

### Intended use

Weekly ranking of active customers by expected value of a retention offer (`telco score`), and scoring of single
customers through `/churn/score`. A person decides on the campaign; the list is a recommendation.

### Not intended for

- Real customers of a real operator without retraining and validation on that operator's data.
- Decisions about an individual's price, credit or service level.
- Estimating the effect of an offer: the model predicts churn without an offer, not the change an offer makes
  (uplift; recommendations §2).

### Performance (holdout of 1,409 customers, scored once; `reports/churn_models.md`)

| Metric | LightGBM (calibrated) | Logistic regression | Rules baseline | Base rate |
|---|---|---|---|---|
| ROC-AUC | 0.862 (0.842 to 0.881) | 0.858 | 0.669 | 0.500 |
| PR-AUC | 0.679 (0.629 to 0.729) | 0.680 | 0.390 | 0.265 |
| Brier score | 0.129 (0.119 to 0.140) | 0.130 | 0.173 | 0.195 |
| Expected calibration error | 0.020 | 0.020 | 0.010 | 0.000 |

LightGBM and logistic regression are not distinguishable on the holdout (paired AUC difference 0.004, 95% CI −0.004
to 0.012). LightGBM was chosen for its lower cross-validated Brier score and because its SHAP drivers per customer feed
the messages; the logistic model's odds ratios agree with SHAP on direction for all 22 features (rank correlation
0.91).

### Subgroups (out-of-fold, training part)

| Group | Customers | Churn rate | Mean predicted | ROC-AUC |
|---|---|---|---|---|
| Female | 2,744 | 0.267 | 0.267 | 0.863 |
| Male | 2,890 | 0.263 | 0.264 | 0.857 |
| Not senior | 4,693 | 0.234 | 0.235 | 0.865 |
| Senior citizen | 941 | 0.422 | 0.417 | 0.801 |
| Month-to-month | 3,109 | 0.428 | 0.423 | 0.773 |
| One year | 1,182 | 0.107 | 0.115 | 0.790 |
| Two year | 1,343 | 0.028 | 0.033 | 0.773 |

Calibration holds in every group. Ranking is weaker for senior citizens (0.80) and within contract types (0.77 to
0.79), because contract explains much of the overall AUC. Gender is an input with no predictive value (odds ratio
0.97, p = 0.72); senior citizen status is an input with a small effect (odds ratio 1.12, p = 0.23). Both are
protected or sensitive attributes in many jurisdictions: before real use, drop them, retrain and confirm the
performance is unchanged.

### Caveats

- The snapshot has no time dimension: no out-of-time test is possible (D-007).
- Probabilities for the snapshot's active customers in `telco score` are in-sample (D-023).
- The expected value depends on assumptions that have not been measured (save rate, offer cost, backfire) and on
  IBM's undisclosed CLTV (recommendations §3).

## 2. Ticket classifiers (`artifacts/ticket_models.joblib`, `telco package-tickets`)

| | |
|---|---|
| Type | three TF-IDF (1–2-grams) + logistic regression pipelines, class-balanced: theme (14 classes), sentiment (neg/neu/pos), urgency (high/normal) (D-024) |
| Training data | 1,000 opening tweets to eight US telecom support accounts (3 October to 3 December 2017), labelled by an LLM (Claude) with `docs/labelling_guide.md`; 200 also hand-labelled to measure label quality (D-020) |
| Label quality | human vs LLM Cohen's kappa: theme 0.65, sentiment 0.50, urgency 0.65; the LLM labels lean negative and under-flag urgency (`reports/nlp_classifiers.md`) |
| Use | routing and counting support messages by theme and urgency |

### Performance (trained on October, tested on November onwards; `reports/ticket_serving.md`)

| Target | Macro-F1 (LLM labels, 300 tweets) | Macro-F1 (hand labels, 100 tweets) | Phase 6 embeddings model |
|---|---|---|---|
| Theme | 0.44 (0.37 to 0.49) | 0.41 | 0.58 |
| Sentiment | 0.50 (0.46 to 0.53) | 0.35 | 0.61 |
| Urgency | 0.71 (0.64 to 0.77) | 0.73 | 0.73 |

### Not intended for

- Sentiment as a measure of customer satisfaction: positive messages are almost absent from the training labels and
  the model misses praise.
- Text outside US telecom customer support in English, or language after 2017 (new products, plans and slang).
- Decisions about individual customers: the tweets are not linked to customer records.
