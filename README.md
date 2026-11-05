# Telco retention: churn, survival, targeting and support-text NLP

Customer-retention analytics on two public data sets that are deliberately **not linked** (D-001):
- **IBM Telco Customer Churn** (7,043 fictional customers): who is likely to leave, when, and who is worth a
  retention offer under a fixed budget;
- **Customer Support on Twitter** (83,138 opening tweets to eight US telecom brands, 2017): what customers complain
  about, classified by theme, sentiment and urgency, and compared with IBM's churn reasons at the level of drivers.

For a retention manager: [reports/recommendations.md](reports/recommendations.md).

## Results

All money figures are **estimates under assumptions** in [configs/config.yaml](configs/config.yaml) (save rate 30%,
offer 50, contact 2, backfire 2%, 500 contacts); there is no treatment data, which is why the first recommendation is
a randomised test.

| Question | Answer | Report |
|---|---|---|
| Can churn be predicted? | LightGBM, calibrated: ROC-AUC 0.862 (0.842 to 0.881), Brier 0.129 on a 1,409-customer holdout; tied with logistic regression (0.858), far above the month-to-month rule (0.669) | [churn_models.md](reports/churn_models.md) |
| When do customers leave? | month-to-month customers: 30% gone by month 12, 51% by month 36; two-year contracts have 0.08 times the churn hazard | [survival.md](reports/survival.md) |
| Who to contact? | ranking by expected value of the offer: 539k (503k to 572k) at 500 contacts, against 461k by churn probability, 264k by the current rule and 110k at random; first in every sensitivity scenario | [targeting.md](reports/targeting.md) |
| How big a test? | 73 customers per arm detect a 30% save rate, 596 per arm a 10% one | [targeting.md](reports/targeting.md) §4 |
| What do customers complain about? | 13 themes found by clustering sentence embeddings, then a 14-class taxonomy; reliability is 28% of complaints, customer service 15% | [themes.md](reports/themes.md), [themes_to_churn.md](reports/themes_to_churn.md) |
| Can it be classified? | out of time (October → November): themes macro-F1 0.58 with embeddings vs 0.44 TF-IDF; urgency 0.73; labels by an LLM, checked against 200 hand labels (kappa 0.50 to 0.65) | [nlp_classifiers.md](reports/nlp_classifiers.md) |
| How do complaints relate to churn? | public complaints are about reliability; IBM's churners cite price (27%) and customer service (24%); customer service is large in both | [themes_to_churn.md](reports/themes_to_churn.md) |

Deployment: a weekly contact list (`telco score`), a FastAPI service and a Docker image ([scoring.md](reports/scoring.md),
[ticket_serving.md](reports/ticket_serving.md)).

## How to run

Requires [uv](https://docs.astral.sh/uv/). Everything runs in the project's uv virtual environment.

```bash
uv sync --all-extras
```

```bash
uv run telco all
```

```bash
uv run pytest
```

Data: `fetch` downloads IBM's file; the extended IBM sample and the Twitter data come from Kaggle with the Kaggle CLI
([data/README.md](data/README.md)). Steps, in order:

| Step | Does | Writes |
|---|---|---|
| `fetch`, `data` | download (SHA-256 pinned), clean, data contract, leakage audit | `data/processed/customers.parquet`, lineage tables |
| `eda` | decision-focused exploration | [eda.md](reports/eda.md) |
| `train` | baselines, logistic regression, LightGBM, calibration, SHAP; saves the model | [churn_models.md](reports/churn_models.md), `artifacts/churn_model.joblib` |
| `survival` | Kaplan-Meier, Cox, remaining lifetime | [survival.md](reports/survival.md) |
| `target` | policies at the budget, sensitivity, messages, uplift test design | [targeting.md](reports/targeting.md) |
| `score` | contact list for the active customers (or `--input` a file in IBM's columns) | `outputs/contact_list.csv`, [scoring.md](reports/scoring.md) |
| `themes` | telecom tweets, embeddings, clusters, themes (needs the `nlp` extra) | [themes.md](reports/themes.md) |
| `classify` | theme, sentiment, urgency classifiers; human vs LLM agreement; churn drivers (needs the `nlp` extra) | [nlp_classifiers.md](reports/nlp_classifiers.md), [themes_to_churn.md](reports/themes_to_churn.md) |
| `package-tickets` | servable TF-IDF ticket classifiers | [ticket_serving.md](reports/ticket_serving.md), `artifacts/ticket_models.joblib` |

`import-labels` validates hand labels from `data/labels/human_to_label.csv`. The `nlp` extra (sentence embeddings,
torch) is only needed for `themes` and `classify`; CI installs the core only. A scheduled workflow
(`.github/workflows/score.yml`) retrains and scores weekly.

## API

```bash
uv run uvicorn --factory telco_retention.api:create_app --port 8000
```

```bash
docker build -t telco-retention .
```

```bash
docker run -p 8000:8000 -v "$PWD/artifacts:/app/artifacts:ro" telco-retention
```

`GET /health`; `POST /churn/score` (`{"customers": [...]}` in IBM's values; returns churn probability, value at risk,
expected value of an offer, whether to contact, top drivers and the message); `POST /tickets/classify`
(`{"texts": [...]}`; returns theme, sentiment and urgency). Models are read from `artifacts/`; interactive docs at
`/docs`.

## Documentation

| Document | Contents |
|---|---|
| [business_case.md](docs/business_case.md) | decisions, baseline, cost of errors, what the data can support |
| [methodology.md](docs/methodology.md) | validation, models, assumptions and failure modes per phase |
| [decisions.md](docs/decisions.md) | D-001 to D-025: every judgement call and why |
| [model_card.md](docs/model_card.md) | churn model and ticket classifiers: use, performance, subgroups, limits |
| [limitations.md](docs/limitations.md) | data, modelling, decision and engineering limits |
| [monitoring_plan.md](docs/monitoring_plan.md) | what to watch, thresholds and actions |
| [data_lineage.md](docs/data_lineage.md) | sources, filters and row counts, labels, scoring |
| [labelling_guide.md](docs/labelling_guide.md) | the taxonomy and label definitions |

## Layout

| Path | Contents |
|---|---|
| `src/telco_retention/` | `data/` (fetch, clean, contract, leakage audit), `eda.py`, `models/` (design, training, evaluation, SHAP), `survival.py`, `targeting.py`, `scoring.py` (batch scoring), `api.py` (FastAPI), `nlp/` (tweets, themes, classifiers, insight, served ticket models), `cli.py` |
| `configs/` | paths, seed and every assumption |
| `data/labels/` | sampled tweet ids, LLM labels, hand labels, sentiment failure review (ids and labels only) |
| `reports/` | generated reports, tables and figures; `recommendations.md` |
| `tests/` | pytest on small fixtures (no network) |
| `Dockerfile`, `.github/workflows/` | serving image; CI (lint, types, tests, image smoke test) and the scheduled scoring run |
