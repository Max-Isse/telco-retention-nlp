# Telco retention: churn, survival, targeting and support-text NLP

Customer-retention analytics on two public data sets that are deliberately **not linked**:
- **IBM Telco Customer Churn** (7,043 customers): who is likely to leave, when, and who is worth an offer under a
  fixed budget;
- **Customer Support on Twitter** (telecom brands): which complaint themes drive dissatisfaction, connected to churn
  drivers at the level of themes and service types (D-001).

**Status:** phases 0–7 of 8 complete (data, leakage audit, EDA, churn models, survival, retention targeting,
support-text themes, classifiers and churn drivers, batch scoring and the API). See [docs/phase_plan.md](docs/phase_plan.md).

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

- `fetch` downloads IBM's file and checks its pinned SHA-256 (the extended IBM sample comes from Kaggle; see
  [data/README.md](data/README.md)); `data` cleans it, enforces the data contract, audits
  leakage and writes `data/processed/customers.parquet`; `eda` writes [reports/eda.md](reports/eda.md); `train` fits the baselines, logistic regression and LightGBM
  and writes [reports/churn_models.md](reports/churn_models.md); `survival` writes Kaplan-Meier, Cox and
  remaining-lifetime results to [reports/survival.md](reports/survival.md); `target` compares targeting policies
  at the budget, with sensitivity, messages and the uplift test design ([reports/targeting.md](reports/targeting.md));
  `themes` filters the Twitter support data to US telecoms and finds complaint themes ([reports/themes.md](reports/themes.md));
  `classify` trains theme, sentiment and urgency classifiers out of time and links themes to churn drivers
  ([reports/nlp_classifiers.md](reports/nlp_classifiers.md), [reports/themes_to_churn.md](reports/themes_to_churn.md)).
- `score` writes the ranked contact list for the budget (`outputs/contact_list.csv`, summary in
  [reports/scoring.md](reports/scoring.md)); `package-tickets` saves the servable ticket classifiers
  ([reports/ticket_serving.md](reports/ticket_serving.md)). A scheduled workflow (`.github/workflows/score.yml`)
  retrains and scores weekly.
- The `nlp` extra (sentence embeddings, CPU torch) is needed only for `themes` and `classify`; CI installs the core only.

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
(`{"texts": [...]}`; returns theme, sentiment and urgency). Models come from `artifacts/` (`telco train`,
`telco package-tickets`); interactive docs at `/docs`.
- Every path, seed and business assumption is in [configs/config.yaml](configs/config.yaml).

## Layout

| Path | Contents |
|---|---|
| `src/telco_retention/` | `data/` (fetch, clean, contract, leakage audit), `eda.py`, `models/` (design, training, evaluation, SHAP), `survival.py`, `targeting.py`, `scoring.py` (batch scoring), `api.py` (FastAPI), `nlp/` (tweets, themes, classifiers, insight, served ticket models), `cli.py` |
| `configs/` | config and assumptions |
| `docs/` | standards, business case, methodology, phase plan, decisions, lineage |
| `reports/` | generated reports, tables and figures |
| `tests/` | pytest on small fixtures (no network) |
