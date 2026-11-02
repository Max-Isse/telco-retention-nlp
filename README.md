# Telco retention: churn, survival, targeting and support-text NLP

Customer-retention analytics on two public data sets that are deliberately **not linked**:
- **IBM Telco Customer Churn** (7,043 customers): who is likely to leave, when, and who is worth an offer under a
  fixed budget;
- **Customer Support on Twitter** (telecom brands): which complaint themes drive dissatisfaction, connected to churn
  drivers at the level of themes and service types (D-001).

**Status:** phases 0–4 of 8 complete (data, leakage audit, EDA, churn models, survival, retention targeting). See [docs/phase_plan.md](docs/phase_plan.md).

## How to run

Requires [uv](https://docs.astral.sh/uv/). Everything runs in the project's uv virtual environment.

```bash
uv sync
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
  at the budget, with sensitivity, messages and the uplift test design ([reports/targeting.md](reports/targeting.md)).
- Every path, seed and business assumption is in [configs/config.yaml](configs/config.yaml).

## Layout

| Path | Contents |
|---|---|
| `src/telco_retention/` | `data/` (fetch, clean, contract, leakage audit), `eda.py`, `models/` (design, training, evaluation, SHAP), `survival.py`, `targeting.py`, `cli.py` |
| `configs/` | config and assumptions |
| `docs/` | standards, business case, methodology, phase plan, decisions, lineage |
| `reports/` | generated reports, tables and figures |
| `tests/` | pytest on small fixtures (no network) |
