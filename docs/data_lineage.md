# Data lineage

```
IBM GitHub: Telco-Customer-Churn.csv (SHA-256 pinned)   ─┐ verified against each other (D-002)
Kaggle: IBM extended sample, telco_extended.xlsx (pinned) ─┘
  └─ data/raw/ (manifest: data/raw_manifest.json)         telco fetch
       └─ clean: types, blanks, duplicates                telco data (src/telco_retention/data/load.py)
            └─ pandera contract + leakage audit           src/telco_retention/data/validate.py
                 └─ data/processed/customers.parquet      model input for phases 2-4
```

| Step | Rows | Note | Decision |
|---|---|---|---|
| read (extended layout) | 7,043 | 33 columns; matches IBM's basic file except `Dependents` for 551 | D-002 |
| `TotalCharges` blank → 0 where tenure = 0 | 7,043 | 11 blank, all with tenure 0 | D-005 |
| drop duplicate `customer_id` | 7,043 | 0 removed | |
| contract passed; processed table saved | 7,043 | churn rate 26.5% | D-003 |

Generated each run: `reports/tables/lineage.csv` and `reports/tables/leakage_audit.csv`.

## Support tweets (`telco themes`)

| Step | Rows | Decision |
|---|---|---|
| all tweets in the dataset | 2,811,774 | |
| replies by the 8 US telecom brands | 147,639 | D-018 |
| customer tweets answered by those brands | 139,707 | |
| opening tweets (not replies) | 85,819 | |
| at least 4 words after cleaning | 83,351 | |
| duplicate texts removed | 83,138 | |
| themes assigned (20 clusters → 13 themes) | 83,138 | D-019 |

Generated each run: `reports/tables/nlp_lineage.csv`.

## Labels and served models

| Artifact | Built from | Step | Decision |
|---|---|---|---|
| `data/labels/label_sample_ids.csv` | 700 October + 300 November-onwards opening tweets, drawn at random once with the sizes in `nlp.label_sample` | committed; the ids file is the record | D-020 |
| `data/labels/llm_labels.csv` | the 1,000 sampled tweets, labelled by an LLM with `docs/labelling_guide.md` | | D-020 |
| `data/labels/human_labels.csv` | 200 of them (100 per period), labelled by hand, without the LLM labels | `telco import-labels` | D-020 |
| `artifacts/churn_model.joblib` (git-ignored) | `customers.parquet`, stratified 80% training part | `telco train` | D-009, D-010 |
| `artifacts/ticket_models.joblib` (git-ignored) | the 1,000 LLM labels and the tweets' cleaned text | `telco package-tickets` | D-024 |

Label files hold tweet ids and labels only; tweet text stays in `data/processed/` (git-ignored).

## Scoring

```
customers.parquet (churn = 0: 5,174 active customers) or a file in IBM's columns (--input)
  └─ prepare (same typing as clean) → design matrix → churn_model.joblib → calibrated p
       └─ value at risk (CLTV, else charges x margin x horizon) → expected value → contact list    telco score
            ├─ outputs/contact_list.csv, outputs/score_run.json (input hash, model, assumptions; git-ignored)
            └─ reports/scoring.md
```

The scheduled workflow (`.github/workflows/score.yml`) runs `fetch → data → train → score` weekly on the pinned
data and keeps the outputs as a 14-day workflow artifact (D-025).

