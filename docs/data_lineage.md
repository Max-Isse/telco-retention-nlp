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

