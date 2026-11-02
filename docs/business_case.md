# Business case

## The decisions

1. **Who to contact with a retention offer, and with what message**, under a limited retention budget
   (`business.budget_contacts`, ASSUMPTION). Made by the retention manager each campaign cycle.
2. **Which complaint themes to fix first**, made by support and operations leads from the themes customers raise.

## Baseline (what happens without a model)

- Random contact, or a simple rule: month-to-month contract and tenure of at most 12 months. On IBM's data that rule
  flags 28% of customers and finds 55% of churners at 51% precision (`reports/eda.md`).
- Complaint themes read anecdotally from tickets, without volumes or trends.

## Cost of each error (ASSUMPTIONS, `business` in `configs/config.yaml`)

| Outcome | What happens | Cost used |
|---|---|---|
| Contact a customer who would have left, offer works | value kept | + value at risk − offer − contact |
| Contact a customer who would have left, offer fails | customer leaves anyway | − offer − contact |
| Contact a customer who would have stayed | offer wasted; contact can trigger churn | − offer − contact − backfire × value |
| Miss a customer who leaves | future value lost | − value at risk (no spend) |

- Value at risk comes from IBM's CLTV when the extended file is used. Otherwise it is a documented CLV estimate:
  monthly charges × margin over a discounted horizon, with remaining lifetime from the survival model (phase 3).
- Save rate, offer cost, contact cost and backfire rate are assumptions: no treatment data exists. They are varied
  in sensitivity analysis. Measuring true uplift needs a randomised campaign, designed in phase 4.

## What the data can and cannot support

- **IBM Telco** is a fictional customer base published by IBM as a sample: realistic structure, one snapshot, no
  time dimension, and no record of offers made. Rankings and calibration can be validated; uplift cannot.
- **Twitter customer support** (Kaggle, ThoughtVector) is real public text from brands' support accounts. It is not
  linked to Telco customers, so the NLP work stands alone and connects to churn only at the level of themes and
  service types (D-001).
