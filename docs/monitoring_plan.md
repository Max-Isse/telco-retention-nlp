# Monitoring plan

What to watch once the contact list and the API are used, the threshold that triggers action, and the action.
**Status:** this is a plan. In place today: input validation (pydantic on the API, the pandera contract in
`telco data`), the run manifest of each scoring run (`outputs/score_run.json`: input hash, model, assumptions, score
distribution), request IDs on every API response, and `/health`. The checks below marked *(to build)* are not
implemented.

## 1. Inputs (every scoring run)

| Check | Threshold | Action |
|---|---|---|
| Schema and allowed values (contract, payment method, services; tenure and charges ranges) | any failure | reject the record (API: 422; batch: the run stops with the count of bad rows) |
| Population stability index (PSI) of tenure, monthly charges, contract, internet service, payment method against the previous run's active base *(to build)* | 0.10 watch, 0.25 act | above 0.25: check for a product, price or data-feed change before using the list |
| Share of fibre and of month-to-month customers | ±5 points from the previous run | as above; these two drive most of the score |

## 2. Scores (every scoring run, from the manifest)

| Check | Threshold | Action |
|---|---|---|
| Mean churn probability of the active base (0.165 at launch) | ±0.03 | investigate inputs (section 1) before acting on the list |
| Customers with positive expected value (2,326 at launch) | ±20% | as above; or the assumptions changed |
| Overlap of this week's list with last week's *(to build)* | below 50% without an input change | check for a scoring or data error |

## 3. Outcomes (after each campaign window, on the randomised test and the permanent control)

| Check | Threshold | Action |
|---|---|---|
| Calibration on the control group: predicted vs actual churn by decile, expected calibration error | ECE above 0.05 | recalibrate on recent outcomes (calibrator only, D-010) |
| Ranking on the control group: ROC-AUC (0.86 on the holdout) | below 0.80 | retrain on recent outcomes; compare against the current model on the same period |
| Measured save rate (treated vs control churn) against the assumed 30% | outside the test's 95% interval | update `business.save_rate`, re-run `telco target`, revisit the budget (recommendations §4) |
| Realised value per contact | negative for two campaigns | stop the campaign type; review the offer |

Retraining rule: retrain on the latest labelled period; promote the new model only if its Brier score on the most
recent out-of-time period is not worse than the current model's (paired bootstrap), and record the decision.

## 4. Support text (weekly)

| Check | Threshold | Action |
|---|---|---|
| Weekly theme shares (`reports/themes_to_churn.md` §1) | a theme's share above twice its 4-week average | alert the owning team (outage spikes: network operations) |
| Share predicted "Other" (4.9% at launch) | above 10% | new topics: re-read a sample, extend the taxonomy (`docs/labelling_guide.md`) |
| Monthly re-label of 100 new tweets by a person *(to build)* | macro-F1 drop of 0.05 on theme or urgency | relabel and retrain; check the guide still fits |

## 5. Service

| Check | Threshold | Action |
|---|---|---|
| `/health` | "degraded" or unreachable | the image's health check marks the container unhealthy and the orchestrator restarts it; a missing model needs `telco train` or `telco package-tickets` |
| 4xx share by endpoint | above 5% of requests | a caller sends values outside IBM's vocabulary: fix the caller or extend the schema deliberately |
| p95 latency, 5xx share | 1 s; any 5xx | logs by request ID |
| Scheduled workflow | a failed run | rerun manually (`workflow_dispatch`); the previous list stays valid for one more week |

## Owners

Retention manager: sections 2 and 3 (decisions and outcomes). Data scientist: sections 1, 3 (retraining) and 4.
Platform: section 5.
