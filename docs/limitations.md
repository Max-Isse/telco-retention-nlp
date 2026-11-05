# Limitations

## Data

- **IBM Telco is fictional.** It is a sample IBM publishes for analytics demos: realistic structure, but the churn
  rate (26.5%), the CLTV formula and every relationship in it describe this sample, not a real operator.
- **One snapshot, no dates.** There is no out-of-time test for the churn model (D-007), and survival on tenure relies
  on a cohort reading of the churn flag that IBM's description does not state (D-011).
- **No treatment data.** No offer was ever recorded, so the save rate, offer cost and backfire rate are assumptions
  and uplift cannot be estimated (D-015 to D-017).
- **Extended vs basic file.** The extended sample (Kaggle) matches IBM's file on every shared field except
  `Dependents` for 551 customers (D-002); `dependents` is one of the strongest predictors.
- **The tweets are not linked to the customers** (D-001), are public (customers who complain in public are not a
  random sample of customers), come from eight US brands, and end in December 2017.

## Modelling

- Gender and senior citizen status are model inputs; neither should be in a real deployment without review
  (`docs/model_card.md`).
- LightGBM and logistic regression are statistically tied; the choice rests on the Brier score and on SHAP drivers
  for messages, not on accuracy.
- 11 of 21 Cox covariates violate proportional hazards, so the unstratified hazard ratios are averages over tenure
  (D-012); remaining lifetime uses each contract's own monthly hazard and ignores the other covariates (D-013).
- Labels for the text models come from an LLM (kappa with the human labeller 0.50 to 0.65), and the human sample is
  one person's labels. The classifiers inherit the LLM's lean towards negative sentiment and normal urgency.
- Small theme classes (Wi-Fi: 3 test tweets) have wide uncertainty; the out-of-time test covers one event-heavy
  period (the iPhone X launch).

## Decision layer

- Profit figures are estimates under assumptions; at a 10% save rate the profit at the budget falls by about 70%, and
  with a transparent CLV instead of IBM's CLTV by about 95% (`reports/targeting.md` §2).
- Ranking by expected value targets likely churners with high value, not persuadable customers; only an uplift model
  fitted on randomised data can target the offer's effect.
- Messages are rules on SHAP drivers (an assumption), not measured best offers.
- The churn-driver comparison between tweets and IBM churn reasons compares shares, not effects.

## Engineering

- The served ticket models are TF-IDF (no torch); they give up 0.14 macro-F1 on themes and 0.11 on sentiment against
  the embeddings models (D-024). An image with CPU torch and the sentence-transformer is the upgrade.
- The API has no authentication, rate limiting or persistent prediction log; it is meant to sit behind a gateway.
- The scheduled workflow retrains on the same pinned data each week, so its model only changes if the code or the
  data pin changes; it shows the mechanics of a scheduled run, not a production feed.
- The Docker image is built and smoke-tested in CI, not on a production host.
- The contact list from the scheduled workflow is a workflow artifact; in a public repository signed-in GitHub users
  can download it (fine for IBM's fictional sample, not for real customers; D-025).
