# Phase plan

Each phase ends with tests green, lint clean and a summary (built, results, assumptions, open questions).
Standards: `docs/project_standards.md`. Environment: uv venv (`uv sync`, `uv run telco <step>`); no Docker needed
until the API image.

| Phase | Scope | Main outputs | Status |
|---|---|---|---|
| 0 | Repo, uv venv, standards, business case, phase plan, decisions | this file, `docs/`, `configs/config.yaml`, CI | done |
| 1 | IBM Telco download (checksum pinned), cleaning, pandera contract, leakage audit (test), decision-focused EDA, lineage | `data/`, `eda.py`, `reports/eda.md`, D-002–D-005 | done |
| 2 | Churn models: base rate and rules baselines, logistic regression (odds ratios, CIs), LightGBM; stratified CV + holdout, bootstrap CIs; calibration (recalibrate if needed); segments by contract, tenure, service; SHAP vs coefficients | `models/`, `reports/churn_models.md` | |
| 3 | Survival: Kaplan–Meier by contract, Cox PH on tenure, PH assumption checks, hazard ratios; expected remaining lifetime for CLV | `survival/`, `reports/survival.md` | |
| 4 | Retention targeting: value at risk (CLTV or CLV estimate), expected value per customer, policies at a fixed budget (random, top probability, top EV, rules), profit curves, sensitivity; randomised uplift test design (sample size, holdout) | `targeting/`, `reports/targeting.md` | |
| 5 | Support text: Twitter dataset filtered to telecom brands (filter and volumes documented), cleaning, threading customer tweet → brand reply; theme discovery with sentence embeddings + BERTopic or clustering, labelled and validated by reading samples | `nlp/`, `reports/themes.md` | |
| 6 | Hand-labelled sample and taxonomy (LLM labels only with measured agreement); TF-IDF + logistic vs embedding classifier, out-of-time split, macro-F1, confusion matrix; sentiment and urgency vs an off-the-shelf model, failure cases; theme volumes over time; themes → churn drivers | `reports/nlp_classifiers.md`, `reports/themes_to_churn.md` | |
| 7 | Batch scoring (customer file → ranked contact list for a budget), FastAPI (`/churn/score`, `/tickets/classify`, `/health`), Dockerfile, scheduled GitHub Actions workflow | `serving/`, `docker/`, `.github/workflows/` | |
| 8 | Recommendations for a retention manager, model card, limitations, monitoring plan, lineage, how-to-run, README | `reports/recommendations.md`, `docs/` | |
