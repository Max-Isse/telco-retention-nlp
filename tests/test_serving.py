"""Batch scoring, contact list and the API, on models trained on the 298-row IBM fixture (no network)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

from telco_retention.config import Config, load_config, with_root

FIXTURES = Path(__file__).parent / "fixtures"
CUSTOMER = {
    "customer_id": "TEST-1",
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "No",
    "dependents": "No",
    "tenure": 2,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 85.0,
}
Setup = tuple[Config, pd.DataFrame, dict[str, Any]]


@pytest.fixture(scope="module")
def setup(tmp_path_factory: pytest.TempPathFactory) -> Setup:
    from telco_retention.data.load import RENAME, clean
    from telco_retention.models.report import evaluate_all, save_model
    from telco_retention.models.train import train_all

    base = load_config()
    raw = dict(base.raw)
    raw["models"] = {**base.section("models"), "cv_folds": 3, "bootstrap": 50}
    cfg = with_root(Config(raw=raw, root=base.root), tmp_path_factory.mktemp("serve"))
    df = pd.read_csv(FIXTURES / "telco_sample300.csv", dtype=str, keep_default_na=False)
    customers = clean(df.rename(columns=RENAME), "basic").customers
    res = train_all(customers, cfg)
    path = save_model(res, evaluate_all(res, cfg), cfg)
    return cfg, customers, joblib.load(path)


def test_raw_file_scores_like_the_cleaned_table(setup: Setup) -> None:
    from telco_retention.scoring import prepare, score

    cfg, customers, bundle = setup
    raw = pd.read_csv(FIXTURES / "telco_sample300.csv", dtype=str, keep_default_na=False)
    a = score(customers, bundle, cfg, drivers=False)
    b = score(prepare(raw), bundle, cfg, drivers=False)
    np.testing.assert_allclose(a.p_churn, b.p_churn)


def test_contact_list_respects_budget_and_positive_ev(setup: Setup) -> None:
    from telco_retention.scoring import run_batch

    cfg, customers, bundle = setup
    raw = dict(cfg.raw)
    raw["business"] = {**cfg.section("business"), "budget_contacts": 10}
    run = run_batch(customers[customers.churn == 0].reset_index(drop=True), bundle, Config(raw, cfg.root), "fixture")
    c = run.contacts
    assert len(c) <= 10 and (c.ev > 0).all() and c.ev.is_monotonic_decreasing
    assert c["rank"].tolist() == list(range(1, len(c) + 1))
    assert run.manifest["value_sources"] == {"charges x margin x horizon": len(run.scored)}  # basic file: no CLTV


def test_api_scores_and_validates(setup: Setup) -> None:
    from fastapi.testclient import TestClient

    from telco_retention.api import create_app

    cfg, _, _ = setup
    with TestClient(create_app(cfg)) as client:
        h = client.get("/health").json()
        assert h["status"] == "degraded" and h["churn_model"] and h["ticket_models"] is None
        r = client.post("/churn/score", json={"customers": [CUSTOMER, {**CUSTOMER, "cltv": 6000.0}]})
        assert r.status_code == 200 and r.headers["X-Request-ID"]
        a, b = r.json()
        assert 0 < a["p_churn"] < 1 and a["p_churn"] == b["p_churn"]
        assert b["value"] == 6000.0 and b["value_source"] == "cltv"
        bad = {**CUSTOMER, "contract": "Weekly"}
        assert client.post("/churn/score", json={"customers": [bad]}).status_code == 422
        assert client.post("/tickets/classify", json={"texts": ["internet down"]}).status_code == 503


def test_ticket_models_classify(tmp_path: Path) -> None:
    from telco_retention.nlp.tickets import classify_texts, fit_ticket_models

    cfg = with_root(load_config(), tmp_path)
    words = {1: "internet outage down", 3: "slow speed mbps", 9: "bill charged twice"}
    rows = [
        {
            "tweet_id": 1000 * code + 100 * (period == "test") + i,
            "period": period,
            "clean": f"my {w} again {i}",
            "theme_code": code,
            "sentiment": "neg" if i % 2 else "neu",
            "urgency": "high" if code == 1 else "normal",
        }
        for period in ("train", "test")
        for code, w in words.items()
        for i in range(20)
    ]
    bundle, scores = fit_ticket_models(pd.DataFrame(rows), cfg)
    assert scores.set_index("target").loc["theme_code", "macro_f1"] > 0.9
    out = classify_texts(["@ATT my internet outage is down again https://t.co/x"], bundle)[0]
    assert out["theme_code"] == 1 and out["urgency"] == "high" and "@ATT" not in out["clean_text"]
