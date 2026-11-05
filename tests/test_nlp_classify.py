"""Classifiers, agreement, urgency rule, driver table and label import (no network: the transformer is mocked)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from telco_retention.config import load_config, with_root


def test_agreement_kappa() -> None:
    from telco_retention.nlp.classify import agreement

    human = pd.DataFrame(
        {
            "tweet_id": [1, 2, 3, 4],
            "theme_code": [1, 2, 3, 3],
            "sentiment": ["neg"] * 4,
            "urgency": ["high", "normal", "high", "normal"],
        }
    )
    llm = human.assign(theme_code=[1, 2, 3, 1])
    a = agreement(human, llm).set_index("label")
    assert a.loc["theme_code", "agreement"] == 0.75 and a.loc["urgency", "cohen_kappa"] == 1.0


def test_urgency_rule_flags_outages_and_cancellations() -> None:
    from telco_retention.nlp.classify import urgency_rule

    got = urgency_rule(pd.Series(["internet is down since 8am", "love the new plan", "ready to cancel"]))
    assert got.tolist() == ["high", "normal", "high"]


def _labelled(n_per_class: int = 30) -> tuple[pd.DataFrame, np.ndarray]:
    """Three separable themes in text and in a fake embedding space (test data only)."""
    rng = np.random.default_rng(0)
    words = {1: "outage internet down", 3: "slow speed mbps", 9: "bill charged twice"}
    rows, vecs = [], []
    for period in ("train", "test"):
        for code, w in words.items():
            for i in range(n_per_class):
                rows.append(
                    {
                        "row": len(rows) + 1,
                        "tweet_id": len(rows) + 1,
                        "period": period,
                        "theme_code": code,
                        "sentiment": "neg" if i % 3 else "neu",
                        "urgency": "high" if code == 1 else "normal",
                        "clean": f"{w} {i}",
                    }
                )
                v = np.zeros(8)
                v[[1, 3, 9].index(code)] = 1
                vecs.append(v + rng.normal(0, 0.1, 8))
    return pd.DataFrame(rows), np.array(vecs)


def test_classifiers_beat_the_majority_baseline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from telco_retention.nlp import classify

    cfg = with_root(load_config(), tmp_path)
    df, x = _labelled()
    monkeypatch.setattr(classify, "off_the_shelf_sentiment", lambda texts: np.array(["neg"] * len(texts)))
    res = classify.run_classifiers(df, x, pd.Series(["Internet outage"] * len(df)), cfg)
    scores = res.theme_scores.set_index("model").macro_f1
    assert scores["embeddings + logistic"] > 0.9 > scores["majority class"]
    assert res.meta["test_rows"] == 90 and res.confusion.to_numpy().sum() == 90
    assert set(res.urgency_scores.model) == {"embeddings + logistic", "keyword rule"}
    assert res.cluster_check["accuracy_all_1000"] == pytest.approx(1 / 3)


def test_driver_table(tmp_path: Path) -> None:
    from telco_retention.nlp.insight import driver_table

    cfg = with_root(load_config(), tmp_path)
    pred = pd.DataFrame({"theme_code": [1, 1, 3, 9], "sentiment": ["neg", "neu", "neg", "neg"]})
    customers = pd.DataFrame(
        {
            "churn": [1, 0, 1, 0],
            "churn_reason": ["Network reliability", None, "Price too high", None],
            "internet_service": ["Fiber optic", "DSL", "Fiber optic", "DSL"],
            "payment_method": ["Electronic check"] * 4,
            "tech_support": ["No"] * 4,
            "phone_service": [1] * 4,
            "streaming_tv": ["Yes"] * 4,
        }
    )
    d = driver_table(pred, customers, cfg).set_index("driver")
    rel = d.loc["Reliability (outages, drops, Wi-Fi, coverage)"]
    assert rel.tweet_share == 0.5 and rel.tweet_share_negative == 0.25 and rel.ibm_churn_reason_share == 0.5
    assert rel.churn_rate_with == 1.0 and rel.churn_rate_without == 0.0


def test_import_labels_rejects_incomplete_rows(tmp_path: Path) -> None:
    from telco_retention.cli import cmd_import_labels

    cfg = with_root(load_config(), tmp_path)
    labels = cfg.path("raw_dir").parent / "labels"
    labels.mkdir(parents=True)
    good = pd.DataFrame(
        {
            "row": [1, 2],
            "tweet_id": [10, 20],
            "text": ["a", "b"],
            "theme_code": ["1", "14"],
            "sentiment": ["NEG", "pos"],
            "urgency": ["high", "normal"],
            "notes": ["", ""],
        }
    )
    good.to_csv(labels / "human_to_label.csv", index=False, encoding="utf-8-sig")
    cmd_import_labels(cfg)
    out = pd.read_csv(labels / "human_labels.csv")
    assert "text" not in out.columns and out.sentiment.tolist() == ["neg", "pos"]
    good.assign(urgency=["high", ""]).to_csv(labels / "human_to_label.csv", index=False, encoding="utf-8-sig")
    with pytest.raises(SystemExit, match="1 rows incomplete"):
        cmd_import_labels(cfg)
