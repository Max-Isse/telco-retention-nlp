"""Design matrix, baselines, calibration, metrics and the full training + report path on 298 real IBM rows
(`fixtures/telco_sample300.csv`: up to 50 customers per contract type and churn class)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from telco_retention.config import Config, load_config, with_root

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def customers() -> pd.DataFrame:
    from telco_retention.data.load import RENAME, clean

    raw = pd.read_csv(FIXTURES / "telco_sample300.csv", dtype=str, keep_default_na=False)
    return clean(raw.rename(columns=RENAME), "basic").customers


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    base = load_config()
    raw = dict(base.raw)
    raw["models"] = {**base.section("models"), "cv_folds": 3, "bootstrap": 50}
    return with_root(Config(raw=raw, root=base.root), tmp_path)


def test_design_matrix_is_full_rank_and_leak_free(customers: pd.DataFrame, cfg: Config) -> None:
    from telco_retention.data.load import features
    from telco_retention.models.design import design_matrix

    x = design_matrix(features(customers, cfg))
    assert np.linalg.matrix_rank(np.c_[np.ones(len(x)), x.to_numpy()]) == x.shape[1] + 1  # no redundant dummies
    assert "contract=month-to-month" not in x.columns and "contract=two_year" in x.columns  # reference dropped
    assert not {"churn", "customer_id", "total_charges"} & set(x.columns)
    assert x.dtypes.eq(float).all()


def test_baselines_and_calibrator() -> None:
    from telco_retention.models.train import BaseRateModel, Calibrator, RulesModel

    x = pd.DataFrame({"tenure": [3.0, 30.0, 5.0, 2.0], "contract=one_year": [0.0, 0.0, 1.0, 0.0]})
    y = np.array([1, 0, 0, 1])
    rules = RulesModel(12, list(x.columns)).fit(x, y)
    assert rules.predict_proba(x)[:, 1].tolist() == [1.0, 0.0, 0.0, 1.0]  # flagged: month-to-month and short tenure
    assert BaseRateModel().fit(x, y).predict_proba(x)[0, 1] == 0.5
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 5000)
    yy = (rng.uniform(0, 1, 5000) < p**2).astype(int)  # p is overconfident: the truth is p squared
    for method in ("sigmoid", "isotonic"):
        cal = Calibrator(method).fit(p, yy).transform(p)
        assert np.mean((cal - yy) ** 2) < np.mean((p - yy) ** 2)
    assert np.array_equal(Calibrator("none").fit(p, yy).transform(p), p)


def test_metrics_and_bootstrap_cis() -> None:
    from telco_retention.models.evaluate import bootstrap, ci, metrics

    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 600)
    good, noise = np.clip(y * 0.6 + rng.uniform(0, 0.4, 600), 0, 1), rng.uniform(0, 1, 600)
    assert metrics(y, good)["roc_auc"] > 0.9 > metrics(y, noise)["roc_auc"]
    b = bootstrap(y, {"good": good}, 200, seed=2)["good"]
    lo, hi = ci(b[:, 0])
    assert lo <= metrics(y, good)["roc_auc"] <= hi


def test_training_holds_out_customers_and_writes_the_report(customers: pd.DataFrame, cfg: Config) -> None:
    from telco_retention.models.report import evaluate_all, save_model, write_report
    from telco_retention.models.train import train_all

    res = train_all(customers, cfg)
    assert not set(res.train.customer_id) & set(res.holdout.customer_id)  # holdout customers never trained on
    assert len(res.train) + len(res.holdout) == len(customers)
    assert abs(res.train.churn.mean() - res.holdout.churn.mean()) < 0.05  # stratified
    for col in ("p_logistic_cal", "p_lightgbm_cal", "p_rules"):
        assert res.train[col].between(0, 1).all()
    ev = evaluate_all(res, cfg)
    assert ev["chosen"] in ("logistic", "lightgbm") and len(ev["holdout"]) == 6
    path = write_report(res, ev, cfg, save_model(res, ev, cfg))
    assert "no time dimension" in path.read_text(encoding="utf-8")
    import joblib

    bundle = joblib.load(cfg.path("artifacts_dir") / "churn_model.joblib")
    assert bundle["columns"] == res.columns and bundle["name"] == ev["chosen"]
