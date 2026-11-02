"""Expected value, policy ordering, bootstrap, messages, the uplift test design, and the targeting run end to end."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from telco_retention.config import Config, load_config, with_root

FIXTURES = Path(__file__).parent / "fixtures"


def test_expected_and_realised_value_formulas() -> None:
    from telco_retention.targeting import Assumptions, expected_value, realised_value

    a = Assumptions(save_rate=0.3, offer_cost=50, contact_cost=2, backfire_rate=0.02)
    ev = expected_value(np.array([0.5]), np.array([1000.0]), a)[0]
    assert ev == pytest.approx(0.5 * 1000 * 0.3 - 52 - 0.5 * 0.02 * 1000)
    real = realised_value(np.array([1, 0]), np.array([1000.0, 1000.0]), a)
    assert real.tolist() == [pytest.approx(300 - 52), pytest.approx(-52 - 20)]
    # with calibrated p, the realised value equals the EV in expectation
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200_000)
    churn = (rng.uniform(0, 1, p.size) < p).astype(int)
    v = np.full(p.size, 1000.0)
    assert realised_value(churn, v, a).mean() == pytest.approx(expected_value(p, v, a).mean(), rel=0.02)


def test_uplift_test_needs_fewer_customers_for_bigger_effects(tmp_path: Path) -> None:
    from telco_retention.targeting import uplift_test_design

    d = uplift_test_design(0.5, with_root(load_config(), tmp_path))
    assert d.customers_per_arm.is_monotonic_decreasing
    # textbook check: 50% vs 40% churn (save rate 0.2), alpha 0.05, power 0.8 -> about 385-390 per arm
    assert 380 <= int(d.loc[d.true_save_rate == 0.2, "customers_per_arm"].iloc[0]) <= 395


def test_message_rules_pick_the_first_matching_driver(tmp_path: Path) -> None:
    from telco_retention.targeting import message_for

    cfg = with_root(load_config(), tmp_path)
    assert message_for(["tenure", "contract=two_year"], cfg).startswith("Welcome call")
    assert message_for(["payment_method=mailed_check"], cfg).startswith("Bill credit")
    assert message_for(["male"], cfg) == cfg.section("targeting")["default_message"]


def test_bootstrap_interval_contains_the_point_estimate() -> None:
    from telco_retention.targeting import bootstrap_profit

    rng = np.random.default_rng(1)
    realised = rng.normal(10, 50, 2000)
    order = np.argsort(-realised)
    frame = pd.DataFrame(index=range(2000))
    ci = bootstrap_profit(frame, {"best": order}, realised, budget=200, n=200, seed=2).iloc[0]
    assert ci.profit_ci_low < realised[order[:200]].sum() * 1.05 and ci.profit_ci_high > 0


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    base = load_config()
    raw = dict(base.raw)
    raw["models"] = {**base.section("models"), "cv_folds": 3, "bootstrap": 30}
    raw["business"] = {**base.section("business"), "budget_contacts": 40}
    raw["targeting"] = {**base.section("targeting"), "bootstrap": 30}
    return with_root(Config(raw=raw, root=base.root), tmp_path)


def test_targeting_end_to_end(cfg: Config) -> None:
    from telco_retention.data.load import RENAME, clean
    from telco_retention.models.report import evaluate_all
    from telco_retention.models.train import train_all
    from telco_retention.targeting import run_targeting, write_targeting_report

    raw = pd.read_csv(FIXTURES / "telco_sample300.csv", dtype=str, keep_default_na=False)
    customers = clean(raw.rename(columns=RENAME), "basic").customers
    customers["cltv"] = 2000 + customers["tenure"] * 50  # the basic file has no CLTV: test value only
    res_models = train_all(customers, cfg)
    chosen = evaluate_all(res_models, cfg)["chosen"]
    bundle = {"name": "lightgbm", "model": res_models.models["lightgbm"], "columns": res_models.columns}
    res = run_targeting(customers, np.full(len(customers), 12.0), bundle, cfg)
    at = res.at_budget.set_index("policy")
    assert at.loc["top expected value", "expected_profit"] >= at.loc["random", "expected_profit"]
    top = res.frame.ev.to_numpy()
    assert set(res.messages.customer_id) == set(res.frame.customer_id.to_numpy()[np.argsort(-top, kind="stable")[:40]])
    assert chosen in ("logistic", "lightgbm")
    assert "randomised test" in write_targeting_report(res, cfg).read_text(encoding="utf-8")
