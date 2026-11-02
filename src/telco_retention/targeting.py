"""Retention targeting: expected value per customer, policies at a fixed budget, profit curves, sensitivity, messages,
and the design of the randomised test that would measure true uplift.

Expected value of contacting customer i (D-015), with value at risk V_i, churn probability p_i and the assumptions
save rate s, offer cost c_o, contact cost c_c, backfire rate b (`business`):
    EV_i = p_i * V_i * s  -  c_o  -  c_c  -  (1 - p_i) * b * V_i
Policies are ranked on predicted quantities, then scored on what actually happened (churn labels), with s and b applied
in expectation: a contacted churner yields s * V - c_o - c_c, a contacted stayer -c_o - c_c - b * V (D-016).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from telco_retention.config import Config
from telco_retention.data.load import features
from telco_retention.models.design import design_matrix
from telco_retention.models.train import Calibrator, RulesModel, factories


@dataclass(frozen=True)
class Assumptions:
    save_rate: float
    offer_cost: float
    contact_cost: float
    backfire_rate: float

    @classmethod
    def from_config(cls, cfg: Config, **overrides: float) -> Assumptions:
        b = cfg.section("business")
        base = {
            "save_rate": float(b["save_rate"]),
            "offer_cost": float(b["offer_cost"]),
            "contact_cost": float(b["contact_cost"]),
            "backfire_rate": float(b["backfire_rate"]),
        }
        return cls(**{**base, **overrides})


def oof_probabilities(customers: pd.DataFrame, cfg: Config, model: str) -> np.ndarray:
    """Calibrated out-of-fold churn probability for every customer (model and calibration as chosen in phase 2)."""
    x = design_matrix(features(customers, cfg))
    y = customers["churn"].astype(int).to_numpy()
    folds = list(
        StratifiedKFold(int(cfg.section("models")["cv_folds"]), shuffle=True, random_state=cfg.seed).split(x, y)
    )
    make = factories(cfg, list(x.columns))[model]
    raw = np.zeros(len(x))
    for tr, va in folds:
        raw[va] = make().fit(x.iloc[tr], y[tr]).predict_proba(x.iloc[va])[:, 1]
    method = choose_calibration(raw, y, folds, cfg)
    cal = np.zeros(len(x))
    for tr, va in folds:  # each fold calibrated by a calibrator fitted on the other folds
        cal[va] = Calibrator(method).fit(raw[tr], y[tr]).transform(raw[va])
    return cal


def choose_calibration(raw: np.ndarray, y: np.ndarray, folds: list[Any], cfg: Config) -> str:
    best, best_score = "none", np.inf
    for method in cfg.section("models")["calibration_methods"]:
        out = np.zeros(len(y))
        for tr, va in folds:
            out[va] = Calibrator(method).fit(raw[tr], y[tr]).transform(raw[va])
        score = float(np.mean((out - y) ** 2))
        if score < best_score:
            best, best_score = method, score
    return best


def clv_estimate(customers: pd.DataFrame, remaining_months: np.ndarray, cfg: Config) -> np.ndarray:
    """Margin on monthly charges over the expected remaining months (survival, phase 3), discounted monthly."""
    b = cfg.section("business")
    monthly_rate = (1 + float(b["annual_discount_rate"])) ** (1 / 12) - 1
    # average discount factor over the expected remaining months (simple approximation of a discounted annuity)
    discount = np.array(
        [np.mean((1 + monthly_rate) ** -np.arange(1, max(round(float(m)), 1) + 1)) for m in remaining_months]
    )
    return customers["monthly_charges"].to_numpy(float) * float(b["monthly_margin"]) * remaining_months * discount


def expected_value(p: np.ndarray, value: np.ndarray, a: Assumptions) -> np.ndarray:
    return p * value * a.save_rate - a.offer_cost - a.contact_cost - (1 - p) * a.backfire_rate * value


def realised_value(churn: np.ndarray, value: np.ndarray, a: Assumptions) -> np.ndarray:
    """Value of contacting each customer given what happened (save and backfire applied in expectation)."""
    return np.where(
        churn == 1,
        a.save_rate * value - a.offer_cost - a.contact_cost,
        -a.offer_cost - a.contact_cost - a.backfire_rate * value,
    )


def policy_orders(frame: pd.DataFrame, cfg: Config, seed: int) -> dict[str, np.ndarray]:
    """Contact order (customer positions) for each policy; the first k entries are contacted at budget k."""
    rng = np.random.default_rng(seed)
    n = len(frame)
    x = design_matrix(features(frame, cfg))
    flagged = RulesModel(int(cfg.section("models")["rules_max_tenure"]), list(x.columns))._flag(x)
    rules_order = np.r_[rng.permutation(np.where(flagged)[0]), rng.permutation(np.where(~flagged)[0])]
    return {
        "random": rng.permutation(n),
        "rules (month-to-month, tenure <= 12)": rules_order,
        "top churn probability": np.argsort(-frame["p_churn"].to_numpy(), kind="stable"),
        "top expected value": np.argsort(-frame["ev"].to_numpy(), kind="stable"),
    }


def profit_curves(frame: pd.DataFrame, orders: dict[str, np.ndarray], realised: np.ndarray) -> pd.DataFrame:
    out = {"contacted": np.arange(0, len(frame) + 1)}
    for name, order in orders.items():
        out[name] = np.r_[0.0, np.cumsum(realised[order])]
    return pd.DataFrame(out)


def bootstrap_profit(
    frame: pd.DataFrame, orders: dict[str, np.ndarray], realised: np.ndarray, budget: int, n: int, seed: int
) -> pd.DataFrame:
    """CI of profit at the budget: resample customers, keep each policy's ranking, contact the top `budget` share."""
    rng = np.random.default_rng(seed)
    size = len(frame)
    rows = []
    rank_of = {name: np.empty(size, dtype=int) for name in orders}
    for name, order in orders.items():
        rank_of[name][order] = np.arange(size)
    samples = {name: np.empty(n) for name in orders}
    for i in range(n):
        idx = rng.integers(0, size, size)
        for name in orders:
            pick = idx[np.argsort(rank_of[name][idx], kind="stable")[:budget]]
            samples[name][i] = realised[pick].sum()
    for name, vals in samples.items():
        lo, hi = np.percentile(vals, [2.5, 97.5])
        rows.append({"policy": name, "profit_ci_low": float(lo), "profit_ci_high": float(hi)})
    return pd.DataFrame(rows)


def message_for(drivers: list[str], cfg: Config) -> str:
    tc = cfg.section("targeting")
    for d in drivers:
        for rule in tc["messages"]:
            if d.startswith(rule["driver"]):
                return str(rule["action"])
    return str(tc["default_message"])


def top_drivers(model: Any, x: pd.DataFrame, k: int = 3) -> list[list[str]]:
    """The k features pushing each customer's churn risk up most (positive SHAP contributions)."""
    values = shap.TreeExplainer(model).shap_values(x)
    values = values[1] if isinstance(values, list) else values
    cols = np.array(x.columns)
    out = []
    for row in values:
        idx = [i for i in np.argsort(-row) if row[i] > 0][:k]
        out.append(list(cols[idx]))
    return out


def uplift_test_design(p_control: float, cfg: Config) -> pd.DataFrame:
    """Customers per arm to detect a relative churn reduction among treated, two-sided two-proportion z-test."""
    tc = cfg.section("targeting")["uplift_test"]
    power = NormalIndPower()
    rows = []
    for s in tc["save_rates"]:
        p_treat = p_control * (1 - float(s))
        es = proportion_effectsize(p_control, p_treat)
        n = power.solve_power(effect_size=es, alpha=float(tc["alpha"]), power=float(tc["power"]), ratio=1.0)
        rows.append(
            {
                "true_save_rate": float(s),
                "churn_control": p_control,
                "churn_treated": p_treat,
                "customers_per_arm": int(np.ceil(n)),
                "total": int(2 * np.ceil(n)),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class TargetingResult:
    frame: pd.DataFrame
    curves: pd.DataFrame
    at_budget: pd.DataFrame
    sensitivity: pd.DataFrame
    test_design: pd.DataFrame
    messages: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)


def at_budget_table(
    frame: pd.DataFrame, orders: dict[str, np.ndarray], realised: np.ndarray, budget: int
) -> pd.DataFrame:
    rows = []
    for name, order in orders.items():
        pick = order[:budget]
        rows.append(
            {
                "policy": name,
                "contacted": budget,
                "churners_reached": int(frame["churn"].to_numpy()[pick].sum()),
                "precision": float(frame["churn"].to_numpy()[pick].mean()),
                "value_at_risk_reached": float(
                    frame.loc[frame.index[pick][frame["churn"].to_numpy()[pick] == 1], "value"].sum()
                ),
                "expected_profit": float(realised[pick].sum()),
            }
        )
    return pd.DataFrame(rows)


def run_targeting(
    customers: pd.DataFrame, remaining_months: np.ndarray, model_bundle: dict[str, Any], cfg: Config
) -> TargetingResult:
    tc, b = cfg.section("targeting"), cfg.section("business")
    budget = int(b["budget_contacts"])
    a = Assumptions.from_config(cfg)
    frame = customers[["customer_id", "churn", "contract", "tenure", "monthly_charges", "cltv"]].copy()
    frame["p_churn"] = oof_probabilities(customers, cfg, model_bundle["name"])
    frame["clv_estimate"] = clv_estimate(customers, remaining_months, cfg)
    frame["value"] = frame[tc["value_source"]].to_numpy(float)
    frame["ev"] = expected_value(frame.p_churn.to_numpy(), frame.value.to_numpy(), a)
    realised = realised_value(frame.churn.to_numpy(), frame.value.to_numpy(), a)
    orders = policy_orders(customers.assign(p_churn=frame.p_churn, ev=frame.ev), cfg, cfg.seed)
    curves = profit_curves(frame, orders, realised)
    at_b = at_budget_table(frame, orders, realised, budget).merge(
        bootstrap_profit(frame, orders, realised, budget, int(tc["bootstrap"]), cfg.seed), on="policy"
    )
    best_k = int(np.argmax(curves["top expected value"].to_numpy()))

    sens = []
    s_cfg = tc["sensitivity"]
    base = {"save_rate": a.save_rate, "offer_cost": a.offer_cost, "backfire_rate": a.backfire_rate}
    scenarios: list[tuple[str, dict[str, float], str]] = [("base", {}, tc["value_source"])]
    scenarios += [
        (f"{k} = {v}", {k: float(v)}, tc["value_source"]) for k in base for v in s_cfg[k] if float(v) != base[k]
    ]
    scenarios += [(f"value_source = {v}", {}, v) for v in s_cfg["value_source"] if v != tc["value_source"]]
    for label, over, source in scenarios:
        aa = Assumptions.from_config(cfg, **over)
        value = frame[source].to_numpy(float)
        ev = expected_value(frame.p_churn.to_numpy(), value, aa)
        real = realised_value(frame.churn.to_numpy(), value, aa)
        ords = policy_orders(customers.assign(p_churn=frame.p_churn, ev=ev), cfg, cfg.seed)
        row: dict[str, Any] = {"scenario": label}
        for name, order in ords.items():
            row[name] = float(real[order[:budget]].sum())
        cum = np.r_[0.0, np.cumsum(real[ords["top expected value"]])]
        row["best_contacts_top_ev"] = int(np.argmax(cum))
        row["customers_with_positive_ev"] = int((ev > 0).sum())
        sens.append(row)

    p_targeted = float(frame.p_churn.to_numpy()[orders["top expected value"][:budget]].mean())
    x_all = design_matrix(features(customers, cfg)).reindex(columns=model_bundle["columns"], fill_value=0.0)
    targeted = orders["top expected value"][:budget]
    drivers = top_drivers(model_bundle["model"], x_all.iloc[targeted]) if model_bundle["name"] == "lightgbm" else []
    msgs = pd.DataFrame(
        {
            "customer_id": frame.customer_id.to_numpy()[targeted],
            "p_churn": frame.p_churn.to_numpy()[targeted],
            "value": frame.value.to_numpy()[targeted],
            "ev": frame.ev.to_numpy()[targeted],
            "top_drivers": [", ".join(d) for d in drivers] if drivers else "",
            "message": [message_for(d, cfg) for d in drivers] if drivers else tc["default_message"],
        }
    )
    meta = {
        "budget": budget,
        "assumptions": a.__dict__,
        "value_source": tc["value_source"],
        "best_contacts_top_ev": best_k,
        "best_profit_top_ev": float(curves["top expected value"].max()),
        "positive_ev_customers": int((frame.ev > 0).sum()),
        "p_targeted": p_targeted,
        "cltv_vs_clv_corr": float(np.corrcoef(frame.cltv, frame.clv_estimate)[0, 1]),
        "model": model_bundle["name"],
    }
    return TargetingResult(frame, curves, at_b, pd.DataFrame(sens), uplift_test_design(p_targeted, cfg), msgs, meta)


def _md(df: pd.DataFrame, floatfmt: str = ",.0f") -> str:
    return str(df.astype(object).to_markdown(index=False, floatfmt=floatfmt)) if len(df) else "_none_"


def write_targeting_report(res: TargetingResult, cfg: Config) -> Path:
    tables, figs = cfg.path("tables_dir"), cfg.path("figures_dir")
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    res.at_budget.to_csv(tables / "targeting_at_budget.csv", index=False)
    res.sensitivity.to_csv(tables / "targeting_sensitivity.csv", index=False)
    res.test_design.to_csv(tables / "targeting_uplift_test_design.csv", index=False)
    res.curves.iloc[::10].to_csv(tables / "targeting_profit_curves.csv", index=False)
    m = res.meta
    budget = m["budget"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name in res.curves.columns[1:]:
        ax.plot(res.curves.contacted, res.curves[name], label=name)
    ax.axvline(budget, color="0.4", ls="--", lw=1, label=f"budget {budget}")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel("customers contacted (in policy order)")
    ax.set_ylabel("expected profit (assumed units)")
    ax.set_title("Profit curve by targeting policy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "targeting_profit_curves.png", dpi=120)
    plt.close(fig)
    msg_counts = res.messages["message"].value_counts().rename_axis("message").reset_index(name="customers")
    seg = (
        res.frame.iloc[np.argsort(-res.frame.ev.to_numpy(), kind="stable")[:budget]]
        .groupby("contract", observed=True)
        .size()
        .rename("customers")
        .reset_index()
    )
    a = m["assumptions"]
    d = res.test_design
    n30 = int(d.loc[d.true_save_rate == 0.3, "customers_per_arm"].iloc[0]) if (d.true_save_rate == 0.3).any() else 0
    text = f"""<!-- GENERATED by `telco target` - do not edit by hand. -->
# Retention targeting

**Estimates under assumptions** (`business` in `configs/config.yaml`): save rate {a["save_rate"]:.0%}, offer cost
{a["offer_cost"]:.0f}, contact cost {a["contact_cost"]:.0f}, backfire rate {a["backfire_rate"]:.0%}, budget {budget:,}
contacts; value at risk = **{m["value_source"]}**. There is no treatment data: the save rate is assumed, which is why a
randomised test (section 4) is needed before any of this is trusted. Design: D-014 to D-017.

Churn probabilities are calibrated and out-of-fold for every customer ({m["model"]}, as chosen in phase 2), so no
customer is scored by a model that saw them. Policies rank customers on predictions; profit is scored on actual churn,
with the save and backfire rates applied in expectation.

## 1. Policies at the budget

{_md(res.at_budget.assign(precision=res.at_budget.precision.map(lambda v: f"{v:.1%}")))}

`profit_ci_*`: 95% bootstrap interval ({cfg.section("targeting")["bootstrap"]:,} resamples of customers).
{m["positive_ev_customers"]:,} customers have positive expected value; the top-EV profit curve peaks at
{m["best_contacts_top_ev"]:,} contacts ({m["best_profit_top_ev"]:,.0f}), so the budget of {budget:,} binds.

Contract mix of the {budget:,} top-EV customers:

{_md(seg)}

![profit curves](figures/targeting_profit_curves.png)

## 2. Sensitivity (profit at the budget by policy; each row re-ranks)

{_md(res.sensitivity)}

IBM's CLTV and the transparent CLV estimate (margin on monthly charges over expected remaining life, discounted)
correlate at {m["cltv_vs_clv_corr"]:.2f}; CLTV comes from an undisclosed IBM formula and is about ten times larger, so
the value source changes both the scale of the profit and who is targeted.

## 3. What to say: messages from each customer's top churn drivers

Drivers are the features with the largest positive SHAP contributions for the customer (LightGBM); the first matching
rule in `targeting.messages` picks the offer (an ASSUMPTION to be tested, not a measured effect).

{_md(msg_counts)}

## 4. Measuring true uplift: the randomised test

Contacting customers changes the outcome, so the save rate cannot be learned from history without a control group.
Design: take the customers the policy would contact, randomise them into treatment (contacted) and control (not
contacted), and compare churn after the campaign window. Required customers per arm to detect a given relative churn
reduction with a two-sided z-test (alpha {cfg.section("targeting")["uplift_test"]["alpha"]}, power
{cfg.section("targeting")["uplift_test"]["power"]:.0%}), at the predicted churn rate of the top-EV customers
({m["p_targeted"]:.1%}):

{_md(res.test_design, ",.3f")}

- A 30% save rate needs about {n30:,} customers per arm;
  smaller effects need several campaigns or a larger targeted group.
- Keep a permanent random holdout (for example 10% of each campaign) to keep measuring uplift, then fit an uplift
  model (two-model or transformed-outcome) on the test data to target persuadable customers rather than likely churners.
"""
    path = cfg.path("reports_dir") / "targeting.md"
    path.write_text(text, encoding="utf-8")
    return path
