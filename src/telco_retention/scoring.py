"""Batch and on-request churn scoring with the deployed model: probability, value at risk, expected value of an
offer, top drivers and the suggested message, and the ranked contact list under the budget (D-023).

The deployed model is the phase 2 bundle (`artifacts/churn_model.joblib`: the chosen model refitted on the training
part, with its calibrator). Expected value uses the same formula and assumptions as phase 4 (`targeting.py`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from telco_retention.config import Config
from telco_retention.data.load import CATEGORICAL, RENAME, YES_NO, features
from telco_retention.models.design import design_matrix
from telco_retention.targeting import Assumptions, expected_value, message_for, top_drivers


def load_bundle(cfg: Config) -> dict[str, Any]:
    import joblib

    path = cfg.path("artifacts_dir") / "churn_model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `telco train` first")
    bundle: dict[str, Any] = joblib.load(path)
    return bundle


def prepare(records: pd.DataFrame) -> pd.DataFrame:
    """Type raw records in IBM's vocabulary (Yes/No, category labels) the way `clean()` types the training data."""
    out = records.rename(columns={c: RENAME.get(c, c) for c in records.columns}).copy()
    for c in ["tenure", "monthly_charges", "total_charges", "cltv"]:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["senior_citizen"] = out["senior_citizen"].astype(str).str.strip().map({"1": 1, "0": 0, "Yes": 1, "No": 0})
    for c in YES_NO:
        out[c] = out[c].astype(str).str.strip().map({"Yes": 1, "No": 0})
    for c in CATEGORICAL:
        out[c] = out[c].astype(str).str.strip().astype("category")
    required = ["tenure", "monthly_charges", "senior_citizen", *YES_NO]
    bad = out[required].isna().any(axis=1)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} records have missing or invalid values in {required}")
    return out


def value_at_risk(customers: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """IBM's CLTV where present (phase 4's value source); otherwise margin on monthly charges over the configured
    horizon, discounted monthly (the survival-based estimate needs the Cox model, not deployed)."""
    b = cfg.section("business")
    monthly_rate = (1 + float(b["annual_discount_rate"])) ** (1 / 12) - 1
    horizon = int(b["clv_horizon_months"])
    discount = float(np.mean((1 + monthly_rate) ** -np.arange(1, horizon + 1)))
    fallback = customers["monthly_charges"].to_numpy(float) * float(b["monthly_margin"]) * horizon * discount
    if "cltv" in customers:
        cltv = pd.to_numeric(customers["cltv"], errors="coerce").to_numpy(float)
        has = ~np.isnan(cltv)
        return np.where(has, cltv, fallback), np.where(has, "cltv", "charges x margin x horizon")
    return fallback, np.full(len(customers), "charges x margin x horizon")


def describe(column: str, value: float) -> str:
    """A driver in words with the customer's value: `tenure 2`, `no contract=two_year`, `internet_service=fiber_optic`.
    A 0/1 column pushes risk up either by being present or by being absent, so the value is part of the reason."""
    if column in ("tenure", "monthly_charges"):
        return f"{column} {value:g}"
    return column if value >= 0.5 else f"no {column}"


def score(customers: pd.DataFrame, bundle: dict[str, Any], cfg: Config, drivers: bool = True) -> pd.DataFrame:
    """One row per customer: calibrated churn probability, value at risk, expected value of an offer, drivers."""
    x = design_matrix(features(customers, cfg)).reindex(columns=bundle["columns"], fill_value=0.0)
    raw = bundle["model"].predict_proba(x)[:, 1]
    p = bundle["calibrator"].transform(raw)
    value, source = value_at_risk(customers, cfg)
    ev = expected_value(p, value, Assumptions.from_config(cfg))
    ids = customers["customer_id"] if "customer_id" in customers else pd.Series(range(1, len(customers) + 1))
    out = pd.DataFrame({"customer_id": ids.to_numpy(), "p_churn": p, "value": value, "value_source": source, "ev": ev})
    if drivers and bundle["name"] == "lightgbm":
        d = top_drivers(bundle["model"], x)
        out["top_drivers"] = ["; ".join(describe(c, float(x.iloc[i][c])) for c in v) for i, v in enumerate(d)]
        out["message"] = [message_for(v, cfg) for v in d]
    else:
        out["top_drivers"] = ""
        out["message"] = str(cfg.section("targeting")["default_message"])
    return out


def contact_list(scored: pd.DataFrame, budget: int) -> pd.DataFrame:
    """Highest expected value first; only customers with positive expected value, at most `budget` of them."""
    ranked = scored[scored.ev > 0].sort_values("ev", ascending=False, kind="stable").head(budget)
    return ranked.reset_index(drop=True).assign(rank=lambda d: np.arange(1, len(d) + 1))


@dataclass
class ScoreRun:
    scored: pd.DataFrame
    contacts: pd.DataFrame
    manifest: dict[str, Any]


def run_batch(customers: pd.DataFrame, bundle: dict[str, Any], cfg: Config, source: str) -> ScoreRun:
    budget = int(cfg.section("business")["budget_contacts"])
    scored = score(customers, bundle, cfg, drivers=False)
    contacts = contact_list(scored, budget)
    # drivers (SHAP) only for the customers on the list
    with_drivers = score(customers[customers.customer_id.isin(contacts.customer_id)], bundle, cfg)
    contacts = contacts.drop(columns=["top_drivers", "message"]).merge(
        with_drivers[["customer_id", "top_drivers", "message"]], on="customer_id"
    )
    digest = hashlib.sha256(pd.util.hash_pandas_object(customers, index=False).to_numpy().tobytes()).hexdigest()
    manifest = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "input": source,
        "input_rows": len(customers),
        "input_hash": digest[:16],
        "model": bundle["name"],
        "calibration": bundle["calibrator"].method,
        "budget": budget,
        "assumptions": Assumptions.from_config(cfg).__dict__,
        "contacted": len(contacts),
        "positive_ev_customers": int((scored.ev > 0).sum()),
        "expected_value_of_list": float(contacts.ev.sum()),
        "p_churn_mean": float(scored.p_churn.mean()),
        "p_churn_mean_contacted": float(contacts.p_churn.mean()) if len(contacts) else float("nan"),
        "p_churn_quantiles": {str(q): float(scored.p_churn.quantile(q)) for q in (0.1, 0.5, 0.9)},
        "value_sources": scored.value_source.value_counts().to_dict(),
    }
    return ScoreRun(scored, contacts, manifest)


def write_outputs(run: ScoreRun, cfg: Config) -> list[Path]:
    """The contact list and run manifest (git-ignored `outputs/`), and a summary report (committed)."""
    out = cfg.path("outputs_dir")
    out.mkdir(parents=True, exist_ok=True)
    cols = ["rank", "customer_id", "p_churn", "value", "value_source", "ev", "top_drivers", "message"]
    p1 = out / "contact_list.csv"
    run.contacts[cols].to_csv(p1, index=False, float_format="%.4f")
    p2 = out / "score_run.json"
    p2.write_text(json.dumps(run.manifest, indent=2, default=str) + "\n", encoding="utf-8")
    m = run.manifest
    msgs = run.contacts.message.value_counts().rename_axis("message").reset_index(name="customers")
    decile = pd.qcut(run.scored.p_churn, 10, labels=False, duplicates="drop")
    by_decile = (
        run.scored.assign(decile=(decile + 1).astype(int))
        .groupby("decile")
        .agg(
            customers=("p_churn", "size"), p_churn_mean=("p_churn", "mean"), positive_ev=("ev", lambda s: (s > 0).sum())
        )
        .reset_index()
    )
    report = f"""<!-- GENERATED by `telco score` - do not edit by hand. -->
# Batch scoring run

Run {m["run_at"]} on `{m["input"]}`: {m["input_rows"]:,} customers (input hash `{m["input_hash"]}`), model
**{m["model"]}** ({m["calibration"]} calibration), budget {m["budget"]} contacts, assumptions {m["assumptions"]}.

- Customers with positive expected value: **{m["positive_ev_customers"]:,}**; contacted: **{m["contacted"]}**.
- Expected value of the list (under the assumptions): **{m["expected_value_of_list"]:,.0f}**.
- Mean churn probability: {m["p_churn_mean"]:.3f} for all scored, {m["p_churn_mean_contacted"]:.3f} for those contacted.
- Value at risk from: {m["value_sources"]}.

The list itself (`outputs/contact_list.csv`: rank, customer id, probability, value, expected value, top drivers,
message) and the run manifest (`outputs/score_run.json`) are written outside git.

## Scored customers by churn-probability decile

{by_decile.astype(object).to_markdown(index=False, floatfmt=".3f")}

## Messages on the list

{msgs.to_markdown(index=False) if len(msgs) else "_none_"}

## Caveat (D-023)

IBM's sample is one snapshot, so the active customers scored here are also customers the model was trained on: their
probabilities are in-sample. In production the model is trained on last period's outcomes and scores this period's
active base. The evaluated performance of this policy is phase 4's, on out-of-fold probabilities.
"""
    p3 = cfg.path("reports_dir") / "scoring.md"
    p3.write_text(report, encoding="utf-8")
    return [p1, p2, p3]
