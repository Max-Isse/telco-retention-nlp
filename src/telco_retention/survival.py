"""Survival analysis on tenure: which reading of the data, Kaplan-Meier by contract, Cox PH with checks, remaining life.

Two readings of IBM's churn flag are tested (D-011):
- snapshot (IBM's wording, "left within the last month"): each customer is at risk only in their last month, so the
  analysis is left-truncated at tenure - 1;
- cohort: churners are recorded at the tenure at which they left and stayers are censored at their current tenure.
The snapshot reading implies a monthly hazard so high that almost no customer would survive a year, which contradicts
the 32% of customers with 49+ months of tenure; `consistency_check` computes this. The cohort reading is coherent and
is used for every result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test

from telco_retention.config import Config
from telco_retention.models.design import design_matrix

MIN_DURATION = 0.5  # customers in their first month (tenure 0) are censored at half a month


def survival_frame(customers: pd.DataFrame, design: str = "cohort") -> pd.DataFrame:
    exit_ = np.maximum(customers["tenure"].to_numpy(float), MIN_DURATION)
    out = pd.DataFrame({"duration": exit_, "event": customers["churn"].astype(int).to_numpy()}, index=customers.index)
    if design == "snapshot":
        out["entry"] = np.maximum(customers["tenure"].to_numpy(float) - 1, 0.0)
    return out


def km_curves(customers: pd.DataFrame, by: str | None = None, design: str = "cohort") -> dict[str, KaplanMeierFitter]:
    sf = survival_frame(customers, design)
    groups = {"all": customers.index} if by is None else customers.groupby(by, observed=True).groups
    out = {}
    for name, idx in groups.items():
        s = sf.loc[idx]
        km = KaplanMeierFitter(label=str(name))
        km.fit(s["duration"], s["event"], entry=s["entry"] if design == "snapshot" else None)
        out[str(name)] = km
    return out


def survival_at(km: KaplanMeierFitter, months: list[int]) -> list[float]:
    return [float(km.survival_function_at_times(m).iloc[0]) for m in months]


def consistency_check(customers: pd.DataFrame) -> pd.DataFrame:
    """What each reading implies, against what the data shows."""
    snap, coh = km_curves(customers, design="snapshot")["all"], km_curves(customers)["all"]
    at1 = customers[customers.tenure == 1]
    rows = [
        ("monthly hazard at tenure 1 (snapshot reading: churners / customers with tenure 1)", float(at1.churn.mean())),
        ("P(reach 48 months), snapshot reading", survival_at(snap, [48])[0]),
        ("P(reach 48 months), cohort reading", survival_at(coh, [48])[0]),
        ("observed share of customers with tenure >= 49 months", float((customers.tenure >= 49).mean())),
        ("churn rate in the data (one month, if the snapshot reading held)", float(customers.churn.mean())),
    ]
    return pd.DataFrame(rows, columns=["quantity", "value"])


def fit_cox(
    customers: pd.DataFrame, features: pd.DataFrame, strata: str | None = None
) -> tuple[CoxPHFitter, pd.DataFrame]:
    """Cox PH on tenure; covariates = the churn design matrix without tenure (tenure is the time axis)."""
    x = design_matrix(features).drop(columns=["tenure"])
    if strata:
        x = x.drop(columns=[c for c in x.columns if c.startswith(f"{strata}=")])
        x[strata] = customers[strata].astype(str).to_numpy()
    df = pd.concat([x, survival_frame(customers)], axis=1)
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(df, duration_col="duration", event_col="event", strata=[strata] if strata else None)
    return cph, df


def ph_test(cph: CoxPHFitter, df: pd.DataFrame) -> pd.DataFrame:
    """Schoenfeld-residual test against rank(time), per covariate."""
    res = proportional_hazard_test(cph, df, time_transform="rank")
    out = res.summary.reset_index().rename(columns={"index": "covariate"})
    return out[["covariate", "test_statistic", "p"]].sort_values("p").reset_index(drop=True)


def hazard_ratios(cph: CoxPHFitter) -> pd.DataFrame:
    s = cph.summary.reset_index()
    return pd.DataFrame(
        {
            "covariate": s["covariate"],
            "hazard_ratio": s["exp(coef)"],
            "hr_ci_low": s["exp(coef) lower 95%"],
            "hr_ci_high": s["exp(coef) upper 95%"],
            "p": s["p"],
        }
    ).sort_values("hazard_ratio")


def monthly_hazard(customers: pd.DataFrame, by: str) -> pd.DataFrame:
    """Discrete monthly hazard by tenure month, cohort reading: churners at t / customers with tenure >= t."""
    rows = []
    for level, g in customers.groupby(by, observed=True):
        ten, churn = g["tenure"].to_numpy(int), g["churn"].to_numpy(int)
        for t in range(1, int(ten.max()) + 1):
            at_risk = int((ten >= t).sum())
            events = int(((ten == t) & (churn == 1)).sum())
            rows.append({by: level, "tenure": t, "at_risk": at_risk, "churners": events, "hazard": events / at_risk})
    return pd.DataFrame(rows)


def remaining_lifetime(customers: pd.DataFrame, by: str, horizon: int, tail_months: int = 12) -> pd.DataFrame:
    """Expected months still a customer within `horizon`, from the monthly hazard of the customer's group.

    Conditional on having reached tenure t: E = sum_{k=1..H} prod_{j=1..k} (1 - h(t + j)). Beyond the observed tenure
    range the hazard is the group's pooled hazard over its last `tail_months` months (an extrapolation).
    """
    haz = monthly_hazard(customers, by)
    out = np.zeros(len(customers))
    tenure = customers["tenure"].to_numpy(int)
    groups = customers[by].astype(str).to_numpy()
    for level, g in haz.groupby(by, observed=True):
        g = g.set_index("tenure")
        max_t = int(g.index.max())
        last = g.loc[g.index > max_t - tail_months]
        tail = float(last.churners.sum() / last.at_risk.sum())
        h = np.array([g.hazard.get(t, tail) for t in range(0, max_t + horizon + 2)], dtype=float)
        for i in np.where(groups == str(level))[0]:
            future = h[tenure[i] + 1 : tenure[i] + 1 + horizon]
            out[i] = float(np.cumprod(1 - future).sum())
    return pd.DataFrame({"customer_id": customers["customer_id"].to_numpy(), "expected_remaining_months": out})


def _md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return str(df.astype(object).to_markdown(index=False, floatfmt=floatfmt)) if len(df) else "_none_"


def _row(name: str, km: KaplanMeierFitter, months: list[int]) -> dict[str, Any]:
    return {"curve": name, **dict(zip(map(str, months), survival_at(km, months), strict=True))}


def run_survival(customers: pd.DataFrame, feats: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    horizon = int(cfg.section("business")["clv_horizon_months"])
    months = [6, 12, 24, 36, 48, 60, 72]
    overall = km_curves(customers)["all"]
    by_contract = km_curves(customers, "contract")
    snapshot = km_curves(customers, design="snapshot")["all"]
    surv_rows = [_row("all customers", overall, months)]
    surv_rows += [_row(k, v, months) for k, v in by_contract.items()]
    surv_rows.append(_row("all customers, snapshot reading (rejected)", snapshot, months))
    medians = pd.DataFrame(
        [
            {"curve": k, "median_tenure_months": v.median_survival_time_}
            for k, v in {"all": overall, **by_contract}.items()
        ]
    )
    sf = survival_frame(customers)
    logrank = multivariate_logrank_test(sf["duration"], customers["contract"].astype(str), sf["event"])
    cox, cox_df = fit_cox(customers, feats)
    cox_s, cox_s_df = fit_cox(customers, feats, strata="contract")
    remaining = remaining_lifetime(customers, "contract", horizon)
    rem_tbl = (
        customers.assign(expected_remaining_months=remaining["expected_remaining_months"].to_numpy())
        .groupby("contract", observed=True)["expected_remaining_months"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )
    return {
        "overall": overall,
        "snapshot": snapshot,
        "by_contract": by_contract,
        "consistency": consistency_check(customers),
        "survival": pd.DataFrame(surv_rows),
        "medians": medians,
        "logrank_stat": float(logrank.test_statistic),
        "logrank_p": float(logrank.p_value),
        "hr": hazard_ratios(cox),
        "ph": ph_test(cox, cox_df),
        "hr_strat": hazard_ratios(cox_s),
        "ph_strat": ph_test(cox_s, cox_s_df),
        "cox_concordance": float(cox.concordance_index_),
        "cox_strat_concordance": float(cox_s.concordance_index_),
        "remaining": remaining,
        "remaining_by_contract": rem_tbl,
        "hazard_by_contract": monthly_hazard(customers, "contract"),
        "horizon": horizon,
    }


def _figures(res: dict[str, Any], figs: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    res["overall"].plot_survival_function(ax=axes[0], ci_show=True, label="cohort reading (used)")
    res["snapshot"].plot_survival_function(ax=axes[0], ci_show=False, label="snapshot reading (rejected)")
    axes[0].set_title("Two readings of the churn flag")
    for km in res["by_contract"].values():
        km.plot_survival_function(ax=axes[1], ci_show=True)
    axes[1].set_title("Survival by contract (cohort reading)")
    for ax in axes:
        ax.set_xlabel("tenure (months)")
        ax.set_ylabel("P(still a customer)")
        ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(figs / "survival_km.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    for level, g in res["hazard_by_contract"].groupby("contract", observed=True):
        g = g.set_index("tenure")
        smooth = g.churners.rolling(6, min_periods=1).sum() / g.at_risk.rolling(6, min_periods=1).sum()
        ax.plot(smooth.index, smooth.values, label=str(level))
    ax.set_xlabel("tenure (months)")
    ax.set_ylabel("monthly churn hazard (6-month window)")
    ax.set_title("Monthly churn hazard by tenure and contract")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "survival_hazard.png", dpi=120)
    plt.close(fig)


def write_survival_report(res: dict[str, Any], cfg: Config) -> Path:
    tables, figs = cfg.path("tables_dir"), cfg.path("figures_dir")
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    names = ["consistency", "survival", "medians", "hr", "ph", "hr_strat", "ph_strat", "remaining_by_contract"]
    for name in [*names, "hazard_by_contract"]:
        res[name].to_csv(tables / f"survival_{name}.csv", index=False)
    proc = cfg.path("processed_dir")
    proc.mkdir(parents=True, exist_ok=True)
    res["remaining"].to_parquet(proc / "expected_remaining_months.parquet", index=False)
    _figures(res, figs)
    ph_bad = res["ph"][res["ph"].p < 0.05]
    text = f"""<!-- GENERATED by `telco survival` - do not edit by hand. -->
# Survival on tenure

## 0. Which reading of the data? (D-011)

IBM describes the churn flag as "left within the last month". Read literally, the data is a monthly snapshot and each
customer is at risk only in their last month (left truncation). The alternative cohort reading records churners at
the tenure at which they left and censors stayers at their current tenure. What each implies:

{_md(res["consistency"])}

Under the snapshot reading almost no new customer would reach four years, yet a third of customers have; and a
26.5% churn rate in one month is far beyond any real telco. The snapshot reading is inconsistent with the data, so the
**cohort reading is used throughout**. IBM's customers are fictional; absolute survival figures describe this sample.

## 1. Kaplan-Meier

P(still a customer) at each tenure (months):

{_md(res["survival"])}

Median tenure (months; `inf` where survival stays above 50% within 72 months):

{_md(res["medians"], ".1f")}

Log-rank test across contract types: statistic {res["logrank_stat"]:.0f}, p = {res["logrank_p"]:.2g}.

![Kaplan-Meier](figures/survival_km.png)

![hazard](figures/survival_hazard.png)

## 2. Cox proportional hazards

Covariates: the churn design matrix without tenure (tenure is the time axis). Concordance {res["cox_concordance"]:.3f}.
Hazard ratios < 1 lower the monthly churn hazard; relative to month-to-month, DSL and electronic check.

{_md(res["hr"])}

### Proportional-hazards check (Schoenfeld residuals against rank of time)

{len(ph_bad)} of {len(res["ph"])} covariates reject proportionality at 5% (their effect changes with tenure; the hazard
ratio above is then an average over tenure):

{_md(res["ph"])}

### Model stratified by contract

Contract gets its own baseline hazard, so no proportionality is assumed for it. Concordance
{res["cox_strat_concordance"]:.3f} (lower by construction: the contract effect is in the baselines, not the score).

{_md(res["hr_strat"])}

PH check of the stratified model:

{_md(res["ph_strat"])}

## 3. Expected remaining lifetime (input to customer value)

For each customer: expected months still a customer within a {res["horizon"]}-month horizon, from the monthly hazard of
their contract type, conditional on their current tenure. Beyond 72 months the hazard is extrapolated as each
contract's pooled hazard over its last 12 observed months. Saved to `data/processed/expected_remaining_months.parquet`.

{_md(res["remaining_by_contract"], ".1f")}
"""
    path = cfg.path("reports_dir") / "survival.md"
    path.write_text(text, encoding="utf-8")
    return path
