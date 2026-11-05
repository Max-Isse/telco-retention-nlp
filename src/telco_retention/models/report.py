"""reports/churn_models.md, tables, figures, and the chosen model saved for targeting (generated; do not edit)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from telco_retention.config import Config
from telco_retention.data.load import features
from telco_retention.eda import add_segments
from telco_retention.models.design import design_matrix
from telco_retention.models.evaluate import bootstrap, calibration_table, ci, metrics, segment_table
from telco_retention.models.explain import odds_ratios, shap_vs_logistic
from telco_retention.models.train import TrainResult

HOLDOUT = {
    "base_rate": "p_base_rate",
    "rules": "p_rules",
    "logistic (raw)": "p_logistic_raw",
    "logistic (calibrated)": "p_logistic",
    "lightgbm (raw)": "p_lightgbm_raw",
    "lightgbm (calibrated)": "p_lightgbm",
}
OOF = {"base_rate": "p_base_rate", "rules": "p_rules", "logistic": "p_logistic_cal", "lightgbm": "p_lightgbm_cal"}


def _md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return str(df.astype(object).to_markdown(index=False, floatfmt=floatfmt)) if len(df) else "_none_"


def evaluate_all(res: TrainResult, cfg: Config) -> dict[str, Any]:
    mc = cfg.section("models")
    ho, tr = res.holdout, res.train
    boots = bootstrap(ho.churn, {k: ho[v] for k, v in HOLDOUT.items()}, int(mc["bootstrap"]), cfg.seed)
    rows = []
    for name, col in HOLDOUT.items():
        m, b = metrics(ho.churn, ho[col]), boots[name]
        rows.append(
            {
                "model": name,
                "roc_auc": m["roc_auc"],
                "roc_auc_ci": "{:.3f} to {:.3f}".format(*ci(b[:, 0])),
                "pr_auc": m["pr_auc"],
                "pr_auc_ci": "{:.3f} to {:.3f}".format(*ci(b[:, 1])),
                "brier": m["brier"],
                "brier_ci": "{:.3f} to {:.3f}".format(*ci(b[:, 2])),
                "log_loss": m["log_loss"],
                "ece": m["ece"],
            }
        )
    holdout_tbl = pd.DataFrame(rows)
    diff = boots["lightgbm (calibrated)"] - boots["logistic (calibrated)"]
    point = [
        metrics(ho.churn, ho.p_lightgbm)[k] - metrics(ho.churn, ho.p_logistic)[k]
        for k in ("roc_auc", "pr_auc", "brier")
    ]
    paired = pd.DataFrame(
        {
            "metric": ["roc_auc", "pr_auc", "brier"],
            "lightgbm_minus_logistic": point,
            "ci_low": [ci(diff[:, i])[0] for i in range(3)],
            "ci_high": [ci(diff[:, i])[1] for i in range(3)],
            "share_of_resamples_lightgbm_better": [
                float((diff[:, 0] > 0).mean()),
                float((diff[:, 1] > 0).mean()),
                float((diff[:, 2] < 0).mean()),
            ],
        }
    )
    oof_tbl = pd.DataFrame([{"model": k, **metrics(tr.churn, tr[v])} for k, v in OOF.items()])
    seg_df = add_segments(tr, cfg)
    segs = {
        by: segment_table(seg_df, by, {"logistic": "p_logistic_cal", "lightgbm": "p_lightgbm_cal"})
        for by in ("contract", "tenure_band", "internet_service", "gender", "senior_citizen")
    }
    calib = pd.concat(
        [
            calibration_table(ho.churn, ho[c]).assign(model=n)
            for n, c in HOLDOUT.items()
            if "logistic" in n or "lightgbm" in n
        ]
    )
    x_tr = design_matrix(features(tr, cfg))[res.columns]
    odds = odds_ratios(x_tr, tr.churn.astype(int).to_numpy())
    x_ho = design_matrix(features(ho, cfg)).reindex(columns=res.columns, fill_value=0.0)
    shap_tbl, rho, shap_values = shap_vs_logistic(res.models["lightgbm"], res.models["logistic"], x_ho)
    chosen = min(("logistic", "lightgbm"), key=lambda n: float(oof_tbl.loc[oof_tbl.model == n, "brier"].iloc[0]))
    return {
        "holdout": holdout_tbl,
        "paired": paired,
        "oof": oof_tbl,
        "segments": segs,
        "calibration": calib,
        "odds": odds,
        "shap": shap_tbl,
        "rho": rho,
        "shap_values": shap_values,
        "x_holdout": x_ho,
        "chosen": chosen,
    }


def save_model(res: TrainResult, ev: dict[str, Any], cfg: Config) -> Path:
    path = cfg.path("artifacts_dir") / "churn_model.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    name = ev["chosen"]
    joblib.dump(
        {
            "name": name,
            "model": res.models[name],
            "calibrator": res.calibrators[name],
            "columns": res.columns,
            "chosen_by": "lowest cross-fitted out-of-fold Brier score (training part)",
            "meta": res.meta,
        },
        path,
    )
    return path


def _figures(res: TrainResult, ev: dict[str, Any], cfg: Config) -> list[Path]:
    out = cfg.path("figures_dir")
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for model, g in ev["calibration"].groupby("model"):
        if "calibrated" in model or "raw" in model:
            ax.plot(g.mean_predicted, g.observed_rate, marker="o", ms=3, label=model)
    ax.plot([0, 1], [0, 1], color="0.6", ls="--", lw=1)
    ax.set_xlabel("mean predicted churn probability")
    ax.set_ylabel("observed churn rate")
    ax.set_title("Reliability on the holdout")
    ax.legend(fontsize=7)
    fig.tight_layout()
    paths.append(out / "churn_reliability.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    top = ev["shap"].head(12).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    axes[0].barh(top.feature, top.lgbm_mean_abs_shap * top.lgbm_direction, color="C0")
    axes[0].set_title("LightGBM: mean |SHAP| x direction (log-odds)")
    axes[1].barh(top.feature, top.logistic_coef_std, color="C1")
    axes[1].set_title("Logistic: coefficient per SD (log-odds)")
    for ax in axes:
        ax.axvline(0, color="0.5", lw=0.8)
    fig.tight_layout()
    paths.append(out / "churn_shap_vs_logistic.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)
    return paths


def write_report(res: TrainResult, ev: dict[str, Any], cfg: Config, model_path: Path) -> Path:
    tables = cfg.path("tables_dir")
    tables.mkdir(parents=True, exist_ok=True)
    for name in ("holdout", "paired", "oof", "calibration", "odds", "shap"):
        ev[name].to_csv(tables / f"churn_{name}.csv", index=False)
    for by, t in ev["segments"].items():
        t.to_csv(tables / f"churn_segments_{by}.csv", index=False)
    res.calibration_choice.to_csv(tables / "churn_calibration_choice.csv", index=False)
    figs = _figures(res, ev, cfg)
    m = res.meta
    odds = ev["odds"].copy()
    odds = odds.reindex(odds.coef.abs().sort_values(ascending=False).index)
    choice = res.calibration_choice.pivot(
        index="model", columns="method", values="oof_brier_cross_fitted"
    ).reset_index()
    chosen_cal = {k: v.method for k, v in res.calibrators.items()}
    reports = cfg.path("reports_dir")
    fig_list = ", ".join(f"`{p.relative_to(reports).as_posix()}`" for p in figs)
    seg_text = "\n\n".join(f"By {by.replace('_', ' ')}:\n\n{_md(t)}" for by, t in ev["segments"].items())
    text = f"""<!-- GENERATED by `telco train` - do not edit by hand. -->
# Churn models

Telco has **no time dimension** (one snapshot per customer), so there is no out-of-time test. Substitute: a stratified
holdout of {m["holdout_rows"]:,} customers ({m["holdout_churn_rate"]:.1%} churn), scored once, with bootstrap 95% CIs
({cfg.section("models")["bootstrap"]:,} resamples), and stratified {cfg.section("models")["cv_folds"]}-fold CV on the
{m["train_rows"]:,} training customers for every choice. {m["n_features"]} features from `features()` (outcome-derived
fields excluded and tested, D-003; design: D-008). Decisions: D-007 to D-010.

## 1. Holdout performance

{_md(ev["holdout"])}

LightGBM minus logistic regression (both calibrated), paired bootstrap on the same resamples:

{_md(ev["paired"])}

Cross-validated (out-of-fold, training part; calibrated models calibrated by the other folds):

{_md(ev["oof"])}

**Model used for targeting: {ev["chosen"]}** (lowest cross-validated Brier score; calibrated probabilities matter
because targeting multiplies them by value at risk). Saved to `{model_path.relative_to(cfg.root).as_posix()}`.

## 2. Calibration

Calibration method per model, chosen by cross-fitted out-of-fold Brier score (lower is better):
{", ".join(f"{k}: **{v}**" for k, v in chosen_cal.items())}.

{_md(choice, ".5f")}

![reliability](figures/churn_reliability.png)

## 3. Segments (cross-validated predictions on the training part, calibrated)

Segment AUCs are within-segment ranking quality; `mean_p` vs `churn_rate` is calibration in the segment.

{seg_text}

## 4. Logistic regression: odds ratios (unpenalised, training part)

Pseudo-R² {ev["odds"].attrs.get("pseudo_r2", float("nan")):.3f}. Odds ratios > 1 raise churn odds. Tenure and monthly
charges per unit and per standard deviation (`or_per_sd`); categorical effects relative to month-to-month contract,
DSL internet and electronic check payment. **Collinearity:** the service flags explain
{ev["odds"].attrs.get("charges_r2_on_services", float("nan")):.1%} of the variance of monthly charges, so the separate
effects of charges, internet type and streaming services are not identifiable (wide intervals below); their joint
effect is. Read individual service odds ratios with that in mind (D-008).

{_md(odds, ".3f")}

## 5. LightGBM SHAP vs logistic coefficients (holdout)

Spearman rank correlation of importance (mean |SHAP| vs |coefficient per SD|): **{ev["rho"]:.2f}**. Directions agree
for {int(ev["shap"].direction_agrees.sum())} of {len(ev["shap"])} features.

{_md(ev["shap"])}

![SHAP vs logistic](figures/churn_shap_vs_logistic.png)

Figures: {fig_list}.
"""
    path = reports / "churn_models.md"
    path.write_text(text, encoding="utf-8")
    return path
