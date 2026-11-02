"""Metrics with bootstrap CIs, paired comparisons, calibration tables and segment performance."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def ece(y: ArrayLike, p: ArrayLike, bins: int = 10) -> float:
    t = calibration_table(y, p, bins)
    return float(np.sum(t.rows * np.abs(t.mean_predicted - t.observed_rate)) / t.rows.sum())


def calibration_table(y: ArrayLike, p: ArrayLike, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    df["bin"] = np.minimum((df.p * bins).astype(int), bins - 1)
    return (
        df.groupby("bin")
        .agg(rows=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
        .reset_index()
    )


def metrics(y: ArrayLike, p: ArrayLike) -> dict[str, float]:
    y, p = np.asarray(y, int), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    both = 0 < y.sum() < len(y)
    return {
        "roc_auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if y.any() else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": ece(y, p),
    }


def bootstrap(y: ArrayLike, preds: dict[str, ArrayLike], n: int, seed: int) -> dict[str, np.ndarray]:
    """Resample customers with replacement; returns per model an (n, 3) array of ROC-AUC, PR-AUC, Brier."""
    y = np.asarray(y, int)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    out = {}
    for name, p in preds.items():
        p = np.asarray(p, float)
        vals = np.empty((n, 3))
        for i, ix in enumerate(idx):
            yy, pp = y[ix], p[ix]
            vals[i] = (roc_auc_score(yy, pp), average_precision_score(yy, pp), brier_score_loss(yy, pp))
        out[name] = vals
    return out


def ci(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def segment_table(df: pd.DataFrame, by: str, preds: dict[str, str]) -> pd.DataFrame:
    rows = []
    for level, g in df.groupby(by, observed=True):
        row = {by: level, "customers": len(g), "churners": int(g.churn.sum()), "churn_rate": float(g.churn.mean())}
        for name, col in preds.items():
            m = metrics(g.churn, g[col])
            row[f"{name}_roc_auc"] = m["roc_auc"]
            row[f"{name}_pr_auc"] = m["pr_auc"]
            row[f"{name}_mean_p"] = float(g[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)
