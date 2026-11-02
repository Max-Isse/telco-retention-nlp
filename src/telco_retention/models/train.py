"""Split, baselines, logistic regression and LightGBM, out-of-fold predictions, calibration (D-007, D-009, D-010).

Protocol: a stratified holdout (20%) is set aside and scored once. Every choice (calibration method per model) uses
out-of-fold (OOF) predictions from stratified 5-fold CV on the training part. Final models are refitted on the whole
training part, calibrated with a calibrator fitted on their OOF predictions, and score the holdout.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from telco_retention.config import Config
from telco_retention.data.load import features
from telco_retention.models.design import design_matrix

Factory = Callable[[], Any]


class RulesModel:
    """Rules baseline as a probability: churn rate of flagged vs unflagged customers, learned on training rows."""

    def __init__(self, max_tenure: int, columns: list[str]) -> None:
        self.max_tenure = max_tenure
        self.columns = columns
        self.rates = (0.0, 0.0)

    def _flag(self, x: pd.DataFrame) -> np.ndarray:
        not_m2m = x[[c for c in self.columns if c.startswith("contract=")]].sum(axis=1) > 0
        return ((~not_m2m) & (x["tenure"] <= self.max_tenure)).to_numpy()

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> RulesModel:
        f = self._flag(x)
        self.rates = (float(y[~f].mean()), float(y[f].mean()))
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = np.where(self._flag(x), self.rates[1], self.rates[0])
        return np.c_[1 - p, p]


class BaseRateModel:
    def fit(self, x: pd.DataFrame, y: np.ndarray) -> BaseRateModel:
        self.rate = float(np.mean(y))
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.c_[np.full(len(x), 1 - self.rate), np.full(len(x), self.rate)]


def factories(cfg: Config, columns: list[str]) -> dict[str, Factory]:
    mc = cfg.section("models")
    lp = {**mc["lightgbm"], "random_state": cfg.seed, "deterministic": True, "force_row_wise": True, "verbose": -1}
    return {
        "base_rate": BaseRateModel,
        "rules": lambda: RulesModel(int(mc["rules_max_tenure"]), columns),
        "logistic": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=float(mc["logistic"]["C"]), max_iter=2000)
        ),
        "lightgbm": lambda: lgb.LGBMClassifier(objective="binary", **lp),
    }


class Calibrator:
    """none | sigmoid (Platt, on the logit of p) | isotonic; fitted on OOF predictions."""

    def __init__(self, method: str) -> None:
        self.method = method
        self.model: Any = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> Calibrator:
        if self.method == "sigmoid":
            z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
            self.model = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
        elif self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if self.method == "sigmoid":
            z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
            return np.asarray(self.model.predict_proba(z)[:, 1])
        if self.method == "isotonic":
            return np.asarray(self.model.predict(p))
        return p


@dataclass
class TrainResult:
    train: pd.DataFrame  # customer rows of the training part, with fold and OOF predictions per model
    holdout: pd.DataFrame  # customer rows of the holdout, with final predictions per model
    columns: list[str]
    models: dict[str, Any]
    calibrators: dict[str, Calibrator]
    calibration_choice: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)


def split(customers: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    mc = cfg.section("models")
    tr, ho = train_test_split(
        customers, test_size=float(mc["holdout_share"]), stratify=customers["churn"], random_state=cfg.seed
    )
    return tr.reset_index(drop=True), ho.reset_index(drop=True)


def train_all(customers: pd.DataFrame, cfg: Config) -> TrainResult:
    mc = cfg.section("models")
    tr, ho = split(customers, cfg)
    x_tr, x_ho = design_matrix(features(tr, cfg)), design_matrix(features(ho, cfg))
    columns = list(x_tr.columns)
    x_ho = x_ho.reindex(columns=columns, fill_value=0.0)
    y_tr = tr["churn"].astype(int).to_numpy()
    folds = list(StratifiedKFold(int(mc["cv_folds"]), shuffle=True, random_state=cfg.seed).split(x_tr, y_tr))
    tr = tr.assign(fold=-1)
    for k, (_, va) in enumerate(folds):
        tr.loc[va, "fold"] = k
    makers = factories(cfg, columns)
    models, calibrators, choice_rows = {}, {}, []
    for name, make in makers.items():
        oof = np.zeros(len(tr))
        for tr_idx, va_idx in folds:
            oof[va_idx] = make().fit(x_tr.iloc[tr_idx], y_tr[tr_idx]).predict_proba(x_tr.iloc[va_idx])[:, 1]
        tr[f"p_{name}"] = oof
        final = make().fit(x_tr, y_tr)
        models[name] = final
        ho[f"p_{name}_raw"] = final.predict_proba(x_ho)[:, 1]
        if name in ("logistic", "lightgbm"):
            # choose the calibration by cross-fitted Brier on the OOF predictions (calibrator fitted on other folds)
            scores, cross_fitted = {}, {}
            for method in mc["calibration_methods"]:
                cal_oof = np.zeros(len(tr))
                for tr_idx, va_idx in folds:
                    cal = Calibrator(method).fit(oof[tr_idx], y_tr[tr_idx])
                    cal_oof[va_idx] = cal.transform(oof[va_idx])
                scores[method], cross_fitted[method] = brier_score_loss(y_tr, cal_oof), cal_oof
                choice_rows.append({"model": name, "method": method, "oof_brier_cross_fitted": scores[method]})
            best = min(scores, key=lambda m: scores[m])
            calibrators[name] = Calibrator(best).fit(oof, y_tr)
            tr[f"p_{name}_cal"] = cross_fitted[best]  # calibrated OOF, each fold calibrated by the others
            ho[f"p_{name}"] = calibrators[name].transform(ho[f"p_{name}_raw"].to_numpy())
        else:
            ho[f"p_{name}"] = ho[f"p_{name}_raw"]
    meta = {
        "train_rows": len(tr),
        "holdout_rows": len(ho),
        "train_churn_rate": float(y_tr.mean()),
        "holdout_churn_rate": float(ho["churn"].mean()),
        "n_features": len(columns),
    }
    choice = pd.DataFrame(choice_rows)
    return TrainResult(tr, ho, columns, models, calibrators, choice, meta)
