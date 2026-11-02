"""Explanations: logistic odds ratios with CIs (statsmodels) and SHAP for LightGBM, compared side by side."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from scipy.stats import spearmanr


def odds_ratios(x: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Unpenalised logistic regression for inference: coefficient, odds ratio and 95% CI, p-value per feature.

    Numeric features are per 1 unit (tenure: month; monthly_charges: currency unit) and also per standard deviation.
    """
    fit = sm.Logit(y, sm.add_constant(x)).fit(disp=False, maxiter=200)
    conf = fit.conf_int()
    sd = x.std()
    out = pd.DataFrame(
        {
            "feature": fit.params.index,
            "coef": fit.params.values,
            "odds_ratio": np.exp(fit.params.values),
            "or_ci_low": np.exp(conf[0].values),
            "or_ci_high": np.exp(conf[1].values),
            "p_value": fit.pvalues.values,
        }
    )
    out["or_per_sd"] = [
        np.exp(c * sd.get(f, np.nan)) if f in ("tenure", "monthly_charges") else np.nan
        for f, c in zip(out.feature, out.coef, strict=True)
    ]
    out.attrs["pseudo_r2"] = float(fit.prsquared)
    # collinearity check: how much of monthly charges the service flags explain (wide CIs on services if high)
    services = [c for c in x.columns if c not in ("monthly_charges", "tenure")]
    out.attrs["charges_r2_on_services"] = float(
        sm.OLS(x["monthly_charges"], sm.add_constant(x[services])).fit().rsquared
    )
    out.attrs["converged"] = bool(fit.mle_retvals.get("converged", False))
    return out[out.feature != "const"].reset_index(drop=True)


def shap_vs_logistic(lgbm: Any, logistic_pipeline: Any, x: pd.DataFrame) -> tuple[pd.DataFrame, float, Any]:
    """Mean |SHAP| (log-odds) and its direction per feature vs the logistic coefficient on standardised features."""
    explainer = shap.TreeExplainer(lgbm)
    values = explainer.shap_values(x)
    values = values[1] if isinstance(values, list) else values
    mean_abs = np.abs(values).mean(axis=0)
    direction = [
        float(np.corrcoef(x[c], values[:, i])[0, 1]) if x[c].std() > 0 else float("nan")
        for i, c in enumerate(x.columns)
    ]
    coef_std = logistic_pipeline[-1].coef_[0]
    table = pd.DataFrame(
        {
            "feature": x.columns,
            "lgbm_mean_abs_shap": mean_abs,
            "lgbm_direction": np.sign(direction),
            "logistic_coef_std": coef_std,
        }
    )
    table["lgbm_rank"] = table.lgbm_mean_abs_shap.rank(ascending=False).astype(int)
    table["logistic_rank"] = table.logistic_coef_std.abs().rank(ascending=False).astype(int)
    table["direction_agrees"] = np.sign(table.logistic_coef_std) == table.lgbm_direction
    rho = float(spearmanr(table.lgbm_mean_abs_shap, table.logistic_coef_std.abs()).statistic)
    return table.sort_values("lgbm_rank").reset_index(drop=True), rho, values
