"""One design matrix for every model, built only from `features()` (outcome-derived fields never enter; D-003).

Recoding (D-008): the add-on services are 3-level fields whose "No internet service" level duplicates
`internet_service == "No"` (and "No phone service" duplicates `phone_service == 0`). Left as dummies they make the
logistic design singular, so each becomes a 0/1 "has service" flag and the internet/phone fields carry the rest.
`total_charges` is close to tenure x monthly charges (r ≈ 0.83 with tenure) and is left out of the design, so the
logistic coefficients of tenure and charges stay interpretable.
"""

from __future__ import annotations

import pandas as pd

ADDONS = ["online_security", "online_backup", "device_protection", "tech_support", "streaming_tv", "streaming_movies"]
# reference level of each multi-level categorical (the most common or most natural baseline)
REFERENCE = {
    "contract": "Month-to-month",
    "internet_service": "DSL",
    "payment_method": "Electronic check",
}
NUMERIC = ["tenure", "monthly_charges"]
BINARY = ["senior_citizen", "partner", "dependents", "phone_service", "paperless_billing"]


def design_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Numeric design (float): binaries, has-service flags, one-hot categoricals without their reference level."""
    out = pd.DataFrame(index=features.index)
    out["male"] = (features["gender"].astype(str) == "Male").astype(float)
    for c in BINARY:
        out[c] = features[c].astype(float)
    out["multiple_lines"] = (features["multiple_lines"].astype(str) == "Yes").astype(float)
    for c in ADDONS:
        out[c] = (features[c].astype(str) == "Yes").astype(float)
    for col, ref in REFERENCE.items():
        values = features[col].astype(str)
        for level in sorted(values.unique()):
            if level != ref:
                name = f"{col}={level}".replace(" ", "_").replace("(", "").replace(")", "").lower()
                out[name] = (values == level).astype(float)
    for c in NUMERIC:
        out[c] = features[c].astype(float)
    return out


def align(design: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Same columns, same order as the training design (unseen levels -> 0, missing columns -> 0)."""
    return design.reindex(columns=columns, fill_value=0.0)
