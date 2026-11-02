"""Fetch, pin and load the IBM Telco Customer Churn data into one clean, typed customer table.

Two source layouts are supported (D-002):
- basic: IBM's official CSV (7,043 x 21): customerID, demographics, services, charges, Churn (Yes/No);
- extended: IBM's extended sample (xlsx, 33 columns) adding location, Churn Score, CLTV and Churn Reason.
Both are mapped to snake_case names; outcome-derived fields are kept in the raw frame for the leakage audit but never
enter `features()` (D-003). Cleaning decisions and their impact are returned for data lineage.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from telco_retention.config import Config

# basic CSV header / extended xlsx header -> project name
RENAME = {
    "customerID": "customer_id",
    "CustomerID": "customer_id",
    "gender": "gender",
    "Gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Senior Citizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure",
    "Tenure Months": "tenure",
    "PhoneService": "phone_service",
    "Phone Service": "phone_service",
    "MultipleLines": "multiple_lines",
    "Multiple Lines": "multiple_lines",
    "InternetService": "internet_service",
    "Internet Service": "internet_service",
    "OnlineSecurity": "online_security",
    "Online Security": "online_security",
    "OnlineBackup": "online_backup",
    "Online Backup": "online_backup",
    "DeviceProtection": "device_protection",
    "Device Protection": "device_protection",
    "TechSupport": "tech_support",
    "Tech Support": "tech_support",
    "StreamingTV": "streaming_tv",
    "Streaming TV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Streaming Movies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "Paperless Billing": "paperless_billing",
    "PaymentMethod": "payment_method",
    "Payment Method": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "Monthly Charges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Total Charges": "total_charges",
    "Churn": "churn",
    "Churn Label": "churn_label",
    "Churn Value": "churn_value",
    "Churn Score": "churn_score",
    "Churn Reason": "churn_reason",
    "CLTV": "cltv",
}
YES_NO = ["partner", "dependents", "phone_service", "paperless_billing"]
CATEGORICAL = [
    "gender",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "payment_method",
]
NUMERIC = ["tenure", "monthly_charges", "total_charges"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(cfg: Config, force: bool = False) -> Path:
    """Download IBM's CSV (if missing), verify the pinned checksum when set, record it in the manifest."""
    tc = cfg.section("telco")
    raw = cfg.path("raw_dir")
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / tc["file"]
    if force or not path.exists():
        urllib.request.urlretrieve(tc["url"], path)  # data file over HTTPS from IBM's GitHub; nothing is executed
    digest = sha256_file(path)
    if tc.get("sha256") and digest != tc["sha256"]:
        raise ValueError(f"{path.name}: SHA-256 {digest} does not match the pinned {tc['sha256']}")
    manifest = cfg.path("manifest")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    rows = sum(1 for _ in path.open(encoding="utf-8")) - 1
    entries: dict[str, Any] = {"telco": {"source": tc["url"], "file": path.name, "sha256": digest, "rows": rows}}
    extended = raw / tc["extended_file"]
    if extended.exists():
        entries["telco_extended"] = {
            "source": tc.get("extended_source"),
            "file": extended.name,
            "sha256": sha256_file(extended),
            "rows": len(pd.read_excel(extended)),
        }
    manifest.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    return path


def read_source(cfg: Config) -> tuple[pd.DataFrame, str]:
    """The extended file when present in data/raw/, otherwise IBM's basic CSV; columns renamed to snake_case."""
    tc = cfg.section("telco")
    extended = cfg.path("raw_dir") / tc["extended_file"]
    if extended.exists():
        digest = sha256_file(extended)
        if tc.get("extended_sha256") and digest != tc["extended_sha256"]:
            raise ValueError(f"{extended.name}: SHA-256 {digest} does not match the pinned {tc['extended_sha256']}")
        df, layout = pd.read_excel(extended), "extended"
    else:
        df, layout = pd.read_csv(cfg.path("raw_dir") / tc["file"], dtype=str, keep_default_na=False), "basic"
    df = df.rename(columns={c: RENAME.get(c, re.sub(r"\W+", "_", str(c)).strip("_").lower()) for c in df.columns})
    return df, layout


@dataclass
class LoadResult:
    customers: pd.DataFrame  # typed, cleaned, incl. outcome-derived fields (for the audit only)
    layout: str
    steps: list[dict[str, Any]] = field(default_factory=list)  # lineage: rows and decisions per step


def clean(df: pd.DataFrame, layout: str) -> LoadResult:
    steps: list[dict[str, Any]] = [{"step": f"read ({layout})", "rows": len(df), "note": f"{df.shape[1]} columns"}]
    out = df.copy()
    for c in ["tenure", "monthly_charges", "total_charges", "churn_score", "cltv"]:
        if c in out:
            out[c] = pd.to_numeric(out[c].astype(str).str.strip().replace("", None), errors="coerce")
    blank_tc = out["total_charges"].isna()
    zero_tenure = out["tenure"] == 0
    # TotalCharges is blank only for customers in their first month (tenure 0): nothing billed yet -> 0
    out.loc[blank_tc & zero_tenure, "total_charges"] = 0.0
    steps.append(
        {
            "step": "total_charges blank -> 0 where tenure = 0",
            "rows": len(out),
            "note": f"{int(blank_tc.sum())} blank; {int((blank_tc & zero_tenure).sum())} with tenure 0 set to 0",
        }
    )
    if "churn" not in out and "churn_label" in out:
        out["churn"] = out["churn_label"]
    out["churn"] = out["churn"].astype(str).str.strip().map({"Yes": 1, "No": 0}).astype("Int64")
    out["senior_citizen"] = out["senior_citizen"].astype(str).str.strip().map({"1": 1, "0": 0, "Yes": 1, "No": 0})
    for c in YES_NO:
        out[c] = out[c].astype(str).str.strip().map({"Yes": 1, "No": 0})
    for c in CATEGORICAL:
        out[c] = out[c].astype(str).str.strip().astype("category")
    before = len(out)
    out = out.drop_duplicates(subset="customer_id")
    steps.append({"step": "drop duplicate customer_id", "rows": len(out), "note": f"{before - len(out)} removed"})
    return LoadResult(customers=out.reset_index(drop=True), layout=layout, steps=steps)


def features(customers: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Model inputs: every column except the id, the target and the outcome-derived fields (D-003)."""
    tc, forbidden = cfg.section("telco"), set(cfg.section("leakage")["forbidden_features"])
    drop = {tc["id"], tc["target"], *forbidden}
    # location columns of the extended file are identifiers/geography, not used in phase 2 (D-004)
    drop |= {
        c
        for c in customers.columns
        if c in {"count", "country", "state", "city", "zip_code", "lat_long", "latitude", "longitude", "cltv"}
    }
    return customers.drop(columns=[c for c in customers.columns if c in drop])


def save_processed(cfg: Config, res: LoadResult) -> Path:
    out = cfg.path("processed_dir")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "customers.parquet"
    res.customers.to_parquet(path, index=False)
    return path
