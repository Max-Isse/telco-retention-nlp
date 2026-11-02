"""Data contract for the cleaned customer table (pandera) and the leakage audit."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pandera.pandas as pa

from telco_retention.config import Config
from telco_retention.data.load import features

YES_NO_INT = pa.Check.isin([0, 1])
SERVICE = ["Yes", "No", "No internet service"]


def customer_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "customer_id": pa.Column(str, unique=True, nullable=False),
            "gender": pa.Column("category", pa.Check.isin(["Female", "Male"])),
            "senior_citizen": pa.Column(int, YES_NO_INT),
            "partner": pa.Column(int, YES_NO_INT),
            "dependents": pa.Column(int, YES_NO_INT),
            "tenure": pa.Column(float, pa.Check.in_range(0, 72)),
            "phone_service": pa.Column(int, YES_NO_INT),
            "multiple_lines": pa.Column("category", pa.Check.isin(["Yes", "No", "No phone service"])),
            "internet_service": pa.Column("category", pa.Check.isin(["DSL", "Fiber optic", "No"])),
            **{
                c: pa.Column("category", pa.Check.isin(SERVICE))
                for c in [
                    "online_security",
                    "online_backup",
                    "device_protection",
                    "tech_support",
                    "streaming_tv",
                    "streaming_movies",
                ]
            },
            "contract": pa.Column("category", pa.Check.isin(["Month-to-month", "One year", "Two year"])),
            "paperless_billing": pa.Column(int, YES_NO_INT),
            "payment_method": pa.Column("category"),
            "monthly_charges": pa.Column(float, pa.Check.gt(0)),
            "total_charges": pa.Column(float, pa.Check.ge(0)),
            "churn": pa.Column("Int64", pa.Check.isin([0, 1]), nullable=False),
        },
        strict=False,
        coerce=True,
        name="customers",
    )


def validate(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Run the contract lazily; one row per failed check (empty = contract satisfied)."""
    try:
        customer_schema().validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        cases = exc.failure_cases
        return [
            {"column": r["column"], "check": str(r["check"]), "failure": str(r["failure_case"])}
            for _, r in cases.head(50).iterrows()
        ]
    return []


def leakage_audit(customers: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Every source column, whether it is outcome-derived, and whether it reaches the model features."""
    forbidden = set(cfg.section("leakage")["forbidden_features"])
    feats = set(features(customers, cfg).columns)
    target = cfg.section("telco")["target"]
    rows = []
    for c in customers.columns:
        role = "target" if c == target else ("outcome-derived" if c in forbidden else "candidate")
        rows.append({"column": c, "role": role, "in_features": c in feats})
    return pd.DataFrame(rows)
