"""Loading, cleaning, the data contract, the leakage audit and the EDA helpers.

`fixtures/telco_basic.csv`: 15 real rows of IBM's file (both churn classes in every contract type, the two kinds of
blank TotalCharges row, and one repeated customer to exercise de-duplication).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from telco_retention.config import Config, load_config, with_root

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    c = with_root(load_config(), tmp_path)
    raw = c.path("raw_dir")
    raw.mkdir(parents=True)
    shutil.copyfile(FIXTURES / "telco_basic.csv", raw / c.section("telco")["file"])
    return c


def _loaded(cfg: Config) -> pd.DataFrame:
    from telco_retention.data.load import clean, read_source

    raw, layout = read_source(cfg)
    assert layout == "basic"
    return clean(raw, layout).customers


def test_cleaning_types_blanks_and_duplicates(cfg: Config) -> None:
    from telco_retention.data.load import clean, read_source

    raw, layout = read_source(cfg)
    res = clean(raw, layout)
    df = res.customers
    assert len(raw) == 15 and len(df) == 14  # the repeated customer is dropped
    assert df.total_charges.notna().all() and (df.loc[df.tenure == 0, "total_charges"] == 0).all()
    assert set(df.churn.unique()) == {0, 1} and df.senior_citizen.isin([0, 1]).all()
    assert [s["rows"] for s in res.steps] == [15, 15, 14]


def test_contract_passes_and_catches_violations(cfg: Config) -> None:
    from telco_retention.data.validate import validate

    df = _loaded(cfg)
    assert validate(df) == []
    broken = df.copy()
    broken.loc[0, "tenure"] = 99
    broken.loc[1, "monthly_charges"] = -5
    assert {e["column"] for e in validate(broken)} >= {"tenure", "monthly_charges"}


def test_outcome_derived_fields_never_reach_the_features(cfg: Config) -> None:
    """D-003: Churn Reason / Score / Category (extended IBM file) exist only because the outcome is known."""
    from telco_retention.data.load import RENAME, clean, features
    from telco_retention.data.validate import leakage_audit

    raw = pd.read_csv(FIXTURES / "telco_basic.csv", dtype=str, keep_default_na=False)
    extended = raw.rename(columns={"customerID": "CustomerID", "tenure": "Tenure Months", "Churn": "Churn Label"})
    extended["Churn Value"] = (extended["Churn Label"] == "Yes").astype(int).astype(str)
    extended["Churn Score"] = "80"
    extended["Churn Reason"] = "Competitor offered more data"
    extended["Churn Category"] = "Competitor"
    extended["CLTV"] = "4000"
    extended = extended.rename(columns={c: RENAME.get(c, c.replace(" ", "_").lower()) for c in extended.columns})
    df = clean(extended, "extended").customers
    feats = features(df, cfg)
    forbidden = set(cfg.section("leakage")["forbidden_features"])
    assert not forbidden & set(feats.columns)
    assert {"churn", "customer_id", "cltv"}.isdisjoint(feats.columns)
    audit = leakage_audit(df, cfg).set_index("column")
    assert audit.loc["churn_reason", "role"] == "outcome-derived" and not audit.loc["churn_reason", "in_features"]
    assert audit.loc["contract", "in_features"]


def test_fetch_rejects_a_changed_file(cfg: Config, tmp_path: Path) -> None:
    from telco_retention.config import Config as C
    from telco_retention.data.load import fetch

    raw = dict(cfg.raw)
    raw["telco"] = {**cfg.section("telco"), "url": (FIXTURES / "telco_basic.csv").as_uri(), "sha256": "0" * 64}
    with pytest.raises(ValueError, match="does not match"):
        fetch(C(raw=raw, root=tmp_path), force=True)


def test_eda_helpers_and_report(cfg: Config) -> None:
    from telco_retention.eda import rules_baseline, wilson, write_eda

    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi and wilson(0, 0)[0] != wilson(0, 0)[0]  # NaN for empty segments
    df = _loaded(cfg)
    rb = rules_baseline(df)
    assert 0 <= rb["precision"] <= 1 and rb["flagged"] == int(
        ((df.contract == "Month-to-month") & (df.tenure <= 12)).sum()
    )
    path = write_eda(df, cfg, "basic")
    assert "Base churn rate" in path.read_text(encoding="utf-8")
