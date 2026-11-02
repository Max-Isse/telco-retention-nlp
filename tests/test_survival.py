"""Survival functions on generated cohorts with known monthly hazards (test data only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _cohort(n: int, hazard: float, rng: np.random.Generator, label: str) -> pd.DataFrame:
    """Customers who churn each month with probability `hazard`, censored at a uniform current tenure (1..72)."""
    life = rng.geometric(hazard, n)  # month of churn
    censor = rng.integers(1, 73, n)
    tenure = np.minimum(life, censor)
    return pd.DataFrame(
        {
            "customer_id": [f"{label}{i}" for i in range(n)],
            "tenure": tenure.astype(float),
            "churn": (life <= censor).astype(int),
            "contract": label,
        }
    )


@pytest.fixture
def cohorts() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.concat([_cohort(4000, 0.02, rng, "A"), _cohort(4000, 0.04, rng, "B")], ignore_index=True)


def test_kaplan_meier_recovers_a_constant_hazard(cohorts: pd.DataFrame) -> None:
    from telco_retention.survival import km_curves, survival_at

    km = km_curves(cohorts, "contract")
    assert survival_at(km["A"], [24])[0] == pytest.approx(0.98**24, abs=0.03)
    assert survival_at(km["B"], [24])[0] == pytest.approx(0.96**24, abs=0.03)


def test_monthly_hazard_and_remaining_lifetime(cohorts: pd.DataFrame) -> None:
    from telco_retention.survival import monthly_hazard, remaining_lifetime

    haz = monthly_hazard(cohorts, "contract")
    pooled = haz.groupby("contract").apply(lambda g: g.churners.sum() / g.at_risk.sum(), include_groups=False)
    assert pooled["A"] == pytest.approx(0.02, abs=0.004) and pooled["B"] == pytest.approx(0.04, abs=0.006)
    rem = remaining_lifetime(cohorts, "contract", horizon=24)
    expected_a = sum(0.98**k for k in range(1, 25))  # memoryless: same for every tenure
    got_a = rem.expected_remaining_months[cohorts.contract == "A"].mean()
    assert got_a == pytest.approx(expected_a, rel=0.05)
    assert rem.expected_remaining_months[cohorts.contract == "B"].mean() < got_a


def test_hand_counted_hazard() -> None:
    from telco_retention.survival import monthly_hazard

    df = pd.DataFrame({"tenure": [1, 1, 2, 3], "churn": [1, 0, 1, 0], "contract": "A", "customer_id": list("abcd")})
    h = monthly_hazard(df, "contract").set_index("tenure")
    assert h.loc[1, "at_risk"] == 4 and h.loc[1, "churners"] == 1
    assert h.loc[2, "at_risk"] == 2 and h.loc[2, "hazard"] == 0.5
    assert h.loc[3, "hazard"] == 0.0


def test_snapshot_reading_is_flagged_as_inconsistent(cohorts: pd.DataFrame) -> None:
    from telco_retention.survival import consistency_check

    c = consistency_check(cohorts).set_index("quantity")["value"]
    assert c["P(reach 48 months), cohort reading"] > c["P(reach 48 months), snapshot reading"]
