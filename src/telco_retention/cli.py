"""Command-line entry point: `telco <step>` (or `telco all`)."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable

import pandas as pd

from telco_retention.config import Config, load_config

log = logging.getLogger("telco")


def cmd_fetch(cfg: Config, force: bool) -> None:
    from telco_retention.data.load import fetch

    log.info("fetched %s", fetch(cfg, force=force))


def cmd_data(cfg: Config) -> None:
    """Load, clean, validate (contract), audit leakage, write lineage and the processed table."""
    from telco_retention.data.load import clean, read_source, save_processed
    from telco_retention.data.validate import leakage_audit, validate

    raw, layout = read_source(cfg)
    res = clean(raw, layout)
    errors = validate(res.customers)
    if errors:
        raise SystemExit(f"customer contract failed: {errors[:5]}")
    tables = cfg.path("tables_dir")
    tables.mkdir(parents=True, exist_ok=True)
    audit = leakage_audit(res.customers, cfg)
    audit.to_csv(tables / "leakage_audit.csv", index=False)
    path = save_processed(cfg, res)
    lineage = pd.DataFrame(
        [*res.steps, {"step": "contract passed; saved processed table", "rows": len(res.customers), "note": path.name}]
    )
    lineage.to_csv(tables / "lineage.csv", index=False)
    leaks = audit[(audit.role == "outcome-derived") & audit.in_features]
    if len(leaks):
        raise SystemExit(f"outcome-derived columns reached the features: {leaks.column.tolist()}")
    log.info(
        "data: %s",
        json.dumps({"layout": layout, "rows": len(res.customers), "churn_rate": float(res.customers.churn.mean())}),
    )


def cmd_eda(cfg: Config) -> None:
    from telco_retention.eda import write_eda

    df = pd.read_parquet(cfg.path("processed_dir") / "customers.parquet")
    layout = "extended" if "cltv" in df.columns else "basic"
    log.info("eda report %s", write_eda(df, cfg, layout))


def cmd_train(cfg: Config) -> None:
    from telco_retention.models.report import evaluate_all, save_model, write_report
    from telco_retention.models.train import train_all

    customers = pd.read_parquet(cfg.path("processed_dir") / "customers.parquet")
    res = train_all(customers, cfg)
    ev = evaluate_all(res, cfg)
    path = save_model(res, ev, cfg)
    log.info("churn models: chosen %s; report %s", ev["chosen"], write_report(res, ev, cfg, path))


def cmd_survival(cfg: Config) -> None:
    from telco_retention.data.load import features
    from telco_retention.survival import run_survival, write_survival_report

    customers = pd.read_parquet(cfg.path("processed_dir") / "customers.parquet")
    res = run_survival(customers, features(customers, cfg), cfg)
    log.info("survival report %s", write_survival_report(res, cfg))


def cmd_target(cfg: Config) -> None:
    import joblib

    from telco_retention.targeting import run_targeting, write_targeting_report

    customers = pd.read_parquet(cfg.path("processed_dir") / "customers.parquet")
    remaining = pd.read_parquet(cfg.path("processed_dir") / "expected_remaining_months.parquet")
    months = customers[["customer_id"]].merge(remaining, on="customer_id")["expected_remaining_months"].to_numpy()
    bundle = joblib.load(cfg.path("artifacts_dir") / "churn_model.joblib")
    res = run_targeting(customers, months, bundle, cfg)
    log.info("targeting report %s", write_targeting_report(res, cfg))


STEPS = ["fetch", "data", "eda", "train", "survival", "target"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="telco", description=__doc__)
    parser.add_argument("step", choices=[*STEPS, "all"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true", help="fetch: download again")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    steps: dict[str, Callable[[], None]] = {
        "fetch": lambda: cmd_fetch(cfg, args.force),
        "data": lambda: cmd_data(cfg),
        "eda": lambda: cmd_eda(cfg),
        "train": lambda: cmd_train(cfg),
        "survival": lambda: cmd_survival(cfg),
        "target": lambda: cmd_target(cfg),
    }
    for name in STEPS if args.step == "all" else [args.step]:
        steps[name]()


if __name__ == "__main__":
    main()
