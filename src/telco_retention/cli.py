"""Command-line entry point: `telco <step>` (or `telco all`)."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path

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


def cmd_themes(cfg: Config) -> None:
    from telco_retention.nlp.data import load_raw, save, telecom_threads
    from telco_retention.nlp.report import write_themes_report
    from telco_retention.nlp.themes import run_themes

    res = telecom_threads(load_raw(cfg), cfg)
    save(res.tweets, cfg)
    th = run_themes(res.tweets, cfg)
    log.info("themes: k=%s; report %s", th["k"], write_themes_report(res.tweets, res.steps, th, cfg))


def cmd_import_labels(cfg: Config) -> None:
    """Validate the hand-labelled CSV and keep ids + labels only (the text stays out of git)."""
    src = cfg.path("raw_dir").parent / "labels" / "human_to_label.csv"
    df = pd.read_csv(src, encoding="utf-8-sig", dtype=str).fillna("")
    for col in ("theme_code", "sentiment", "urgency"):
        df[col] = df[col].str.strip().str.lower()
    bad = df[
        ~df.theme_code.isin([str(i) for i in range(1, 15)])
        | ~df.sentiment.isin(["neg", "neu", "pos"])
        | ~df.urgency.isin(["high", "normal"])
    ]
    if len(bad):
        raise SystemExit(f"{len(bad)} rows incomplete or invalid, first rows: {bad.row.head(10).tolist()}")
    out = df[["row", "tweet_id", "theme_code", "sentiment", "urgency", "notes"]].assign(labeller="human")
    out["theme_code"] = out.theme_code.astype(int)
    out["tweet_id"] = out.tweet_id.astype("int64")
    path = src.parent / "human_labels.csv"
    out.to_csv(path, index=False)
    log.info("imported %s human labels to %s", len(out), path)


def cmd_classify(cfg: Config) -> None:
    from telco_retention.nlp.classify import agreement, labelled_frame, run_classifiers
    from telco_retention.nlp.insight import driver_table, predict_all, weekly_volumes, write_reports
    from telco_retention.nlp.themes import cached_embeddings, save_path

    tweets = pd.read_parquet(cfg.path("processed_dir") / "telecom_tweets.parquet")
    x = cached_embeddings(tweets, cfg)
    labels_dir = cfg.path("raw_dir").parent / "labels"
    df, xl = labelled_frame(cfg, tweets, x, labels_dir / "llm_labels.csv")
    themes = pd.read_parquet(save_path(cfg))[["tweet_id", "theme"]]
    cluster_theme = df[["tweet_id"]].merge(themes, on="tweet_id", how="left")["theme"]
    res = run_classifiers(df, xl, cluster_theme, cfg)
    human_path = labels_dir / "human_labels.csv"
    agree = human_eval = None
    if human_path.exists():
        from telco_retention.nlp.classify import evaluate_on_human

        human = pd.read_csv(human_path)
        from telco_retention.nlp.classify import label_bias

        llm_cols = df[["tweet_id", "theme_code", "sentiment", "urgency"]]
        agree = agreement(human, llm_cols)
        agree.attrs["bias"] = label_bias(human, llm_cols)
        human_eval = evaluate_on_human(res, human, tweets, x, cfg)
    pred = predict_all(res, tweets, x)
    customers = pd.read_parquet(cfg.path("processed_dir") / "customers.parquet")
    paths = write_reports(
        res, pred, weekly_volumes(pred, cfg), driver_table(pred, customers, cfg), agree, human_eval, cfg
    )
    log.info("classifiers: best %s; reports %s", res.meta["best_theme_model"], [str(p) for p in paths])


def cmd_score(cfg: Config, input_path: str | None) -> None:
    """Score the active customers (or a file of customers in IBM's columns) and write the ranked contact list."""
    from telco_retention.scoring import load_bundle, prepare, run_batch, write_outputs

    if input_path:
        path = Path(input_path)
        raw = pd.read_csv(path, dtype=str, keep_default_na=False) if path.suffix == ".csv" else pd.read_parquet(path)
        customers, source = prepare(raw), path.name
    else:
        all_customers = pd.read_parquet(cfg.path("processed_dir") / "customers.parquet")
        customers, source = (
            all_customers[all_customers.churn == 0].reset_index(drop=True),
            "active customers (churn = 0)",
        )
    run = run_batch(customers, load_bundle(cfg), cfg, source)
    log.info("scored %s customers, %s on the list; %s", len(customers), len(run.contacts), write_outputs(run, cfg))


def cmd_package_tickets(cfg: Config) -> None:
    """Evaluate and save the servable (TF-IDF) ticket classifiers."""
    import joblib

    from telco_retention.nlp.tickets import fit_ticket_models, write_ticket_report

    tweets = pd.read_parquet(cfg.path("processed_dir") / "telecom_tweets.parquet")
    labels_dir = cfg.path("raw_dir").parent / "labels"
    df = pd.read_csv(labels_dir / "llm_labels.csv").merge(tweets[["tweet_id", "clean"]], on="tweet_id")
    human_path = labels_dir / "human_labels.csv"
    human = pd.read_csv(human_path) if human_path.exists() else None
    bundle, scores = fit_ticket_models(df, cfg, human)
    path = cfg.path("artifacts_dir") / "ticket_models.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    log.info("ticket models %s; report %s", path, write_ticket_report(scores, cfg))


STEPS = ["fetch", "data", "eda", "train", "survival", "target", "score", "themes", "classify", "package-tickets"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="telco", description=__doc__)
    parser.add_argument("step", choices=[*STEPS, "all", "import-labels"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true", help="fetch: download again")
    parser.add_argument("--input", default=None, help="score: CSV/Parquet of customers in IBM's columns")
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
        "themes": lambda: cmd_themes(cfg),
        "classify": lambda: cmd_classify(cfg),
        "score": lambda: cmd_score(cfg, args.input),
        "package-tickets": lambda: cmd_package_tickets(cfg),
        "import-labels": lambda: cmd_import_labels(cfg),
    }
    for name in STEPS if args.step == "all" else [args.step]:
        steps[name]()


if __name__ == "__main__":
    main()
