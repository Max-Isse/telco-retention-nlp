"""Theme volumes over time and the link from complaint themes to churn drivers (D-022), plus the phase 6 report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from telco_retention.config import Config
from telco_retention.nlp.classify import ClassifyResult, label_names


def predict_all(res: ClassifyResult, tweets: pd.DataFrame, x: np.ndarray) -> pd.DataFrame:
    model = res.models["best_theme"]
    pred = model.predict(tweets["clean"]) if res.meta["best_theme_model"].startswith("tf-idf") else model.predict(x)
    sent = res.models["sentiment"].predict(x)
    urg = res.models["urgency"].predict(x)
    return tweets[["tweet_id", "brand", "created"]].assign(theme_code=pred, sentiment=sent, urgency=urg)


def weekly_volumes(pred: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    names = label_names(cfg)
    p = pred[pred.created >= pd.Timestamp("2017-10-01", tz="UTC")]
    p = p.assign(week=p.created.dt.tz_convert(None).dt.to_period("W-SUN").dt.start_time, theme=p.theme_code.map(names))
    counts = p.groupby(["week", "theme"]).size().unstack(fill_value=0)
    counts = counts[counts.sum(axis=1) >= 500]  # partial weeks at the edges have too few tweets
    return counts.div(counts.sum(axis=1), axis=0)


def driver_table(pred: pd.DataFrame, customers: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    drivers: dict[str, Any] = cfg.section("nlp")["drivers"]
    churners = customers[customers.churn == 1]
    reasons = churners["churn_reason"].fillna("(missing)")
    rows = []
    for name, d in drivers.items():
        row: dict[str, Any] = {
            "driver": name,
            "tweet_share": float(pred.theme_code.isin(d["themes"]).mean()),
            "tweet_share_negative": float((pred.theme_code.isin(d["themes"]) & (pred.sentiment == "neg")).mean()),
            "ibm_churn_reason_share": float(reasons.isin(d["churn_reasons"]).mean()),
        }
        feat = d.get("telco_feature")
        if feat:
            col, val = feat["column"], feat["value"]
            has = customers[col].astype(str) == str(val)
            row["telco_feature"] = f"{col} = {val}"
            row["churn_rate_with"] = float(customers.loc[has, "churn"].mean())
            row["churn_rate_without"] = float(customers.loc[~has, "churn"].mean())
            row["customers_with"] = int(has.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return str(df.astype(object).to_markdown(index=False, floatfmt=floatfmt)) if len(df) else "_none_"


def write_reports(
    res: ClassifyResult,
    pred: pd.DataFrame,
    weekly: pd.DataFrame,
    drivers: pd.DataFrame,
    agreement: pd.DataFrame | None,
    human_eval: pd.DataFrame | None,
    cfg: Config,
) -> list[Path]:
    tables, figs, reports = cfg.path("tables_dir"), cfg.path("figures_dir"), cfg.path("reports_dir")
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    for name, df in [
        ("nlp_theme_scores", res.theme_scores),
        ("nlp_theme_per_class", res.theme_per_class),
        ("nlp_sentiment_scores", res.sentiment_scores),
        ("nlp_urgency_scores", res.urgency_scores),
        ("nlp_weekly_theme_shares", weekly.reset_index()),
        ("themes_to_churn_drivers", drivers),
    ]:
        df.to_csv(tables / f"{name}.csv", index=False)
    res.confusion.to_csv(tables / "nlp_theme_confusion.csv")
    names = label_names(cfg)

    fig, ax = plt.subplots(figsize=(10, 5))
    top = weekly.mean().sort_values(ascending=False).index[:7]
    for theme in top:
        ax.plot(weekly.index, weekly[theme], marker="o", label=theme)
    ax.set_ylabel("share of the week's support tweets")
    ax.set_title("Weekly theme shares (classifier applied to all 83k tweets)")
    ax.legend(fontsize=7)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figs / "nlp_weekly_themes.png", dpi=120)
    plt.close(fig)

    conf = res.confusion.copy()
    conf.index = [f"{c} {names[c][:22]}" for c in conf.index]
    conf.columns = [str(c) for c in conf.columns]
    fail = res.sentiment_failures.copy()
    fail_counts = fail.groupby(["sentiment", "pred_sent_ots"]).size().rename("tweets").reset_index()
    review_path = cfg.path("raw_dir").parent / "labels" / "sentiment_failure_review.csv"
    review = pd.read_csv(review_path) if review_path.exists() else pd.DataFrame(columns=["row", "category"])
    review_counts = (
        review.groupby("category").size().rename("tweets").reset_index().sort_values("tweets", ascending=False)
    )
    fail_examples = (
        fail.merge(review[["row", "category"]], on="row", how="left")
        .groupby("category", dropna=False)
        .head(2)
        .assign(clean=lambda d: d.clean.str.slice(0, 120))[["category", "sentiment", "pred_sent_ots", "clean"]]
    )
    m = res.meta
    if agreement is not None:
        b = agreement.attrs.get("bias", {})
        nan = float("nan")
        agree_text = (
            _md(agreement)
            + "\n\nKappa strength uses the Landis and Koch bands. Where the LLM labels lean, as shares of the "
            + f"{len(agreement) and int(agreement.tweets.iloc[0])} tweets: human neutral but LLM negative "
            + f"{b.get('human_neu_llm_neg', nan):.1%}, the reverse {b.get('human_neg_llm_neu', nan):.1%}; human "
            + f"urgent but LLM normal {b.get('human_high_llm_normal', nan):.1%}, the reverse "
            + f"{b.get('human_normal_llm_high', nan):.1%}. The LLM labels are more negative and less often urgent "
            + "than the human's. Classifiers trained on them inherit that lean, so quote the scores against the hand "
            + "labels (section 2)."
        )
    else:
        agree_text = "_Pending: the human labels (`data/labels/human_labels.csv`) are not in yet._"
    human_text = _md(human_eval) if human_eval is not None else "_Pending the human labels._"
    cl = res.cluster_check
    classifiers = f"""<!-- GENERATED by `telco classify` - do not edit by hand. -->
# Support-text classifiers

Labels: 1,000 tweets labelled by an **LLM (Claude)** following `docs/labelling_guide.md` (D-020); 200 of them were
also labelled by hand, to measure the LLM labels' quality. **Out-of-time split:** {m["train_rows"]} October tweets
train, {m["test_rows"]} tweets from November onwards test (the iPhone X launch falls in the test period).

## 1. Label quality: human vs LLM

{agree_text}

## 2. Themes (14 classes)

{_md(res.theme_scores)}

Best: **{m["best_theme_model"]}**. Per class on the test period:

{_md(res.theme_per_class)}

Confusion matrix (rows = label, columns = prediction, codes as in the guide):

{_md(conf.reset_index().rename(columns={"index": "label"}), ".0f")}

Phase 5's cluster themes, read off as a zero-label classifier against the LLM labels of all 1,000 tweets: accuracy
{cl["accuracy_all_1000"]:.2f}, macro-F1 {cl["macro_f1_all_1000"]:.2f} ({cl["share_labelled_other"]:.0%} of tweets are
"Other", a class the clusters do not have).

Classifiers scored against the hand labels (the 200 human-labelled tweets; test-period rows only where marked):

{human_text}

## 3. Sentiment

{_md(res.sentiment_scores)}

Where the off-the-shelf model disagrees with the labels (test period), by label and prediction:

{_md(fail_counts, ".0f")}

Why it disagrees: each disagreement was read and categorised (`data/labels/sentiment_failure_review.csv`):

{_md(review_counts, ".0f")}

Examples per category:

{_md(fail_examples, ".0f")}

The largest group is a definitional difference, not an error: the model scores outage vocabulary ("down", "not working",
"no service") as negative, while the labelling guide calls a factual problem report neutral and leaves the problem to
the theme. Sarcasm is the clearest true failure: complaints phrased as praise (\"Good job guys! Thank you!\" for a
cancelled pre-order) are scored positive.

## 4. Urgency

{_md(res.urgency_scores)}
"""
    p1 = reports / "nlp_classifiers.md"
    p1.write_text(classifiers, encoding="utf-8")

    share = (
        pred.assign(theme=pred.theme_code.map(names))
        .groupby("theme")
        .agg(
            tweets=("tweet_id", "size"),
            negative=("sentiment", lambda s: float((s == "neg").mean())),
            urgent=("urgency", lambda s: float((s == "high").mean())),
        )
        .reset_index()
        .assign(share=lambda d: d.tweets / d.tweets.sum())
        .sort_values("tweets", ascending=False)
    )
    share.to_csv(tables / "nlp_theme_volumes.csv", index=False)
    weekly_tbl = weekly[top].reset_index().assign(week=lambda d: d.week.dt.strftime("%Y-%m-%d"))
    insight = f"""<!-- GENERATED by `telco classify` - do not edit by hand. -->
# From complaint themes to churn drivers

The Twitter data and IBM's customers are **not linked** (D-001): nothing below joins a tweet to a customer. The two
meet only at the level of churn drivers: what telecom customers complain about in public, what IBM's churners say made
them leave, and how churn differs for customers with the related service.

## 1. Theme volumes, sentiment and urgency (all {len(pred):,} tweets, classifier predictions)

{_md(share)}

Weekly theme shares (weeks with 500+ tweets):

{_md(weekly_tbl)}

![weekly themes](figures/nlp_weekly_themes.png)

## 2. Themes and churn drivers (D-022)

`tweet_share`: share of support tweets in the driver's themes. `ibm_churn_reason_share`: share of IBM's 1,869 churners
giving a reason in that driver. `churn_rate_with/without`: IBM churn for customers with and without the related feature.

{_md(drivers)}

Reading: the drivers that dominate public complaints (reliability, TV) are not the ones IBM's churners name most
(price and competitor offers, customer service); customer service is large in both. Shares are not comparable as
rates: tweets over-represent acute, visible failures, and churn reasons are what a leaving customer chose from a list.
"""
    p2 = reports / "themes_to_churn.md"
    p2.write_text(insight, encoding="utf-8")
    return [p1, p2]
