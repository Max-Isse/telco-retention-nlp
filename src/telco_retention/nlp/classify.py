"""Theme, sentiment and urgency classifiers on the labelled sample, out-of-time; label agreement (D-020, D-021).

Labels: 1,000 tweets labelled by an LLM (Claude) with `docs/labelling_guide.md`, 200 of them also hand-labelled.
Split by time: October 2017 trains, November onwards tests. Models: TF-IDF + logistic regression (baseline) and
sentence embeddings + logistic regression; for sentiment also an off-the-shelf Twitter model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline

from telco_retention.config import Config

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
URGENT_WORDS = (
    "down",
    "outage",
    "out ",
    "not working",
    "no internet",
    "no service",
    "no signal",
    "cancel",
    "switch",
    "asap",
    "urgent",
    "can't use",
    "cannot",
    "now",
    "today",
    "all day",
    "hours",
)


def label_names(cfg: Config) -> dict[int, str]:
    return {int(k): str(v) for k, v in cfg.section("nlp")["labels"].items()}


def labelled_frame(cfg: Config, tweets: pd.DataFrame, x: np.ndarray, path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Labels joined to tweet text and embedding rows (same order)."""
    labels = pd.read_csv(path)
    pos = pd.Series(np.arange(len(tweets)), index=tweets["tweet_id"].to_numpy())
    df = labels.merge(tweets[["tweet_id", "clean", "created", "brand"]], on="tweet_id", how="left")
    return df, x[pos.loc[df["tweet_id"]].to_numpy()]


def tfidf_lr(seed: int) -> Any:
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True),
        LogisticRegression(C=5.0, max_iter=3000, class_weight="balanced", random_state=seed),
    )


def embed_lr(seed: int) -> Any:
    return LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced", random_state=seed)


def macro_f1_ci(y: np.ndarray, pred: np.ndarray, n: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        vals.append(f1_score(y[idx], pred[idx], average="macro", zero_division=0))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def per_class(y: np.ndarray, pred: np.ndarray, names: dict[int, str]) -> pd.DataFrame:
    labels = sorted(names)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    return pd.DataFrame({"theme": [names[c] for c in labels], "precision": p, "recall": r, "f1": f, "test_rows": s})


@dataclass
class ClassifyResult:
    theme_scores: pd.DataFrame
    theme_per_class: pd.DataFrame
    confusion: pd.DataFrame
    cluster_check: dict[str, float]
    sentiment_scores: pd.DataFrame
    sentiment_failures: pd.DataFrame
    urgency_scores: pd.DataFrame
    models: dict[str, Any]
    test: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)


def off_the_shelf_sentiment(texts: list[str]) -> np.ndarray:
    """Off-the-shelf Twitter sentiment (RoBERTa). Loads the cached PyTorch weights without network calls when present
    (the hub's safetensors conversion lookup can hang); downloads them otherwise."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    try:
        tok = AutoTokenizer.from_pretrained(SENTIMENT_MODEL, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            SENTIMENT_MODEL, local_files_only=True, use_safetensors=False
        )
    except OSError:
        tok = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL, use_safetensors=False)
    clf = pipeline("sentiment-analysis", model=model, tokenizer=tok, device=-1, truncation=True)  # type: ignore[call-overload]
    mapping = {"negative": "neg", "neutral": "neu", "positive": "pos"}
    return np.array([mapping[r["label"].lower()] for r in clf(texts, batch_size=32)])


def urgency_rule(texts: pd.Series) -> np.ndarray:
    low = texts.str.lower()
    return np.where(low.apply(lambda t: any(w in t for w in URGENT_WORDS)), "high", "normal")


def run_classifiers(df: pd.DataFrame, x: np.ndarray, cluster_theme: pd.Series, cfg: Config) -> ClassifyResult:
    names = label_names(cfg)
    seed, boot = cfg.seed, 1000
    tr, te = (df["period"] == "train").to_numpy(), (df["period"] == "test").to_numpy()
    y = df["theme_code"].to_numpy()
    rows, models, preds = [], {}, {}
    candidates = {
        "majority class": (DummyClassifier(strategy="most_frequent"), "text"),
        "tf-idf + logistic": (tfidf_lr(seed), "text"),
        "embeddings + logistic": (embed_lr(seed), "emb"),
    }
    for name, (model, kind) in candidates.items():
        xtr = df.loc[tr, "clean"] if kind == "text" else x[tr]
        xte = df.loc[te, "clean"] if kind == "text" else x[te]
        model.fit(xtr, y[tr])
        pred = model.predict(xte)
        preds[name], models[name] = pred, model
        lo, hi = macro_f1_ci(y[te], pred, boot, seed)
        rows.append(
            {
                "model": name,
                "macro_f1": f1_score(y[te], pred, average="macro", zero_division=0),
                "macro_f1_ci": f"{lo:.2f} to {hi:.2f}",
                "accuracy": float((pred == y[te]).mean()),
            }
        )
    best = max(("tf-idf + logistic", "embeddings + logistic"), key=lambda n: f1_score(y[te], preds[n], average="macro"))
    code_of = {v: k for k, v in names.items()}
    cl = cluster_theme.map(code_of).to_numpy()
    cluster_check = {
        "accuracy_all_1000": float((cl == y).mean()),
        "macro_f1_all_1000": float(f1_score(y, cl, average="macro", zero_division=0)),
        "share_labelled_other": float((y == 14).mean()),
    }
    labels_sorted = sorted(names)
    conf = pd.DataFrame(
        confusion_matrix(y[te], preds[best], labels=labels_sorted), index=labels_sorted, columns=labels_sorted
    )
    conf.index.name = "true \\ predicted"

    # sentiment: own classifier (embeddings) vs off-the-shelf model, on the test period
    ys = df["sentiment"].to_numpy()
    own_s = embed_lr(seed).fit(x[tr], ys[tr])
    pred_own = own_s.predict(x[te])
    pred_ots = off_the_shelf_sentiment(df.loc[te, "clean"].tolist())
    s_rows = []
    for name, pred in [
        ("embeddings + logistic (trained on October)", pred_own),
        (f"off-the-shelf {SENTIMENT_MODEL}", pred_ots),
    ]:
        s_rows.append(
            {
                "model": name,
                "macro_f1": f1_score(ys[te], pred, average="macro", zero_division=0),
                "accuracy": float((pred == ys[te]).mean()),
                "neg_recall": float(((pred == "neg") & (ys[te] == "neg")).sum() / max((ys[te] == "neg").sum(), 1)),
            }
        )
    test = df.loc[te].assign(pred_theme=preds[best], pred_sent_own=pred_own, pred_sent_ots=pred_ots)
    failures = test[test.sentiment != test.pred_sent_ots][["row", "clean", "sentiment", "pred_sent_ots"]]

    # urgency: own classifier vs keyword rule
    yu = df["urgency"].to_numpy()
    own_u = embed_lr(seed).fit(x[tr], yu[tr])
    pred_u = own_u.predict(x[te])
    rule_u = urgency_rule(df.loc[te, "clean"])
    u_rows = []
    for name, pred in [("embeddings + logistic", pred_u), ("keyword rule", rule_u)]:
        tp = int(((pred == "high") & (yu[te] == "high")).sum())
        u_rows.append(
            {
                "model": name,
                "macro_f1": f1_score(yu[te], pred, average="macro", zero_division=0),
                "high_precision": tp / max(int((pred == "high").sum()), 1),
                "high_recall": tp / max(int((yu[te] == "high").sum()), 1),
            }
        )
    test["pred_urgency"] = pred_u
    models.update({"best_theme": models[best], "sentiment": own_s, "urgency": own_u})
    meta = {"best_theme_model": best, "train_rows": int(tr.sum()), "test_rows": int(te.sum())}
    return ClassifyResult(
        pd.DataFrame(rows),
        per_class(y[te], preds[best], names),
        conf,
        cluster_check,
        pd.DataFrame(s_rows),
        failures,
        pd.DataFrame(u_rows),
        models,
        test,
        meta,
    )


def kappa_band(k: float) -> str:
    """Landis and Koch (1977) descriptive bands for Cohen's kappa."""
    for upper, name in [(0.0, "poor"), (0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")]:
        if k <= upper:
            return name
    return "almost perfect"


def label_bias(human: pd.DataFrame, llm: pd.DataFrame) -> dict[str, float]:
    """Where the LLM labels lean, on the shared tweets."""
    m = human.merge(llm, on="tweet_id", suffixes=("_human", "_llm"))
    return {
        "human_neu_llm_neg": float(((m.sentiment_human == "neu") & (m.sentiment_llm == "neg")).mean()),
        "human_neg_llm_neu": float(((m.sentiment_human == "neg") & (m.sentiment_llm == "neu")).mean()),
        "human_high_llm_normal": float(((m.urgency_human == "high") & (m.urgency_llm == "normal")).mean()),
        "human_normal_llm_high": float(((m.urgency_human == "normal") & (m.urgency_llm == "high")).mean()),
    }


def agreement(human: pd.DataFrame, llm: pd.DataFrame) -> pd.DataFrame:
    """Cohen's kappa and raw agreement between human and LLM labels on the shared tweets."""
    m = human.merge(llm, on="tweet_id", suffixes=("_human", "_llm"))
    rows = []
    for col in ("theme_code", "sentiment", "urgency"):
        a, b = m[f"{col}_human"].astype(str), m[f"{col}_llm"].astype(str)
        k = float(cohen_kappa_score(a, b))
        rows.append(
            {
                "label": col,
                "tweets": len(m),
                "agreement": float((a == b).mean()),
                "cohen_kappa": k,
                "strength": kappa_band(k),
            }
        )
    return pd.DataFrame(rows)


def evaluate_on_human(
    res: ClassifyResult, human: pd.DataFrame, tweets: pd.DataFrame, x: np.ndarray, cfg: Config
) -> pd.DataFrame:
    """Models scored against the hand labels; classifiers only on test-period tweets (October ones were trained on)."""
    ids = pd.read_csv(cfg.path("raw_dir").parent / "labels" / "label_sample_ids.csv")
    h = human.merge(ids[["tweet_id", "period"]], on="tweet_id").merge(tweets[["tweet_id", "clean"]], on="tweet_id")
    pos = pd.Series(np.arange(len(tweets)), index=tweets["tweet_id"].to_numpy())
    xh = x[pos.loc[h["tweet_id"]].to_numpy()]
    te = (h["period"] == "test").to_numpy()
    theme_model = res.models["best_theme"]
    theme_pred = (
        theme_model.predict(h["clean"])
        if res.meta["best_theme_model"].startswith("tf-idf")
        else theme_model.predict(xh)
    )
    sent_own = res.models["sentiment"].predict(xh)
    sent_ots = off_the_shelf_sentiment(h["clean"].tolist())
    urg = res.models["urgency"].predict(xh)
    rows = []
    for name, y, pred, mask in [
        (f"themes: {res.meta['best_theme_model']}", h.theme_code.to_numpy(), theme_pred, te),
        ("sentiment: embeddings + logistic", h.sentiment.to_numpy(), sent_own, te),
        ("sentiment: off-the-shelf", h.sentiment.to_numpy(), sent_ots, np.ones(len(h), bool)),
        ("urgency: embeddings + logistic", h.urgency.to_numpy(), urg, te),
    ]:
        rows.append(
            {
                "model": name,
                "human_labelled_rows": int(mask.sum()),
                "macro_f1": f1_score(y[mask], pred[mask], average="macro", zero_division=0),
                "accuracy": float((y[mask] == pred[mask]).mean()),
            }
        )
    return pd.DataFrame(rows)
