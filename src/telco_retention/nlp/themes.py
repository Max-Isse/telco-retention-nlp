"""Theme discovery: sentence embeddings, k-means, class-based TF-IDF terms, examples near each centre (D-019).

Clusters are labelled by a person reading their top terms and examples (labels in `nlp.theme_labels`); the labels are
validated against a hand-labelled sample in phase 6.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score

from telco_retention.config import Config


def embed(texts: list[str], model_name: str, batch_size: int = 256) -> np.ndarray:
    from sentence_transformers import SentenceTransformer  # heavy import, only when embedding

    model = SentenceTransformer(model_name, device="cpu")
    return np.asarray(model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False))


def cached_embeddings(tweets: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Embeddings of `clean`, cached on disk under a hash of the tweet ids and model name."""
    nc = cfg.section("nlp")
    key = hashlib.sha256((nc["embedding_model"] + ",".join(map(str, tweets.tweet_id))).encode()).hexdigest()[:16]
    path = cfg.path("processed_dir") / f"telecom_embeddings_{key}.npy"
    if path.exists():
        return np.load(path)
    vectors = embed(tweets["clean"].tolist(), nc["embedding_model"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors


def choose_k(x: np.ndarray, candidates: list[int], sample: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=min(sample, len(x)), replace=False)
    rows = []
    for k in candidates:
        labels = KMeans(k, n_init=4, random_state=seed).fit_predict(x[idx])
        rows.append({"k": k, "silhouette": float(silhouette_score(x[idx], labels))})
    return pd.DataFrame(rows)


def cluster(x: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    km = KMeans(k, n_init=4, random_state=seed).fit(x)
    return km.labels_, km.cluster_centers_


def ctfidf_terms(texts: pd.Series, labels: np.ndarray, top: int = 10, min_df: int = 5) -> dict[int, list[str]]:
    """Class-based TF-IDF (as in BERTopic): words frequent in a cluster and rare across clusters.

    The vocabulary is built over tweets (a term must occur in at least `min_df` tweets), then counts are summed per
    cluster.
    """
    vec = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=min_df, lowercase=True)
    counts_by_tweet = vec.fit_transform(texts.to_numpy())
    clusters = np.unique(labels)
    member = csr_matrix((np.ones(len(labels)), (np.searchsorted(clusters, labels), np.arange(len(labels)))))
    counts = np.asarray((member @ counts_by_tweet).todense(), dtype=float)
    tf = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    idf = np.log(1 + counts.sum(axis=1).mean() / np.maximum(counts.sum(axis=0), 1))
    scores = tf * idf
    vocab = np.array(vec.get_feature_names_out())
    return {int(c): list(vocab[np.argsort(-scores[i])[:top]]) for i, c in enumerate(clusters)}


def examples(
    tweets: pd.DataFrame, x: np.ndarray, labels: np.ndarray, centres: np.ndarray, n: int
) -> dict[int, list[str]]:
    out = {}
    for c in range(len(centres)):
        idx = np.where(labels == c)[0]
        sims = x[idx] @ centres[c]
        out[c] = tweets["clean"].to_numpy()[idx[np.argsort(-sims)[:n]]].tolist()
    return out


def run_themes(tweets: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    nc = cfg.section("nlp")
    x = cached_embeddings(tweets, cfg)
    k_table = choose_k(x, [int(k) for k in nc["cluster_k"]], int(nc["silhouette_sample"]), cfg.seed)
    k = int(nc.get("k") or k_table.sort_values("silhouette", ascending=False).k.iloc[0])
    labels, centres = cluster(x, k, cfg.seed)
    terms = ctfidf_terms(tweets["clean"], labels)
    return {"x": x, "k_table": k_table, "k": k, "labels": labels, "centres": centres, "terms": terms}


def save_path(cfg: Config) -> Path:
    return cfg.path("processed_dir") / "telecom_tweets_themes.parquet"
