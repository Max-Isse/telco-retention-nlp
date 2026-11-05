"""Telecom support conversations from the Twitter customer-support dataset: filter, thread, clean (D-018).

Unit: a customer's **opening tweet** (not a reply to anything) whose first reply came from one of the configured US
phone/internet brands, paired with that brand reply. Customer handles are removed; author ids in the source are
already anonymised numbers.
"""

from __future__ import annotations

import html
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from telco_retention.config import Config

URL = re.compile(r"https?://\S+|www\.\S+")
HANDLE = re.compile(r"@\w+")
EMOJI = re.compile(
    "[\U0001f000-\U0001faff\U00002600-\U000027bf\U0001f900-\U0001f9ff‍️\U0001f1e6-\U0001f1ff]",
    flags=re.UNICODE,
)
SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Remove handles, URLs and emoji, unescape HTML entities, collapse whitespace (case kept for transformers)."""
    t = html.unescape(str(text))
    t = URL.sub(" ", t)
    t = HANDLE.sub(" ", t)
    t = EMOJI.sub(" ", t)
    return SPACES.sub(" ", t).strip()


def emoji_count(text: str) -> int:
    return len(EMOJI.findall(str(text)))


def load_raw(cfg: Config) -> pd.DataFrame:
    """The full dataset, cached as Parquet next to the zip on first use."""
    nc = cfg.section("nlp")
    zpath = cfg.root / nc["source_zip"]
    cache = zpath.parent.parent / "twcs.parquet"
    if not cache.exists():
        with zipfile.ZipFile(zpath) as z:
            name = next(n for n in z.namelist() if n.endswith("twcs.csv") and ".." not in n)
            with z.open(name) as f:
                df = pd.read_csv(f, dtype={"author_id": "string", "text": "string", "response_tweet_id": "string"})
        df.to_parquet(cache, index=False)
    return pd.read_parquet(cache)


@dataclass
class ThreadResult:
    tweets: pd.DataFrame  # one row per opening customer tweet with its first brand reply
    steps: list[dict[str, Any]] = field(default_factory=list)


def telecom_threads(raw: pd.DataFrame, cfg: Config) -> ThreadResult:
    nc = cfg.section("nlp")
    brands: dict[str, str] = nc["brands"]
    steps = [{"step": "all tweets in the dataset", "rows": len(raw)}]
    replies = raw[(~raw.inbound) & raw.author_id.isin(list(brands)) & raw.in_response_to_tweet_id.notna()].copy()
    steps.append({"step": "replies by the configured telecom brands", "rows": len(replies)})
    replies["created"] = pd.to_datetime(replies.created_at, format="%a %b %d %H:%M:%S %z %Y", utc=True)
    replies["parent"] = replies.in_response_to_tweet_id.astype("int64")
    first = replies.sort_values("created").drop_duplicates("parent", keep="first")
    customers = raw[raw.inbound].merge(
        first[["parent", "author_id", "text", "created"]].rename(
            columns={"parent": "tweet_id", "author_id": "brand_account", "text": "reply_text", "created": "reply_at"}
        ),
        on="tweet_id",
    )
    steps.append({"step": "customer tweets answered by a telecom brand", "rows": len(customers)})
    openers = customers[customers.in_response_to_tweet_id.isna()].copy()
    steps.append({"step": "opening tweets (not replies to anything)", "rows": len(openers)})
    openers["created"] = pd.to_datetime(openers.created_at, format="%a %b %d %H:%M:%S %z %Y", utc=True)
    openers["clean"] = openers.text.map(clean_text)
    openers["emojis"] = openers.text.map(emoji_count)
    openers["words"] = openers.clean.map(lambda s: len(s.split()))
    keep = openers[openers.words >= int(nc["min_words"])]
    steps.append({"step": f"at least {nc['min_words']} words after cleaning", "rows": len(keep)})
    keep = keep.drop_duplicates("clean")
    steps.append({"step": "duplicate texts removed", "rows": len(keep)})
    out = pd.DataFrame(
        {
            "tweet_id": keep.tweet_id.to_numpy(),
            "brand_account": keep.brand_account.to_numpy(),
            "brand": keep.brand_account.map(brands).to_numpy(),
            "created": keep.created.to_numpy(),
            "text": keep.text.to_numpy(),
            "clean": keep.clean.to_numpy(),
            "emojis": keep.emojis.to_numpy(),
            "words": keep.words.to_numpy(),
            "reply_text": keep.reply_text.map(clean_text).to_numpy(),
            "reply_minutes": ((keep.reply_at - keep.created).dt.total_seconds() / 60).to_numpy(),
        }
    ).sort_values("created", ignore_index=True)
    return ThreadResult(out, steps)


def volumes(tweets: pd.DataFrame) -> pd.DataFrame:
    g = tweets.groupby("brand")
    out = g.agg(
        tweets=("tweet_id", "size"),
        first=("created", "min"),
        last=("created", "max"),
        median_words=("words", "median"),
        median_reply_minutes=("reply_minutes", "median"),
    ).reset_index()
    out["share"] = out.tweets / out.tweets.sum()
    return out.sort_values("tweets", ascending=False)


def monthly_volumes(tweets: pd.DataFrame) -> pd.DataFrame:
    m = tweets.assign(month=tweets.created.dt.strftime("%Y-%m"))
    return m.groupby("month").size().rename("tweets").reset_index()


def save(tweets: pd.DataFrame, cfg: Config) -> Path:
    out = cfg.path("processed_dir") / "telecom_tweets.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    tweets.to_parquet(out, index=False)
    return out
