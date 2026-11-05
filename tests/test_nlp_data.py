"""Cleaning, threading and the class-based TF-IDF on a handful of constructed tweets (test data only, no model)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from telco_retention.config import load_config, with_root


def test_clean_text_removes_handles_urls_emoji_and_entities() -> None:
    from telco_retention.nlp.data import clean_text, emoji_count

    raw = "@sprintcare my bill is wrong &amp; nobody helps 😡😡 https://t.co/abc  !!"
    assert clean_text(raw) == "my bill is wrong & nobody helps !!"
    assert emoji_count(raw) == 2


def _tweets() -> pd.DataFrame:
    rows = [
        # an opener answered by Sprint, then the brand's two replies (only the first counts)
        (
            1,
            "111",
            True,
            "Tue Oct 31 10:00:00 +0000 2017",
            "@sprintcare my internet keeps dropping every night",
            "2,3",
            None,
        ),
        (2, "sprintcare", False, "Tue Oct 31 10:05:00 +0000 2017", "@111 Sorry! Please DM us", None, 1.0),
        (3, "sprintcare", False, "Tue Oct 31 11:00:00 +0000 2017", "@111 following up", None, 1.0),
        # a customer reply inside a thread (not an opener)
        (4, "111", True, "Tue Oct 31 10:06:00 +0000 2017", "@sprintcare done, sent you a DM now", "5", 2.0),
        (5, "sprintcare", False, "Tue Oct 31 10:10:00 +0000 2017", "@111 thanks", None, 4.0),
        # an opener answered by a non-telecom brand
        (6, "222", True, "Tue Oct 31 12:00:00 +0000 2017", "@AmazonHelp where is my parcel today", "7", None),
        (7, "AmazonHelp", False, "Tue Oct 31 12:01:00 +0000 2017", "@222 Let us check", None, 6.0),
        # a too-short opener answered by Verizon
        (8, "333", True, "Tue Oct 31 13:00:00 +0000 2017", "@VerizonSupport hi", "9", None),
        (9, "VerizonSupport", False, "Tue Oct 31 13:02:00 +0000 2017", "@333 Hello!", None, 8.0),
    ]
    cols = ["tweet_id", "author_id", "inbound", "created_at", "text", "response_tweet_id", "in_response_to_tweet_id"]
    return pd.DataFrame(rows, columns=cols)


def test_threading_keeps_only_telecom_openers_with_their_first_reply(tmp_path: Path) -> None:
    from telco_retention.nlp.data import telecom_threads

    res = telecom_threads(_tweets(), with_root(load_config(), tmp_path))
    out = res.tweets
    assert out.tweet_id.tolist() == [1]  # not the in-thread reply (4), not Amazon (6), not the 1-word opener (8)
    row = out.iloc[0]
    assert row.brand == "Sprint" and row.reply_text == "Sorry! Please DM us" and row.reply_minutes == 5.0
    assert row.clean == "my internet keeps dropping every night"
    assert [s["rows"] for s in res.steps] == [9, 4, 3, 2, 1, 1]


def test_ctfidf_finds_the_distinctive_words_of_each_cluster() -> None:
    from telco_retention.nlp.themes import ctfidf_terms

    texts = pd.Series(["bill charged twice on my bill"] * 6 + ["internet outage again no internet"] * 6)
    labels = np.array([0] * 6 + [1] * 6)
    terms = ctfidf_terms(texts, labels, top=3)
    own0 = set(" ".join(texts[:6]).split()) | {"charged twice", "twice bill", "bill charged"}
    assert all(set(term.split()) <= own0 for term in terms[0])  # only words of cluster 0's tweets
    assert not {"internet", "outage"} & set(" ".join(terms[0]).split())
    assert {"internet", "outage", "outage internet", "internet outage", "again"} & set(terms[1])
