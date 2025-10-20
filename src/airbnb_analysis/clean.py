#!/usr/bin/env python3
"""
clean.py — Preprocess Airbnb reviews data.

Usage:
    python clean.py --input /path/to/airbnb_reviews.csv --output /path/to/airbnb_reviews_clean.csv

What it does:
1) Reads raw CSV
2) Normalizes/combines language columns
3) Parses dates (ds) and adds year, month, day, week, quarter
4) Cleans review text (message) into clean_message, adds message_len_chars and message_len_words
5) Parses topics/subtopics to Python lists (when possible)
6) Keeps a compact set of useful columns
"""
import argparse
import ast
import re
from typing import List, Optional

import numpy as np
import pandas as pd

# ---------- Text utilities ----------

def clean_text(s: Optional[str]) -> str:
    """Lowercase, remove punctuation/numbers, collapse whitespace; optional stopword removal."""
    if not isinstance(s, str):
        return ""
    x = s.lower()
    x = re.sub(r"(https?://\S+)|(\S+@\S+)", " ", x)
    x = re.sub(r"[^a-z\s']", " ", x)
    x = re.sub(r"\s+", " ", x).strip()

    return x


# ---------- Column helpers ----------
def coalesce_series(*series: pd.Series) -> pd.Series:
    """Return the first non-null value across multiple series (like SQL COALESCE)."""
    out = series[0].copy()
    for s in series[1:]:
        out = out.fillna(s)
    return out


def parse_maybe_list(col: pd.Series) -> pd.Series:
    """Convert stringified list (e.g., "['a','b']") to Python list where possible; else keep NaN/None."""
    def _parse(x):
        if isinstance(x, list):
            return x
        if pd.isna(x):
            return None
        if isinstance(x, str):
            x_strip = x.strip()
            if len(x_strip) >= 2 and x_strip[0] in "[(" and x_strip[-1] in ")]":
                try:
                    v = ast.literal_eval(x_strip)
                    if isinstance(v, (list, tuple)):
                        return list(v)
                except Exception:
                    return None
        return None
    return col.apply(_parse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input airbnb_reviews.csv")
    ap.add_argument("--output", required=True, help="Path to save cleaned CSV")
    args = ap.parse_args()

    # --- Load ---
    df = pd.read_csv(args.input)

    # --- Dates ---
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["year"] = df["ds"].dt.year
    df["month"] = df["ds"].dt.month
    df["day"] = df["ds"].dt.day
    df["week"] = df["ds"].dt.isocalendar().week.astype("Int64")
    df["quarter"] = df["ds"].dt.quarter

    # --- Language unification ---
    df["language_final"] = coalesce_series(
        df["language"],
        df["Trustpilot: language"],
        df["Google Play: language"]
    )

    # --- Message cleaning ---
    df["clean_message"] = df["message"].apply(lambda x: clean_text(x))
    df["message_len_chars"] = df["message"].fillna("").astype(str).str.len()
    df["message_len_words"] = df["message"].fillna("").astype(str).str.split().apply(len)

    # --- Topics/Subtopics parsing ---
    df["topics_parsed"] = parse_maybe_list(df["topics"])
    df["subtopics_parsed"] = parse_maybe_list(df["subtopics"])

    # --- Keep a compact set of useful columns ---
    keep_cols = [
        "ds","year","month","day","week","quarter",
        "source","language_final","sentiment","score",
        "message","clean_message","message_len_chars","message_len_words",
        "topics_parsed"
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    clean_df = df[keep_cols].copy()

    # --- Save cleaned CSV ---
    clean_df.to_csv(args.output, index=False)
    print(f"[OK] Saved cleaned dataset to: {args.output}")


if __name__ == "__main__":
    main()
