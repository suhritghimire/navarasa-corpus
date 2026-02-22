#!/usr/bin/env python3
"""
annotation/pipeline.py
========================
3-LLM Ensemble Annotation Pipeline for Sanskrit NavaRasa Classification.

Annotates Sanskrit verses with one of the 9 Navarasa labels using three
independent LLM annotators (GPT-4o, DeepSeek-Chat, Groq/LLaMA-3.1) and
applies consensus filtering.

Requirements (set as environment variables):
    OPENAI_API_KEY     - GPT-4o
    DEEPSEEK_API_KEY   - DeepSeek-Chat
    GROQ_API_KEY       - Groq / LLaMA-3.1

Usage:
    python annotation/pipeline.py \\
        --input  data/raw/your_verses.xlsx \\
        --output data/annotated/output.xlsx \\
        --consensus unanimous        # 'unanimous' | 'majority'
        --batch-size 50              # verses per API call batch
"""

import os
import time
import argparse
import logging
from typing import Optional

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Label space ─────────────────────────────────────────────────────────────
VALID_RASAS = [
    "Shringara", "Hasya", "Karuna", "Raudra", "Veera",
    "Bhayanaka", "Bibhatsa", "Adbhuta", "Shanta"
]

RASA_LIST_STR = ", ".join(VALID_RASAS)

SYSTEM_PROMPT = (
    "You are an expert in Sanskrit literature and Indian aesthetics (rasa theory).\n"
    "Given a Sanskrit verse, classify it into exactly ONE of the nine Navarasa categories:\n"
    f"{RASA_LIST_STR}.\n"
    "Reply with ONLY the rasa name — no explanation, no punctuation, nothing else."
)

# ── Normaliser (shared with consensus_filter.py) ─────────────────────────────
_NORM_MAP = {
    "shantha": "Shanta", "shanta": "Shanta",
    "sringara": "Shringara", "shringara": "Shringara",
    "veera": "Veera", "karuna": "Karuna", "raudra": "Raudra",
    "bhayanaka": "Bhayanaka", "bibhatsa": "Bibhatsa",
    "adbhuta": "Adbhuta", "hasya": "Hasya",
}


def normalize_rasa(val) -> Optional[str]:
    if pd.isna(val):
        return None
    return _NORM_MAP.get(str(val).strip().lower(), None)


# ── GPT-4o annotator ─────────────────────────────────────────────────────────
def annotate_gpt4o(verses: list[str], api_key: str, batch_size: int = 50) -> list[Optional[str]]:
    """Annotate verses using GPT-4o via the OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    client = OpenAI(api_key=api_key)
    results = []

    for i in tqdm(range(0, len(verses), batch_size), desc="GPT-4o"):
        batch = verses[i: i + batch_size]
        for verse in batch:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": verse},
                    ],
                    max_tokens=10,
                    temperature=0,
                )
                raw = response.choices[0].message.content.strip()
                results.append(normalize_rasa(raw))
            except Exception as e:
                log.warning(f"GPT-4o error on verse: {e}")
                results.append(None)
            time.sleep(0.05)  # rate-limit guard

    return results


# ── DeepSeek annotator ───────────────────────────────────────────────────────
def annotate_deepseek(verses: list[str], api_key: str, batch_size: int = 50) -> list[Optional[str]]:
    """Annotate verses using DeepSeek-Chat via the OpenAI-compatible API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    results = []

    for i in tqdm(range(0, len(verses), batch_size), desc="DeepSeek"):
        batch = verses[i: i + batch_size]
        for verse in batch:
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": verse},
                    ],
                    max_tokens=10,
                    temperature=0,
                )
                raw = response.choices[0].message.content.strip()
                results.append(normalize_rasa(raw))
            except Exception as e:
                log.warning(f"DeepSeek error on verse: {e}")
                results.append(None)
            time.sleep(0.05)

    return results


# ── Groq/LLaMA annotator ─────────────────────────────────────────────────────
def annotate_groq(verses: list[str], api_key: str, batch_size: int = 50) -> list[Optional[str]]:
    """Annotate verses using Groq-hosted LLaMA-3.1."""
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("groq package not installed. Run: pip install groq")

    client = Groq(api_key=api_key)
    results = []

    for i in tqdm(range(0, len(verses), batch_size), desc="Groq/LLaMA"):
        batch = verses[i: i + batch_size]
        for verse in batch:
            try:
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": verse},
                    ],
                    max_tokens=10,
                    temperature=0,
                )
                raw = response.choices[0].message.content.strip()
                results.append(normalize_rasa(raw))
            except Exception as e:
                log.warning(f"Groq error on verse: {e}")
                results.append(None)
            time.sleep(0.05)

    return results


# ── Consensus filter ─────────────────────────────────────────────────────────
def apply_consensus(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Add Final_rasa and Majority_rasa columns based on 3-LLM agreement."""
    from collections import Counter

    final_rasas, majority_rasas = [], []

    for _, row in df.iterrows():
        preds = [
            row.get("GPT-4o_rasa"),
            row.get("deepseek-chat_rasa"),
            row.get("groq(gpt-oss-20b)_rasa"),
        ]
        valid = [p for p in preds if p is not None]

        if len(valid) == 0:
            final_rasas.append("Not_Determined")
            majority_rasas.append("Not_Determined")
            continue

        # Unanimous
        if len(set(valid)) == 1 and len(valid) == 3:
            final_rasas.append(valid[0])
            majority_rasas.append(valid[0])
            continue

        # 2-of-3 majority
        counts = Counter(valid)
        top_rasa, top_count = counts.most_common(1)[0]
        if top_count >= 2:
            final_rasas.append("Not_Determined")
            majority_rasas.append(top_rasa)
        else:
            final_rasas.append("Not_Determined")
            majority_rasas.append("Not_Determined")

    df["Final_rasa"]    = final_rasas
    df["Majority_rasa"] = majority_rasas

    unanimous = (df["Final_rasa"] != "Not_Determined").sum()
    majority  = (df["Majority_rasa"] != "Not_Determined").sum()
    log.info(f"Unanimous  : {unanimous:,} ({100*unanimous/len(df):.1f}%)")
    log.info(f"Majority   : {majority:,}  ({100*majority/len(df):.1f}%)")
    log.info(f"Undecided  : {(df['Final_rasa'] == 'Not_Determined').sum():,}")

    return df


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="3-LLM ensemble annotation pipeline")
    parser.add_argument("--input",      required=True, help="Input XLSX with a 'sanskrit_text' column")
    parser.add_argument("--output",     required=True, help="Output XLSX path")
    parser.add_argument("--consensus",  choices=["unanimous", "majority"], default="unanimous")
    parser.add_argument("--batch-size", type=int, default=50, help="Verses per LLM batch")
    parser.add_argument("--text-col",   default="sanskrit_text", help="Column name for verse text")
    args = parser.parse_args()

    openai_key   = os.environ.get("OPENAI_API_KEY")
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    groq_key     = os.environ.get("GROQ_API_KEY")

    log.info(f"Loading {args.input}")
    df = pd.read_excel(args.input)
    log.info(f"  Rows: {len(df):,}")

    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found. Available: {list(df.columns)}")

    verses = df[args.text_col].fillna("").tolist()

    # ── Annotate ──────────────────────────────────────────────────────────
    if openai_key:
        log.info("Running GPT-4o annotation…")
        df["GPT-4o_rasa"] = annotate_gpt4o(verses, openai_key, args.batch_size)
    else:
        log.warning("OPENAI_API_KEY not set — skipping GPT-4o annotation")

    if deepseek_key:
        log.info("Running DeepSeek annotation…")
        df["deepseek-chat_rasa"] = annotate_deepseek(verses, deepseek_key, args.batch_size)
    else:
        log.warning("DEEPSEEK_API_KEY not set — skipping DeepSeek annotation")

    if groq_key:
        log.info("Running Groq/LLaMA annotation…")
        df["groq(gpt-oss-20b)_rasa"] = annotate_groq(verses, groq_key, args.batch_size)
    else:
        log.warning("GROQ_API_KEY not set — skipping Groq annotation")

    # ── Consensus ─────────────────────────────────────────────────────────
    df = apply_consensus(df, args.consensus)

    # Filter to high-confidence rows if requested
    if args.consensus == "unanimous":
        df_out = df[df["Final_rasa"] != "Not_Determined"].copy()
    else:
        df_out = df[df["Majority_rasa"] != "Not_Determined"].copy()
        df_out["Final_rasa"] = df_out["Majority_rasa"]

    log.info(f"Kept ({args.consensus}): {len(df_out):,} rows")
    log.info("\nClass distribution:\n" + df_out["Final_rasa"].value_counts().to_string())

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df_out.to_excel(args.output, index=False)
    log.info(f"Saved → {args.output}")
    log.info(" Annotation pipeline complete.")


if __name__ == "__main__":
    main()
