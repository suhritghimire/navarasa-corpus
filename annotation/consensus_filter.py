#!/usr/bin/env python3
"""
annotation/consensus_filter.py
================================
Apply majority-vote or unanimity consensus filter to LLM annotation outputs.

Given a spreadsheet with GPT-4o, DeepSeek, and Groq rasa predictions,
this script computes:
  - Final_rasa (unanimous agreement → high confidence label)
  - Majority_rasa (2-of-3 agreement → lower confidence)
  - Not_Determined (all 3 disagree → excluded from training)

Usage:
    python annotation/consensus_filter.py \
        --input data/annotated/ramayana_llm_labeled.xlsx \
        --output data/processed/SanskritRasaBank_v1.xlsx \
        --consensus unanimous
"""

import os
import argparse
import pandas as pd


VALID_RASAS = {
    "Shringara", "Hasya", "Karuna", "Raudra", "Veera",
    "Bhayanaka", "Bibhatsa", "Adbhuta", "Shanta"
}

LLM_COLUMNS = ["GPT-4o_rasa", "deepseek-chat_rasa", "groq(gpt-oss-20b)_rasa"]


def normalize_rasa(val) -> str | None:
    """Normalize a raw rasa label to canonical form."""
    if pd.isna(val):
        return None
    val = str(val).strip().lower()
    mapping = {
        "shantha": "Shanta", "shanta": "Shanta",
        "sringara": "Shringara", "shringara": "Shringara",
        "veera": "Veera", "karuna": "Karuna",
        "raudra": "Raudra", "bhayanaka": "Bhayanaka",
        "bibhatsa": "Bibhatsa", "adbhuta": "Adbhuta", "hasya": "Hasya",
    }
    return mapping.get(val, None)


def compute_consensus(row: pd.Series) -> tuple[str, str]:
    """
    Given a row with normalized LLM predictions, compute:
      - final_rasa: unanimous, or Not_Determined
      - majority_rasa: 2-of-3, or Not_Determined
    Returns (final_rasa, majority_rasa)
    """
    preds = [row[f"{c}_n"] for c in LLM_COLUMNS if f"{c}_n" in row.index]
    # Remove None (invalid) predictions
    valid_preds = [p for p in preds if p is not None]

    if len(valid_preds) == 0:
        return "Not_Determined", "Not_Determined"

    # Unanimous
    if len(set(valid_preds)) == 1 and len(valid_preds) == 3:
        return valid_preds[0], valid_preds[0]

    # Majority (2-of-3)
    from collections import Counter
    counts = Counter(valid_preds)
    top_rasa, top_count = counts.most_common(1)[0]
    if top_count >= 2:
        return "Not_Determined", top_rasa

    return "Not_Determined", "Not_Determined"


def main():
    parser = argparse.ArgumentParser(description="Apply consensus filter to LLM annotations")
    parser.add_argument("--input", required=True, help="Input XLSX with LLM predictions")
    parser.add_argument("--output", required=True, help="Output XLSX path")
    parser.add_argument(
        "--consensus", choices=["unanimous", "majority"], default="unanimous",
        help="Which consensus strategy to use for Final_rasa (default: unanimous)"
    )
    args = parser.parse_args()

    print(f"\n Loading: {args.input}")
    df = pd.read_excel(args.input)
    print(f"   Total rows: {len(df):,}")

    # Normalize all LLM columns
    for col in LLM_COLUMNS:
        if col in df.columns:
            df[f"{col}_n"] = df[col].apply(normalize_rasa)
        else:
            print(f"   Column not found: {col}")

    # Compute consensus
    df[["Final_rasa", "Majority_rasa"]] = df.apply(
        compute_consensus, axis=1, result_type="expand"
    )

    # Stats
    unanimous_count = (df["Final_rasa"] != "Not_Determined").sum()
    majority_count  = (df["Majority_rasa"] != "Not_Determined").sum()
    print(f"\n   Unanimous agreement: {unanimous_count:,} ({100*unanimous_count/len(df):.1f}%)")
    print(f"   Majority agreement:  {majority_count:,} ({100*majority_count/len(df):.1f}%)")
    print(f"   Not determined:      {(df['Final_rasa'] == 'Not_Determined').sum():,}")

    # Select only high-confidence rows for output
    if args.consensus == "unanimous":
        df_out = df[df["Final_rasa"] != "Not_Determined"].copy()
        label_col = "Final_rasa"
    else:
        df_out = df[df["Majority_rasa"] != "Not_Determined"].copy()
        label_col = "Majority_rasa"
        df_out["Final_rasa"] = df_out["Majority_rasa"]

    print(f"\n   Kept for training ({args.consensus}): {len(df_out):,} rows")
    print("\n   Class distribution:")
    print(df_out["Final_rasa"].value_counts().to_string())

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_out.to_excel(args.output, index=False)
    print(f"\n   Saved → {args.output}")
    print(" Consensus filtering complete.")


if __name__ == "__main__":
    main()
