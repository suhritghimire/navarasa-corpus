#!/usr/bin/env python3
"""
evaluation/llm_baseline.py
============================
Evaluate zero-shot LLM predictions against Final_rasa ground truth.

Expects a spreadsheet with columns:
    sanskrit_text | Final_rasa | GPT-4o_rasa | deepseek-chat_rasa | groq(gpt-oss-20b)_rasa

Usage:
    python evaluation/llm_baseline.py \
        --data data/annotated/combined_master.xlsx \
        --output results/
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, cohen_kappa_score


VALID_RASAS = {
    "Shringara", "Hasya", "Karuna", "Raudra", "Veera",
    "Bhayanaka", "Bibhatsa", "Adbhuta", "Shanta"
}

LLM_COLUMN_MAP = {
    "GPT-4o":     "GPT-4o_rasa",
    "DeepSeek":   "deepseek-chat_rasa",
    "Groq":       "groq(gpt-oss-20b)_rasa",
}


def normalize_rasa(val):
    """Map raw LLM output to canonical rasa name, or None if invalid."""
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


def evaluate_llm(y_true: pd.Series, y_pred: pd.Series, model_name: str) -> dict:
    mask = y_pred.notna()
    yt = y_true[mask]
    yp = y_pred[mask]
    coverage = mask.sum() / len(y_true)

    acc    = accuracy_score(yt, yp)
    w_f1   = f1_score(yt, yp, average="weighted", zero_division=0)
    m_f1   = f1_score(yt, yp, average="macro", zero_division=0)
    kappa  = cohen_kappa_score(yt, yp)

    print(f"\n{'─'*55}")
    print(f"  {model_name}")
    print(f"{'─'*55}")
    print(f"  Valid responses : {mask.sum():>6,} / {len(y_true):,} ({coverage*100:.1f}%)")
    print(f"  Accuracy        : {acc*100:.2f}%")
    print(f"  Weighted F1     : {w_f1*100:.2f}%")
    print(f"  Macro F1        : {m_f1*100:.2f}%")
    print(f"  Cohen's κ       : {kappa:.4f}")
    print("\n  Per-class report:")
    print(classification_report(yt, yp, zero_division=0, digits=4))

    return {
        "model": model_name,
        "coverage": coverage,
        "accuracy": acc,
        "weighted_f1": w_f1,
        "macro_f1": m_f1,
        "kappa": kappa,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate zero-shot LLM baselines")
    parser.add_argument("--data", type=str, default="data/annotated/combined_master.xlsx")
    parser.add_argument("--output", type=str, default="results/")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"\n Loading: {args.data}")
    df = pd.read_excel(args.data)
    print(f"   Rows: {len(df):,}")

    # Normalize ground-truth
    df["Final_rasa_n"] = df["Final_rasa"].apply(normalize_rasa)

    # Normalize LLM predictions
    for col_label, col_name in LLM_COLUMN_MAP.items():
        if col_name in df.columns:
            df[f"{col_label}_n"] = df[col_name].apply(normalize_rasa)

    # Filter rows where ground truth is valid
    df_valid = df[df["Final_rasa_n"].isin(VALID_RASAS)].copy()
    print(f"   Valid ground-truth rows: {len(df_valid):,}")

    print("\n" + "=" * 55)
    print("  ZERO-SHOT LLM BASELINE EVALUATION")
    print("=" * 55)

    results = []
    for col_label in LLM_COLUMN_MAP:
        norm_col = f"{col_label}_n"
        if norm_col not in df_valid.columns:
            print(f"   Skipping {col_label}: column not found.")
            continue
        r = evaluate_llm(df_valid["Final_rasa_n"], df_valid[norm_col], col_label)
        results.append(r)

    # Summary table
    if results:
        summary = pd.DataFrame(results).set_index("model")
        summary_path = os.path.join(args.output, "llm_baseline_summary.csv")
        summary.to_csv(summary_path)
        print(f"\n\n SUMMARY TABLE\n{'='*55}")
        print(summary[["accuracy", "weighted_f1", "macro_f1", "kappa"]].to_string())
        print(f"\n   Summary saved → {summary_path}")

    print("\n LLM baseline evaluation complete.")


if __name__ == "__main__":
    main()
