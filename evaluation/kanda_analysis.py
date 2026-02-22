#!/usr/bin/env python3
"""
evaluation/kanda_analysis.py
==============================
Kāṇḍa-wise rasa distribution analysis for the Vālmīki Rāmāyaṇa.

Usage:
    python evaluation/kanda_analysis.py \
        --data data/annotated/combined_master.xlsx \
        --output results/
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

RASA_COLORS = {
    "Shringara": "#27ae60",
    "Hasya":     "#f1c40f",
    "Karuna":    "#95a5a6",
    "Raudra":    "#e74c3c",
    "Veera":     "#f39c12",
    "Bhayanaka": "#2c3e50",
    "Bibhatsa":  "#2980b9",
    "Adbhuta":   "#e67e22",
    "Shanta":    "#85c1e9",
}

KANDA_ORDER = ["Bala Kanda", "Ayodhya Kanda", "Aranya Kanda",
               "Kishkindha Kanda", "Sundara Kanda", "Yuddha Kanda"]

KANDA_DESCRIPTIONS = {
    "Bala Kanda": "Auspicious opening; Rāma's birth and martial training",
    "Ayodhya Kanda": "Grief-dominant (karuṇa-pradhāna); exile and Daśaratha's death",
    "Aranya Kanda": "Terror and fury; Sītā's abduction—the epic's emotional crisis",
    "Kishkindha Kanda": "Wonder and valor; the Vānara kingdom and Hanumān's debut",
    "Sundara Kanda": "The Beautiful Book; heroism and poetic beauty combined",
    "Yuddha Kanda": "War climax; maximum Vīra and Raudra—the epic's resolution",
}


def load_and_prepare(data_path: str) -> pd.DataFrame:
    df = pd.read_excel(data_path)
    # Normalize Kanda column name
    kanda_col = next(
        (c for c in df.columns if "kanda" in c.lower() or "book" in c.lower()), None
    )
    rasa_col = next(
        (c for c in df.columns if "final_rasa" in c.lower() or "rasa" in c.lower()), None
    )
    if not kanda_col or not rasa_col:
        raise ValueError(
            f"Could not detect Kanda or Rasa columns. Found: {list(df.columns)}"
        )
    df = df[[kanda_col, rasa_col]].rename(
        columns={kanda_col: "Kanda", rasa_col: "Final_rasa"}
    )
    df = df.dropna(subset=["Kanda"])
    df = df[df["Final_rasa"] != "Not_Determined"]
    return df


def compute_distribution(df: pd.DataFrame) -> pd.DataFrame:
    pivot = pd.crosstab(df["Kanda"], df["Final_rasa"])
    return pivot


def plot_stacked_bar(pivot: pd.DataFrame, output_dir: str):
    rasas = [r for r in RASA_COLORS if r in pivot.columns]
    colors = [RASA_COLORS[r] for r in rasas]

    # Normalize by row
    norm = pivot[rasas].div(pivot[rasas].sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(14, 7))
    bottom = np.zeros(len(norm))

    for rasa, color in zip(rasas, colors):
        vals = norm[rasa].values if rasa in norm.columns else np.zeros(len(norm))
        ax.barh(norm.index, vals, left=bottom, color=color, label=rasa, height=0.65, edgecolor="white")
        bottom += vals

    ax.set_xlabel("Rasa Proportion (%)", fontsize=12)
    ax.set_title("Kāṇḍa-wise Rasa Distribution — Vālmīki Rāmāyaṇa", fontsize=14, pad=16)
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", ncol=3, fontsize=9, framealpha=0.85)
    plt.tight_layout()

    out = os.path.join(output_dir, "kanda_rasa_stacked_bar.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"   Stacked bar chart → {out}")


def plot_heatmap(pivot: pd.DataFrame, output_dir: str):
    rasas = [r for r in RASA_COLORS if r in pivot.columns]
    norm = pivot[rasas].div(pivot[rasas].sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        norm, annot=True, fmt=".2f", cmap="YlOrRd",
        ax=ax, linewidths=0.4, linecolor="lightgrey",
        cbar_kws={"label": "Proportion"}
    )
    ax.set_title("Kāṇḍa Rasa Fingerprint (Row-Normalized)", fontsize=13, pad=16)
    ax.set_ylabel("Kāṇḍa", fontsize=11)
    ax.set_xlabel("Rasa", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out = os.path.join(output_dir, "kanda_rasa_heatmap.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"   Rasa fingerprint heatmap → {out}")


def print_literary_analysis(pivot: pd.DataFrame):
    print("\n" + "=" * 70)
    print("   KĀṆḌA-WISE LITERARY ANALYSIS")
    print("=" * 70)
    for kanda in KANDA_ORDER:
        if kanda not in pivot.index:
            print(f"\n  [{kanda}] ── No data (pending annotation)")
            continue
        row = pivot.loc[kanda]
        top_rasas = row.nlargest(3)
        desc = KANDA_DESCRIPTIONS.get(kanda, "")
        print(f"\n   {kanda}")
        print(f"     {desc}")
        print(f"     Top 3 rasas: ", end="")
        for r, c in top_rasas.items():
            print(f"{r} ({c})", end="  |  ")
        print()


def main():
    parser = argparse.ArgumentParser(description="Kāṇḍa-wise rasa distribution analysis")
    parser.add_argument("--data", type=str, default="data/annotated/combined_master.xlsx")
    parser.add_argument("--output", type=str, default="results/")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"\n Loading: {args.data}")
    df = load_and_prepare(args.data)
    pivot = compute_distribution(df)

    print("\n Kāṇḍa × Rasa crosstab:")
    print(pivot.to_string())

    pivot.to_csv(os.path.join(args.output, "kanda_distribution.csv"))
    print(f"\n   Distribution CSV → {args.output}/kanda_distribution.csv")

    print_literary_analysis(pivot)
    plot_stacked_bar(pivot, args.output)
    plot_heatmap(pivot, args.output)

    print("\n Kāṇḍa analysis complete.")


if __name__ == "__main__":
    main()
