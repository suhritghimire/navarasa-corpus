#!/usr/bin/env python3
"""
evaluation/compute_metrics.py
==============================
Compute and display full metrics for any model's predictions on SanskritRasaBank.

Usage:
    python evaluation/compute_metrics.py --predictions results/predictions/ --output results/
    python evaluation/compute_metrics.py --csv results/MuRIL_classification_report.csv
"""

import os
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
)

RASA_LABELS = [
    "Adbhuta", "Bhayanaka", "Bibhatsa", "Hasya",
    "Karuna", "Raudra", "Shanta", "Shringara", "Veera"
]

RASA_COLORS = {
    "Shringara": "#27ae60",   # Green - Love
    "Hasya":     "#f1c40f",   # Yellow - Laughter
    "Karuna":    "#95a5a6",   # Grey - Grief
    "Raudra":    "#e74c3c",   # Red - Fury
    "Veera":     "#f39c12",   # Gold - Heroism
    "Bhayanaka": "#2c3e50",   # Dark - Terror
    "Bibhatsa":  "#2980b9",   # Blue - Disgust
    "Adbhuta":   "#e67e22",   # Orange - Wonder
    "Shanta":    "#85c1e9",   # Light Blue - Serenity
}


def load_fold_predictions(predictions_dir: str) -> tuple:
    """Load all fold prediction CSVs and concatenate into single arrays."""
    all_true, all_pred = [], []
    fold_files = sorted(glob.glob(os.path.join(predictions_dir, "*.csv")))

    if not fold_files:
        raise FileNotFoundError(f"No CSV files found in {predictions_dir}")

    for fp in fold_files:
        df = pd.read_csv(fp)
        if "true_label" in df.columns and "predicted_label" in df.columns:
            all_true.extend(df["true_label"].tolist())
            all_pred.extend(df["predicted_label"].tolist())
        elif "label" in df.columns and "prediction" in df.columns:
            all_true.extend(df["label"].tolist())
            all_pred.extend(df["prediction"].tolist())
        else:
            print(f"   Skipping {fp}: unrecognised column names.")

    return np.array(all_true), np.array(all_pred)


def print_metrics(y_true, y_pred, model_name: str = "Model"):
    """Print full classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    w_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    m_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"  {model_name} — Performance Summary")
    print(f"{'='*60}")
    print(f"  Accuracy      : {acc*100:.2f}%")
    print(f"  Weighted F1   : {w_f1*100:.2f}%")
    print(f"  Macro F1      : {m_f1*100:.2f}%")
    print(f"  Cohen's κ     : {kappa:.4f}")
    print(f"{'='*60}")

    print("\nPer-class report:")
    labels = sorted(set(y_true.tolist() + y_pred.tolist()))
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    return {"accuracy": acc, "weighted_f1": w_f1, "macro_f1": m_f1, "kappa": kappa}


def plot_confusion_matrix(y_true, y_pred, output_path: str, model_name: str = "Model"):
    """Generate and save a normalized confusion matrix heatmap."""
    labels = sorted(set(y_true.tolist() + y_pred.tolist()))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="YlOrRd",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="lightgrey"
    )
    plt.title(f"{model_name} — Confusion Matrix (Normalized)", fontsize=13, pad=16)
    plt.ylabel("True Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    os.makedirs(output_path, exist_ok=True)
    fn = os.path.join(output_path, f"{model_name.replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(fn, dpi=150)
    plt.close()
    print(f"   Confusion matrix saved → {fn}")


def plot_per_class_f1(y_true, y_pred, output_path: str, model_name: str = "Model"):
    """Generate a horizontal bar chart of per-class F1 scores."""
    labels = sorted(set(y_true.tolist() + y_pred.tolist()))
    f1_scores = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    colors = [RASA_COLORS.get(l, "#7f8c8d") for l in labels]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, f1_scores * 100, color=colors, edgecolor="white", height=0.65)
    ax.axvline(80, color="orange", linestyle="--", linewidth=1.2, label="80% threshold")
    ax.axvline(90, color="green", linestyle="--", linewidth=1.2, label="90% threshold")

    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{score*100:.1f}%", va="center", fontsize=10)

    ax.set_xlabel("F1 Score (%)", fontsize=11)
    ax.set_title(f"{model_name} — Per-Class F1 Score", fontsize=13, pad=16)
    ax.set_xlim(0, 108)
    ax.legend(fontsize=9)
    plt.tight_layout()

    fn = os.path.join(output_path, f"{model_name.replace(' ', '_')}_per_class_f1.png")
    plt.savefig(fn, dpi=150)
    plt.close()
    print(f"   Per-class F1 chart saved → {fn}")


def main():
    parser = argparse.ArgumentParser(description="Compute NavaRasa classification metrics")
    parser.add_argument("--predictions", type=str, default=None,
                        help="Directory containing prediction CSVs (one per fold)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Single pre-computed classification report CSV")
    parser.add_argument("--output", type=str, default="results/",
                        help="Directory to save figures and reports")
    parser.add_argument("--model-name", type=str, default="MuRIL", help="Model name for labels")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    figures_dir = os.path.join(args.output, "confusion_matrices")

    if args.predictions:
        print(f"\n Loading predictions from: {args.predictions}")
        y_true, y_pred = load_fold_predictions(args.predictions)
        metrics = print_metrics(y_true, y_pred, model_name=args.model_name)
        plot_confusion_matrix(y_true, y_pred, figures_dir, model_name=args.model_name)
        plot_per_class_f1(y_true, y_pred, args.output, model_name=args.model_name)

        # Save summary CSV
        report_dict = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        df_report = pd.DataFrame(report_dict).transpose()
        report_path = os.path.join(args.output, f"{args.model_name}_classification_report.csv")
        df_report.to_csv(report_path)
        print(f"   Classification report saved → {report_path}")

    elif args.csv:
        print(f"\n Loading pre-computed report: {args.csv}")
        df = pd.read_csv(args.csv, index_col=0)
        print(df.to_string())

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
