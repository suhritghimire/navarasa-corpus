#!/usr/bin/env python3
"""
notebooks/consolidate_notebooks.py
====================================
Utility script: merges all Dataset_creation/*.ipynb files into a single
clean notebook  (notebooks/navarasa_dataset_creation.ipynb).

De-duplicates cells by source content, groups pip-installs into one
setup cell, and preserves logical pipeline order.

Usage:
    python notebooks/consolidate_notebooks.py
"""

import json
import os
import hashlib
from copy import deepcopy

# ── Config ───────────────────────────────────────────────────────────────────
NOTEBOOK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Dataset_creation"
)

NOTEBOOKS = [
    "dataset_creation.ipynb",
    "rasa-ramayana.ipynb",
    "Rasas_ramayana-2.ipynb",
    "Untitled8-2.ipynb",
    "Untitled8_2.ipynb",
    "Navarasa.ipynb",
]

SECTION_TITLES = [
    "## PART 1 -- Initial Dataset Loading & Batch Labeling (OpenAI + Gemini)",
    "## PART 2 -- Gemini Parallel Labeling Pipeline",
    "## PART 3 -- Second Pass Labeling & Result Merging",
    "## PART 4 -- Dataset Cleaning & Filtering",
    "## PART 5 -- Agreement Filtering & Label Standardization",
    "## PART 6 -- Final Model Training, Evaluation & Export",
]

OUTPUT_NOTEBOOK = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "navarasa_dataset_creation.ipynb"
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def cell_fingerprint(source):
    if isinstance(source, list):
        source = "".join(source)
    return hashlib.md5(source.strip().encode()).hexdigest()


def source_text(cell):
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def is_empty_cell(cell):
    return source_text(cell).strip() == ""


def is_pure_install(cell):
    if cell.get("cell_type") != "code":
        return False
    first = next((l.strip() for l in source_text(cell).splitlines() if l.strip()), "")
    return first.startswith(("!pip", "! pip", "%pip"))


def clean_outputs(outputs):
    cleaned = []
    for out in outputs:
        if out.get("output_type") == "display_data":
            data = out.get("data", {})
            if not ("text/html" in data or "text/plain" in data):
                if all(k.startswith("application/vnd") for k in data):
                    continue
        if out.get("output_type") == "error" and \
                "KeyboardInterrupt" in "".join(out.get("traceback", [])):
            continue
        cleaned.append(out)
    return cleaned


def strip_cell(cell):
    c = deepcopy(cell)
    c.pop("id", None)
    c["metadata"] = {}
    if c.get("cell_type") == "code":
        c["outputs"] = clean_outputs(c.get("outputs", []))
        c["execution_count"] = None
    return c


def make_md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def make_separator(title):
    return {"cell_type": "markdown", "metadata": {}, "source": ["---\n", "\n", f"{title}\n"]}


def get_cells(nb_path):
    with open(nb_path, encoding="utf-8") as f:
        return json.load(f).get("cells", [])


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Pass 1: collect all unique install lines
    all_install_lines = set()
    for nb_name in NOTEBOOKS:
        nb_path = os.path.join(NOTEBOOK_DIR, nb_name)
        if not os.path.exists(nb_path):
            continue
        for cell in get_cells(nb_path):
            if is_pure_install(cell):
                for line in source_text(cell).splitlines():
                    line = line.strip()
                    if line:
                        all_install_lines.add(line)

    install_cell = None
    if all_install_lines:
        install_cell = {
            "cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [],
            "source": ["# Install all required packages\n"] + [l + "\n" for l in sorted(all_install_lines)],
        }

    # Pass 2: merge
    seen = set()
    all_cells = []

    all_cells.append(make_md(
        "# Navarasa Dataset Creation Pipeline\n\n"
        "Complete workflow for building the Sanskrit Navarasa classification dataset "
        "from the Valmiki Ramayana.\n\n"
        "## Pipeline Overview\n"
        "1. **Load raw dataset** -- Valmiki Ramayana shlokas (Sanskrit + English)\n"
        "2. **Batch labeling** -- OpenAI gpt-4o-mini batch API, classify into 9 Rasas\n"
        "3. **Gemini labeling** -- Google Gemini as a second annotator\n"
        "4. **Merge results** -- Unify outputs from all batch runs\n"
        "5. **Clean & filter** -- Standardize labels, remove noise\n"
        "6. **Agreement filtering** -- Keep only samples where annotators agree\n"
        "7. **Train & evaluate** -- 5-fold CV baseline on the final dataset\n"
    ))

    if install_cell:
        all_cells.append(make_md("## Setup: Install Dependencies\n"))
        all_cells.append(install_cell)

    for nb_name, section_title in zip(NOTEBOOKS, SECTION_TITLES):
        nb_path = os.path.join(NOTEBOOK_DIR, nb_name)
        if not os.path.exists(nb_path):
            continue

        section_cells = []
        for cell in get_cells(nb_path):
            if is_empty_cell(cell) or is_pure_install(cell):
                continue
            if cell.get("cell_type") == "markdown" and source_text(cell).strip() in ("", "---"):
                continue
            fp = cell_fingerprint(source_text(cell))
            if fp in seen:
                continue
            seen.add(fp)
            section_cells.append(strip_cell(cell))

        if section_cells:
            all_cells.append(make_separator(section_title))
            all_cells.extend(section_cells)
            print(f"  {len(section_cells):3d} unique cells  <- {nb_name}")
        else:
            print(f"    0 new cells  <- {nb_name} (all duplicates)")

    notebook = {
        "nbformat": 4, "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
            "colab": {"provenance": []},
        },
        "cells": all_cells,
    }

    os.makedirs(os.path.dirname(OUTPUT_NOTEBOOK), exist_ok=True)
    with open(OUTPUT_NOTEBOOK, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

    print(f"\nSaved: {OUTPUT_NOTEBOOK}")
    print(f"Total cells: {len(all_cells)}")
    size_mb = os.path.getsize(OUTPUT_NOTEBOOK) / 1024 / 1024
    print(f"File size  : {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
