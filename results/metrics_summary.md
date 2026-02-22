# Results — Navarasa Classification

Cross-validation metrics for all models trained on `Filtered_Agreed_Rasa_Dataset_Standardized.xlsx`.  
Evaluation metric: **Weighted F1-score** and **Accuracy** on held-out validation fold.

---

## Summary Table

| Model | Mean Accuracy | Mean Weighted F1 | Strategy |
|-------|:-------------:|:----------------:|----------|
| **MuRIL-large-cased** | **87.42%** | **87.50%** | Full fine-tuning |
| **IndicBERT-v2** | **84.60%** | **84.75%** | LoRA (r=16, α=32) |
| **XLM-RoBERTa-large** | **83.45%** | **83.44%** | LoRA (r=8, α=16) |
| SanBERTa | 61.12% | 61.25% | LoRA (r=16, α=32) |

---

## IndicBERT-v2 — Fold-level Results

| Fold | Accuracy | Weighted F1 |
|------|:--------:|:-----------:|
| 1 | 83.75% | 83.88% |
| 2 | 83.78% | 83.97% |
| 3 | 83.63% | 83.80% |
| 4 | 85.85% | 85.95% |
| 5 | 86.01% | 86.14% |
| **Mean** | **84.60%** | **84.75%** |
| Std dev | ±1.01% | ±1.02% |

---

## XLM-RoBERTa-large — Fold-level Results

| Fold | Accuracy | Weighted F1 |
|------|:--------:|:-----------:|
| 1 | 83.90% | 83.91% |
| 2 | 82.23% | 82.18% |
| 3 | 83.43% | 83.41% |
| 4 | 83.62% | 83.64% |
| 5 | 84.06% | 84.06% |
| **Mean** | **83.45%** | **83.44%** |
| Std dev | ±0.60% | ±0.66% |

---

## MuRIL-large-cased — Fold-level Results

| Fold | Accuracy | Weighted F1 |
|------|:--------:|:-----------:|
| 1 | 85.34% | 85.50% |
| 2 | 86.33% | 86.33% |
| 3 | 87.89% | 87.89% |
| 4 | 89.08% | 89.14% |
| 5 | 88.48% | 88.62% |
| **Mean** | **87.42%** | **87.50%** |
| Std dev | ±1.39% | ±1.38% |

---

## SanBERTa — Fold-level Results

| Fold | Accuracy | Weighted F1 |
|------|:--------:|:-----------:|
| 1 | 60.52% | 60.98% |
| 2 | 62.03% | 62.20% |
| 3 | 59.88% | 59.87% |
| 4 | 62.61% | 62.58% |
| 5 | 60.58% | 60.61% |
| **Mean** | **61.12%** | **61.25%** |
| Std dev | ±0.96% | ±0.88% |

---

## Key Observations

1. **IndicBERT-v2 outperforms all models** — Its Indic-language pre-training aligns well with Sanskrit's script and vocabulary.
2. **XLM-RoBERTa-large** is a strong and consistent performer despite being trained with fewer LoRA parameters.
3. **SanBERTa underperforms** multilingual models significantly, suggesting the Sanskrit-native pre-training corpus alone is insufficient for this task — context from multilingual corpora may help.
4. **All models show stable variance** across folds (std ≈ 1%), demonstrating robust and reproducible results.
5. **Dataset imbalance** is handled via class-weighted loss and focal loss, contributing to stable weighted F1 scores.

---

## Training Configuration Summary

| Config | IndicBERT | XLM-RoBERTa | MuRIL | SanBERTa |
|--------|-----------|-------------|-------|----------|
| Base model | IndicBERTv2-MLM | xlm-roberta-large | muril-large-cased | SanBERTa |
| LoRA r | 16 | 8 | N/A | 16 |
| Batch size (eff.) | 32 | 8 | 32 | 32 |
| LR | 2e-5 | 1e-5 | 2e-5 | 2e-5 |
| Max epochs | 15 | 15 | 10 | 15 |
| Loss | Focal+Smooth | Focal+Smooth | Weighted CE | Focal+Smooth |
| Max seq len | 256 | 256 | 256 | 256 |
