#!/usr/bin/env python3
"""
============================================================
NAVARASA CLASSIFICATION — IndicBERT + XLM-RoBERTa Training
============================================================
Trains two models in a single pipeline:
  - XLM-RoBERTa-large  (xlm-roberta-large)
  - IndicBERT-v2       (ai4bharat/IndicBERTv2-MLM-only)

Features:
  - 5-fold stratified cross-validation
  - LoRA parameter-efficient fine-tuning (PEFT)
  - Focal Loss + Label Smoothing
  - Automatic checkpoint resuming
  - Class-balanced weighted sampling

Usage:
    python train.py

Dataset is expected at: ../../data/Filtered_Agreed_Rasa_Dataset_Standardized.xlsx
Models are saved to:    saved_models/<model>_fold<n>_best/
"""

import os
import sys
import gc
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# ============================================================
# CONFIGURATION
# ============================================================

# Directory layout: this script lives in models/indicbert_xlmroberta/
# Dataset is two levels up at data/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_PATH = os.path.join(REPO_ROOT, "data", "Filtered_Agreed_Rasa_Dataset_Standardized.xlsx")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "training_checkpoint.json")

# Create output directories
for dir_name in ['saved_models', 'results', 'checkpoints', 'logs']:
    os.makedirs(os.path.join(BASE_DIR, dir_name), exist_ok=True)

MAX_LENGTH = 256

MODELS_CONFIG = [
    {
        'name': 'XLM-RoBERTa-large',
        'model_name': 'xlm-roberta-large',
        'lora_r': 8,
        'lora_alpha': 16,
        'batch_size': 4,
        'learning_rate': 1e-5,
        'epochs': 15,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        'target_modules': ['query', 'value', 'key', 'dense']
    },
    {
        'name': 'IndicBERT',
        'model_name': 'ai4bharat/IndicBERTv2-MLM-only',
        'lora_r': 16,
        'lora_alpha': 32,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'epochs': 15,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        'target_modules': ['query', 'value', 'key', 'dense']
    }
]

# ============================================================
# LOSS FUNCTIONS
# ============================================================

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Label smoothing to reduce overconfidence."""
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


def compute_class_weights(labels):
    """Compute balanced class weights for weighted loss."""
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return torch.tensor(class_weights, dtype=torch.float32)


# ============================================================
# CUSTOM TRAINER
# ============================================================

class CustomTrainer(Trainer):
    """HuggingFace Trainer with Focal Loss + Label Smoothing."""

    def __init__(self, *args, class_weights=None, use_focal=True,
                 use_label_smoothing=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.use_focal = use_focal
        self.use_label_smoothing = use_label_smoothing

        if use_focal:
            self.focal_loss = FocalLoss(gamma=2.0, alpha=class_weights)
        if use_label_smoothing:
            self.label_smoothing = None  # Lazy init (need num classes)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.use_label_smoothing and self.label_smoothing is None:
            self.label_smoothing = LabelSmoothingLoss(
                classes=logits.shape[-1], smoothing=0.1
            )

        if self.use_focal and self.use_label_smoothing:
            loss = self.focal_loss(logits, labels) + self.label_smoothing(logits, labels)
        elif self.use_focal:
            loss = self.focal_loss(logits, labels)
        elif self.use_label_smoothing:
            loss = self.label_smoothing(logits, labels)
        else:
            loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples['sanskrit_text'],
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH
    )


# ============================================================
# CHECKPOINT MANAGER
# ============================================================

class CheckpointManager:
    """Manages training state so interrupted runs can be resumed."""

    def __init__(self, checkpoint_file):
        self.checkpoint_file = checkpoint_file
        self.load()

    def load(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                self.state = json.load(f)
            print(f"\n Resuming from checkpoint")
            print(f"   Completed folds: {self.state['completed_folds']}")
        else:
            self.state = {
                'completed_folds': [],
                'completed_models_in_fold': [],
                'current_fold': 1,
                'current_model_idx': 0,
                'fold_metrics': [],
                'best_models_info': {}
            }
            print("\n Starting fresh training")

    def save(self):
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def is_model_completed(self, fold_num, model_name):
        return f"{model_name}_fold{fold_num}" in self.state['best_models_info']

    def mark_model_completed(self, fold_num, model_name, metrics, model_path):
        key = f"{model_name}_fold{fold_num}"
        self.state['best_models_info'][key] = {
            'path': model_path,
            'accuracy': metrics['accuracy'],
            'f1': metrics['f1']
        }
        self.state['completed_models_in_fold'].append(model_name)
        self.state['fold_metrics'].append({
            'fold': fold_num,
            'model': model_name,
            'accuracy': metrics['accuracy'],
            'f1': metrics['f1']
        })
        self.save()

    def mark_fold_completed(self, fold_num):
        if fold_num not in self.state['completed_folds']:
            self.state['completed_folds'].append(fold_num)
        self.state['completed_models_in_fold'] = []
        self.state['current_fold'] = fold_num + 1
        self.state['current_model_idx'] = 0
        self.save()

    def get_next_task(self):
        if len(self.state['completed_folds']) >= 5:
            return None, None, None
        current_fold = self.state['current_fold']
        completed_in_fold = {
            key.replace(f"_fold{current_fold}", "")
            for key in self.state['best_models_info']
            if f"_fold{current_fold}" in key
        }
        for i, cfg in enumerate(MODELS_CONFIG):
            if cfg['name'] not in completed_in_fold:
                return current_fold, i, cfg
        if len(completed_in_fold) == len(MODELS_CONFIG):
            self.mark_fold_completed(current_fold)
            return self.get_next_task()
        return None, None, None


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("NAVARASA CLASSIFICATION — IndicBERT + XLM-RoBERTa")
    print("=" * 60)

    # Hardware detection
    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() else "cpu"
    print(f" Device: {device}")

    # Validate dataset
    if not os.path.exists(DATA_PATH):
        print(f" Dataset not found at: {DATA_PATH}")
        print("   Expected: ../../data/Filtered_Agreed_Rasa_Dataset_Standardized.xlsx")
        sys.exit(1)

    # ── Load data ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    df = pd.read_excel(DATA_PATH)
    df = df[['sanskrit_text', 'Final_rasa']].dropna()
    df['Final_rasa'] = df['Final_rasa'].replace({'Shanta': 'Shantha'})

    unique_rasas = sorted(df['Final_rasa'].unique())
    label2id = {r: i for i, r in enumerate(unique_rasas)}
    id2label = {i: r for i, r in enumerate(unique_rasas)}
    df['label'] = df['Final_rasa'].map(label2id)

    print(f"Total samples : {len(df)}")
    print(f"Classes       : {unique_rasas}")
    print("\nClass distribution:")
    for rasa, count in df['Final_rasa'].value_counts().sort_index().items():
        print(f"  {rasa}: {count}")

    # Save label mapping
    with open(os.path.join(BASE_DIR, "results", "label_mapping.pkl"), "wb") as f:
        pickle.dump({'label2id': label2id, 'id2label': id2label}, f)

    # ── Training loop ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X = df['sanskrit_text'].values
    y = df['label'].values

    checkpoint = CheckpointManager(CHECKPOINT_FILE)

    while True:
        fold_num, model_idx, cfg = checkpoint.get_next_task()

        if fold_num is None:
            print("\n All training completed!")
            break

        print(f"\n{'=' * 60}")
        print(f"TASK: Fold {fold_num}/5 — Model: {cfg['name']}")
        print(f"{'=' * 60}")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            if fold + 1 != fold_num:
                continue

            train_texts, val_texts = X[train_idx], X[val_idx]
            train_labels, val_labels = y[train_idx], y[val_idx]

            print(f"Train: {len(train_texts)} samples | Val: {len(val_texts)} samples")

            # Build HuggingFace datasets
            train_dataset = Dataset.from_pandas(
                pd.DataFrame({'sanskrit_text': train_texts, 'label': train_labels})
            )
            val_dataset = Dataset.from_pandas(
                pd.DataFrame({'sanskrit_text': val_texts, 'label': val_labels})
            )

            # Class weights
            class_weights = compute_class_weights(train_labels)
            if device == "cuda":
                class_weights = class_weights.cuda()
            elif device == "mps":
                class_weights = class_weights.to("mps")

            # Model output path
            best_path = os.path.join(BASE_DIR, "saved_models", f"{cfg['name']}_fold{fold_num}_best")

            if checkpoint.is_model_completed(fold_num, cfg['name']):
                print(" Already completed — skipping")
                continue

            # ── Load tokenizer & model ──────────────────────
            print("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])

            print(f"Loading {cfg['name']}...")
            base_model = AutoModelForSequenceClassification.from_pretrained(
                cfg['model_name'],
                num_labels=len(unique_rasas),
                id2label=id2label,
                label2id=label2id,
                torch_dtype=torch.float32,
                use_cache=False,
                ignore_mismatched_sizes=True
            )

            # Apply LoRA
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=cfg['lora_r'],
                lora_alpha=cfg['lora_alpha'],
                target_modules=cfg['target_modules'],
                lora_dropout=0.1,
                bias='none',
                modules_to_save=["classifier"]
            )
            model = get_peft_model(base_model, lora_config)
            model.print_trainable_parameters()

            # ── Tokenize ─────────────────────────────────────
            print("Tokenizing...")
            train_enc = train_dataset.map(
                lambda x: tokenize_function(x, tokenizer), batched=True
            ).remove_columns(['sanskrit_text'])
            val_enc = val_dataset.map(
                lambda x: tokenize_function(x, tokenizer), batched=True
            ).remove_columns(['sanskrit_text'])

            data_collator = DataCollatorWithPadding(
                tokenizer=tokenizer, padding='longest', max_length=MAX_LENGTH
            )

            # ── Training arguments ────────────────────────────
            output_dir = os.path.join(BASE_DIR, "checkpoints", f"{cfg['name']}_fold{fold_num}")
            training_args = TrainingArguments(
                output_dir=output_dir,
                eval_strategy="epoch",
                save_strategy="epoch",
                learning_rate=cfg['learning_rate'],
                per_device_train_batch_size=cfg['batch_size'],
                per_device_eval_batch_size=cfg['batch_size'],
                num_train_epochs=cfg['epochs'],
                weight_decay=cfg['weight_decay'],
                warmup_ratio=cfg['warmup_ratio'],
                lr_scheduler_type="cosine",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                save_total_limit=3,
                logging_dir=os.path.join(BASE_DIR, "logs", f"{cfg['name']}_fold{fold_num}"),
                logging_steps=50,
                fp16=False,
                bf16=False,
                report_to="none",
                gradient_accumulation_steps=2,
                max_grad_norm=1.0,
                dataloader_num_workers=2,
                gradient_checkpointing=True,
                optim="adamw_torch",
            )

            # ── Trainer ───────────────────────────────────────
            trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=train_enc,
                eval_dataset=val_enc,
                data_collator=data_collator,
                class_weights=class_weights,
                use_focal=True,
                use_label_smoothing=True,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )

            # Resume from checkpoint if available
            last_checkpoint = None
            if os.path.exists(output_dir):
                ckpts = [f for f in os.listdir(output_dir) if f.startswith("checkpoint-")]
                if ckpts:
                    last_checkpoint = os.path.join(output_dir, sorted(ckpts)[-1])
                    print(f" Resuming from: {last_checkpoint}")

            # ── Train ─────────────────────────────────────────
            print("\nStarting training...")
            trainer.train(resume_from_checkpoint=last_checkpoint)

            # Save best model
            print(f"\n Saving to {best_path}")
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)

            # Evaluate
            preds = trainer.predict(val_enc)
            pred_labels = np.argmax(preds.predictions, axis=-1)

            acc = accuracy_score(val_labels, pred_labels)
            f1 = precision_recall_fscore_support(val_labels, pred_labels, average='weighted')[2]

            print(f"\n Results — {cfg['name']} Fold {fold_num}:")
            print(f"   Accuracy : {acc:.4f}")
            print(f"   F1-score : {f1:.4f}")

            checkpoint.mark_model_completed(
                fold_num, cfg['name'],
                {'accuracy': acc, 'f1': f1},
                best_path
            )

            # Cleanup
            del model, base_model, tokenizer, trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"\n COMPLETED: {cfg['name']} fold {fold_num}")
            print(f"{'=' * 60}")

    print("\n" + "=" * 60)
    print(" TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nResults  → {BASE_DIR}/results/")
    print(f"Models   → {BASE_DIR}/saved_models/")


if __name__ == "__main__":
    main()
