"""
Comprehensive Evaluation Script
================================
Evaluates MCI classification using foundational tabular models (TabPFN, TabICL, RealMLP)
with multiple feature configurations and evaluation paradigms.

Evaluation Modes:
- LOO with embeddings only
- LOO with text features only
- LOO with early fusion (normalized embeddings + weighted features concatenated)
- LOO with late fusion (separate model per modality, averaged probabilities)
- Few-shot (k=1,2,3,5 per class, 10 episodes) for embeddings, features, early fusion, late fusion

Repeat 3 times for statistical analysis.

Usage:
    python comprehensive_evaluation.py --embedding paraphrase-multilingual-MiniLM-L12-v2
    python comprehensive_evaluation.py --embedding google/embeddinggemma-300m --device cuda

Embedding cache:
    Embeddings are cached as {sanitized_model_name}_{dataset_stem}.npy next to the data file.
    If the cache exists it is loaded automatically; otherwise embeddings are computed and saved.
"""

import argparse
import os
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional
import inspect

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score, average_precision_score
)
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Feature columns
TEXT_FEATURES = [
    'text_speech_rate', 'text_ttr', 'text_noun_ratio', 'text_verb_ratio',
    'text_pronoun_ratio', 'text_pronoun_to_noun_ratio', 'text_mean_frequency',
    'text_coherence', 'text_repetitiveness', 'text_idea_densitity',
    'text_syntactic_complexity'
]

DATA_PATH = "data/english_slovene_chinese_korean_data_preprocessed_04022026.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize_model_name(name: str) -> str:
    """Turn a model name / path into a filesystem-safe string."""
    return name.replace('/', '_').replace('\\', '_').replace(' ', '_')


def embedding_cache_path(embedding_model: str, data_path: str) -> Path:
    """
    Return the expected cache path:
        <data_dir>/<sanitized_model>_<dataset_stem>.npy
    """
    data_path = Path(data_path)
    model_tag = sanitize_model_name(embedding_model)
    return data_path.parent / f"{model_tag}_{data_path.stem}.npy"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Compute classification metrics."""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Macro_F1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'F1_Target': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        'Precision': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'Recall': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
    }

    if len(np.unique(y_true)) == 2 and len(y_proba) > 0:
        try:
            metrics['AUROC'] = roc_auc_score(y_true, y_proba)
            metrics['AUPRC'] = average_precision_score(y_true, y_proba)
        except Exception:
            metrics['AUROC'] = np.nan
            metrics['AUPRC'] = np.nan
    else:
        metrics['AUROC'] = np.nan
        metrics['AUPRC'] = np.nan

    return metrics


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

class EmbeddingGenerator:
    """Generate embeddings from text using various models."""

    def __init__(self, model_name: str, device: str = 'cuda'):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        if 'embeddinggemma' in self.model_name.lower():
            from transformers import AutoTokenizer, AutoModel
            import torch
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            self.model_type = 'gemma'
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.model_type = 'sentence_transformer'

    def encode(self, texts: list) -> np.ndarray:
        if self.model_type == 'gemma':
            import torch
            embeddings = []
            batch_size = 16
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding with Gemma"):
                batch = texts[i:i + batch_size]
                inputs = self.tokenizer(
                    batch, return_tensors='pt', padding=True,
                    truncation=True, max_length=512
                ).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_emb)
            return np.vstack(embeddings)
        else:
            return self.model.encode(texts, show_progress_bar=True)


def load_or_compute_embeddings(embedding_model: str, df: pd.DataFrame,
                                data_path: str, device: str) -> np.ndarray:
    """
    Load cached embeddings if they exist, otherwise compute and save them.

    Cache path: <data_dir>/<sanitized_model>_<dataset_stem>.npy
    """
    cache = embedding_cache_path(embedding_model, data_path)

    if cache.exists():
        print(f"\nLoading cached embeddings from: {cache}")
        embeddings = np.load(cache)
        print(f"  Embedding shape: {embeddings.shape}")
        return embeddings

    print(f"\nNo cache found at: {cache}")
    print(f"Generating embeddings with: {embedding_model}")
    embedder = EmbeddingGenerator(embedding_model, device)
    texts = df['transcript_patient'].fillna('').tolist()
    embeddings = embedder.encode(texts)
    print(f"  Embedding shape: {embeddings.shape}")

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, embeddings)
    print(f"  Embeddings saved to: {cache}")

    return embeddings


# ---------------------------------------------------------------------------
# Classifier wrapper
# ---------------------------------------------------------------------------

class FoundationalClassifier:
    """Wrapper for foundational tabular classifiers."""

    def __init__(self, model_type: str, device: str = 'cuda'):
        self.model_type = model_type
        self.device = device
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.model_type == 'tabpfn':
            from tabpfn import TabPFNClassifier
            sig = inspect.signature(TabPFNClassifier.__init__)
            params = list(sig.parameters.keys())
            if 'N_ensemble_configurations' in params:
                self.model = TabPFNClassifier(device=self.device, N_ensemble_configurations=16)
            else:
                self.model = TabPFNClassifier(device=self.device)
            self.model.fit(X, y)

        elif self.model_type == 'tabicl':
            try:
                from tabicl import TabICLClassifier
                self.model = TabICLClassifier(device=self.device)
            except ImportError:
                from pytabkit.models.sklearn.sklearn_interfaces import TabICL_TD_Classifier
                self.model = TabICL_TD_Classifier(device=self.device)
            self.model.fit(X, y)

        elif self.model_type == 'realmlp':
            from pytabkit import RealMLP_TD_Classifier
            self.model = RealMLP_TD_Classifier(device=self.device)
            self.model.fit(X, y)

        elif self.model_type == 'lr':
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(max_iter=1000, random_state=42)
            self.model.fit(X, y)

        elif self.model_type == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y)

        elif self.model_type == 'lgbm':
            from lightgbm import LGBMClassifier
            self.model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
            self.model.fit(X, y)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def preprocess(X_train: np.ndarray, X_test: np.ndarray):
    """Impute then standardize; fit on train only."""
    imp = SimpleImputer(strategy='median')
    X_train = imp.fit_transform(X_train)
    X_test = imp.transform(X_test)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test


def safe_proba(proba: np.ndarray) -> float:
    """Extract P(class=1) safely."""
    if proba.ndim == 2 and proba.shape[1] >= 2:
        return float(proba[0, 1])
    return 0.5


# ---------------------------------------------------------------------------
# LOO evaluation — single representation
# ---------------------------------------------------------------------------

def run_loo_evaluation(X: np.ndarray, y: np.ndarray, model_type: str,
                       device: str = 'cuda') -> dict:
    """Leave-One-Out evaluation on a single feature matrix."""
    loo = LeaveOneOut()
    y_true_all, y_pred_all, y_proba_all = [], [], []

    for train_idx, test_idx in tqdm(loo.split(X), total=len(y),
                                    desc=f"{model_type} LOO"):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train, X_test = preprocess(X_train, X_test)

        try:
            clf = FoundationalClassifier(model_type, device)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            pred = pred[0] if hasattr(pred, '__len__') and len(pred) == 1 else pred
            proba = clf.predict_proba(X_test)

            y_true_all.append(y_test[0])
            y_pred_all.append(int(pred))
            y_proba_all.append(safe_proba(proba))

        except Exception as e:
            print(f"  Error in LOO iteration: {e}")
            y_true_all.append(y_test[0])
            y_pred_all.append(0)
            y_proba_all.append(0.5)

    return compute_metrics(np.array(y_true_all), np.array(y_pred_all), np.array(y_proba_all))


# ---------------------------------------------------------------------------
# LOO early fusion
# ---------------------------------------------------------------------------

def run_early_fusion(X_emb: np.ndarray, X_feat: np.ndarray, y: np.ndarray,
                     model_type: str, device: str = 'cuda') -> dict:
    """
    Early fusion: normalize modalities separately, reweight features to compensate
    for dimensionality imbalance (d >> 11), concatenate, train one model.
    """
    loo = LeaveOneOut()
    y_true_all, y_pred_all, y_proba_all = [], [], []

    for train_idx, test_idx in tqdm(loo.split(X_emb), total=len(y),
                                    desc=f"{model_type} Early Fusion"):
        X_emb_tr, X_emb_te = X_emb[train_idx], X_emb[test_idx]
        X_feat_tr, X_feat_te = X_feat[train_idx], X_feat[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_emb_tr, X_emb_te = preprocess(X_emb_tr, X_emb_te)
        X_feat_tr, X_feat_te = preprocess(X_feat_tr, X_feat_te)

        weight = np.sqrt(X_emb_tr.shape[1] / X_feat_tr.shape[1])
        X_train = np.hstack([X_emb_tr, X_feat_tr * weight])
        X_test = np.hstack([X_emb_te, X_feat_te * weight])

        try:
            clf = FoundationalClassifier(model_type, device)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            pred = pred[0] if hasattr(pred, '__len__') and len(pred) == 1 else pred
            proba = clf.predict_proba(X_test)

            y_true_all.append(y_test[0])
            y_pred_all.append(int(pred))
            y_proba_all.append(safe_proba(proba))

        except Exception as e:
            print(f"  Error in early fusion: {e}")
            y_true_all.append(y_test[0])
            y_pred_all.append(0)
            y_proba_all.append(0.5)

    return compute_metrics(np.array(y_true_all), np.array(y_pred_all), np.array(y_proba_all))


# ---------------------------------------------------------------------------
# LOO late fusion — two separate models, averaged probabilities
# ---------------------------------------------------------------------------

def run_late_fusion(X_emb: np.ndarray, X_feat: np.ndarray, y: np.ndarray,
                    model_type: str, device: str = 'cuda') -> dict:
    """
    Late fusion: train one model on embeddings and a separate model of the same
    family on symbolic features. Average their predicted P(class=1) to obtain the
    fused probability; threshold at 0.5 for the final label.
    """
    loo = LeaveOneOut()
    y_true_all, y_pred_all, y_proba_all = [], [], []

    for train_idx, test_idx in tqdm(loo.split(X_emb), total=len(y),
                                    desc=f"{model_type} Late Fusion"):
        X_emb_tr, X_emb_te = X_emb[train_idx], X_emb[test_idx]
        X_feat_tr, X_feat_te = X_feat[train_idx], X_feat[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_emb_tr, X_emb_te = preprocess(X_emb_tr, X_emb_te)
        X_feat_tr, X_feat_te = preprocess(X_feat_tr, X_feat_te)

        try:
            # Model 1 — embeddings
            clf_emb = FoundationalClassifier(model_type, device)
            clf_emb.fit(X_emb_tr, y_train)
            proba_emb = safe_proba(clf_emb.predict_proba(X_emb_te))

            # Model 2 — symbolic features
            clf_feat = FoundationalClassifier(model_type, device)
            clf_feat.fit(X_feat_tr, y_train)
            proba_feat = safe_proba(clf_feat.predict_proba(X_feat_te))

            # Average logits (probability space average; equivalent to geometric
            # mean of odds when both are well-calibrated)
            fused_proba = (proba_emb + proba_feat) / 2.0
            fused_pred = int(fused_proba >= 0.5)

            y_true_all.append(y_test[0])
            y_pred_all.append(fused_pred)
            y_proba_all.append(fused_proba)

        except Exception as e:
            print(f"  Error in late fusion: {e}")
            y_true_all.append(y_test[0])
            y_pred_all.append(0)
            y_proba_all.append(0.5)

    return compute_metrics(np.array(y_true_all), np.array(y_pred_all), np.array(y_proba_all))


# ---------------------------------------------------------------------------
# Few-shot — single representation
# ---------------------------------------------------------------------------

def run_fewshot_evaluation(X: np.ndarray, y: np.ndarray, model_type: str,
                           k: int, n_episodes: int = 10,
                           device: str = 'cuda', seed: int = 42) -> dict:
    """
    Few-shot with LOO outer loop.

    For each held-out test sample:
      - Sample k examples per class from the remaining n-1 samples.
      - Train on the 2k support samples (preprocessing fit on support only).
      - Repeat for n_episodes with different random draws.
      - Aggregate: majority vote for labels, mean for probabilities.
    """
    rng = np.random.RandomState(seed)
    loo = LeaveOneOut()
    y_true_all, y_pred_all, y_proba_all = [], [], []

    for train_idx, test_idx in tqdm(loo.split(X), total=len(y),
                                    desc=f"{model_type} Few-shot k={k}"):
        X_pool, X_test = X[train_idx], X[test_idx]
        y_pool, y_test = y[train_idx], y[test_idx]

        episode_preds, episode_probas = [], []

        for _ in range(n_episodes):
            class_0_idx = np.where(y_pool == 0)[0]
            class_1_idx = np.where(y_pool == 1)[0]
            if len(class_0_idx) < k or len(class_1_idx) < k:
                continue

            sel = np.concatenate([
                rng.choice(class_0_idx, size=k, replace=False),
                rng.choice(class_1_idx, size=k, replace=False),
            ])
            X_train, y_train = X_pool[sel], y_pool[sel]
            X_train_p, X_test_p = preprocess(X_train.copy(), X_test.copy())

            try:
                clf = FoundationalClassifier(model_type, device)
                clf.fit(X_train_p, y_train)
                pred = clf.predict(X_test_p)
                pred = pred[0] if hasattr(pred, '__len__') and len(pred) == 1 else pred
                proba = clf.predict_proba(X_test_p)
                episode_preds.append(int(pred))
                episode_probas.append(safe_proba(proba))
            except Exception:
                continue

        if episode_preds:
            y_true_all.append(y_test[0])
            y_pred_all.append(int(np.round(np.mean(episode_preds))))
            y_proba_all.append(float(np.mean(episode_probas)))

    if not y_true_all:
        return {m: np.nan for m in ['Accuracy', 'Balanced_Accuracy', 'Macro_F1',
                                    'F1_Target', 'Precision', 'Recall', 'AUROC', 'AUPRC']}
    return compute_metrics(np.array(y_true_all), np.array(y_pred_all), np.array(y_proba_all))


# ---------------------------------------------------------------------------
# Few-shot — early fusion
# ---------------------------------------------------------------------------

def run_fewshot_early_fusion(X_emb: np.ndarray, X_feat: np.ndarray, y: np.ndarray,
                             model_type: str, k: int, n_episodes: int = 10,
                             device: str = 'cuda', seed: int = 42) -> dict:
    """
    Few-shot early fusion with LOO outer loop.
    Both modalities are preprocessed separately within each episode, then concatenated
    with feature reweighting before training a single model.
    """
    rng = np.random.RandomState(seed)
    loo = LeaveOneOut()
    y_true_all, y_pred_all, y_proba_all = [], [], []

    for train_idx, test_idx in tqdm(loo.split(X_emb), total=len(y),
                                    desc=f"{model_type} Few-shot Early Fusion k={k}"):
        X_emb_pool, X_emb_test = X_emb[train_idx], X_emb[test_idx]
        X_feat_pool, X_feat_test = X_feat[train_idx], X_feat[test_idx]
        y_pool, y_test = y[train_idx], y[test_idx]

        episode_preds, episode_probas = [], []

        for _ in range(n_episodes):
            class_0_idx = np.where(y_pool == 0)[0]
            class_1_idx = np.where(y_pool == 1)[0]
            if len(class_0_idx) < k or len(class_1_idx) < k:
                continue

            sel = np.concatenate([
                rng.choice(class_0_idx, size=k, replace=False),
                rng.choice(class_1_idx, size=k, replace=False),
            ])
            X_emb_tr, X_feat_tr, y_tr = X_emb_pool[sel], X_feat_pool[sel], y_pool[sel]

            X_emb_tr, X_emb_te = preprocess(X_emb_tr.copy(), X_emb_test.copy())
            X_feat_tr, X_feat_te = preprocess(X_feat_tr.copy(), X_feat_test.copy())

            weight = np.sqrt(X_emb_tr.shape[1] / X_feat_tr.shape[1])
            X_train = np.hstack([X_emb_tr, X_feat_tr * weight])
            X_test_ep = np.hstack([X_emb_te, X_feat_te * weight])

            try:
                clf = FoundationalClassifier(model_type, device)
                clf.fit(X_train, y_tr)
                pred = clf.predict(X_test_ep)
                pred = pred[0] if hasattr(pred, '__len__') and len(pred) == 1 else pred
                proba = clf.predict_proba(X_test_ep)
                episode_preds.append(int(pred))
                episode_probas.append(safe_proba(proba))
            except Exception:
                continue

        if episode_preds:
            y_true_all.append(y_test[0])
            y_pred_all.append(int(np.round(np.mean(episode_preds))))
            y_proba_all.append(float(np.mean(episode_probas)))

    if not y_true_all:
        return {m: np.nan for m in ['Accuracy', 'Balanced_Accuracy', 'Macro_F1',
                                    'F1_Target', 'Precision', 'Recall', 'AUROC', 'AUPRC']}
    return compute_metrics(np.array(y_true_all), np.array(y_pred_all), np.array(y_proba_all))


# ---------------------------------------------------------------------------
# Few-shot — late fusion (two separate models, averaged probabilities)
# ---------------------------------------------------------------------------

def run_fewshot_late_fusion(X_emb: np.ndarray, X_feat: np.ndarray, y: np.ndarray,
                            model_type: str, k: int, n_episodes: int = 10,
                            device: str = 'cuda', seed: int = 42) -> dict:
    """
    Few-shot late fusion with LOO outer loop.

    For each episode, two models of the same family are trained on the same 2k
    support samples — one on embeddings, one on symbolic features.  Their
    P(class=1) predictions are averaged to form the fused probability.
    """
    rng = np.random.RandomState(seed)
    loo = LeaveOneOut()
    y_true_all, y_pred_all, y_proba_all = [], [], []

    for train_idx, test_idx in tqdm(loo.split(X_emb), total=len(y),
                                    desc=f"{model_type} Few-shot Late Fusion k={k}"):
        X_emb_pool, X_emb_test = X_emb[train_idx], X_emb[test_idx]
        X_feat_pool, X_feat_test = X_feat[train_idx], X_feat[test_idx]
        y_pool, y_test = y[train_idx], y[test_idx]

        episode_preds, episode_probas = [], []

        for _ in range(n_episodes):
            class_0_idx = np.where(y_pool == 0)[0]
            class_1_idx = np.where(y_pool == 1)[0]
            if len(class_0_idx) < k or len(class_1_idx) < k:
                continue

            sel = np.concatenate([
                rng.choice(class_0_idx, size=k, replace=False),
                rng.choice(class_1_idx, size=k, replace=False),
            ])
            X_emb_tr, X_feat_tr, y_tr = X_emb_pool[sel], X_feat_pool[sel], y_pool[sel]

            X_emb_tr_p, X_emb_te_p = preprocess(X_emb_tr.copy(), X_emb_test.copy())
            X_feat_tr_p, X_feat_te_p = preprocess(X_feat_tr.copy(), X_feat_test.copy())

            try:
                clf_emb = FoundationalClassifier(model_type, device)
                clf_emb.fit(X_emb_tr_p, y_tr)
                p_emb = safe_proba(clf_emb.predict_proba(X_emb_te_p))

                clf_feat = FoundationalClassifier(model_type, device)
                clf_feat.fit(X_feat_tr_p, y_tr)
                p_feat = safe_proba(clf_feat.predict_proba(X_feat_te_p))

                fused = (p_emb + p_feat) / 2.0
                episode_preds.append(int(fused >= 0.5))
                episode_probas.append(fused)
            except Exception:
                continue

        if episode_preds:
            y_true_all.append(y_test[0])
            y_pred_all.append(int(np.round(np.mean(episode_preds))))
            y_proba_all.append(float(np.mean(episode_probas)))

    if not y_true_all:
        return {m: np.nan for m in ['Accuracy', 'Balanced_Accuracy', 'Macro_F1',
                                    'F1_Target', 'Precision', 'Recall', 'AUROC', 'AUPRC']}
    return compute_metrics(np.array(y_true_all), np.array(y_pred_all), np.array(y_proba_all))


# ---------------------------------------------------------------------------
# Per-language evaluation
# ---------------------------------------------------------------------------

def evaluate_language(language: str, df: pd.DataFrame, embeddings: np.ndarray,
                      models: list, device: str, run_id: int) -> list:
    """Evaluate all configurations for a single language."""
    results = []

    lang_mask = df['language'].str.lower() == language.lower()
    lang_df = df[lang_mask].reset_index(drop=True)
    lang_emb = embeddings[lang_mask]

    if len(lang_df) == 0:
        print(f"  No data for {language}")
        return results

    print(f"\n{'='*60}")
    print(f"LANGUAGE: {language.upper()} (Run {run_id})")
    print(f"{'='*60}")
    print(f"Samples: {len(lang_df)}")
    print(f"Class distribution: {dict(lang_df['binary_label'].value_counts())}")

    X_feat = lang_df[TEXT_FEATURES].values
    X_emb = lang_emb
    y = lang_df['binary_label'].values

    # Majority class baseline (only once, on run 1)
    if run_id == 1:
        majority_class = int(np.bincount(y).argmax())
        y_pred_majority = np.full_like(y, majority_class)
        baseline_metrics = compute_metrics(y, y_pred_majority, y_pred_majority.astype(float))
        results.append({
            'Run': run_id, 'Language': language, 'Model': 'majority_baseline',
            'Mode': 'baseline', **baseline_metrics
        })
        print(f"\n--- Majority Baseline (class={majority_class}) ---")
        print(f"  F1={baseline_metrics['Macro_F1']:.4f}, BalAcc={baseline_metrics['Balanced_Accuracy']:.4f}")

    for model_type in models:
        print(f"\n{'─'*50}")
        print(f"Model: {model_type.upper()}")
        print(f"{'─'*50}")

        seed = 42 + run_id * 100

        # ------------------------------------------------------------------
        # Full-data LOO
        # ------------------------------------------------------------------
        for tag, X in [('loo_embeddings', X_emb), ('loo_features', X_feat)]:
            label = tag.replace('_', ' ').title()
            print(f"\n[LOO] {label}")
            try:
                metrics = run_loo_evaluation(X, y, model_type, device)
                results.append({'Run': run_id, 'Language': language, 'Model': model_type,
                                'Mode': tag, **metrics})
                print(f"  F1={metrics['Macro_F1']:.4f}, BalAcc={metrics['Balanced_Accuracy']:.4f}")
            except Exception as e:
                print(f"  Error: {e}")

        print(f"\n[LOO] Early Fusion")
        try:
            metrics = run_early_fusion(X_emb, X_feat, y, model_type, device)
            results.append({'Run': run_id, 'Language': language, 'Model': model_type,
                            'Mode': 'loo_early_fusion', **metrics})
            print(f"  F1={metrics['Macro_F1']:.4f}, BalAcc={metrics['Balanced_Accuracy']:.4f}")
        except Exception as e:
            print(f"  Error: {e}")

        print(f"\n[LOO] Late Fusion (separate models, averaged P)")
        try:
            metrics = run_late_fusion(X_emb, X_feat, y, model_type, device)
            results.append({'Run': run_id, 'Language': language, 'Model': model_type,
                            'Mode': 'loo_late_fusion', **metrics})
            print(f"  F1={metrics['Macro_F1']:.4f}, BalAcc={metrics['Balanced_Accuracy']:.4f}")
        except Exception as e:
            print(f"  Error: {e}")

        # ------------------------------------------------------------------
        # Few-shot
        # ------------------------------------------------------------------
        for k in [1, 2, 3, 5]:
            print(f"\n[Few-shot k={k}] Embeddings")
            try:
                metrics = run_fewshot_evaluation(X_emb, y, model_type, k=k,
                                                 n_episodes=3, device=device, seed=seed)
                results.append({'Run': run_id, 'Language': language, 'Model': model_type,
                                'Mode': f'fewshot_k{k}', **metrics})
                print(f"  F1={metrics['Macro_F1']:.4f}, BalAcc={metrics['Balanced_Accuracy']:.4f}")
            except Exception as e:
                print(f"  Error: {e}")

            print(f"\n[Few-shot k={k}] Features")
            try:
                metrics = run_fewshot_evaluation(X_feat, y, model_type, k=k,
                                                 n_episodes=3, device=device, seed=seed)
                results.append({'Run': run_id, 'Language': language, 'Model': model_type,
                                'Mode': f'fewshot_features_k{k}', **metrics})
                print(f"  F1={metrics['Macro_F1']:.4f}, BalAcc={metrics['Balanced_Accuracy']:.4f}")
            except Exception as e:
                print(f"  Error: {e}")

            print(f"\n[Few-shot k={k}] Early Fusion")
            try:
                metrics = run_fewshot_early_fusion(X_emb, X_feat, y, model_type, k=k,
                                                   n_episodes=3, device=device, seed=seed)
                results.append({'Run': run_id, 'Language': language, 'Model': model_type,
                                'Mode': f'fewshot_early_fusion_k{k}', **metrics})
                print(f"  F1={metrics['Macro_F1']:.4f}, BalAcc={metrics['Balanced_Accuracy']:.4f}")
            except Exception as e:
                print(f"  Error: {e}")

            print(f"\n[Few-shot k={k}] Late Fusion (separate models, averaged P)")
            try:
                metrics = run_fewshot_late_fusion(X_emb, X_feat, y, model_type, k=k,
                                                  n_episodes=3, device=device, seed=seed)
                results.append({'Run': run_id, 'Language': language, 'Model': model_type,
                                'Mode': f'fewshot_late_fusion_k{k}', **metrics})
                print(f"  F1={metrics['Macro_F1']:.4f}, BalAcc={metrics['Balanced_Accuracy']:.4f}")
            except Exception as e:
                print(f"  Error: {e}")

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(results_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Aggregate results across runs and compute mean ± std."""
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    metrics_cols = ['Accuracy', 'Balanced_Accuracy', 'Macro_F1', 'F1_Target',
                    'Precision', 'Recall', 'AUROC', 'AUPRC']

    summary_rows = []
    for (lang, model, mode), group in results_df.groupby(['Language', 'Model', 'Mode']):
        row = {'Language': lang, 'Model': model, 'Mode': mode}
        for m in metrics_cols:
            vals = group[m].dropna()
            row[f'{m}_mean'] = vals.mean() if len(vals) else np.nan
            row[f'{m}_std'] = vals.std() if len(vals) else np.nan
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_dir / "aggregated_results.csv", index=False)

    for metric in ['Macro_F1', 'Balanced_Accuracy']:
        pivot = summary_df.pivot_table(
            index=['Model', 'Mode'], columns='Language', values=f'{metric}_mean'
        )
        pivot['Mean'] = pivot.mean(axis=1)
        pivot.sort_values('Mean', ascending=False, inplace=True)
        pivot.to_csv(summary_dir / f"pivot_{metric}.csv")

    print(f"\nAggregated results saved to: {summary_dir}")
    return summary_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Comprehensive MCI Classification Evaluation")
    parser.add_argument('--embedding', type=str,
                        default='paraphrase-multilingual-MiniLM-L12-v2',
                        help='Embedding model name or HuggingFace path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--n_runs', type=int, default=3,
                        help='Number of repetitions for statistical analysis')
    parser.add_argument('--output_dir', type=str, default='comprehensive_results',
                        help='Root output directory')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['tabpfn', 'realmlp'],
                        help='Models to evaluate: tabpfn, tabicl, realmlp, lr, rf, lgbm')
    parser.add_argument('--languages', type=str, nargs='+', default=None,
                        help='Filter to specific languages (e.g. --languages slovene english)')
    parser.add_argument('--precomputed_embeddings', type=str, default=None,
                        help=(
                            'Explicit path to a precomputed embeddings .npy file. '
                            'If omitted, the script looks for '
                            '<sanitized_model>_<dataset_stem>.npy next to the data '
                            'file and generates/saves it if missing.'
                        ))
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("COMPREHENSIVE MCI CLASSIFICATION EVALUATION")
    print("=" * 60)
    print(f"Embedding model : {args.embedding}")
    print(f"Models          : {args.models}")
    print(f"Device          : {args.device}")
    print(f"Runs            : {args.n_runs}")
    print(f"Output          : {output_dir}")

    # Load data
    print("\nLoading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Total samples: {len(df)}")
    df['binary_label'] = df['class'].apply(
        lambda x: 1 if str(x).lower() in ['mci', 'patient'] else 0
    )

    # Embeddings — explicit path overrides auto-detection
    if args.precomputed_embeddings:
        print(f"\nLoading embeddings from explicit path: {args.precomputed_embeddings}")
        embeddings = np.load(args.precomputed_embeddings)
        print(f"  Embedding shape: {embeddings.shape}")
    else:
        embeddings = load_or_compute_embeddings(args.embedding, df, DATA_PATH, args.device)

    # Save run config
    config = {
        'embedding_model': args.embedding,
        'models': args.models,
        'n_runs': args.n_runs,
        'device': args.device,
        'timestamp': timestamp,
        'embedding_dim': int(embeddings.shape[1]),
        'late_fusion_strategy': 'separate_models_averaged_probability',
    }
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Run evaluation
    all_results = []
    languages = df['language'].str.lower().unique()
    if args.languages:
        languages = [l for l in languages if l in [x.lower() for x in args.languages]]
        print(f"Filtering to languages: {languages}")

    for run_id in range(1, args.n_runs + 1):
        print(f"\n{'#' * 60}")
        print(f"RUN {run_id}/{args.n_runs}")
        print(f"{'#' * 60}")

        for language in languages:
            results = evaluate_language(
                language, df, embeddings, args.models, args.device, run_id
            )
            all_results.extend(results)

        # Intermediate save
        pd.DataFrame(all_results).to_csv(
            output_dir / f"results_run_{run_id}.csv", index=False
        )

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "all_results.csv", index=False)

    print("\n" + "=" * 60)
    print("AGGREGATING RESULTS")
    print("=" * 60)
    summary_df = aggregate_results(results_df, output_dir)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY (Mean ± Std across runs)")
    print("=" * 60)
    for mode in sorted(results_df['Mode'].unique()):
        print(f"\n--- {mode} ---")
        for _, row in summary_df[summary_df['Mode'] == mode].iterrows():
            print(f"  {row['Language']:10s} {row['Model']:10s}: "
                  f"F1={row['Macro_F1_mean']:.4f}±{row['Macro_F1_std']:.4f}  "
                  f"BalAcc={row['Balanced_Accuracy_mean']:.4f}±{row['Balanced_Accuracy_std']:.4f}")

    print(f"\n{'=' * 60}")
    print(f"All results saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()