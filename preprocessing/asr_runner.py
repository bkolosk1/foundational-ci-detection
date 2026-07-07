#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, make_scorer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# -----------------------
# Config
# -----------------------
EMBEDDINGS_ROOT = Path("asr_outputs")
CONTROLS_DIR = EMBEDDINGS_ROOT / "controls"
PATIENTS_DIR = EMBEDDINGS_ROOT / "patients"

# Grid search parameters
PARAM_GRID = {
    'lr__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'lr__penalty': ['l2'],
    'lr__solver': ['lbfgs'],
    'lr__max_iter': [1000]
}

N_SPLITS = 5  # number of folds
N_REPEATS = 5  # number of repetitions
RANDOM_STATE = 42

# -----------------------
# Data Loading
# -----------------------
def load_embeddings(group_dir: Path, embedding_type: str):
    """Load all embeddings of given type from a directory.
    
    Args:
        group_dir: Path to controls or patients directory
        embedding_type: 'whisper' or 'wav2vec2'
    
    Returns:
        List of numpy arrays
    """
    pattern = f"*.{embedding_type}.npy"
    files = sorted(group_dir.glob(pattern))
    embeddings = []
    for f in files:
        emb = np.load(f)
        embeddings.append(emb)
    return embeddings

def prepare_dataset(embedding_type: str):
    """Prepare X, y for given embedding type.
    
    Returns:
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,) with 0=control, 1=patient
        names: list of sample names
    """
    # Load controls (label=0)
    controls_emb = load_embeddings(CONTROLS_DIR, embedding_type)
    controls_names = [f.stem.replace(f".{embedding_type}", "") 
                      for f in sorted(CONTROLS_DIR.glob(f"*.{embedding_type}.npy"))]
    
    # Load patients (label=1)
    patients_emb = load_embeddings(PATIENTS_DIR, embedding_type)
    patients_names = [f.stem.replace(f".{embedding_type}", "") 
                      for f in sorted(PATIENTS_DIR.glob(f"*.{embedding_type}.npy"))]
    
    # Combine
    X = np.vstack([np.array(controls_emb), np.array(patients_emb)])
    y = np.array([0] * len(controls_emb) + [1] * len(patients_emb))
    names = controls_names + patients_names
    
    print(f"  Loaded {len(controls_emb)} controls, {len(patients_emb)} patients")
    print(f"  Feature dim: {X.shape[1]}")
    
    return X, y, names

def prepare_combined_dataset():
    """Prepare dataset with concatenated whisper + wav2vec2 features."""
    # Load both types
    X_whisper, y, names = prepare_dataset('whisper')
    X_wav2vec2, _, _ = prepare_dataset('wav2vec2')
    
    # Concatenate features
    X_combined = np.hstack([X_whisper, X_wav2vec2])
    
    print(f"\nCombined dataset:")
    print(f"  Total samples: {X_combined.shape[0]}")
    print(f"  Combined feature dim: {X_combined.shape[1]} (whisper: {X_whisper.shape[1]}, wav2vec2: {X_wav2vec2.shape[1]})")
    
    return X_combined, y, names

# -----------------------
# Training & Evaluation
# -----------------------
def train_and_evaluate(X, y, model_name: str):
    """Train LR with grid search and evaluate using repeated CV."""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    
    # Create pipeline with scaler and LR (scaling happens inside each fold)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced'))
    ])
    
    # Setup repeated cross-validation
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, 
        n_repeats=N_REPEATS, 
        random_state=RANDOM_STATE
    )
    
    # Grid search
    print(f"\nRunning grid search with {N_REPEATS}x{N_SPLITS}-fold CV ({N_SPLITS * N_REPEATS} total fits per parameter)...")
    grid_search = GridSearchCV(
        pipeline, 
        PARAM_GRID, 
        cv=cv, 
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    grid_search.fit(X, y)
    
    # Best model results
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    # Get std of best model across all CV folds
    best_idx = grid_search.best_index_
    cv_results = grid_search.cv_results_
    test_scores = cv_results['split0_test_score']  # This will give us access to splits
    
    # Calculate mean and std across all repeated folds for best params
    best_test_scores = []
    for i in range(N_SPLITS * N_REPEATS):
        score_key = f'split{i}_test_score'
        best_test_scores.append(cv_results[score_key][best_idx])
    
    cv_mean = np.mean(best_test_scores)
    cv_std = np.std(best_test_scores)
    
    print(f"\n✓ Best parameters: {best_params}")
    print(f"✓ CV ROC-AUC: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"  (Mean ± Std over {N_SPLITS * N_REPEATS} folds)")
    
    # Train final model on full dataset for reporting
    best_model.fit(X, y)
    y_pred = best_model.predict(X)
    y_proba = best_model.predict_proba(X)[:, 1]
    
    print(f"\nClassification Report (Full Dataset):")
    print(classification_report(y, y_pred, target_names=['Control', 'Patient']))
    
    print(f"\nConfusion Matrix (Full Dataset):")
    cm = confusion_matrix(y, y_pred)
    print(f"              Predicted")
    print(f"              Control  Patient")
    print(f"Actual Control   {cm[0,0]:4d}     {cm[0,1]:4d}")
    print(f"       Patient   {cm[1,0]:4d}     {cm[1,1]:4d}")
    
    roc_auc = roc_auc_score(y, y_proba)
    print(f"\nROC-AUC (Full Dataset): {roc_auc:.4f}")
    
    return {
        'model': best_model,
        'best_params': best_params,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'roc_auc_full': roc_auc,
        'all_cv_scores': best_test_scores
    }

# -----------------------
# Main
# -----------------------
def main():
    print("🔬 Training Logistic Regression on ASR Embeddings")
    print(f"Using {N_REPEATS}x{N_SPLITS}-fold Repeated Stratified CV\n")
    
    results = {}
    
    # 1. Whisper only
    print("\n" + "="*60)
    print("1. WHISPER EMBEDDINGS ONLY")
    print("="*60)
    X_whisper, y_whisper, _ = prepare_dataset('whisper')
    results['whisper'] = train_and_evaluate(X_whisper, y_whisper, "Whisper-only")
    
    # 2. Wav2Vec2 only
    print("\n" + "="*60)
    print("2. WAV2VEC2 EMBEDDINGS ONLY")
    print("="*60)
    X_wav2vec2, y_wav2vec2, _ = prepare_dataset('wav2vec2')
    results['wav2vec2'] = train_and_evaluate(X_wav2vec2, y_wav2vec2, "Wav2Vec2-only")
    
    # 3. Combined
    print("\n" + "="*60)
    print("3. COMBINED EMBEDDINGS (WHISPER + WAV2VEC2)")
    print("="*60)
    X_combined, y_combined, _ = prepare_combined_dataset()
    results['combined'] = train_and_evaluate(X_combined, y_combined, "Combined")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n{'Model':<15} {'CV ROC-AUC (mean±std)':<30} {'Best C':<10}")
    print("-" * 70)
    for name, res in results.items():
        cv_str = f"{res['cv_mean']:.4f} ± {res['cv_std']:.4f}"
        best_c = res['best_params']['lr__C']
        print(f"{name:<15} {cv_str:<30} {best_c:<10}")
    

if __name__ == "__main__":
    main()