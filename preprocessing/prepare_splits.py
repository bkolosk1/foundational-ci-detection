#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import subprocess
import numpy as np
import torch
import torchaudio
from typing import List, Iterable, Dict, Tuple
import re
import json
import yaml
from collections import defaultdict, Counter
from sklearn.model_selection import RepeatedStratifiedKFold

# =======================
# YOUR READER (UNCHANGED)
# =======================
TARGET_SR = 16000

# 1) File discovery: pass multiple extensions (case-insensitive)
def list_audio_files(folder: str | Path, exts: Iterable[str] = (".m4a", ".wav")) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    exts = {e.lower() for e in exts}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)

# 2) Robust loader: try torchaudio -> fallback to ffmpeg pipe
def load_audio_16k_mono(path: str | Path) -> torch.Tensor:
    """
    Returns mono 16kHz float32 tensor of shape [1, T].
    """
    path = str(path)

    # First try torchaudio.load (fast path)
    try:
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)  # mono
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        return wav.to(torch.float32)
    except Exception:
        # Fallback: use ffmpeg to decode & resample to mono 16k PCM, then wrap as tensor
        # ffmpeg -i input -ac 1 -ar 16000 -f s16le pipe:1
        cmd = [
            "ffmpeg", "-v", "error", "-i", path,
            "-ac", "1", "-ar", str(TARGET_SR),
            "-f", "s16le", "pipe:1"
        ]
        proc = subprocess.run(cmd, capture_output=True, check=True)
        # Convert raw int16 PCM to float32 [-1, 1]
        audio_i16 = np.frombuffer(proc.stdout, dtype=np.int16)
        if audio_i16.size == 0:
            # empty / unreadable
            return torch.zeros(1, 0, dtype=torch.float32)
        audio_f32 = (audio_i16.astype(np.float32) / 32768.0).reshape(1, -1)
        return torch.from_numpy(audio_f32)

# =======================
# CV UTILITIES
# =======================

# Choose how to turn a filename stem into a patient ID:
#   'stem'   -> use the whole stem (e.g., "HP_01_cookie_theft" stays that full stem)
#   'prefix' -> part before '_' or '-' (e.g., "HP_01_cookie_theft" -> "HP")
ID_MODE = "stem"  # change to "prefix" if you want to collapse multiple sessions per patient

def infer_patient_id(stem: str, mode: str = ID_MODE) -> str:
    if mode == "stem":
        return stem
    tok = re.split(r"[_\-]", stem)[0]
    return tok if tok else stem

def collect_patients_with_your_reader(
    controls_dir: str | Path,
    patients_dir: str | Path,
    ctrl_exts: Iterable[str],
    pat_exts: Iterable[str],
) -> Tuple[List[str], List[str], Dict[str, List[str]], Dict[str, List[str]], List[Path], List[Path]]:
    """
    Uses *your* list_audio_files exactly, with per-group extensions.
    Returns:
      control_ids, patient_ids, control_files_map, patient_files_map, ctrl_files, pat_files
    """
    # EXACT usage per your snippet:
    ctrl_files = list_audio_files(controls_dir, exts=ctrl_exts)
    pat_files  = list_audio_files(patients_dir, exts=pat_exts)

    control_files_map: Dict[str, List[str]] = defaultdict(list)
    patient_files_map: Dict[str, List[str]] = defaultdict(list)

    for p in ctrl_files:
        pid = infer_patient_id(p.stem)
        control_files_map[pid].append(str(p))

    for p in pat_files:
        pid = infer_patient_id(p.stem)
        patient_files_map[pid].append(str(p))

    control_ids = sorted(control_files_map.keys())
    patient_ids = sorted(patient_files_map.keys())
    return control_ids, patient_ids, control_files_map, patient_files_map, ctrl_files, pat_files

def check_feasible(n_splits: int, y: List[int]) -> None:
    counts = Counter(y)
    bad = [cls for cls, cnt in counts.items() if cnt < n_splits]
    if bad:
        raise ValueError(
            f"Each class must have at least {n_splits} patients for Stratified {n_splits}-fold CV. "
            f"Counts: {dict(counts)}"
        )

def build_repeated_stratified(
    control_ids: List[str], patient_ids: List[str], n_splits: int = 5, n_repeats: int = 5, seed: int = 42
) -> Dict:
    all_ids = control_ids + patient_ids
    y = [0] * len(control_ids) + [1] * len(patient_ids)
    check_feasible(n_splits, y)

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    idx2id = {i: pid for i, pid in enumerate(all_ids)}
    splits = []
    repeat_no = 0
    fold_in_repeat = 0

    for _, (train_idx, test_idx) in enumerate(rskf.split(all_ids, y), start=1):
        if fold_in_repeat == 0:
            repeat_no += 1
        fold_no = (fold_in_repeat % n_splits) + 1

        train_ids = [idx2id[i] for i in train_idx]
        test_ids  = [idx2id[i] for i in test_idx]

        splits.append({
            "repeat": repeat_no,
            "fold": fold_no,
            "train_ids": train_ids,
            "test_ids": test_ids,
        })

        fold_in_repeat += 1
        if fold_in_repeat == n_splits:
            fold_in_repeat = 0

    return {"all_ids": all_ids, "labels": y, "splits": splits}

def to_basename_map(d: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Convert {pid: [full/paths]} -> {pid: [basename.ext]} and force plain dict (no defaultdict)."""
    return {pid: [Path(p).name for p in paths] for pid, paths in dict(d).items()}

def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def save_yaml(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

# =======================
# MAIN
# =======================
if __name__ == "__main__":
    # EXACT paths & extension sets like your snippet:
    CONTROLS_DIR = "recordings/Controls"
    PATIENTS_DIR = "recordings/patients"

    CTRL_EXTS = (".mp3", ".wav") 
    PAT_EXTS  = (".m4a", ".wav")  

    # CV params
    N_SPLITS  = 5
    N_REPEATS = 5
    SEED      = 42

    OUT_DIR = Path("cv_splits")
    OUT_JSON = OUT_DIR / f"repeated_stratified_{N_SPLITS}x{N_REPEATS}_seed{SEED}.json"
    OUT_YAML = OUT_DIR / f"repeated_stratified_{N_SPLITS}x{N_REPEATS}_seed{SEED}.yaml"

    # 1) Use YOUR reader exactly to discover files (with per-group exts)
    ctrl_ids, pat_ids, ctrl_map, pat_map, ctrl_files, pat_files = collect_patients_with_your_reader(
        CONTROLS_DIR, PATIENTS_DIR, CTRL_EXTS, PAT_EXTS
    )

    # Debug: verify discovery respects extensions
    print(f"Discovered control files ({len(ctrl_files)}):")
    for p in ctrl_files:
        print("  -", p.name)
    print(f"Discovered patient files ({len(pat_files)}):")
    for p in pat_files:
        print("  -", p.name)

    print(f"\nPatient-ID mode: {ID_MODE}")
    print(f"→ Controls -> {len(ctrl_ids)} unique IDs")
    print(f"→ Patients -> {len(pat_ids)} unique IDs")

    if not ctrl_ids or not pat_ids:
        raise SystemExit(
            f"\nFound {len(ctrl_ids)} control IDs and {len(pat_ids)} patient IDs — need both > 0.\n"
            f"Check folders and extension filters.\n"
            f"Controls dir: {CONTROLS_DIR}  (exts={CTRL_EXTS})\n"
            f"Patients dir: {PATIENTS_DIR}  (exts={PAT_EXTS})"
        )

    # 2) Build 5x5 Repeated Stratified CV at the patient level
    cv = build_repeated_stratified(ctrl_ids, pat_ids, n_splits=N_SPLITS, n_repeats=N_REPEATS, seed=SEED)

    # 3) Pack metadata (convert defaultdict -> dict and paths -> basenames to avoid YAML errors)
    ctrl_map_base = to_basename_map(ctrl_map)
    pat_map_base  = to_basename_map(pat_map)

    meta = {
        "params": {
            "n_splits": N_SPLITS,
            "n_repeats": N_REPEATS,
            "random_state": SEED,
            "strategy": "RepeatedStratifiedKFold",
            "id_mode": ID_MODE,
            "controls_dir": str(Path(CONTROLS_DIR).resolve()),
            "patients_dir": str(Path(PATIENTS_DIR).resolve()),
            "ctrl_exts": list(CTRL_EXTS),
            "pat_exts": list(PAT_EXTS),
        },
        "label_map": {"control": 0, "patient": 1},
        "counts": {
            "controls": len(ctrl_ids),
            "patients": len(pat_ids),
            "total": len(ctrl_ids) + len(pat_ids),
        },
        # ✅ store only basenames (after last '/'), YAML-safe dicts
        "patient_files": {
            "controls": ctrl_map_base,
            "patients": pat_map_base,
        },
        "notes": (
            "IDs are derived from filename stems by default (ID_MODE='stem'). "
            "Set ID_MODE='prefix' to collapse multiple sessions per subject. "
            "Stored file names are basenames only (after last '/')."
        ),
    }

    to_save = {"meta": meta, "splits": cv["splits"], "all_ids": cv["all_ids"]}

    # 4) Save JSON + YAML
    save_json(to_save, OUT_JSON)
    save_yaml(to_save, OUT_YAML)

    print("\n✅ Saved CV splits:")
    print(f"• JSON: {OUT_JSON}")
    print(f"• YAML: {OUT_YAML}")
    print(f"• Total folds stored: {len(cv['splits'])}  (expected {N_SPLITS * N_REPEATS})")
