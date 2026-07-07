#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
import whisper

from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

# -----------------------
# Config (edit if needed)
# -----------------------
INPUT_DIRS = {
    "controls": "recordings/Controls",
    "patients": "recordings/patients",
}

OUT_ROOT = Path("asr_outputs")  # outputs/<group>/<stem>.<ext>

# Multilingual Whisper: "tiny" | "base" | "small" | "medium" | "large"
WHISPER_SIZE = "small"

# Multilingual wav2vec2 (no tokenizer needed)
W2V2_NAME = "facebook/wav2vec2-large-xlsr-53"

TARGET_SR = 16000
AUDIO_EXTS = (".mp3", ".m4a", ".wav", ".flac", ".mp4", ".aac", ".ogg")

# -----------------------
# Helpers
# -----------------------
def list_audio_files(root: str) -> List[Path]:
    p = Path(root)
    if not p.exists():
        return []
    return sorted([x for x in p.iterdir() if x.is_file() and x.suffix.lower() in AUDIO_EXTS])

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_audio_16k_mono(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.size(0) > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)  # mono
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.to(torch.float32)  # [1, T]

def chunk_indices(total_len: int, chunk_len: int) -> List[Tuple[int, int]]:
    if total_len <= 0:  
        return [(0, 0)]
    n = math.ceil(total_len / chunk_len)
    return [(i * chunk_len, min((i + 1) * chunk_len, total_len)) for i in range(n)]

def whisper_file_embedding(model: whisper.Whisper, wav_16k: torch.Tensor, device: str) -> np.ndarray:
    """Mean over encoder outputs of 30s-chunks (pad_or_trim) -> (D,)"""
    CHUNK_SAMPLES = 30 * TARGET_SR
    audio_np = wav_16k.squeeze(0).cpu().numpy()
    L = audio_np.shape[0]
    if L == 0:
        return np.zeros((model.dims.n_audio_state,), dtype=np.float32)

    vecs = []
    for s, e in chunk_indices(L, CHUNK_SAMPLES):
        chunk = whisper.pad_or_trim(audio_np[s:e])  # exactly 30s
        mel = whisper.log_mel_spectrogram(chunk).to(device)
        with torch.no_grad():
            enc = model.encoder(mel.unsqueeze(0))  # [1, T', D]
            vec = enc.mean(dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)
            vecs.append(vec)
    return np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)

def wav2vec2_file_embedding(fe: Wav2Vec2FeatureExtractor, model: Wav2Vec2Model, wav_16k: torch.Tensor, device: str) -> np.ndarray:
    """Mean over last_hidden_state frames -> (D,)"""
    audio_np = wav_16k.squeeze(0).cpu().numpy()
    inputs = fe(audio_np, sampling_rate=TARGET_SR, return_tensors="pt")
    with torch.no_grad():
        out = model(inputs.input_values.to(device))
        last = out.last_hidden_state  # [1, T, D]
        vec = last.mean(dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return vec

# -----------------------
# Main
# -----------------------
def main():
    # Optional CLI overrides:
    #   python script.py <controls_dir> <patients_dir>
    if len(sys.argv) == 3:
        INPUT_DIRS["controls"] = sys.argv[1]
        INPUT_DIRS["patients"] = sys.argv[2]

    ensure_dir(OUT_ROOT)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load multilingual models
    print("Loading Whisper (multilingual)...")
    whisper_model = whisper.load_model(WHISPER_SIZE).to(device)

    print("Loading wav2vec2 XLSR-53 (multilingual)...")
    w2v2_fe = Wav2Vec2FeatureExtractor.from_pretrained(W2V2_NAME)
    w2v2_model = Wav2Vec2Model.from_pretrained(W2V2_NAME).to(device)

    for group, in_dir in INPUT_DIRS.items():
        files = list_audio_files(in_dir)
        if not files:
            print(f"⚠️  No audio in: {in_dir}")
            continue

        out_group_dir = OUT_ROOT / group
        ensure_dir(out_group_dir)

        for fpath in files:
            print(f"\n🔊 Processing [{group}] {fpath.name}")
            stem = fpath.stem

            # Load
            wav_16k = load_audio_16k_mono(fpath)

            # ---- Whisper transcript ----
            tr = whisper_model.transcribe(str(fpath))
            transcript = (tr.get("text") or "").strip()
            (out_group_dir / f"{stem}.whisper.txt").write_text(transcript + "\n", encoding="utf-8")

            # ---- Whisper embedding ----
            emb_whisper = whisper_file_embedding(whisper_model, wav_16k, device)
            np.save(out_group_dir / f"{stem}.whisper.npy", emb_whisper)

            # ---- Wav2Vec2 embedding ----
            emb_w2v2 = wav2vec2_file_embedding(w2v2_fe, w2v2_model, wav_16k, device)
            np.save(out_group_dir / f"{stem}.wav2vec2.npy", emb_w2v2)

    print("\n✅ Finished. Files saved under:")
    print("   asr_outputs/<controls|patients>/<stem>.whisper.txt | .whisper.npy | .wav2vec2.npy")

if __name__ == "__main__":
    main()
