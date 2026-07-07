from collections import Counter
import numpy as np
import librosa
import parselmouth
from textgrid import TextGrid as TG

FILLER_WHITELIST = {"hmm", "uh", "um", "okay", "ok", "right", "alright", "wait", "wait a second"}

def read_textgrid_intervals(textgrid_path, tier_name):
    tg = TG.fromFile(textgrid_path)
    tier = next(t for t in tg.tiers if t.name == tier_name)
    # Normalize text once here
    def norm(s): 
        return (s or "").strip()
    return [{"xmin": itv.minTime, "xmax": itv.maxTime, "text": norm(itv.mark)} for itv in tier.intervals]

def pauses_and_tokens_from_intervals(intervals):
    """Pauses = intervals with empty text; tokens = non-empty texts."""
    pauses = []
    tokens = []
    token_durs = []
    for itv in intervals:
        d = itv["xmax"] - itv["xmin"]
        if itv["text"] == "":
            pauses.append(d)
        else:
            tokens.append(itv["text"])
            token_durs.append(d)
    return pauses, tokens, token_durs

def f0_features_from_y(y, sr):
    # Track F0 + voiced flag
    f0, vflag, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr, frame_length=2048, hop_length=256)
    if f0 is None or vflag is None:
        return np.nan, np.nan, np.nan  # no F0
    mask = (vflag.astype(bool)) & np.isfinite(f0)
    if not np.any(mask):
        return np.nan, np.nan, np.nan
    f0_valid = f0[mask]
    # Convert only valid F0 to semitones relative to 55 Hz
    f0_st = librosa.hz_to_midi(f0_valid) - librosa.hz_to_midi(55.0)
    return (np.median(f0_st),
            np.percentile(f0_st, 95) - np.percentile(f0_st, 5),
            np.subtract(*np.percentile(f0_st, [75, 25])))

def intensity_features_db(y, sr):
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=256).ravel()
    # Convert to dB relative
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    finite = np.isfinite(rms_db)
    if not np.any(finite):
        return np.nan, np.nan
    vals = rms_db[finite]
    med = np.median(vals)
    iqr = np.subtract(*np.percentile(vals, [75, 25]))
    return med, iqr

def praat_voice_measures(audio_path):
    snd = parselmouth.Sound(str(audio_path))
    pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
    jitter_local = parselmouth.praat.call(pp, "Get jitter (local)", 0, 0, 75, 500, 1.3)
    ppq5 = parselmouth.praat.call(pp, "Get jitter (ppq5)", 0, 0, 75, 500, 1.3)
    shimmer_local = parselmouth.praat.call([snd, pp], "Get shimmer (local)", 0, 0, 75, 500, 1.3, 1.6)
    apq5 = parselmouth.praat.call([snd, pp], "Get shimmer (apq5)", 0, 0, 75, 500, 1.3, 1.6)
    harm = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr_med = parselmouth.praat.call(harm, "Get quantile", 0, 0, 0.5)
    return dict(jitter_local=jitter_local, ppq5=ppq5,
                shimmer_local=shimmer_local, apq5=apq5, hnr_med=hnr_med)

def nPVI(durs):
    d = np.asarray(durs, float)
    if len(d) < 2 or np.any(d <= 0):
        return np.nan
    diffs = np.abs(np.diff(d))
    avgs = (d[:-1] + d[1:]) / 2
    return 100 * np.mean(diffs / avgs)

def syllable_nuclei_times(y, sr):
    # Onset proxy—language-free; tune delta/wait as needed
    env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(env, sr=sr, hop_length=512)
    peaks = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=0.1, wait=3)
    return times[peaks]

def summarize_pause_stats(pauses):
    if len(pauses) == 0:
        return dict(median=np.nan, p95=np.nan, pct_time=0.0, long_rate_per_min=0.0)
    arr = np.asarray(pauses, float)
    med = np.median(arr)
    p95 = np.percentile(arr, 95)
    return med, p95

def features_from(audio_path, textgrid_path, tier="silences"):
    # Load
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    dur = max(1e-9, len(y) / sr)

    # Intervals → pauses + tokens
    intervals = read_textgrid_intervals(textgrid_path, tier)
    pauses, tokens, token_durs = pauses_and_tokens_from_intervals(intervals)

    # Pause metrics (thresholded)
    pause_med, pause_p95 = summarize_pause_stats(pauses)
    long_pause_rate = (sum(p >= 0.5 for p in pauses) / (dur/60.0)) if dur > 0 else np.nan
    pct_pause_time = 100.0 * (np.sum(pauses) / dur)

    # Nuclei & rhythm
    nuc = syllable_nuclei_times(y, sr)
    inter = np.diff(nuc) if len(nuc) > 1 else np.array([])
    nuclei_rate = len(nuc) / dur
    rhythm_cv = (np.std(inter) / np.mean(inter)) if len(inter) > 1 and np.mean(inter) > 0 else np.nan
    rhythm_npvi = nPVI(inter) if len(inter) > 1 else np.nan

    # Prosody
    F0_med_st, F0_range_95_5_st, F0_IQR_st = f0_features_from_y(y, sr)
    inten_med_db, inten_iqr_db = intensity_features_db(y, sr)

    # Fillers (normalize to lower-case, punctuation stripped)
    def norm_tok(t):
        return "".join(ch for ch in t.lower() if ch.isalpha() or ch.isspace()).strip()
    normed = [norm_tok(t) for t in tokens]
    counts = Counter(normed)
    filler_counts = {k:v for k,v in counts.items() if k in FILLER_WHITELIST}
    filler_rates = {f"rate_{k}": v/(dur/60.0) for k,v in filler_counts.items()}
    med_dur_by_type = {f"meddur_{k}": np.median([d for t,d in zip(normed, token_durs) if t==k])
                       for k in filler_counts}

    # Voice quality
    vq = praat_voice_measures(audio_path)

    return {
        "duration_s": dur,
        "%pause_time": pct_pause_time,
        "pause_med_s": pause_med,
        "pause_p95_s": pause_p95,
        "long_pause_rate_per_min": long_pause_rate,
        "nuclei_rate_per_s": nuclei_rate,
        "rhythm_cv": rhythm_cv,
        "rhythm_nPVI": rhythm_npvi,
        "F0_med_st": F0_med_st,
        "F0_range95_5_st": F0_range_95_5_st,
        "F0_IQR_st": F0_IQR_st,
        "intensity_med_db": inten_med_db,
        "intensity_IQR_db": inten_iqr_db,
        **vq,
        **filler_rates,
        **med_dur_by_type,
    }
