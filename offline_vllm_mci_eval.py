#!/usr/bin/env python3
"""
offline_vllm_mci_eval.py  (now uses OpenAI-compatible API via .env)

Evaluation for:
- Zero-shot classification
- Episodic few-shot (k-shot) classification

Few-shot protocol is aligned with comprehensive_evaluation.py:
  - Outer LOO loop: each sample is held out exactly once as the test item.
  - Inner episodic loop: for each held-out sample, draw k examples per class
    from the remaining n-1 samples, E times.
  - Aggregate per-sample predictions across episodes (majority vote for labels,
    mean for probabilities).

Output format is aligned with comprehensive_evaluation.py for direct comparison.

Supports three input "modalities" (variants):
- transcript_only:    transcript_patient only
- linguistic_only:    text_* features only
- full_data:          transcript_patient + text_* features

Usage examples:
  # Zero-shot (all variants)
  python offline_vllm_mci_eval.py --input_csv data.csv

  # Zero-shot + few-shot (k=1,2,3,5, 10 episodes per k)
  python offline_vllm_mci_eval.py --input_csv data.csv --k 1 2 3 5 --episodes 10

  # Override model/endpoint from .env
  python offline_vllm_mci_eval.py --input_csv data.csv --model my-model --base_url http://host/v1
"""

import argparse
import asyncio
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

load_dotenv(".env")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TEXT_FEATURES = [
    "text_speech_rate",
    "text_ttr",
    "text_noun_ratio",
    "text_verb_ratio",
    "text_pronoun_ratio",
    "text_pronoun_to_noun_ratio",
    "text_mean_frequency",
    "text_coherence",
    "text_repetitiveness",
    "text_idea_densitity",
    "text_syntactic_complexity",
]

REQUIRED_CSV_COLS = (
    ["ID", "class", "dataset", "task", "language", "transcript_patient"] + TEXT_FEATURES
)

VARIANTS = {
    # (use_transcript_patient, use_text_features)
    "full_data":       (True,  True),
    "transcript_only": (True,  False),
    "linguistic_only": (False, True),
}

PROMPTS = {
    "simple_classifier": {
        "system": (
            "You are a binary classifier for a research dataset (non-diagnostic). "
            "Use only the provided transcript and/or linguistic metrics. "
            "Inputs may be English, Slovene, Mandarin, or Korean; treat multilingualism, "
            "accent, dialect, and topical content as neutral. "
            "Ignore demographic/identity attributes and stereotypes. Assume no class base-rate. "
            "Do not reveal reasoning. "
            "Your output must be exactly ONE WORD: Control or Patient."
        ),
        "instruction": (
            'Classify strictly as "Control" or "Patient" using only evidence present '
            "in the provided fields. "
            "If a section is missing, ignore it. Output exactly one word: Control or Patient."
        ),
        "output_format": "Return exactly one word: Control or Patient",
    }
}


# ---------------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------------

def normalize_true_class(x: Any) -> str:
    s = str(x).strip().lower()
    if s in {"mci", "patient"}:
        return "Patient"
    if s in {"hc", "control"}:
        return "Control"
    raise ValueError(f"Unknown class value: {x!r}")


def true_to_binary(label: str) -> int:
    return 1 if label == "Patient" else 0


def normalize_pred_text(s: str) -> str:
    t = (s or "").strip()
    tl = t.lower()
    if tl.startswith("patient"):
        return "Patient"
    if tl.startswith("control"):
        return "Control"
    return "Control"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_float(v: Any, nd: int) -> str:
    try:
        if v is None:
            return "NA"
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return "NA"
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def truncate_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    return s[:max_chars] + " ...[TRUNCATED]"


def format_linguistic_block(row: Dict[str, Any], nd: int) -> str:
    lines = []
    for col in TEXT_FEATURES:
        clean = col.replace("text_", "").replace("_", " ").title()
        lines.append(f"- {clean}: {_fmt_float(row.get(col), nd)}")
    return "\n".join(lines)


def build_prompt(
    row: Dict[str, Any],
    template: Dict[str, str],
    use_transcript: bool,
    use_text: bool,
    feature_decimals: int,
    max_transcript_chars: int,
    support_examples: Optional[List[Tuple[Dict[str, Any], str]]] = None,
) -> str:
    parts: List[str] = [template["system"]]

    # Few-shot examples (optional)
    if support_examples:
        parts.append("### EXAMPLES (labeled)")
        for ex_row, ex_label in support_examples:
            ex_parts = [
                f"Example ID: {ex_row.get('ID','NA')}  Language: {ex_row.get('language','NA')}"
            ]
            if use_transcript:
                ex_parts.append(
                    "[TRANSCRIPT]\n"
                    + truncate_text(str(ex_row.get("transcript_patient", "")), max_transcript_chars)
                )
            if use_text:
                ex_parts.append(
                    "[LINGUISTIC METRICS]\n" + format_linguistic_block(ex_row, feature_decimals)
                )
            ex_parts.append(f"Label: {ex_label}")
            parts.append("\n\n".join(ex_parts))

    # Query case
    q_parts = [
        f"### DATA INPUT FOR ID: {row.get('ID','NA')}  Language: {row.get('language','NA')}"
    ]
    if use_transcript:
        q_parts.append(
            "[TRANSCRIPT]\n"
            + truncate_text(str(row.get("transcript_patient", "")), max_transcript_chars)
        )
    if use_text:
        q_parts.append(
            "[LINGUISTIC METRICS]\n" + format_linguistic_block(row, feature_decimals)
        )

    parts.append("\n\n".join(q_parts))
    parts.append("### INSTRUCTIONS\n" + template["instruction"])
    parts.append("### OUTPUT FORMAT\n" + template["output_format"])
    return "\n\n".join([p for p in parts if p.strip()])


def short_model_name(model_path: str) -> str:
    return model_path.rstrip("/").split("/")[-1]


# ---------------------------------------------------------------------------
# Metrics (aligned with comprehensive_evaluation.py)
# ---------------------------------------------------------------------------

def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    return {
        "Accuracy":          float(accuracy_score(y_true, y_pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Macro_F1":          float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1_Target":         float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "Precision":         float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "Recall":            float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "AUROC":             np.nan,
        "AUPRC":             np.nan,
    }


# ---------------------------------------------------------------------------
# Async generation
# ---------------------------------------------------------------------------

async def _single_generate(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    temperature: float,
    sem: asyncio.Semaphore,
    guided_choice: bool = True,
    max_tokens: int = 512,
    debug: bool = False,
    max_retries: int = 3,
) -> str:
    extra = {"structured_outputs": {"choice": ["Control", "Patient"]}} if guided_choice else {}
    backoff = 1.0
    async with sem:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=60.0,
                    extra_body=extra,
                )
                choice = resp.choices[0]
                content = choice.message.content
                if debug and not content:
                    print(
                        f"    [DBG] finish_reason={choice.finish_reason!r} "
                        f"content={content!r} logprobs={choice.logprobs}"
                    )
                return content or ""
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                else:
                    print(f"    [WARN] Generation failed after {max_retries} attempts: {e}")
                    return ""
    return ""


async def async_generate(
    client: AsyncOpenAI,
    model: str,
    prompts: List[str],
    temperature: float,
    sem: asyncio.Semaphore,
    guided_choice: bool = True,
    debug: bool = False,
) -> List[str]:
    tasks = [
        _single_generate(client, model, p, temperature, sem, guided_choice, debug=debug)
        for p in prompts
    ]
    return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Few-shot: LOO outer loop + inner episodic sampling
#
# Protocol (identical to comprehensive_evaluation.py):
#   For each held-out test sample i:
#     For episode in range(n_episodes):
#       Sample k examples per class from the remaining n-1 samples (support set).
#       Build prompt with 2k in-context demonstrations.
#       Classify sample i.
#     Aggregate episode predictions by majority vote; probabilities by mean.
# ---------------------------------------------------------------------------

def _sample_support_indices(
    y_pool: np.ndarray,
    k: int,
    rng: np.random.RandomState,
) -> Optional[np.ndarray]:
    """Return indices into y_pool for k samples per class, or None if impossible."""
    class_0 = np.where(y_pool == 0)[0]
    class_1 = np.where(y_pool == 1)[0]
    if len(class_0) < k or len(class_1) < k:
        return None
    sel = np.concatenate([
        rng.choice(class_0, size=k, replace=False),
        rng.choice(class_1, size=k, replace=False),
    ])
    return sel


async def run_fewshot_loo(
    lang_df: pd.DataFrame,
    k: int,
    n_episodes: int,
    client: AsyncOpenAI,
    model: str,
    temperature: float,
    sem: asyncio.Semaphore,
    template: Dict[str, str],
    use_transcript: bool,
    use_text: bool,
    feature_decimals: int,
    max_transcript_chars: int,
    guided_choice: bool,
    seed: int,
    debug: bool,
) -> Dict[str, float]:
    """
    Few-shot evaluation with the same LOO + episodic protocol as
    comprehensive_evaluation.py.

    Returns aggregated metrics over all test samples.
    """
    rng = np.random.RandomState(seed)
    rows = lang_df.to_dict(orient="records")
    y = lang_df["binary_label"].values
    n = len(lang_df)

    # Per-sample episode predictions (list of lists)
    # sample_episode_preds[i] = list of binary predictions across episodes
    sample_episode_preds: List[List[int]] = [[] for _ in range(n)]

    for test_idx in range(n):
        # Pool = everything except the held-out sample
        pool_indices = np.array([j for j in range(n) if j != test_idx])
        y_pool = y[pool_indices]
        pool_rows = [rows[j] for j in pool_indices]

        test_row = rows[test_idx]

        for _ in range(n_episodes):
            sel = _sample_support_indices(y_pool, k, rng)
            if sel is None:
                continue  # not enough samples per class — skip episode

            support: List[Tuple[Dict[str, Any], str]] = []
            for idx in sel:
                label_str = "Patient" if y_pool[idx] == 1 else "Control"
                support.append((pool_rows[idx], label_str))

            prompt = build_prompt(
                row=test_row,
                template=template,
                use_transcript=use_transcript,
                use_text=use_text,
                feature_decimals=feature_decimals,
                max_transcript_chars=max_transcript_chars,
                support_examples=support,
            )

            raw = await _single_generate(
                client, model, prompt, temperature, sem, guided_choice, debug=debug
            )
            pred_str = normalize_pred_text(raw)
            sample_episode_preds[test_idx].append(1 if pred_str == "Patient" else 0)

    # Aggregate: majority vote per sample
    y_true_all, y_pred_all = [], []
    for i in range(n):
        ep_preds = sample_episode_preds[i]
        if not ep_preds:
            continue
        y_true_all.append(int(y[i]))
        # majority vote via rounding the mean (ties -> 1, consistent with tabular script)
        y_pred_all.append(int(np.round(np.mean(ep_preds))))

    if not y_true_all:
        return {m: np.nan for m in
                ["Accuracy", "Balanced_Accuracy", "Macro_F1", "F1_Target",
                 "Precision", "Recall", "AUROC", "AUPRC"]}

    return compute_metrics(y_true_all, y_pred_all)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(results_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    metric_cols = ["Accuracy", "Balanced_Accuracy", "Macro_F1", "F1_Target",
                   "Precision", "Recall", "AUROC", "AUPRC"]

    summary_rows = []
    for (lang, model, mode), group in results_df.groupby(["Language", "Model", "Mode"]):
        row = {"Language": lang, "Model": model, "Mode": mode}
        for m in metric_cols:
            vals = group[m].dropna()
            row[f"{m}_mean"] = vals.mean() if len(vals) else np.nan
            row[f"{m}_std"]  = vals.std()  if len(vals) else np.nan
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_dir / "aggregated_results.csv", index=False)
    return summary_df


def ensure_csv_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_CSV_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")


# ---------------------------------------------------------------------------
# Per-language evaluation
# ---------------------------------------------------------------------------

async def evaluate_language(
    language: str,
    lang_df: pd.DataFrame,
    client: AsyncOpenAI,
    model: str,
    temperature: float,
    sem: asyncio.Semaphore,
    model_name: str,
    templates: Dict[str, Dict[str, str]],
    variants: List[str],
    k_values: List[int],
    n_episodes: int,
    feature_decimals: int,
    max_transcript_chars: int,
    seed: int,
    guided_choice: bool = True,
    debug: bool = False,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """Evaluate all configurations for a single language."""
    results = []
    pred_df = lang_df[["ID", "class", "language"]].copy()

    n_samples  = len(lang_df)
    n_patient  = int(lang_df["binary_label"].sum())
    n_control  = n_samples - n_patient

    print(f"\n{'='*60}")
    print(f"LANGUAGE: {language.upper()}")
    print(f"{'='*60}")
    print(f"Samples: {n_samples} (Patient: {n_patient}, Control: {n_control})")

    for template_name, tmpl in templates.items():
        for variant_name in variants:
            use_trans, use_text = VARIANTS[variant_name]

            # ------------------------------------------------------------------
            # Zero-shot (LOO for consistency: each sample classified without
            # demonstrations, but we still iterate individually so the outer
            # loop structure matches the few-shot case)
            # ------------------------------------------------------------------
            mode = f"zero_shot_{variant_name}"
            print(f"\n  [{mode}]")

            prompts = [
                build_prompt(
                    row=r.to_dict(),
                    template=tmpl,
                    use_transcript=use_trans,
                    use_text=use_text,
                    feature_decimals=feature_decimals,
                    max_transcript_chars=max_transcript_chars,
                    support_examples=None,
                )
                for _, r in lang_df.iterrows()
            ]

            raw_outs = await async_generate(
                client, model, prompts, temperature, sem, guided_choice, debug
            )
            preds = [normalize_pred_text(x) for x in raw_outs]
            pred_df[mode] = preds
            if debug and preds:
                print(f"    [sample] raw={raw_outs[0]!r:20s} -> {preds[0]}")

            y_true = lang_df["binary_label"].tolist()
            y_pred = [1 if p == "Patient" else 0 for p in preds]
            metrics = compute_metrics(y_true, y_pred)

            results.append({
                "Run": 1, "Language": language, "Model": model_name,
                "Mode": mode, **metrics,
            })
            print(f"    F1={metrics['Macro_F1']:.4f}, BalAcc={metrics['Balanced_Accuracy']:.4f}")

            # ------------------------------------------------------------------
            # Few-shot — LOO outer loop + inner episodic sampling
            # Identical protocol to comprehensive_evaluation.py:
            #   - Each sample is held out once as the test item.
            #   - k examples per class are drawn from the remaining n-1 samples.
            #   - Repeated for n_episodes; predictions aggregated by majority vote.
            # ------------------------------------------------------------------
            for k in k_values:
                mode_fs = f"fewshot_k{k}_{variant_name}"
                print(f"\n  [{mode_fs}] ({n_episodes} episodes per test sample)")

                if n_control < k or n_patient < k:
                    print(
                        f"    Skipped: need {k} per class, "
                        f"have min({n_control}, {n_patient})={min(n_control, n_patient)}"
                    )
                    continue

                fs_metrics = await run_fewshot_loo(
                    lang_df=lang_df,
                    k=k,
                    n_episodes=n_episodes,
                    client=client,
                    model=model,
                    temperature=temperature,
                    sem=sem,
                    template=tmpl,
                    use_transcript=use_trans,
                    use_text=use_text,
                    feature_decimals=feature_decimals,
                    max_transcript_chars=max_transcript_chars,
                    guided_choice=guided_choice,
                    seed=seed + k * 1000 + hash(language + variant_name) % 10000,
                    debug=debug,
                )

                results.append({
                    "Run": 1, "Language": language, "Model": model_name,
                    "Mode": mode_fs, **fs_metrics,
                })
                print(
                    f"    F1={fs_metrics['Macro_F1']:.4f}, "
                    f"BalAcc={fs_metrics['Balanced_Accuracy']:.4f}"
                )

    return results, pred_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _main():
    ap = argparse.ArgumentParser(description="Online MCI evaluation via OpenAI-compatible API")

    ap.add_argument("--base_url", default=None)
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max_concurrency", type=int, default=None)

    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_dir", default="online_eval_output")
    ap.add_argument("--templates", nargs="+", default=["simple_classifier"])
    ap.add_argument("--variants", nargs="+",
                    default=["transcript_only", "linguistic_only", "full_data"])
    ap.add_argument("--feature_decimals", type=int, default=3)
    ap.add_argument("--max_transcript_chars", type=int, default=6000)
    ap.add_argument("--k", type=int, nargs="*", default=[],
                    help="Few-shot k values (e.g. --k 1 2 3 5)")
    ap.add_argument("--episodes", type=int, default=3,
                    help="Episodes per held-out test sample (inner loop)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--languages", nargs="+", default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--guided_choice", action="store_true")

    args = ap.parse_args()

    base_url       = args.base_url       or os.environ.get("SERVER_HOST")
    api_key        = args.api_key        or os.environ.get("API_KEY", "EMPTY")
    model          = args.model          or os.environ.get("MODEL_NAME", "gpt-4-turbo")
    temperature    = args.temperature    if args.temperature is not None \
                                         else float(os.environ.get("TEMPERATURE", "0.0"))
    max_concurrency = args.max_concurrency or int(os.environ.get("MAX_CONCURRENCY", "32"))

    if not base_url:
        raise ValueError("No API base URL provided. Set SERVER_HOST in .env or pass --base_url.")

    model_name = short_model_name(model)

    df = pd.read_csv(args.input_csv)
    ensure_csv_columns(df)
    df["true_label"]   = df["class"].apply(normalize_true_class)
    df["binary_label"] = df["true_label"].apply(true_to_binary)

    templates = {t: PROMPTS[t] for t in args.templates}

    languages = sorted(df["language"].str.lower().unique())
    if args.languages:
        languages = [l for l in languages if l in [x.lower() for x in args.languages]]
        if not languages:
            raise ValueError(
                f"No matching languages. Available: {sorted(df['language'].str.lower().unique())}"
            )

    # Estimate API calls
    lang_counts = df["language"].str.lower().value_counts()
    n_variants  = len(args.variants)
    n_templates = len(args.templates)
    zs_calls    = len(df) * n_variants * n_templates
    fs_calls    = 0
    for lang in languages:
        n     = lang_counts.get(lang, 0)
        n_pat = int(df[df["language"].str.lower() == lang]["binary_label"].sum())
        n_ctl = n - n_pat
        for k in args.k:
            if n_pat >= k and n_ctl >= k:
                # Each of the n samples is a test item; each gets n_episodes calls
                fs_calls += n * n_variants * n_templates * args.episodes
    total_calls = zs_calls + fs_calls

    print("=" * 60)
    print("ONLINE MCI CLASSIFICATION EVALUATION")
    print("=" * 60)
    print(f"Endpoint:        {base_url}")
    print(f"Model:           {model} ({model_name})")
    print(f"Variants:        {args.variants}")
    print(f"Guided choice:   {args.guided_choice}")
    print(f"Few-shot k:      {args.k if args.k else 'none (zero-shot only)'}")
    print(f"Episodes/sample: {args.episodes}")
    print(f"Max concurrency: {max_concurrency}")
    print(f"")
    print(f"API call estimate:")
    print(f"  Zero-shot:  {zs_calls:>6d}  ({len(df)} samples × {n_variants} variants × {n_templates} templates)")
    if args.k:
        print(f"  Few-shot:   {fs_calls:>6d}  ({len(args.k)} k-values × n_samples × {args.episodes} episodes)")
    print(f"  TOTAL:      {total_calls:>6d}")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Printing one example prompt per variant (no API calls)\n")
        first_row = df.iloc[0].to_dict()
        for tmpl_name, tmpl in templates.items():
            for variant_name in args.variants:
                use_trans, use_text = VARIANTS[variant_name]
                prompt = build_prompt(
                    row=first_row,
                    template=tmpl,
                    use_transcript=use_trans,
                    use_text=use_text,
                    feature_decimals=args.feature_decimals,
                    max_transcript_chars=args.max_transcript_chars,
                    support_examples=None,
                )
                print(f"\n{'─'*60}")
                print(
                    f"TEMPLATE: {tmpl_name}  |  VARIANT: {variant_name}  |  "
                    f"ID: {first_row.get('ID','?')}  |  Language: {first_row.get('language','?')}"
                )
                print(f"{'─'*60}")
                print(prompt)
        return

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    sem    = asyncio.Semaphore(max_concurrency)

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.out_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    config = {
        "base_url": base_url, "model": model, "model_short": model_name,
        "templates": args.templates, "variants": args.variants,
        "k_values": args.k, "episodes_per_sample": args.episodes,
        "seed": args.seed, "timestamp": timestamp,
        "max_transcript_chars": args.max_transcript_chars,
        "feature_decimals": args.feature_decimals,
        "temperature": temperature, "max_concurrency": max_concurrency,
        "fewshot_protocol": "LOO_outer_loop_plus_inner_episodic_sampling",
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    all_results: List[Dict[str, Any]] = []
    all_preds:   List[pd.DataFrame]   = []

    for language in languages:
        lang_mask = df["language"].str.lower() == language
        lang_df   = df[lang_mask].reset_index(drop=True)

        results, pred_df = await evaluate_language(
            language=language,
            lang_df=lang_df,
            client=client,
            model=model,
            temperature=temperature,
            sem=sem,
            model_name=model_name,
            templates=templates,
            variants=args.variants,
            k_values=args.k,
            n_episodes=args.episodes,
            feature_decimals=args.feature_decimals,
            max_transcript_chars=args.max_transcript_chars,
            seed=args.seed,
            guided_choice=args.guided_choice,
            debug=args.debug,
        )
        all_results.extend(results)
        all_preds.append(pred_df)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "all_results.csv", index=False)

    pd.concat(all_preds, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"config": config, "results": all_results}, f,
                  indent=2, ensure_ascii=False, default=str)

    aggregate_results(results_df, output_dir)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for _, row in results_df.iterrows():
        print(
            f"  {row['Language']:10s} {row['Mode']:35s} "
            f"F1={row['Macro_F1']:.4f} BalAcc={row['Balanced_Accuracy']:.4f}"
        )

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"  all_results.csv  <- main output (comparable with comprehensive_evaluation.py)")
    print(f"  predictions.csv  <- per-sample predictions")
    print(f"  config.json      <- run configuration")
    print(f"  metrics.json     <- detailed metrics (JSON)")
    print(f"  summary/         <- aggregated results")
    print(f"{'='*60}")


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()