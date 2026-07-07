"""
aggregate_results.py
====================
Aggregates LLM (online_eval_output) and ML (comprehensive_results) results
into unified tables and visualizations.

Layout:
  Section 1 – LLM Zero-shot  (per language x model x variant)
  Section 2 – LLM Few-shot   (per language x model x k, transcript_only)
  Section 3 – ML LOO         (per language x model x mode)
  Section 4 – ML Few-shot    (per language x model x k, embeddings)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("final_tables")
OUT.mkdir(exist_ok=True)

LANGUAGES = ["english", "korean", "mandarin", "slovene"]
METRIC    = "Macro_F1"
BAL_ACC   = "Balanced_Accuracy"

# ════════════════════════════════════════════════════════════════════════════
# 1. LOAD LLM RESULTS
# ════════════════════════════════════════════════════════════════════════════

def load_best_llm_run(base_dir: str, model_short: str) -> pd.DataFrame:
    base = Path(base_dir)
    best_df, best_rows = None, 0
    for d in sorted(base.iterdir()):
        csv = d / "all_results.csv"
        cfg = d / "config.json"
        if not csv.exists() or not cfg.exists():
            continue
        try:
            c = json.load(open(cfg))
            if c.get("model_short", "") != model_short:
                continue
            df = pd.read_csv(csv)
            if len(df) > best_rows:
                best_df, best_rows = df, len(df)
        except Exception:
            continue
    return best_df if best_df is not None else pd.DataFrame()


llm_gpt   = load_best_llm_run("online_eval_output", "gpt-oss-20b")
llm_gemma = load_best_llm_run("online_eval_output", "medgemma-27b-it")

llm_frames = [df for df in [llm_gpt, llm_gemma] if not df.empty]
llm_df = pd.concat(llm_frames, ignore_index=True) if llm_frames else pd.DataFrame()
if not llm_df.empty:
    llm_df["Language"] = llm_df["Language"].str.lower()

print(f"LLM rows loaded: {len(llm_df)}")
if not llm_df.empty:
    print("  Models:", llm_df["Model"].unique().tolist())

# ════════════════════════════════════════════════════════════════════════════
# 2. LOAD ML RESULTS
# ════════════════════════════════════════════════════════════════════════════

def load_best_ml_run(base_dir: str, model_type: str) -> pd.DataFrame:
    base = Path(base_dir)
    best_df, best_rows = None, 0
    for d in sorted(base.iterdir()):
        csv = d / "all_results.csv"
        cfg = d / "config.json"
        if not csv.exists() or not cfg.exists():
            continue
        try:
            c = json.load(open(cfg))
            if model_type not in c.get("models", []):
                continue
            df = pd.read_csv(csv)
            df = df[df["Model"] == model_type]
            if len(df) > best_rows:
                best_df, best_rows = df, len(df)
        except Exception:
            continue
    return best_df if best_df is not None else pd.DataFrame()


ML_MODELS = ["lr", "rf", "lgbm", "tabpfn", "realmlp"]
ml_frames = []
for m in ML_MODELS:
    df = load_best_ml_run("comprehensive_results", m)
    if not df.empty:
        ml_frames.append(df)
        print(f"ML {m}: {len(df)} rows")

ml_df = pd.concat(ml_frames, ignore_index=True) if ml_frames else pd.DataFrame()
if not ml_df.empty:
    ml_df["Language"] = ml_df["Language"].str.lower()
    numeric_cols = [METRIC, BAL_ACC, "F1_Target", "Precision", "Recall", "AUROC", "AUPRC"]
    for c in numeric_cols:
        if c in ml_df.columns:
            ml_df[c] = pd.to_numeric(ml_df[c], errors="coerce")
    ml_agg = (ml_df
              .groupby(["Language", "Model", "Mode"])[numeric_cols]
              .agg(["mean", "std"])
              .reset_index())
    ml_agg.columns = ["_".join(c).strip("_") for c in ml_agg.columns]


def fmt(v, std=None):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    s = f"{float(v):.3f}"
    if std is not None and not np.isnan(float(std)):
        s += f"±{float(std):.3f}"
    return s


# ════════════════════════════════════════════════════════════════════════════
# 3. TABLE 1 – LLM Zero-shot
# ════════════════════════════════════════════════════════════════════════════
print("\n── TABLE 1: LLM Zero-shot ──")

if not llm_df.empty:
    zs = llm_df[llm_df["Mode"].str.startswith("zero_shot")].copy()
    zs["variant"] = zs["Mode"].str.replace("zero_shot_", "", regex=False)

    tbl1_rows = []
    for model in sorted(zs["Model"].unique()):
        for variant in ["transcript_only", "linguistic_only", "full_data"]:
            row = {"Model": model, "Variant": variant}
            sub = zs[(zs["Model"] == model) & (zs["variant"] == variant)]
            for lang in LANGUAGES:
                v = sub[sub["Language"] == lang][METRIC].values
                row[lang] = fmt(v[0]) if len(v) else "—"
            row["Avg"] = fmt(sub[METRIC].mean()) if len(sub) else "—"
            tbl1_rows.append(row)

    tbl1 = pd.DataFrame(tbl1_rows)
    print(tbl1.to_string(index=False))
    tbl1.to_csv(OUT / "table1_llm_zeroshot.csv", index=False)


# ════════════════════════════════════════════════════════════════════════════
# 4. TABLE 2 – LLM Few-shot (transcript_only)
# ════════════════════════════════════════════════════════════════════════════
print("\n── TABLE 2: LLM Few-shot (transcript_only) ──")

if not llm_df.empty:
    fs = llm_df[llm_df["Mode"].str.startswith("fewshot") &
                llm_df["Mode"].str.endswith("transcript_only")].copy()
    fs["k"] = fs["Mode"].str.extract(r"fewshot_k(\d+)").astype(int)

    tbl2_rows = []
    for model in sorted(fs["Model"].unique()):
        # include zero-shot as k=0 reference
        zs_sub = llm_df[(llm_df["Model"] == model) &
                        (llm_df["Mode"] == "zero_shot_transcript_only")]
        if not zs_sub.empty:
            row = {"Model": model, "k": "0 (ZS)"}
            for lang in LANGUAGES:
                v = zs_sub[zs_sub["Language"] == lang][METRIC].values
                row[lang] = fmt(v[0]) if len(v) else "—"
            row["Avg"] = fmt(zs_sub[METRIC].mean())
            tbl2_rows.append(row)

        for k in sorted(fs["k"].unique()):
            row = {"Model": model, "k": str(k)}
            sub = fs[(fs["Model"] == model) & (fs["k"] == k)]
            for lang in LANGUAGES:
                v = sub[sub["Language"] == lang][METRIC].values
                row[lang] = fmt(v[0]) if len(v) else "—"
            row["Avg"] = fmt(sub[METRIC].mean()) if len(sub) else "—"
            tbl2_rows.append(row)

    tbl2 = pd.DataFrame(tbl2_rows)
    print(tbl2.to_string(index=False))
    tbl2.to_csv(OUT / "table2_llm_fewshot.csv", index=False)


# ════════════════════════════════════════════════════════════════════════════
# 5. TABLE 3 – ML LOO
# ════════════════════════════════════════════════════════════════════════════
print("\n── TABLE 3: ML LOO ──")

LOO_MODES = ["loo_embeddings", "loo_features", "loo_early_fusion", "loo_late_fusion"]

if not ml_df.empty:
    loo = ml_agg[ml_agg["Mode"].isin(LOO_MODES)]

    tbl3_rows = []
    for model in ML_MODELS:
        for mode in LOO_MODES:
            sub = loo[(loo["Model"] == model) & (loo["Mode"] == mode)]
            if sub.empty:
                continue
            row = {"Model": model, "Mode": mode}
            lang_vals = []
            for lang in LANGUAGES:
                lv = sub[sub["Language"] == lang]
                if lv.empty:
                    row[lang] = "—"
                else:
                    m_val = lv[f"{METRIC}_mean"].values[0]
                    s_val = lv[f"{METRIC}_std"].values[0]
                    row[lang] = fmt(m_val, s_val)
                    if not np.isnan(m_val):
                        lang_vals.append(m_val)
            row["Avg"] = fmt(np.mean(lang_vals)) if lang_vals else "—"
            tbl3_rows.append(row)

    tbl3 = pd.DataFrame(tbl3_rows)
    print(tbl3.to_string(index=False))
    tbl3.to_csv(OUT / "table3_ml_loo.csv", index=False)


# ════════════════════════════════════════════════════════════════════════════
# 6. TABLE 4 – ML Few-shot (embeddings)
# ════════════════════════════════════════════════════════════════════════════
print("\n── TABLE 4: ML Few-shot (embeddings) ──")

if not ml_df.empty:
    fs_ml = ml_agg[ml_agg["Mode"].str.match(r"fewshot_k\d+$")]

    tbl4_rows = []
    for model in ML_MODELS:
        for k in [1, 2, 3, 5]:
            mode = f"fewshot_k{k}"
            sub = fs_ml[(fs_ml["Model"] == model) & (fs_ml["Mode"] == mode)]
            if sub.empty:
                continue
            row = {"Model": model, "k": k}
            lang_vals = []
            for lang in LANGUAGES:
                lv = sub[sub["Language"] == lang]
                if lv.empty:
                    row[lang] = "—"
                else:
                    m_val = lv[f"{METRIC}_mean"].values[0]
                    s_val = lv[f"{METRIC}_std"].values[0]
                    row[lang] = fmt(m_val, s_val)
                    if not np.isnan(m_val):
                        lang_vals.append(m_val)
            row["Avg"] = fmt(np.mean(lang_vals)) if lang_vals else "—"
            tbl4_rows.append(row)

    tbl4 = pd.DataFrame(tbl4_rows)
    print(tbl4.to_string(index=False))
    tbl4.to_csv(OUT / "table4_ml_fewshot_emb.csv", index=False)


# ════════════════════════════════════════════════════════════════════════════
# 7. VISUALIZATIONS
# ════════════════════════════════════════════════════════════════════════════

COLORS = {
    "gpt-oss-20b":     "#1f77b4",
    "medgemma-27b-it": "#ff7f0e",
    "lr":              "#2ca02c",
    "rf":              "#d62728",
    "lgbm":            "#9467bd",
    "tabpfn":          "#8c564b",
    "realmlp":         "#e377c2",
}

# ── Fig 1: ML LOO heatmap per language ──────────────────────────────────────
if not ml_df.empty and tbl3_rows:
    LOO_SHORT = {"loo_embeddings": "Emb", "loo_features": "Feat",
                 "loo_early_fusion": "EarlyF", "loo_late_fusion": "LateF"}
    loo_df = ml_agg[ml_agg["Mode"].isin(LOO_MODES)]

    fig, axes = plt.subplots(1, len(LANGUAGES), figsize=(18, 6), sharey=False)
    for ax, lang in zip(axes, LANGUAGES):
        labels, vals = [], []
        for model in ML_MODELS:
            for mode in LOO_MODES:
                sub = loo_df[(loo_df["Model"] == model) &
                             (loo_df["Mode"] == mode) &
                             (loo_df["Language"] == lang)]
                v = sub[f"{METRIC}_mean"].values[0] if not sub.empty else np.nan
                labels.append(f"{model}\n{LOO_SHORT[mode]}")
                vals.append(v)

        vals = np.array(vals, dtype=float)
        colors = ["#2ecc71" if v >= 0.7 else "#e74c3c" if v < 0.5 else "#3498db"
                  for v in vals]
        bars = ax.barh(labels, vals, color=colors, edgecolor="white", height=0.75)
        ax.set_xlim(0.25, 1.0)
        ax.set_title(lang.capitalize(), fontweight="bold", fontsize=11)
        ax.set_xlabel("Macro F1")
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(max(v + 0.005, 0.26), bar.get_y() + bar.get_height() / 2,
                        f"{v:.3f}", va="center", fontsize=7)

    plt.suptitle("ML LOO – Macro F1 by Language\n(green ≥0.70, blue 0.50–0.70, red <0.50)",
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(OUT / "fig1_ml_loo_by_language.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig1_ml_loo_by_language.png")

# ── Fig 2: LLM zero-shot vs few-shot curves ─────────────────────────────────
if not llm_df.empty:
    fig, axes = plt.subplots(1, len(LANGUAGES), figsize=(16, 4), sharey=True)
    for ax, lang in zip(axes, LANGUAGES):
        sub = llm_df[llm_df["Language"] == lang]
        for model in sorted(sub["Model"].unique()):
            color = COLORS.get(model, "gray")
            msub = sub[sub["Model"] == model]
            zs_val = msub[msub["Mode"] == "zero_shot_transcript_only"][METRIC].values
            if len(zs_val):
                ax.axhline(zs_val[0], linestyle="--", color=color, alpha=0.7,
                           label=f"{model} ZS={zs_val[0]:.3f}")
            ks, fvals = [], []
            for k in [1, 2, 3, 5]:
                v = msub[msub["Mode"] == f"fewshot_k{k}_transcript_only"][METRIC].values
                if len(v):
                    ks.append(k)
                    fvals.append(v[0])
            if ks:
                ax.plot(ks, fvals, "o-", color=color, label=f"{model} FS")
        ax.set_title(lang.capitalize(), fontweight="bold")
        ax.set_xlabel("k (shots)")
        ax.set_ylim(0.25, 0.85)
        ax.set_xticks([1, 2, 3, 5])
        ax.axhline(0.5, linestyle=":", color="gray", alpha=0.4)
    axes[0].set_ylabel("Macro F1")
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.15, 1), fontsize=8)
    plt.suptitle("LLM: Zero-shot (dashed) vs Few-shot (transcript only)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "fig2_llm_zeroshot_vs_fewshot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig2_llm_zeroshot_vs_fewshot.png")

# ── Fig 3: LLM vs ML grouped bar (best per system per language) ──────────────
if not ml_df.empty or not llm_df.empty:
    # Build best-per-system rows
    sys_rows = []

    if not llm_df.empty:
        for model in sorted(llm_df["Model"].unique()):
            row = {"System": model, "Type": "LLM"}
            for lang in LANGUAGES:
                best = llm_df[(llm_df["Model"] == model) &
                              (llm_df["Language"] == lang)][METRIC].max()
                row[lang] = best
            sys_rows.append(row)

    if not ml_df.empty:
        for model in ML_MODELS:
            sub = ml_agg[(ml_agg["Model"] == model) & (ml_agg["Mode"].isin(LOO_MODES))]
            if sub.empty:
                continue
            row = {"System": model, "Type": "ML"}
            for lang in LANGUAGES:
                lv = sub[sub["Language"] == lang][f"{METRIC}_mean"]
                row[lang] = lv.max() if len(lv) else np.nan
            sys_rows.append(row)

    if sys_rows:
        sys_df = pd.DataFrame(sys_rows)
        n = len(sys_df)
        x = np.arange(len(LANGUAGES))
        w = 0.8 / n

        fig, ax = plt.subplots(figsize=(13, 5))
        for i, (_, row) in enumerate(sys_df.iterrows()):
            vals = [row[l] if not pd.isna(row[l]) else np.nan for l in LANGUAGES]
            color = COLORS.get(row["System"], f"C{i}")
            offset = (i - n / 2 + 0.5) * w
            ax.bar(x + offset, vals, w * 0.92,
                   label=row["System"], color=color, alpha=0.87,
                   hatch="//" if row["Type"] == "LLM" else "",
                   edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels([l.capitalize() for l in LANGUAGES], fontsize=11)
        ax.set_ylabel("Macro F1")
        ax.set_ylim(0.2, 1.0)
        ax.axhline(0.5, linestyle=":", color="gray", alpha=0.5)
        ax.legend(fontsize=8, ncol=3, loc="upper left")
        ax.set_title("Best Macro F1 per Language: LLM (hatched) vs ML (solid)",
                     fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUT / "fig3_llm_vs_ml.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved fig3_llm_vs_ml.png")

# ── Fig 4: ML few-shot learning curves ──────────────────────────────────────
if not ml_df.empty and tbl4_rows:
    fig, axes = plt.subplots(1, len(LANGUAGES), figsize=(16, 4), sharey=True)
    for ax, lang in zip(axes, LANGUAGES):
        for model in ML_MODELS:
            ks, vals, stds = [], [], []
            for k in [1, 2, 3, 5]:
                sub = ml_agg[(ml_agg["Model"] == model) &
                             (ml_agg["Mode"] == f"fewshot_k{k}") &
                             (ml_agg["Language"] == lang)]
                if not sub.empty:
                    mv = sub[f"{METRIC}_mean"].values[0]
                    sv = sub[f"{METRIC}_std"].values[0]
                    ks.append(k)
                    vals.append(mv)
                    stds.append(0 if np.isnan(sv) else sv)
            if ks:
                ax.errorbar(ks, vals, yerr=stds, fmt="o-",
                            color=COLORS.get(model, "gray"), label=model,
                            capsize=3, linewidth=1.5)
        ax.set_title(lang.capitalize(), fontweight="bold")
        ax.set_xlabel("k")
        ax.set_xticks([1, 2, 3, 5])
        ax.set_ylim(0.25, 0.95)
        ax.axhline(0.5, linestyle=":", color="gray", alpha=0.4)
    axes[0].set_ylabel("Macro F1")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.1, 1), fontsize=8)
    plt.suptitle("ML Few-shot (embeddings): Macro F1 Learning Curves", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "fig4_ml_fewshot_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig4_ml_fewshot_curves.png")

print(f"\nAll outputs -> {OUT}/")
print("Files:", sorted([f.name for f in OUT.iterdir()]))
