# Foundational CI Detection

**Multilingual Cognitive Impairment Detection in the Era of Foundation Models** (LREC 2026).

Classifies cognitive impairment (MCI/patient vs. HC/control) from picture-description
transcripts across four languages, comparing three families of methods under one
Leave-One-Out (LOO) and episodic few-shot protocol:

1. **Classical / gradient-boosting ML** on expert linguistic features (LR, RF, LightGBM)
2. **Tabular foundation models** on text embeddings and fusion (TabPFN, RealMLP)
3. **Prompted LLMs** as zero-/few-shot classifiers (gpt-oss-20b, medgemma-27b)

Representations: 11 expert linguistic features, frozen transcript embeddings, and
early / late fusion of the two.

## Dataset

| Language | Samples | Class distribution |
|----------|---------|--------------------|
| Mandarin | 259 | MCI 140 / HC 119 |
| English  | 156 | MCI 78 / HC 78 |
| Korean   | 77  | MCI 40 / HC 37 |
| Slovene  | 27  | MCI 12 / HC 15 |

`data/english_slovene_chinese_korean_data_preprocessed_04022026.csv` holds, per
participant: `ID, class, dataset, task, language` and 11 `text_*` linguistic
features (speech rate, TTR, POS ratios, coherence, repetitiveness, idea density,
syntactic complexity, mean word frequency).

Precomputed transcript embeddings are shipped under `data/` so the tabular pipeline
runs **without a GPU** (and without the raw transcripts):

- `data/google_embeddinggemma-300m_*.npy` — `google/embeddinggemma-300m` (768-d)
- `data/paraphrase-multilingual-MiniLM-L12-v2_*.npy` — MiniLM (384-d)

> **Raw transcripts are not distributed here.** The English data derives from the
> Pitt/ADReSS corpus (DementiaBank) and the Korean/Slovene sets from their
> respective providers, all under data-use agreements that restrict redistributing
> transcript text. This repo therefore ships only the derived features and
> embeddings. To run the LLM evaluation or regenerate embeddings, obtain the
> corpora under their DUAs and supply your own CSV with a `transcript_patient`
> column (same row order as the features file).

## Install

```bash
uv venv --python 3.13 && source .venv/bin/activate
uv pip install -e .
```

The tabular pipeline needs `scikit-learn, pandas, numpy, tabpfn, pytabkit, lightgbm,
sentence-transformers`. The LLM pipeline additionally needs `openai, python-dotenv`
and a running OpenAI-compatible endpoint (see `container.def` for a vLLM image).

## Running the pipeline

### 1. Tabular ML / foundation models — `comprehensive_evaluation.py`

LOO and episodic few-shot (k=1,2,3,5) over embeddings-only, features-only, early
fusion, and late fusion, for each language.

```bash
# Full sweep (paper config: TabPFN + RealMLP, embeddinggemma embeddings)
python comprehensive_evaluation.py \
    --embedding google/embeddinggemma-300m \
    --models tabpfn realmlp --n_runs 3 --device cuda

# CPU-friendly single model / language (uses shipped embedding cache)
python comprehensive_evaluation.py \
    --embedding google/embeddinggemma-300m \
    --models lr --languages slovene --n_runs 1 --device cpu
```

Embeddings are auto-loaded from `data/<sanitized_model>_<dataset>.npy` if present,
else computed and cached there. Results → `comprehensive_results/<timestamp>/`.

On a SLURM cluster, `slurm_eval.sh` runs the 4-language × 2-model array inside the
container (`sbatch slurm_eval.sh`).

### 2. LLM zero-/few-shot — `offline_vllm_mci_eval.py`

Talks to an OpenAI-compatible endpoint (configured via `.env`, see `.env.example`).

```bash
cp .env.example .env      # set SERVER_HOST / MODEL_NAME
python offline_vllm_mci_eval.py \
    --input_csv <your_transcript_csv.csv> \
    --k 1 2 3 5 --episodes 10 --guided_choice
```

Three input variants: `transcript_only`, `linguistic_only`, `full_data`.
Results → `online_eval_output/<timestamp>/`.

> This step needs a CSV that includes the `transcript_patient` column. The dataset
> shipped in `data/` has transcripts removed (see licensing note above), so supply
> your own transcript-bearing CSV obtained under the corpus DUAs.

### 3. Aggregate into paper tables — `aggregate_results.py`

Merges the ML (`comprehensive_results/`) and LLM (`online_eval_output/`) runs into
the tables/figures under `results/final_tables/`.

```bash
python aggregate_results.py
```

## Repository layout

```
├── comprehensive_evaluation.py   # tabular ML + foundation models (LOO / few-shot / fusion)
├── offline_vllm_mci_eval.py      # LLM zero-/few-shot via OpenAI-compatible API
├── aggregate_results.py          # merges ML + LLM runs -> results/final_tables/
├── slurm_eval.sh                 # SLURM array launcher
├── container.def                 # Apptainer/Singularity image (vLLM + tabular stack)
├── data/                         # transformed data + precomputed embedding caches
├── cv_splits/                    # repeated stratified 5×5 split definitions
├── preprocessing/                # transcript/feature/ASR extraction (reference; raw audio not shipped)
└── results/final_tables/         # paper tables and figures
```

`preprocessing/` (ASR, TextGrid parsing, feature extraction, split preparation) is
included for reference; it operates on the raw audio/recordings, which are **not**
redistributed here.

## Key results (Macro-F1, LOO)

| Language | Best approach | Macro-F1 |
|----------|---------------|----------|
| English  | TabPFN, late fusion (emb + feat) | 0.816 |
| Korean   | LR, late fusion (emb + feat)     | 0.805 |
| Slovene  | LR / tabular on embeddings       | 0.852 |
| Mandarin | tabular on features              | ~0.71 |

Supervised tabular models with linguistic features and fusion outperform prompted
LLM zero-shot by roughly +0.18 to +0.26 Macro-F1 per language.

## Citation

```bibtex
@inproceedings{foundational-ci-detection-2026,
  title  = {Multilingual Cognitive Impairment Detection in the Era of Foundation Models},
  author = {Hoogland, Damar and Koloski, Boshko and Caporusso, Jaya and Kolenik, Tine
            and Zwitter Vitez, Ana and Pollak, Senja and Manouilidou, Christina and Purver, Matthew},
  booktitle = {Proceedings of LREC},
  year   = {2026}
}
```

## License

See `LICENSE`.
