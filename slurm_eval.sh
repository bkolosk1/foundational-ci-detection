#!/bin/bash
#SBATCH --job-name=mci_eval
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-7          # 4 languages × 2 models = 8 jobs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00      # 3h per job (Mandarin/TabPFN is the slowest)

# ── Job → (language, model) mapping ───────────────────────────
LANGUAGES=(slovene korean english mandarin)
MODELS=(tabpfn realmlp)

N_LANG=${#LANGUAGES[@]}   # 4
LANGUAGE=${LANGUAGES[$(( SLURM_ARRAY_TASK_ID % N_LANG ))]}
MODEL=${MODELS[$(( SLURM_ARRAY_TASK_ID / N_LANG ))]}

# Repo root = submit dir (works with `sbatch` from the repo).
WORKDIR=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}
SIF=${SIF:-${WORKDIR}/container.sif}   # build with: apptainer build container.sif container.def
# All jobs write into the SAME parent dir; each creates its own timestamped subdir
OUTDIR=${WORKDIR}/comprehensive_results/slurm_${SLURM_ARRAY_JOB_ID}

echo "================================================================"
echo "Job ${SLURM_ARRAY_TASK_ID}: language=${LANGUAGE}  model=${MODEL}"
echo "Output parent: ${OUTDIR}"
echo "================================================================"

mkdir -p ${WORKDIR}/logs ${OUTDIR}

# Embeddings for google/embeddinggemma-300m are auto-loaded from
# data/google_embeddinggemma-300m_<dataset>.npy (shipped in this repo); if the
# cache is absent they are computed on first run and saved there.
singularity exec --nv \
    --bind ${WORKDIR}:${WORKDIR} \
    --pwd  ${WORKDIR} \
    ${SIF} \
    python comprehensive_evaluation.py \
        --embedding google/embeddinggemma-300m \
        --device cuda \
        --models  ${MODEL} \
        --n_runs  1 \
        --languages ${LANGUAGE} \
        --output_dir ${OUTDIR}
