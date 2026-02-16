#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Cap-PermLLM Multi-Model Experiment Orchestration (HF backend)
# Runs baseline + ablations + full for each model
# =========================================================

KB="./finance_kb.json"
OUTROOT="runs_final_multimodel_hf"
TEMP="0.0"

# Standardize context length for fairness (set in code if you add truncation)
# NOTE: Your HF generation uses max_new_tokens=512 inside build_llm().

# MODELS=(
#   "meta-llama/Meta-Llama-3-8B-Instruct"
#   "google/gemma-3-12b-it"
#   "OpenPipe/Qwen3-14B-Instruct"
#   "mistralai/Mistral-7B-Instruct-v0.3"
# )

MODELS=(
  "meta-llama/Meta-Llama-3-8B-Instruct"
)

slugify () { echo "$1" | sed 's/\//__/g'; }

run_cfg () {
  local outdir="$1"
  local model="$2"
  local ctx="$3"
  local label="$4"
  local prov="$5"
  local two="$6"
  local taint="$7"

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128" \
bash -c 'exec -a run python agentic_ac_modular.py "$@"' _ \
  --planner_backend hf \
    --planner_model "$model" \
    --kb_path "$KB" \
    --temperature "$TEMP" \
    --outdir "$outdir" \
    --enable_context_eval "$ctx" \
    --enable_label_gate "$label" \
    --enable_cap_provenance "$prov" \
    --enable_two_key "$two" \
    --enable_taint "$taint" \
    --enable_attack_suite 1
}

mkdir -p "$OUTROOT"

for MODEL in "${MODELS[@]}"; do
  MODEL_SLUG="$(slugify "$MODEL")"
  MODEL_OUTROOT="$OUTROOT/$MODEL_SLUG"
  mkdir -p "$MODEL_OUTROOT"

  echo "===================================================="
  echo "MODEL: $MODEL"
  echo "OUT:   $MODEL_OUTROOT"
  echo "===================================================="

  echo "[run] baseline"
  run_cfg "$MODEL_OUTROOT/baseline" "$MODEL" 0 0 0 0 0

  echo "[run] label"
  run_cfg "$MODEL_OUTROOT/label" "$MODEL" 0 1 0 0 0

  echo "[run] label_prov"
  run_cfg "$MODEL_OUTROOT/label_prov" "$MODEL" 0 1 1 0 0

  echo "[run] label_prov_taint"
  run_cfg "$MODEL_OUTROOT/label_prov_taint" "$MODEL" 0 1 1 0 1

  echo "[run] label_prov_taint_two"
  run_cfg "$MODEL_OUTROOT/label_prov_taint_two" "$MODEL" 0 1 1 1 1

  echo "[run] full"
  run_cfg "$MODEL_OUTROOT/full" "$MODEL" 1 1 1 1 1

  echo "[OK] completed model: $MODEL"
done

echo "===================================================="
echo "ALL MODELS DONE. Outputs in: $OUTROOT"
echo "===================================================="
