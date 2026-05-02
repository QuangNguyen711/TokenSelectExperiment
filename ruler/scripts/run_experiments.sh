#!/bin/bash

set -euo pipefail

# Run from repository root: /data/TokenSelectExperiment/ruler
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RULER_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd "${RULER_ROOT}/.." && pwd)

CONFIG_NAME=${CONFIG_NAME:-qwen-token-retrieval}
CONFIG_PATH="${REPO_ROOT}/config/${CONFIG_NAME}.yaml"

if [ ! -f "${CONFIG_PATH}" ]; then
  echo "Error: config file not found: ${CONFIG_PATH}"
  exit 1
fi

CONFIG_BACKUP=$(mktemp)
cp "${CONFIG_PATH}" "${CONFIG_BACKUP}"
restore_config() {
  cp "${CONFIG_BACKUP}" "${CONFIG_PATH}" || true
  rm -f "${CONFIG_BACKUP}" || true
}
trap restore_config EXIT INT TERM

MODEL_NAME=${MODEL_NAME:-qwen2-7b-inst}
SERVER_PORT=${SERVER_PORT:-63333}
NUM_SAMPLES=${NUM_SAMPLES:-100}
RANDOM_SEED=${RANDOM_SEED:-42}
USE_EXISTING_SERVER=${USE_EXISTING_SERVER:-0}

# Optional filter for running only selected experiment names.
# Example: EXPERIMENTS_FILTER="quick-smoke,full-baseline" bash scripts/run_experiments.sh
EXPERIMENTS_FILTER=${EXPERIMENTS_FILTER:-}

contains_csv_item() {
  local csv="$1"
  local item="$2"
  if [ -z "${csv}" ]; then
    return 0
  fi
  IFS=',' read -r -a list <<< "${csv}"
  for v in "${list[@]}"; do
    if [ "${v}" = "${item}" ]; then
      return 0
    fi
  done
  return 1
}

# key1=val1,key2=val2 parser for per-experiment config overrides.
apply_config_overrides() {
  local overrides="$1"

  # Defaults aligned with config/qwen-token-retrieval.yaml
  local rope_base="1000000"
  local rope_scale="1"
  local n_init="128"
  local n_local="512"
  local top_k="8192"
  local max_n_tokens="1048576"
  local adaptive_topk="false"
  local attention_threshold="0.9"
  local l2_norm_pooling="false"
  local weighted_soft_vote="false"
  local union_of_sets="false"
  local dynamic_capacity_union="false"
  local head_wise_adaptive="false"
  local dcu_energy_mode="both"
  local prefill_chunk_size="512"
  local sim_threshold="0.95"
  local max_dynamic_chunk="512"
  local use_dynamic_chunking="false"
  local dynamic_budget_balancing="false"
  local max_len="1048576"
  local chunk_size="8192"
  local conv_type="qwen"
  local truncation="suffix"
  local dtype="bfloat16"

  if [ -n "${overrides}" ]; then
    IFS=',' read -r -a kvs <<< "${overrides}"
    for kv in "${kvs[@]}"; do
      key="${kv%%=*}"
      value="${kv#*=}"
      case "${key}" in
        rope_base) rope_base="${value}" ;;
        rope_scale) rope_scale="${value}" ;;
        n_init) n_init="${value}" ;;
        n_local) n_local="${value}" ;;
        top_k) top_k="${value}" ;;
        max_n_tokens) max_n_tokens="${value}" ;;
        adaptive_topk) adaptive_topk="${value}" ;;
        attention_threshold) attention_threshold="${value}" ;;
        l2_norm_pooling) l2_norm_pooling="${value}" ;;
        weighted_soft_vote) weighted_soft_vote="${value}" ;;
        union_of_sets) union_of_sets="${value}" ;;
        dynamic_capacity_union) dynamic_capacity_union="${value}" ;;
        head_wise_adaptive) head_wise_adaptive="${value}" ;;
        dcu_energy_mode) dcu_energy_mode="${value}" ;;
        prefill_chunk_size) prefill_chunk_size="${value}" ;;
        sim_threshold) sim_threshold="${value}" ;;
        max_dynamic_chunk) max_dynamic_chunk="${value}" ;;
        use_dynamic_chunking) use_dynamic_chunking="${value}" ;;
        dynamic_budget_balancing) dynamic_budget_balancing="${value}" ;;
        max_len) max_len="${value}" ;;
        chunk_size) chunk_size="${value}" ;;
        conv_type) conv_type="${value}" ;;
        truncation) truncation="${value}" ;;
        dtype) dtype="${value}" ;;
        "") ;;
        *)
          echo "[warn] Unknown config override key: ${key}"
          ;;
      esac
    done
  fi

  cat > "${CONFIG_PATH}" << EOF
model:
  type: token-retrieval
  path: Qwen/Qwen2-7B-Instruct
  rope_base: ${rope_base}
  rope_scale: ${rope_scale}
  n_init: ${n_init}
  n_local: ${n_local}
  top_k: ${top_k}
  max_n_tokens: ${max_n_tokens}
  adaptive_topk: ${adaptive_topk}
  attention_threshold: ${attention_threshold}
  l2_norm_pooling: ${l2_norm_pooling}
  weighted_soft_vote: ${weighted_soft_vote}
  union_of_sets: ${union_of_sets}
  dynamic_capacity_union: ${dynamic_capacity_union}
  head_wise_adaptive: ${head_wise_adaptive}
  dcu_energy_mode: "${dcu_energy_mode}"
  prefill_chunk_size: ${prefill_chunk_size}
  sim_threshold: ${sim_threshold}
  max_dynamic_chunk: ${max_dynamic_chunk}
  use_dynamic_chunking: ${use_dynamic_chunking}
  dynamic_budget_balancing: ${dynamic_budget_balancing}

max_len: ${max_len}
chunk_size: ${chunk_size}
conv_type: ${conv_type}
truncation: ${truncation}
dtype: ${dtype}
EOF
}

# Like InfiniteBench style:
# run_experiment <name> <benchmark> <seq_lengths_csv> <tasks_csv> <config_overrides_csv> [num_samples] [random_seed]
# config_overrides example:
#   "top_k=4096,prefill_chunk_size=1024,use_dynamic_chunking=true,max_dynamic_chunk=2048,dynamic_budget_balancing=true"
run_experiment() {
  local exp_name="$1"
  local benchmark="$2"
  local seq_list="$3"
  local task_list="$4"
  local config_overrides="$5"
  local samples="${6:-${NUM_SAMPLES}}"
  local seed="${7:-${RANDOM_SEED}}"

  if ! contains_csv_item "${EXPERIMENTS_FILTER}" "${exp_name}"; then
    return 0
  fi

  apply_config_overrides "${config_overrides}"

  echo "============================================================"
  echo "Running experiment: ${exp_name}"
  echo "model=${MODEL_NAME}, benchmark=${benchmark}, config=${CONFIG_NAME}, samples=${samples}, seed=${seed}"
  echo "seq_lengths=${seq_list:-<default>}, tasks=${task_list:-<default>}"
  echo "config_overrides=${config_overrides:-<default>}"
  echo "hosting_mode=$( [ "${USE_EXISTING_SERVER}" = "1" ] && echo "existing-server" || echo "self-hosted" )"
  echo "============================================================"

  USE_EXISTING_SERVER="${USE_EXISTING_SERVER}" \
  EXPERIMENT_NAME="${exp_name}" \
  SEQ_LENGTHS_OVERRIDE="${seq_list}" \
  TASKS_OVERRIDE="${task_list}" \
  bash "${RULER_ROOT}/scripts/run.sh" "${MODEL_NAME}" "${benchmark}" "${CONFIG_NAME}" "${SERVER_PORT}" "${samples}" "${seed}"
}

# ==============================================================================
# EXPERIMENT SCENARIOS (edit freely)
# ==============================================================================

# Full baseline run
# run_experiment "token-select-baseline-test" "synthetic" \
#   "131072,65536,32768,16384,8192,4096" \
#   "niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery" \
#   "top_k=8192,prefill_chunk_size=512,use_dynamic_chunking=false" \
#   "100" "42"

# run_experiment "dynamic-chunking-1024-sim-anchor-0.95-test" "synthetic" \
#   "131072,65536,32768,16384,8192,4096" \
#   "niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery" \
#   "top_k=8192,prefill_chunk_size=512,use_dynamic_chunking=true,max_dynamic_chunk=1024,sim_threshold=0.95,dynamic_budget_balancing=true" \
#   "100" "42"

# run_experiment "dynamic-chunking-1024-sim-anchor-0.95-no-balancing-test" "synthetic" \
#   "131072,65536,32768,16384,8192,4096" \
#   "niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery" \
#   "top_k=8192,prefill_chunk_size=512,use_dynamic_chunking=true,max_dynamic_chunk=1024,sim_threshold=0.95,dynamic_budget_balancing=false" \
#   "100" "42"

# run_experiment "dynamic-chunking-4096-sim-anchor-0.95-test" "synthetic" \
#   "131072,65536,32768,16384,8192,4096" \
#   "niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery" \
#   "top_k=8192,prefill_chunk_size=512,use_dynamic_chunking=true,max_dynamic_chunk=4096,sim_threshold=0.95,dynamic_budget_balancing=true" \
#   "100" "42"

# run_experiment "dynamic-chunking-4096-sim-anchor-0.95-no-balancing-test" "synthetic" \
#   "131072,65536,32768,16384,8192,4096" \
#   "niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery" \
#   "top_k=8192,prefill_chunk_size=512,use_dynamic_chunking=true,max_dynamic_chunk=4096,sim_threshold=0.95,dynamic_budget_balancing=false" \
#   "100" "42"

run_experiment "dynamic-chunking-4096-sim-0.95-dcu-l2-norm" "synthetic" \
  "131072,65536,32768,16384,8192,4096" \
  "niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery" \
  "top_k=8192,prefill_chunk_size=512,use_dynamic_chunking=true,max_dynamic_chunk=4096,sim_threshold=0.95,dynamic_capacity_union=true,dcu_energy_mode=l2_norm" \
  "100" "42"

echo "All selected experiments finished."
