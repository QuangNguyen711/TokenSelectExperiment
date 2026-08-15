#!/bin/bash
# Breakdown thoi gian prefill: TokenSelect goc vs SCR (sim 0.95, max 4096, no balance)
# Ca hai deu TAT decode selection cache.
set -u

N=${N:-8}
DS=${DS:-kv_retrieval}
ROOT=${ROOT:-result_prefill_breakdown}

# $1=ten  $2=use_dynamic_chunking  $3=sim  $4=max_chunk
write_cfg() {
cat > "/tmp/pb-$1.yaml" <<EOF
model:
  type: token-retrieval
  path: Qwen/Qwen2-7B-Instruct
  rope_base: 1000000
  rope_scale: 1
  n_init: 128
  n_local: 512
  top_k: 8192
  max_n_tokens: 1048576
  adaptive_topk: false
  attention_threshold: 0.9
  l2_norm_pooling: false
  weighted_soft_vote: false
  union_of_sets: false
  dynamic_capacity_union: false
  head_wise_adaptive: false
  dcu_energy_mode: "both"
  prefill_chunk_size: 512
  sim_threshold: $3
  max_dynamic_chunk: $4
  use_dynamic_chunking: $2
  dynamic_budget_balancing: false
  use_cumsum_adaptive: false
  use_hybrid_adaptive: false
  cumsum_threshold: 0.99
  n_tail: 0
  ppl_mode: sum
  query_cache: false
  query_cache_sim_threshold: 0.9

max_len: 1048576
chunk_size: 8192
conv_type: qwen
truncation: suffix
dtype: bfloat16
EOF
}

# $1=ten  $2=profile(on/off)
run() {
    local name=$1 prof=$2
    local out="${ROOT}/${name}-${prof}"
    rm -rf "$out"; mkdir -p "$out"
    echo ">>> ${name} (profiler ${prof})"
    if [ "$prof" = "on" ]; then
        export PREFILL_PROFILE_OUT="$(readlink -f "$out")/breakdown.json"
    else
        unset PREFILL_PROFILE_OUT
    fi
    CUDA_VISIBLE_DEVICES=0 python benchmark/pred.py \
        --config_path "/tmp/pb-${name}.yaml" \
        --output_dir_path "$out" \
        --datasets "$DS" --world_size 1 --rank 0 --num_samples "$N" \
        > "$out/run.log" 2>&1
    unset PREFILL_PROFILE_OUT
}

mkdir -p "$ROOT"
write_cfg baseline "false" 0.95 1024
write_cfg scr      "true"  0.95 4096

for name in baseline scr; do
    run "$name" off
    run "$name" on
done
echo "XONG-BREAKDOWN"
