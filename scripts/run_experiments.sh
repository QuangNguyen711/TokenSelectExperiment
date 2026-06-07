# File: scripts/run_experiments.sh

#!/bin/bash

config_path="config/qwen-token-retrieval.yaml"
world_size=1
datasets="kv_retrieval,longdialogue_qa_eng,math_find,code_debug,passkey,number_string"

run_experiment() {
    local exp_name=$1
    local use_shannon=$2
    local use_cumsum=$3
    local use_hybrid=$4
    local thresh=$5

    local output_dir="result_release_ttft/infinitbench/qwen-${exp_name}"

    # ==========================================================
    # SKIP NẾU ĐÃ CÓ KẾT QUẢ
    # ==========================================================
    if [ -d "$output_dir" ]; then
        echo "⏭️  SKIP: $exp_name"
        echo "    Found existing directory: $output_dir"
        echo
        return 0
    fi

    export CURRENT_EXP=$exp_name

    # ==========================================================
    # GHI ĐÈ CONFIG
    # ==========================================================
    cat << EOF > "$config_path"
model:
  type: token-retrieval
  path: Qwen/Qwen2-7B-Instruct
  rope_base: 1000000
  rope_scale: 1
  n_init: 128
  n_local: 512
  top_k: 8192
  max_n_tokens: 1048576

  adaptive_topk: $use_shannon
  use_cumsum_adaptive: $use_cumsum
  use_hybrid_adaptive: $use_hybrid
  cumsum_threshold: $thresh

  attention_threshold: 0.9
  l2_norm_pooling: false
  weighted_soft_vote: false
  union_of_sets: false
  dynamic_capacity_union: false
  head_wise_adaptive: false
  dcu_energy_mode: "both"
  prefill_chunk_size: 512
  sim_threshold: 0.95
  max_dynamic_chunk: 1024
  use_dynamic_chunking: true
  dynamic_budget_balancing: true

max_len: 1048576
chunk_size: 8192
conv_type: qwen
truncation: suffix
dtype: bfloat16
EOF

    # ==========================================================
    # DỌN PROCESS CŨ
    # ==========================================================
    pkill -f pt_main_thread 2>/dev/null
    sleep 2

    echo "=================================================================="
    echo "🚀 BẮT ĐẦU CHẠY THỰC NGHIỆM: $exp_name"
    echo "🚀 Threshold: $thresh"
    echo "🚀 Output: $output_dir"
    echo "=================================================================="

    bash scripts/multiprocessing-benchmark.sh \
        --config_path "$config_path" \
        --datasets "$datasets" \
        --output_dir_path "$output_dir" \
        --world_size "$world_size"

    echo
    echo "📊 Evaluating..."
    python benchmark/infinitebench_eval.py --result-dir "$output_dir"

    echo
    echo "✅ Finished: $exp_name"
    echo
}

# ==========================================================
# THRESHOLDS
# ==========================================================

THRESHOLDS=(0.95 0.97 0.99)

for thresh in "${THRESHOLDS[@]}"; do

    # ------------------------------------------------------
    # PURE CUMSUM
    # ------------------------------------------------------
    run_experiment \
        "search-thresh-${thresh}-pure-cumsum" \
        "false" \
        "true" \
        "false" \
        "$thresh"

    # ------------------------------------------------------
    # HYBRID
    # ------------------------------------------------------
    run_experiment \
        "search-thresh-${thresh}-hybrid" \
        "false" \
        "false" \
        "true" \
        "$thresh"

done

echo
echo "=========================================================="
echo "🎉 TẤT CẢ THỰC NGHIỆM ĐÃ HOÀN THÀNH!"
echo "=========================================================="