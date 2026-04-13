# File: scripts/run_experiments.sh

#!/bin/sh

config_path="config/qwen-token-retrieval.yaml"
world_size=1
datasets="kv_retrieval,longdialogue_qa_eng,math_find,code_debug,passkey,number_string"

run_experiment() {
    local exp_name=$1
    local l2_norm=$2
    local weighted_vote=$3
    local union_sets=$4
    local top_k_val=$5
    local dynamic_capacity=$6
    local head_wise_adaptive=$7
    local energy_mode=${8:-"both"}
    local p_chunk_size=${9:-512}

    local output_dir="result_release/infinitbench/qwen-${exp_name}"

    export CURRENT_EXP=$exp_name 

    # Ghi đè file config (Thêm dòng dcu_energy_mode)
    cat << EOF > $config_path
model:
  type: token-retrieval
  path: Qwen/Qwen2-7B-Instruct
  rope_base: 1000000
  rope_scale: 1
  n_init: 128
  n_local: 512
  top_k: $top_k_val
  max_n_tokens: 1048576
  adaptive_topk: false
  attention_threshold: 0.9
  l2_norm_pooling: $l2_norm
  weighted_soft_vote: $weighted_vote
  union_of_sets: $union_sets
  dynamic_capacity_union: $dynamic_capacity
  head_wise_adaptive: $head_wise_adaptive
  dcu_energy_mode: "$energy_mode"
  prefill_chunk_size: $p_chunk_size

max_len: 1048576
chunk_size: 8192
conv_type: qwen
truncation: suffix
dtype: bfloat16
EOF

    # Dọn dẹp process cũ
    pkill -f pt_main_thread
    sleep 2 

    bash scripts/multiprocessing-benchmark.sh \
        --config_path $config_path \
        --datasets $datasets \
        --output_dir_path $output_dir \
        --world_size $world_size

    python benchmark/infinitebench_eval.py --result-dir ${output_dir}
}

# ==============================================================================
# CÁC KỊCH BẢN THỬ NGHIỆM
# Cấu trúc tham số:
# run_experiment <Tên> <L2> <Weight> <Union> <TopK> <DCU> <Adaptive> <EnergyMode>
# ==============================================================================

# 3 kịch bản bóc tách để chạy kiểm chứng tốc độ và độ chính xác:
# run_experiment "chunk-256"  "false" "false" "false" 8192 "false" "false" "both" 256
# run_experiment "chunk-512"  "false" "false" "false" 8192 "false" "false" "both" 512
# run_experiment "chunk-2048" "false" "false" "false" 8192 "false" "false" "both" 2048
# run_experiment "chunk-2048-dcu-energy-both"     "false" "false" "false" 8192 "true" "false" "both" 2048