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
    # Tham số thứ 9 (p_chunk_size) ta không cần nữa vì code giờ tự tính chunk rồi

    local output_dir="result_release/infinitbench/qwen-${exp_name}"
    export CURRENT_EXP=$exp_name 

    # GHI ĐÈ FILE CONFIG VỚI NGƯỠNG ALPHA
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

max_len: 1048576
chunk_size: 8192
conv_type: qwen
truncation: suffix
dtype: bfloat16
EOF

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
# GIAI ĐOẠN 2: CHẠY THÍ NGHIỆM ĐÁNH GIÁ (EVALUATION)
# ==============================================================================

# 1. ĐÃ KHẢO SÁT XONG (Tắt baseline đi)
# run_experiment "token-retrieval"     "false" "false" "false" 8192 "false" "false" "both"

# 2. CHẠY KỊCH BẢN: TOKENSELECT THUẦN (Baseline) + DYNAMIC CHUNKING (EV-DC)
run_experiment "tokenselect-dynamic-0.95" "false" "false" "false" 8192 "false" "false" "both"

# Nếu bạn muốn test thêm các thuật toán khác để so sánh trong paper:
# run_experiment "weighted-soft-dynamic" "false" "true"  "false" 8192 "false" "false" "both"