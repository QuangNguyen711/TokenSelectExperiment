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

    local output_dir="result_release/infinitbench/qwen-${exp_name}"

    # Ghi đè file config
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

max_len: 1048576
chunk_size: 8192
conv_type: qwen
truncation: suffix
dtype: bfloat16

EOF

    # Dọn dẹp tiến trình cũ
    pkill pt_main_thread
    sleep 2 # Đợi một chút để VRAM thực sự được giải phóng

    # Chạy benchmark
    bash scripts/multiprocessing-benchmark.sh \
        --config_path $config_path \
        --datasets $datasets \
        --output_dir_path $output_dir \
        --world_size $world_size

    # Chạy eval
    python benchmark/infinitebench_eval.py --result-dir ${output_dir}
}

# ==============================================================================
# CÁC KỊCH BẢN THỬ NGHIỆM
# Cấu trúc tham số:
# run_experiment <Tên> <L2-Norm> <Weighted> <Union> <Top_K> <DCU> <Head-Adaptive>
# ==============================================================================

# 2. Chạy 2 cấu hình SOTA mới (Giải quyết bài toán tối ưu đa mục tiêu)
run_experiment "dynamic-capacity-union" "false" "false" "false" 8192 "true"  "false"
run_experiment "head-wise-adaptive"     "false" "false" "false" 8192 "false" "true"