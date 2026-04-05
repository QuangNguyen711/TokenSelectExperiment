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

    # Xuất tên kịch bản ra biến môi trường để Python đọc
    export CURRENT_EXP=$exp_name 

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

    # Tách chuỗi datasets bằng dấu phẩy và tạo vòng lặp
    IFS=',' read -ra DATASET_ARRAY <<< "$datasets"
    
    for dataset in "${DATASET_ARRAY[@]}"; do
        echo "=========================================================="
        echo "Đang chạy Dataset: $dataset cho Kịch bản: $exp_name"
        echo "=========================================================="
        
        # Bắn tên dataset vào hệ điều hành
        export CURRENT_DATASET=$dataset

        # Dọn dẹp tiến trình cũ trước mỗi dataset cho chắc chắn
        pkill -f pt_main_thread
        sleep 2 

        # Chạy benchmark cho TỪNG dataset một
        bash scripts/multiprocessing-benchmark.sh \
            --config_path $config_path \
            --datasets $dataset \
            --output_dir_path $output_dir \
            --world_size $world_size
    done

    # Chạy eval sau khi tất cả dataset đã hoàn thành
    python benchmark/infinitebench_eval.py --result-dir ${output_dir}
}

# ==============================================================================
# CÁC KỊCH BẢN THỬ NGHIỆM
# Cấu trúc tham số:
# run_experiment <Tên> <L2-Norm> <Weighted> <Union> <Top_K> <DCU> <Head-Adaptive>
# ==============================================================================

# 2. Chạy 2 cấu hình SOTA mới (Giải quyết bài toán tối ưu đa mục tiêu)
run_experiment "qwen-union-of-sets" "false" "false" "true" 8192 "false" "false"
# run_experiment "head-wise-adaptive"     "false" "false" "false" 8192 "false" "true"