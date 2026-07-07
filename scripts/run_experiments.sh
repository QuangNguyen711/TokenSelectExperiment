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
    local sim_thresh=${10:-0.90}
    local max_chunk=${11:-1024}
    local use_dynamic=${12:-"false"}
    local budget_balancing=${13:-"true"}
    local use_adaptive_topk=${14:-"false"}
    local use_cumsum_adaptive=${15:-"false"}
    local use_hybrid_adaptive=${16:-"false"}
    local cumsum_threshold=${17:-0.95}
    local n_tail=${18:-2048}
    local ppl_mode=${19:-"sum"}

    local output_dir="result_release_ttft/infinitbench/qwen-${exp_name}"

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
  adaptive_topk: $use_adaptive_topk
  attention_threshold: 0.9
  l2_norm_pooling: $l2_norm
  weighted_soft_vote: $weighted_vote
  union_of_sets: $union_sets
  dynamic_capacity_union: $dynamic_capacity
  head_wise_adaptive: $head_wise_adaptive
  dcu_energy_mode: "$energy_mode"
  prefill_chunk_size: $p_chunk_size
  sim_threshold: $sim_thresh
  max_dynamic_chunk: $max_chunk
  use_dynamic_chunking: $use_dynamic
  dynamic_budget_balancing: $budget_balancing
  use_cumsum_adaptive: $use_cumsum_adaptive
  use_hybrid_adaptive: $use_hybrid_adaptive
  cumsum_threshold: $cumsum_threshold
  n_tail: $n_tail
  ppl_mode: $ppl_mode

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
# run_experiment <Tên> <L2> <Weight> <Union> <TopK> <DCU> <Adaptive> <EnergyMode> <PrefillChunk> <Sim_Threshold>
# <Max_Chunk_Size> <Use_Dynamic_Chunking> <Budget_Balancing> <Use_Adaptive_TopK> <Use_Cumsum_Adaptive> <Use_Hybrid_Adaptive> <Cumsum_Threshold> <N_Tail> <PPL_Mode>
# ==============================================================================

# Kịch bản A: TokenSelect Gốc (Tắt gộp chunk)
# run_experiment "baseline-512" "false" "false" "false" 8192 "false" "false" "both" 512 0.95 1024 "false" "true"

# # Kịch bản B: Gộp Chunk Động (Bật gộp chunk)
# run_experiment "dynamic-0.95-max-1024" "false" "false" "false" 8192 "false" "false" "both" 512 0.95 1024 "true" "true"

# # Kịch bản C: Gộp Chunk 2048 + L2Norm
# run_experiment "dynamic-0.95-max-2048-l2" "true" "false" "false" 8192 "false" "false" "both" 512 0.95 2048 "true" "true"

# Kịch bản D: Gộp Chunk 2048 + Không Bù Trừ Năng Lượng
# run_experiment "dynamic-0.95-max-2048-no-balance" "false" "false" "false" 8192 "false" "false" "both" 512 0.95 2048 "true" "false"

# run_experiment "token-retrieval" "false" "false" "false" 8192 "false" "false" "both" 512 0.95 1024 "false" "true"

# run_experiment "sim-0.95-max-1024-adaptive-15-no-balance"           "false" "false" "false" 8192 "false" "false" "both" 512 0.95 1024 "true" "false" "true" "false" "false" 0.99

# run_experiment "sim-0.95-max-2048-adaptive-15-no-balance"           "false" "false" "false" 8192 "false" "false" "both" 512 0.95 2048 "true" "false" "true" "false" "false" 0.99

# run_experiment "sim-0.95-max-4096-adaptive-15-no-balance"           "false" "false" "false" 8192 "false" "false" "both" 512 0.95 4096 "true" "false" "true" "false" "false" 0.99

# run_experiment "sim-0.97-max-1024-adaptive-15-no-balance"           "false" "false" "false" 8192 "false" "false" "both" 512 0.97 1024 "true" "false" "true" "false" "false" 0.99

# run_experiment "sim-0.97-max-2048-adaptive-15-no-balance"           "false" "false" "false" 8192 "false" "false" "both" 512 0.97 2048 "true" "false" "true" "false" "false" 0.99

# run_experiment "sim-0.97-max-4096-adaptive-15-no-balance"           "false" "false" "false" 8192 "false" "false" "both" 512 0.97 4096 "true" "false" "true" "false" "false" 0.99

# run_experiment "sim-0.99-max-1024-cumsum-0.99-no-balance"           "false" "false" "false" 8192 "false" "false" "both" 512 0.99 1024 "true" "false" "false" "true" "false" 0.99

# run_experiment "sim-0.99-max-1024-adaptive-15-no-balance-min-norm"  "false" "false" "false" 8192 "false" "false" "both" 512 0.99 1024 "true" "false" "true" "false" "false" 0.99

# run_experiment "sim-0.99-max-2048-adaptive-15-no-balance-min-norm"  "false" "false" "false" 8192 "false" "false" "both" 512 0.99 2048 "true" "false" "true" "false" "false" 0.99

# run_experiment "sim-0.99-max-4096-adaptive-15-no-balance-min-norm"  "false" "false" "false" 8192 "false" "false" "both" 512 0.99 4096 "true" "false" "true" "false" "false" 0.99

# run_experiment "sim-0.95-max-1024-no-balance"                       "false" "false" "false" 8192 "false" "false" "both" 512 0.95 1024 "true" "false" "false" "false" "false" 0.99

# run_experiment "sim-0.95-max-2048-no-balance"                       "false" "false" "false" 8192 "false" "false" "both" 512 0.95 2048 "true" "false" "false" "false" "false" 0.99

# run_experiment "sim-0.95-max-4096-no-balance"                       "false" "false" "false" 8192 "false" "false" "both" 512 0.95 4096 "true" "false" "false" "false" "false" 0.99

# run_experiment "sim-0.97-max-1024-no-balance"                       "false" "false" "false" 8192 "false" "false" "both" 512 0.97 1024 "true" "false" "false" "false" "false" 0.99

# run_experiment "sim-0.97-max-2048-no-balance"                       "false" "false" "false" 8192 "false" "false" "both" 512 0.97 2048 "true" "false" "false" "false" "false" 0.99

# run_experiment "sim-0.97-max-4096-no-balance"                       "false" "false" "false" 8192 "false" "false" "both" 512 0.97 4096 "true" "false" "false" "false" "false" 0.99

# run_experiment "sim-0.99-max-1024-no-balance"                       "false" "false" "false" 8192 "false" "false" "both" 512 0.99 1024 "true" "false" "false" "false" "false" 0.99

# run_experiment "sim-0.99-max-2048-no-balance"                       "false" "false" "false" 8192 "false" "false" "both" 512 0.99 2048 "true" "false" "false" "false" "false" 0.99

# run_experiment "sim-0.99-max-4096-no-balance"                       "false" "false" "false" 8192 "false" "false" "both" 512 0.99 4096 "true" "false" "false" "false" "false" 0.99

# run_experiment "sim-0.99-max-4096-balance"                          "false" "false" "false" 8192 "false" "false" "both" 512 0.99 4096 "true" "true" "false" "false" "false" 0.99

# # exp1: Σ exp(H_h)  (adaptive_topk)
# run_experiment "sumppl-add-4096-test"                                    "false" "false" "false" 8192 "false" "false" "both" 512 0.95 4096 "false" "false" "true"  "false" "false" 0.95

# run_experiment "sumppl-tail-4096"                                   "false" "false" "false" 8192 "false" "false" "both" 512 0.95 4096 "false" "false" "true" "false" "false" 0.95 4096 "sum"

# run_experiment "pplpost-tail-4096"                                  "false" "false" "false" 8192 "false" "false" "both" 512 0.95 4096 "false" "false" "true" "false" "false" 0.95 4096 "post"

# run_experiment "sumppl-tail-4608"                                   "false" "false" "false" 8192 "false" "false" "both" 512 0.95 4096 "false" "false" "true" "false" "false" 0.95 4608 "sum"

# run_experiment "pplpost-tail-4608"                                  "false" "false" "false" 8192 "false" "false" "both" 512 0.95 4096 "false" "false" "true" "false" "false" 0.95 4608 "post"

# run_experiment "sumppl-tail-5120"                                   "false" "false" "false" 8192 "false" "false" "both" 512 0.95 4096 "false" "false" "true" "false" "false" 0.95 5120 "sum"

# run_experiment "pplpost-tail-5120"                                  "false" "false" "false" 8192 "false" "false" "both" 512 0.95 4096 "false" "false" "true" "false" "false" 0.95 5120 "post"

# run_experiment "token-retrieval-4096"                               "false" "false" "false" 4096 "false" "false" "both" 512 0.99 4096 "false" "false" "false" "false" "false" 0.99

# run_experiment "ppl-tail-4096"                                      "false" "false" "false" 8192 "false" "true" "both" 512 0.95 4096 "false" "false" "false" "false" "false" 0.95 4096

# run_experiment "ppl-tail-0"                                         "false" "false" "false" 8192 "false" "true" "both" 512 0.95 4096 "false" "false" "false" "false" "false" 0.95 0

# run_experiment "ppl-tail-256"                                       "false" "false" "false" 8192 "false" "true" "both" 512 0.95 4096 "false" "false" "false" "false" "false" 0.95 256

# run_experiment "ppl-tail-1024"                                      "false" "false" "false" 8192 "false" "true" "both" 512 0.95 4096 "false" "false" "false" "false" "false" 0.95 1024

# run_experiment "ppl-tail-2048"                                      "false" "false" "false" 8192 "false" "true" "both" 512 0.95 4096 "false" "false" "false" "false" "false" 0.95 2048

# exp2: head-wise cumsum union θ=0.9
# run_experiment "headunion-0.9"                                      "false" "false" "false" 8192 "false" "false" "both" 512 0.95 4096 "false" "false" "false" "true"  "false" 0.9

# exp3: head-wise cumsum union θ=0.99
# run_experiment "headunion-0.99"                                     "false" "false" "false" 8192 "false" "false" "both" 512 0.95 4096 "false" "false" "false" "true"  "false" 0.99

# run_experiment "normal_testing"                                     "false" "false" "false" 8192 "false" "false" "both" 512 0.95 4096 "false" "false" "false" "false" "false" 0.95 4096 "fixed"

# run_experiment "thresh-1e7"                                         "false" "false" "false" 8192 "false" "false" "both" 512 0.95 4096 "false" "false" "true" "false" "false" 0.0000001 4096 "thresh"
# run_experiment "thresh-1e8"                                         "false" "false" "false" 8192 "false" "false" "both" 512 0.95 4096 "false" "false" "true" "false" "false" 0.00000001 4096 "thresh"
# run_experiment "thresh-1e9"                                         "false" "false" "false" 8192 "false" "false" "both" 512 0.95 4096 "false" "false" "true" "false" "false" 0.000000001 4096 "thresh"
# run_experiment "thresh-1e10"                                         "false" "false" "false" 8192 "false" "false" "both" 512 0.95 4096 "false" "false" "true" "false" "false" 0.0000000001 4096 "thresh"
# run_experiment "thresh-1e11"                                         "false" "false" "false" 8192 "false" "false" "both" 512 0.95 4096 "false" "false" "true" "false" "false" 0.00000000001 4096 "thresh"
# run_experiment "thresh-1e12"                                         "false" "false" "false" 8192 "false" "false" "both" 512 0.95 4096 "false" "false" "true" "false" "false" 0.000000000001 4096 "thresh"
# run_experiment "thresh-1e13"                                         "false" "false" "false" 8192 "false" "false" "both" 512 0.95 4096 "false" "false" "true" "false" "false" 0.0000000000001 4096 "thresh"
# run_experiment "thresh-1e14"                                         "false" "false" "false" 8192 "false" "false" "both" 512 0.95 4096 "false" "false" "true" "false" "false" 0.00000000000001 4096 "thresh"
run_experiment "thresh-1e15"                                         "false" "false" "false" 8192 "false" "false" "both" 512 0.95 4096 "false" "false" "true" "false" "false" 0.000000000000001 4096 "thresh"
run_experiment "thresh-1e16"                                         "false" "false" "false" 8192 "false" "false" "both" 512 0.95 4096 "false" "false" "true" "false" "false" 0.0000000000000001 4096 "thresh"