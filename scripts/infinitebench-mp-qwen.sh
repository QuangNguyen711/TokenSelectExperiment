#!/bin/sh

config_path=config/qwen-token-retrieval.yaml
output_dir=result_release/infinitbench/qwen-adaptive-0.8-norm-L2-token-retrieval
# output_dir=result_release/infinitbench/qwen-token-retrieval

world_size=1
# datasets="passkey,number_string,kv_retrieval,longdialogue_qa_eng,math_find,code_debug"
datasets="kv_retrieval,longdialogue_qa_eng,math_find,code_debug,passkey,number_string"

pkill pt_main_thread

bash scripts/multiprocessing-benchmark.sh  \
    --config_path $config_path \
    --datasets $datasets \
    --output_dir_path $output_dir \
    --world_size $world_size

# ĐÃ BỎ MERGE.PY VÌ WORLD_SIZE=1 KHÔNG CẦN MERGE

# Chạy thẳng file eval luôn
python benchmark/infinitebench_eval.py --result-dir ${output_dir}