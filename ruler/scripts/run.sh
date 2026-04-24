#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# container: docker.io/cphsieh/ruler:0.1.0
# bash run.sh MODEL_NAME BENCHMARK_NAME CONFIG_NAME PORT [NUM_SAMPLES] [RANDOM_SEED]

if [ $# -lt 4 ] || [ $# -gt 6 ]; then
    echo "Usage: $0 <model_name> <benchmark_name> <config_name> <port> [num_samples] [random_seed]"
    exit 1
fi

# Resolve python interpreter with preference for this repository's virtualenv.
if [ -x "../.venv/bin/python" ]; then
    PYTHON_BIN="../.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
else
    echo "Error: No Python interpreter found. Please install python3 or add python to PATH."
    exit 127
fi

echo "Using Python interpreter: ${PYTHON_BIN}"

USE_EXISTING_SERVER=${USE_EXISTING_SERVER:-0}
SERVER_PID=""
SILENT_SERVER_LOGS=${SILENT_SERVER_LOGS:-1}
SERVER_LOG_FILE=""

cleanup_server() {
    if [ -n "${SERVER_PID}" ]; then
        echo "Stopping server process ${SERVER_PID}..."
        kill "${SERVER_PID}" >/dev/null 2>&1 || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}

trap cleanup_server EXIT INT TERM

# Root Directories
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
ROOT_DIR="${REPO_ROOT}/result/ruler" # the path that stores generated task samples and model predictions.
ENGINE_DIR="." # the path that contains individual engine folders from TensorRT-LLM.

# Model and Tokenizer
source scripts/config_models.sh
if [ -n "${SEQ_LENGTHS_OVERRIDE}" ]; then
    IFS=',' read -r -a SEQ_LENGTHS <<< "${SEQ_LENGTHS_OVERRIDE}"
fi

for seq_len in "${SEQ_LENGTHS[@]}"; do
    if ! [[ "${seq_len}" =~ ^[0-9]+$ ]]; then
        echo "Error: invalid seq length in SEQ_LENGTHS_OVERRIDE: ${seq_len}"
        exit 1
    fi
done

MODEL_NAME=${1}
MODEL_CONFIG=$(MODEL_SELECT ${MODEL_NAME} ${MODEL_DIR} ${ENGINE_DIR})
IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE OPENAI_API_KEY GEMINI_API_KEY AZURE_ID AZURE_SECRET AZURE_ENDPOINT <<< "$MODEL_CONFIG"
if [ -z "${MODEL_PATH}" ]; then
    echo "Model: ${MODEL_NAME} is not supported"
    exit 1
fi

export OPENAI_API_KEY=${OPENAI_API_KEY}
export GEMINI_API_KEY=${GEMINI_API_KEY}
export AZURE_API_ID=${AZURE_ID}
export AZURE_API_SECRET=${AZURE_SECRET}
export AZURE_API_ENDPOINT=${AZURE_ENDPOINT}

# Benchmark and Tasks
source scripts/config_tasks.sh
BENCHMARK=${2}
declare -n TASKS=$BENCHMARK

if [ -n "${TASKS_OVERRIDE}" ]; then
    IFS=',' read -r -a TASKS <<< "${TASKS_OVERRIDE}"
fi

echo $TASKS
if [ -z "${TASKS}" ]; then
    echo "Benchmark: ${BENCHMARK} is not supported"
    exit 1
fi

CONFIG_FILE=${3}
SERVER_PORT=${4}
NUM_SAMPLES_OVERRIDE=${5:-$NUM_SAMPLES}
RANDOM_SEED=${6:-42}
BATCH_SIZE=1
GPUS=${GPUS:-1}
CONFIG_PATH="../config/${CONFIG_FILE}.yaml"
OUTPUT_NAME=${EXPERIMENT_NAME:-${CONFIG_FILE}}

if ! [[ "${NUM_SAMPLES_OVERRIDE}" =~ ^[0-9]+$ ]]; then
    echo "Error: num_samples must be an integer. Got: ${NUM_SAMPLES_OVERRIDE}"
    exit 1
fi

if ! [[ "${RANDOM_SEED}" =~ ^[0-9]+$ ]]; then
    echo "Error: random_seed must be an integer. Got: ${RANDOM_SEED}"
    exit 1
fi

YAML_CONTEXT_LENGTH=0
if [ -f "${CONFIG_PATH}" ]; then
    YAML_CONTEXT_LENGTH=$(${PYTHON_BIN} -c "import yaml,sys; d=yaml.safe_load(open(sys.argv[1])) or {}; m=d.get('model') or {}; v=m.get('max_n_tokens') or d.get('max_len') or 0; print(int(v) if v else 0)" "${CONFIG_PATH}" 2>/dev/null || echo 0)
fi

MAX_CONTEXT_LENGTH=${CONTEXT_LENGTH:-0}
if [ "${MAX_CONTEXT_LENGTH}" -le 0 ]; then
    if [ "${YAML_CONTEXT_LENGTH}" -gt 0 ]; then
        MAX_CONTEXT_LENGTH=${YAML_CONTEXT_LENGTH}
    else
        for seq_len in "${SEQ_LENGTHS[@]}"; do
            if [ "${seq_len}" -gt "${MAX_CONTEXT_LENGTH}" ]; then
                MAX_CONTEXT_LENGTH=${seq_len}
            fi
        done
    fi
fi

echo "Benchmark config: output_name=${OUTPUT_NAME}, num_samples=${NUM_SAMPLES_OVERRIDE}, random_seed=${RANDOM_SEED}, yaml_context_length=${YAML_CONTEXT_LENGTH}, max_context_length=${MAX_CONTEXT_LENGTH}"
echo "Sequence lengths: ${SEQ_LENGTHS[*]}"
echo "Tasks: ${TASKS[*]}"
SKIP_TASKS=${SKIP_TASKS:-}

should_skip_task() {
    local task_name="$1"
    if [ -z "${SKIP_TASKS}" ]; then
        return 1
    fi
    IFS=',' read -r -a skip_list <<< "${SKIP_TASKS}"
    for skip_task in "${skip_list[@]}"; do
        if [ "${task_name}" == "${skip_task}" ]; then
            return 0
        fi
    done
    return 1
}

pkill sft_lr || true

wait_for_server_ready() {
    local server_port="$1"
    local max_retries="${2:-60}"
    local sleep_sec="${3:-2}"
    local i=1
    while [ "$i" -le "$max_retries" ]; do
        if curl -fsS "http://127.0.0.1:${server_port}/get_model_info" >/dev/null 2>&1; then
            return 0
        fi
        sleep "$sleep_sec"
        i=$((i + 1))
    done
    return 1
}

# Start server (you may want to run in other container.)
if [ "${USE_EXISTING_SERVER}" == "1" ]; then
    echo "USE_EXISTING_SERVER=1, skip launching local server."
    if ! wait_for_server_ready "${SERVER_PORT}" 10 1; then
        echo "[error] existing server is not reachable at 127.0.0.1:${SERVER_PORT}"
        exit 1
    fi
else
    pkill -f "pred/serve_sglang.py" || true
    pkill -f "sglang.srt.server" || true

    if [ "${SILENT_SERVER_LOGS}" == "1" ]; then
        SERVER_LOG_DIR=${SERVER_LOG_DIR:-"${REPO_ROOT}/result/ruler/logs"}
        mkdir -p "${SERVER_LOG_DIR}"
        SERVER_LOG_FILE=${SERVER_LOG_FILE:-"${SERVER_LOG_DIR}/server_${MODEL_NAME}_${CONFIG_FILE}_${SERVER_PORT}_$(date +%Y%m%d_%H%M%S).log"}
        echo "Server logs are redirected to: ${SERVER_LOG_FILE}"
    fi

    if [ "$MODEL_FRAMEWORK" == "vllm" ]; then
        if [ "${SILENT_SERVER_LOGS}" == "1" ]; then
            ${PYTHON_BIN} pred/serve_vllm.py \
                --model=${MODEL_PATH} \
                --tensor-parallel-size=${GPUS} \
                --dtype bfloat16 \
                --disable-custom-all-reduce \
                >"${SERVER_LOG_FILE}" 2>&1 &
        else
            ${PYTHON_BIN} pred/serve_vllm.py \
                --model=${MODEL_PATH} \
                --tensor-parallel-size=${GPUS} \
                --dtype bfloat16 \
                --disable-custom-all-reduce \
                &
        fi
        SERVER_PID=$!

    elif [ "$MODEL_FRAMEWORK" == "trtllm" ]; then
        if [ "${SILENT_SERVER_LOGS}" == "1" ]; then
            ${PYTHON_BIN} pred/serve_trt.py \
                --model_path=${MODEL_PATH} \
                >"${SERVER_LOG_FILE}" 2>&1 &
        else
            ${PYTHON_BIN} pred/serve_trt.py \
                --model_path=${MODEL_PATH} \
                &
        fi
        SERVER_PID=$!

    elif [ "$MODEL_FRAMEWORK" == "sglang" ]; then
        if [ "${SILENT_SERVER_LOGS}" == "1" ]; then
            ${PYTHON_BIN} pred/serve_sglang.py \
                --model-path ${MODEL_PATH} \
                --dp ${GPUS} \
                --port ${SERVER_PORT} \
                --context-length ${MAX_CONTEXT_LENGTH} \
                --disable-cuda-graph \
                --mem-fraction-static 0.8 \
                --sgl-conf-file ../config/${CONFIG_FILE}.yaml \
                >"${SERVER_LOG_FILE}" 2>&1 &
        else
            ${PYTHON_BIN} pred/serve_sglang.py \
                --model-path ${MODEL_PATH} \
                --dp ${GPUS} \
                --port ${SERVER_PORT} \
                --context-length ${MAX_CONTEXT_LENGTH} \
                --disable-cuda-graph \
                --mem-fraction-static 0.8 \
                --sgl-conf-file ../config/${CONFIG_FILE}.yaml \
                &
        fi
        SERVER_PID=$!
        # use sglang/test/killall_sglang.sh to kill sglang server if it hangs
    fi

    if ! wait_for_server_ready "${SERVER_PORT}" 120 2; then
        echo "[error] launched server is not ready at 127.0.0.1:${SERVER_PORT}"
        exit 1
    fi
fi

echo "processing data"
# Start client (prepare data / call model API / obtain final metrics)
total_time=0
for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    echo $MAX_SEQ_LENGTH
    DATA_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}/data"
    RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}/${OUTPUT_NAME}"
    PRED_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}/${OUTPUT_NAME}/pred"
    
    mkdir -p ${DATA_DIR}
    mkdir -p ${PRED_DIR}
    
    for TASK in "${TASKS[@]}"; do
        if should_skip_task "${TASK}"; then
            echo "[skip] task=${TASK} (configured via SKIP_TASKS)"
            continue
        fi

        echo "[prepare] task=${TASK} max_seq_len=${MAX_SEQ_LENGTH}"
        ${PYTHON_BIN} data/prepare.py \
            --save_dir ${DATA_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --tokenizer_path ${TOKENIZER_PATH} \
            --tokenizer_type ${TOKENIZER_TYPE} \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --model_template_type ${MODEL_TEMPLATE_TYPE} \
            --num_samples ${NUM_SAMPLES_OVERRIDE} \
            --random_seed ${RANDOM_SEED} \
            ${REMOVE_NEWLINE_TAB}
        if [ $? -ne 0 ]; then
            echo "[error] prepare failed for task=${TASK}, stop benchmark."
            exit 1
        fi
        
        pkill sft_lr
        echo "[infer] task=${TASK} max_seq_len=${MAX_SEQ_LENGTH}"
        if ! wait_for_server_ready "${SERVER_PORT}" 5 1; then
            echo "[error] server is not reachable before inference for task=${TASK}"
            exit 1
        fi
        start_time=$(date +%s)
        ${PYTHON_BIN} pred/call_api.py \
            --data_dir ${DATA_DIR} \
            --save_dir ${PRED_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --server_type ${MODEL_FRAMEWORK} \
            --server_port ${SERVER_PORT} \
            --model_name_or_path ${MODEL_PATH} \
            --temperature ${TEMPERATURE} \
            --top_k ${TOP_K} \
            --top_p ${TOP_P} \
            --random_seed ${RANDOM_SEED} \
            --batch_size ${BATCH_SIZE} \
            --threads ${GPUS} \
            ${STOP_WORDS}
        if [ $? -ne 0 ]; then
            echo "[error] inference failed for task=${TASK}, stop benchmark."
            exit 1
        fi
        end_time=$(date +%s)
        time_diff=$((end_time - start_time))
        total_time=$((total_time + time_diff))
    done
    
    ${PYTHON_BIN} eval/evaluate.py \
        --data_dir ${PRED_DIR} \
        --benchmark ${BENCHMARK}
done

echo "Total time spent on call_api: $total_time seconds"