#!/bin/sh
source scripts/run_experiments.sh

for THRESHOLD in 0.95 0.97 0.99 0.999; do
    echo "========================================"
    echo "RUNNING THRESHOLD: $THRESHOLD"
    echo "========================================"
    # Args: name l2 weight union top_k dcu adaptive energy pchunk sim_thresh max_chunk use_dyn budget_bal
    run_experiment "search-thresh-${THRESHOLD}" \
        "false" "false" "false" 8192 "false" "false" "both" \
        512 $THRESHOLD 4096 "true" "false"
done