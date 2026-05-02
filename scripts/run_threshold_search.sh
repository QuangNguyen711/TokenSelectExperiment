# File: scripts/run_threshold_search.sh
#!/bin/sh

# 1. Nạp (source) file experiments để lấy định nghĩa hàm run_experiment
source scripts/run_experiments.sh

for THRESHOLD in 0.95 0.97 0.99 0.999; do
    echo "========================================"
    echo "RUNNING THRESHOLD: $THRESHOLD"
    echo "========================================"
    
    # 2. Gọi thẳng hàm thay vì dùng lệnh "bash"
    run_experiment "search-thresh-${THRESHOLD}" "false" "false" "false" 8192 "false" "false" "both" 512 $THRESHOLD 4096 "true" "true"
    
done