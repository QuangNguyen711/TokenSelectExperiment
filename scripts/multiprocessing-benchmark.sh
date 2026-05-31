# File: scripts/multiprocessing-benchmark.sh
#!/bin/bash

SHORT=w:,f:,d:,o:,h
LONG=world_size:,config_path:,datasets:,output_dir_path:,help

PARSED=$(getopt --options $SHORT --longoptions $LONG --name "$0" -- "$@")

if [[ $? -ne 0 ]]; then
    echo "Invalid Arguments."
    exit 2
fi

eval set -- "$PARSED"

world_size=8

while true; do
    case "$1" in
        -h|--help)
            echo "Usage: $0 [--world_size <int>] [--config_path <file>] [--datasets <dataset_name>] [--output_dir_path <dir>] [--help]"
            exit
            ;;
        -w|--world_size)
            world_size="$2"
            shift 2
            ;;
        -f|--config_path)
            config_path="$2"
            shift 2
            ;;
        -d|--datasets)
            datasets="$2"
            shift 2
            ;;
        -o|--output_dir_path)
            output_dir_path="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Programming error"
            exit 3
            ;;
    esac
done

# Read external CUDA_VISIBLE_DEVICES into an array if set
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -r -a G_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
else
    # Default to 0, 1, 2... if not set
    G_ARRAY=()
    for ((i=0; i < $world_size; ++i)); do
        G_ARRAY+=($i)
    done
fi

mkdir -p ${output_dir_path}

for ((rank=0; rank < $world_size; ++rank))
do
    TARGET_GPU=${G_ARRAY[$rank]}
    rm -rf ~/.cache/outlines
    CUDA_VISIBLE_DEVICES=${TARGET_GPU} python benchmark/pred.py \
    --config_path ${config_path} \
    --output_dir_path ${output_dir_path} \
    --datasets ${datasets} \
    --world_size ${world_size} \
    --rank ${rank} &
    echo "worker $rank started on RANK $rank"
    sleep 60
done

wait
echo done