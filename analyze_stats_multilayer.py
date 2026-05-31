# File: analyze_stats_multilayer.py

import json
import glob
import numpy as np
from collections import defaultdict, Counter
import os

def get_target_layers(available_layers, num_points=5):
    """Chia đều mảng layer thành 5 điểm mốc."""
    layers_int = sorted([int(l) for l in available_layers])
    if len(layers_int) <= num_points:
        return [str(l) for l in layers_int]
    
    indices = np.linspace(0, len(layers_int) - 1, num_points, dtype=int)
    return [str(layers_int[i]) for i in indices]

def print_layer_stats(name, data_dict, target_layers):
    print(f"\n### {name}\n")
    print("| Layer | Mean | Min | Max | P10 | P50 (Med) | P90 |")
    print("|-------|------|-----|-----|-----|-----------|-----|")
    
    for layer in target_layers:
        if layer not in data_dict or not data_dict[layer]:
            continue
            
        arr = np.array(data_dict[layer])
        print(f"| Layer {layer} | {np.mean(arr):.4f} | {np.min(arr):.4f} | {np.max(arr):.4f} | "
              f"{np.percentile(arr, 10):.4f} | {np.percentile(arr, 50):.4f} | {np.percentile(arr, 90):.4f} |")

def print_block_counts(name, data_dict, target_layers, base_chunk=512):
    """Đếm số lượng chunk chi tiết từ 1 đến 8 blocks in dạng Markdown."""
    print(f"\n### {name} (1 Block = {base_chunk} tokens)\n")
    print("| Layer | 1 Blk | 2 Blks | 3 Blks | 4 Blks | 5 Blks | 6 Blks | 7 Blks | 8 Blks | Total |")
    print("|-------|-------|--------|--------|--------|--------|--------|--------|--------|-------|")

    for layer in target_layers:
        if layer not in data_dict or not data_dict[layer]:
            continue

        arr = np.array(data_dict[layer])
        # Làm tròn lên để tính số block (vd: 512->1, 1024->2, 4096->8)
        blocks = np.ceil(arr / base_chunk).astype(int)
        counts = Counter(blocks)
        total = len(blocks)

        if total == 0: 
            continue

        def get_str(c_val):
            if c_val == 0:
                return "0"
            pct = (c_val / total * 100)
            return f"{c_val} ({pct:.1f}%)"

        # Lấy value cho từng mốc từ 1 đến 8
        c1 = counts.get(1, 0)
        c2 = counts.get(2, 0)
        c3 = counts.get(3, 0)
        c4 = counts.get(4, 0)
        c5 = counts.get(5, 0)
        c6 = counts.get(6, 0)
        c7 = counts.get(7, 0)
        c8 = sum(counts.get(i, 0) for i in range(8, max(blocks.max() + 1, 9)))

        row = f"| Layer {layer} | {get_str(c1)} | {get_str(c2)} | {get_str(c3)} | {get_str(c4)} | {get_str(c5)} | {get_str(c6)} | {get_str(c7)} | {get_str(c8)} | **{total}** |"
        print(row)


output_dir_base = "result_release_ttft/infinitbench"
THRESHOLDS = ["0.95", "0.97", "0.99", "0.999"]

for thresh in THRESHOLDS:
    target_dir = os.path.join(output_dir_base, f"qwen-search-thresh-{thresh}")
    stat_files = glob.glob(os.path.join(target_dir, "*_stats.jsonl")) 
    
    if not stat_files:
        continue
        
    print(f"\n---\n")
    print(f"## 🚀 KẾT QUẢ CHO THRESHOLD: `{thresh}`\n")
    
    merged_consecutive = defaultdict(list)
    merged_lengths = defaultdict(list)
    merged_first_vs_mean = defaultdict(list)
    
    for fpath in stat_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                layer_str = str(data["layer"])
                
                # 1. Consecutive Sims
                merged_consecutive[layer_str].extend(data.get("consecutive_sims", []))
                
                # 2. Lấy ra lengths và first_vs_mean
                c_lengths = data.get("chunk_lengths", [])
                fvm_sims = data.get("first_vs_mean_sims", [])
                
                filtered_lengths = []
                filtered_fvm = []
                
                # LỌC BỎ CHUNK DƯ ( < 512 )
                for i in range(len(c_lengths)):
                    if c_lengths[i] >= 512:
                        filtered_lengths.append(c_lengths[i])
                        if i < len(fvm_sims):
                            filtered_fvm.append(fvm_sims[i])
                
                merged_lengths[layer_str].extend(filtered_lengths)
                merged_first_vs_mean[layer_str].extend(filtered_fvm)
                
    if not merged_consecutive:
        continue
        
    # Lấy 5 layer đại diện
    target_layers = get_target_layers(list(merged_consecutive.keys()), num_points=5)
    print(f"*📌 Đang theo dõi 5 mốc Layers: {target_layers}*")
    
    print_layer_stats("1. Consecutive Token Similarity", merged_consecutive, target_layers)
    print_layer_stats("2. Dynamic Chunk Lengths (Đã bỏ qua chunk < 512)", merged_lengths, target_layers)
    print_block_counts("2.1 Tỷ lệ gộp Chunk (Frequency)", merged_lengths, target_layers)
    print_layer_stats("3. First Token vs Chunk Mean Similarity", merged_first_vs_mean, target_layers)