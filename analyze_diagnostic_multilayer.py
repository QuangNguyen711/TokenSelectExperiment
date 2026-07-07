"""
v2.4: ULTIMATE EDITION
Bản hợp nhất phân tích đầy đủ 7 chỉ số Pipeline & Overlap tổng thể, 
kèm theo chức năng giải phẫu Overlap & Prob Mass theo TỪNG DATASET (Exp 9 & 10).
"""
import json
import glob
import os
import numpy as np
from collections import defaultdict


def get_target_layers(available_layers, num_points=5):
    layers_int = sorted([int(l) for l in available_layers])
    if len(layers_int) <= num_points:
        return [str(l) for l in layers_int]
    indices = np.linspace(0, len(layers_int) - 1, num_points, dtype=int)
    return [str(layers_int[i]) for i in indices]

# --- Helper Functions for Aggregation ---
def med(arr, key, scale=1.0):
    vals = [r[key] * scale for r in arr if key in r and r[key] is not None]
    return float(np.median(vals)) if vals else float("nan")

def avg(arr, key, scale=1.0):
    vals = [r[key] * scale for r in arr if key in r and r[key] is not None]
    return float(np.mean(vals)) if vals else float("nan")

def min_val(arr, key, scale=1.0):
    vals = [r[key] * scale for r in arr if key in r and r[key] is not None]
    return float(np.min(vals)) if vals else float("nan")

def max_val(arr, key, scale=1.0):
    vals = [r[key] * scale for r in arr if key in r and r[key] is not None]
    return float(np.max(vals)) if vals else float("nan")

def med_nested(arr, parent, child, scale=1.0):
    vals = [r[parent][child] * scale for r in arr if r.get(parent) and child in r[parent]]
    return float(np.median(vals)) if vals else float("nan")

def avg_nested(arr, parent, child, scale=1.0):
    vals = [r[parent][child] * scale for r in arr if r.get(parent) and child in r[parent]]
    return float(np.mean(vals)) if vals else float("nan")

def min_nested(arr, parent, child, scale=1.0):
    vals = [r[parent][child] * scale for r in arr if r.get(parent) and child in r[parent]]
    return float(np.min(vals)) if vals else float("nan")

def max_nested(arr, parent, child, scale=1.0):
    vals = [r[parent][child] * scale for r in arr if r.get(parent) and child in r[parent]]
    return float(np.max(vals)) if vals else float("nan")

# --- Print Tables ---
def print_input_cosines(merged, target_layers):
    print("\n### 1. Query similarity (input to pipeline)\n")
    print("| Layer | Cos-PRE (Med) | Cos-PRE (Min) | Cos-POST (Med) | Cos-POST (Min) | Sum-Cos (Med) | Sum-Cos (Min) | Sum-Diff (Med) | Sum-Diff (Mean) |")
    print("|-------|---------------|---------------|----------------|----------------|---------------|---------------|----------------|-----------------|")
    for layer in target_layers:
        recs = merged.get(layer, {}).get("clusters", [])
        if not recs: continue
        print(f"| Layer {layer} | "
              f"{med(recs, 'cos_pre_rope'):.4f} | {min_val(recs, 'cos_pre_rope'):.4f} | "
              f"{med(recs, 'cos_post_rope'):.4f} | {min_val(recs, 'cos_post_rope'):.4f} | "
              f"{med(recs, 'sum_cosine'):.4f} | {min_val(recs, 'sum_cosine'):.4f} | "
              f"{med(recs, 'sum_diff_norm'):.2e} | {avg(recs, 'sum_diff_norm'):.2e} |")

def print_raw_score_stats(merged, target_layers):
    print("\n### 2. Raw Score Magnitude (paged_matmul output, bfloat16)\n")
    print("| Layer | Max Abs (Med) | Max Abs (MAX) | Mean Abs (Med) | Mean Abs (Mean) | bf16 ULP (MAX) |")
    print("|-------|---------------|---------------|----------------|-----------------|----------------|")
    for layer in target_layers:
        recs = merged.get(layer, {}).get("clusters", [])
        if not recs: continue
        print(f"| Layer {layer} | "
              f"{med_nested(recs, 'raw_score', 'max_abs'):.2f} | "
              f"{max_nested(recs, 'raw_score', 'max_abs'):.2f} | "
              f"{med_nested(recs, 'raw_score', 'mean_abs'):.4f} | "
              f"{avg_nested(recs, 'raw_score', 'mean_abs'):.4f} | "
              f"{max_nested(recs, 'raw_score', 'bf16_ulp_at_max'):.2e} |")

def print_per_head_softmax(merged, target_layers):
    print("\n### 3. Per-Head Softmax Sharpness (step 5a)\n")
    print("| Layer | Ent (Med) | Ent (Mean) | Ent (Max) | Ent (Min) | MaxProb_Med (Med) | MaxProb_MAX (Max) |")
    print("|-------|-----------|------------|-----------|-----------|-------------------|-------------------|")
    for layer in target_layers:
        recs = merged.get(layer, {}).get("clusters", [])
        if not recs: continue
        print(f"| Layer {layer} | "
              f"{med(recs, 'per_head_entropy_norm_median'):.4f} | "
              f"{avg(recs, 'per_head_entropy_norm_median'):.4f} | "
              f"{max_val(recs, 'per_head_entropy_norm_median'):.4f} | "
              f"{min_val(recs, 'per_head_entropy_norm_median'):.4f} | "
              f"{med(recs, 'per_head_max_prob_median'):.4f} | "
              f"{max_val(recs, 'per_head_max_prob_max'):.4f} |")

def print_cross_head_divergence(merged, target_layers):
    print("\n### 4. Cross-Head Divergence (step 5b)\n")
    print("| Layer | UniqArgmax (Med) | UniqArgmax (Mean) | VoteTop100 (Med) | VoteTop100 (Mean) | PctAlign (Med) | PctAlign (Mean) | PctAlign (Min) |")
    print("|-------|------------------|-------------------|------------------|-------------------|----------------|-----------------|----------------|")
    for layer in target_layers:
        recs = merged.get(layer, {}).get("clusters", [])
        if not recs: continue
        print(f"| Layer {layer} | "
              f"{med_nested(recs, 'cross_head_divergence', 'unique_argmax_count'):.1f} | "
              f"{avg_nested(recs, 'cross_head_divergence', 'unique_argmax_count'):.1f} | "
              f"{med_nested(recs, 'cross_head_divergence', 'heads_voting_top100'):.1f} | "
              f"{avg_nested(recs, 'cross_head_divergence', 'heads_voting_top100'):.1f} | "
              f"{med_nested(recs, 'cross_head_divergence', 'pct_heads_aligned'):.4f} | "
              f"{avg_nested(recs, 'cross_head_divergence', 'pct_heads_aligned'):.4f} | "
              f"{min_nested(recs, 'cross_head_divergence', 'pct_heads_aligned'):.4f} |")

def print_post_sum(merged, target_layers):
    print("\n### 5. POST-SUM Distribution (step 6)\n")
    print("| Layer | True Ent (Med) | True Ent (Mean) | True Ent (Max) | Sum Max (Med) | Sum Max (Min) | Sum Max (Max) | Sum Mean (Med) | Sum Mean (Mean) |")
    print("|-------|----------------|-----------------|----------------|---------------|---------------|---------------|----------------|-----------------|")
    for layer in target_layers:
        recs = merged.get(layer, {}).get("clusters", [])
        if not recs: continue
        print(f"| Layer {layer} | "
              f"{med(recs, 'true_entropy_norm'):.4f} | {avg(recs, 'true_entropy_norm'):.4f} | {max_val(recs, 'true_entropy_norm'):.4f} | "
              f"{med(recs, 'post_sum_max'):.4f} | {min_val(recs, 'post_sum_max'):.4f} | {max_val(recs, 'post_sum_max'):.4f} | "
              f"{med(recs, 'post_sum_mean'):.4e} | {avg(recs, 'post_sum_mean'):.4e} |")

def print_cutoff_stats(merged, target_layers):
    print("\n### 6. Cutoff Gap & Contested Region (step 7)\n")
    print("| Layer | Gap@Cutoff (Med) | Gap@Cutoff (Min) | Gap@Cutoff (Max) | Cutoff Val (Med) | Contested (Med) | Contested (Mean) | Contested (Max) | Score Range (Med) | Score Range (Max) |")
    print("|-------|------------------|------------------|------------------|------------------|-----------------|------------------|-----------------|-------------------|-------------------|")
    for layer in target_layers:
        recs = merged.get(layer, {}).get("clusters", [])
        if not recs: continue
        print(f"| Layer {layer} | "
              f"{med(recs, 'gap_at_cutoff'):.2e} | {min_val(recs, 'gap_at_cutoff'):.2e} | {max_val(recs, 'gap_at_cutoff'):.2e} | "
              f"{med(recs, 'cutoff_val'):.4f} | "
              f"{med(recs, 'contested_count'):.0f} | {avg(recs, 'contested_count'):.0f} | {max_val(recs, 'contested_count'):.0f} | "
              f"{med(recs, 'score_range'):.4f} | {max_val(recs, 'score_range'):.4f} |")

def print_overlap_stats(merged, target_layers):
    print("\n### 7. Mean Query Overlap (Count vs Prob Mass vs Top-Tier Hit Rate)\n")
    print("| Layer | Count Overlap | Prob Mass | TTHR 1%  | TTHR 2% | TTHR 5% | TTHR 10% | TTHR 20% | TTHR 40% | TTHR 60% | TTHR 80% |")
    print("|-------|---------------|-----------|----------|---------|---------|----------|----------|----------|----------|----------|")
    for layer in target_layers:
        recs = merged.get(layer, {}).get("clusters", [])
        if not recs: continue
        print(f"| Layer {layer} | "
              f"{avg(recs, 'post_sum_overlap', 100):.2f}% | "
              f"{avg(recs, 'prob_mass_overlap', 100):.2f}% | "
              f"{avg(recs, 'tthr_1', 100):.2f}% | "
              f"{avg(recs, 'tthr_2', 100):.2f}% | "
              f"{avg(recs, 'tthr_5', 100):.2f}% | "
              f"{avg(recs, 'tthr_10', 100):.2f}% | "
              f"{avg(recs, 'tthr_20', 100):.2f}% | "
              f"{avg(recs, 'tthr_40', 100):.2f}% | "
              f"{avg(recs, 'tthr_60', 100):.2f}% | "
              f"{avg(recs, 'tthr_80', 100):.2f}% |")

# =============================================================================
# MAIN
# =============================================================================
output_dir_base = "result_release_ttft/infinitbench"
THRESHOLDS = ["0.96"]

for thresh in THRESHOLDS:
    target_dir = os.path.join(output_dir_base, f"qwen-search-thresh-{thresh}")
    diag_files = glob.glob(os.path.join(target_dir, "*_diag.jsonl"))

    if not diag_files:
        continue

    print(f"\n========================================================\n")
    print(f"## 🔬 DIAGNOSTIC v2.4 FOR THRESHOLD: `{thresh}`\n")

    merged = defaultdict(lambda: {"chunk_cosines": [], "clusters": []})
    merged_by_dataset = defaultdict(lambda: defaultdict(lambda: {"clusters": []}))

    for fpath in diag_files:
        dataset_name = os.path.basename(fpath).replace("_diag.jsonl", "")
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                layer_str = str(data["layer"])
                
                # Gộp tổng thể
                merged[layer_str]["chunk_cosines"].extend(data.get("chunk_cosines", []))
                merged[layer_str]["clusters"].extend(data.get("clusters", []))
                
                # Gộp theo Dataset
                merged_by_dataset[dataset_name][layer_str]["clusters"].extend(data.get("clusters", []))

    if not merged: continue

    target_layers = get_target_layers(list(merged.keys()), num_points=5)
    print(f"*📌 Layers: {target_layers}*")

    # In 7 bảng tổng thể
    print_input_cosines(merged, target_layers)
    print_raw_score_stats(merged, target_layers)
    print_per_head_softmax(merged, target_layers)
    print_cross_head_divergence(merged, target_layers)
    print_post_sum(merged, target_layers)
    print_cutoff_stats(merged, target_layers)
    print_overlap_stats(merged, target_layers)

    # In phân tích theo TỪNG DATASET (Exp 9 & 10)
    print(f"\n{'='*60}")
    print(f"📊 DATASET-SPECIFIC OVERLAP ANALYSIS (Exp 9 & 10)")
    print(f"{'='*60}")
    
    for dataset, layers_data in sorted(merged_by_dataset.items()):
        target_layers_ds = get_target_layers(list(layers_data.keys()), num_points=5)
        
        print(f"\n### Dataset: `{dataset}`")
        print("| Layer | Count Overlap (Mean) | Prob Mass Overlap (Mean) | Gap@Cutoff (Min) | Gap@Cutoff (Max) | Gap@Cutoff (Mean)  | Contested (Mean) | TTHR 1% (Mean)  | TTHR 2% (Mean)  | TTHR 5% (Mean)  | TTHR 10% (Mean) | TTHR 20% (Mean) | TTHR 40% (Mean) | TTHR 60% (Mean) | TTHR 80% (Mean) |")
        print("|-------|----------------------|--------------------------|------------------|------------------|--------------------|------------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|")
        
        for layer in target_layers_ds:
            recs = layers_data.get(layer, {}).get("clusters", [])
            if not recs: continue
            
            count_ov = avg(recs, 'post_sum_overlap', 100)
            mass_ov = avg(recs, 'prob_mass_overlap', 100)
            gap_min = min_val(recs, 'gap_at_cutoff')
            gap_max = max_val(recs, 'gap_at_cutoff')
            gap_mean = avg(recs, 'gap_at_cutoff')
            contested_mean = avg(recs, 'contested_count')
            tthr_1 = avg(recs, 'tthr_1', 100)
            tthr_2 = avg(recs, 'tthr_2', 100)
            tthr_5 = avg(recs, 'tthr_5', 100)
            tthr_10 = avg(recs, 'tthr_10', 100)
            tthr_20 = avg(recs, 'tthr_20', 100)
            tthr_40 = avg(recs, 'tthr_40', 100)
            tthr_60 = avg(recs, 'tthr_60', 100)
            tthr_80 = avg(recs, 'tthr_80', 100)

            print(f"| L{layer:<4} | {count_ov:>19.2f}% | {mass_ov:>23.2f}% | {gap_min:>16.2e} | {gap_max:>16.2e} | {gap_mean:>18.2e} | {contested_mean:>16.2f} | {tthr_1:>13.2f}% | {tthr_2:>13.2f}% | {tthr_5:>13.2f}% | {tthr_10:>13.2f}% | {tthr_20:>13.2f}% | {tthr_40:>13.2f}% | {tthr_60:>13.2f}% | {tthr_80:>13.2f}% |")


# =============================================================================
# CROSS-THRESHOLD: KEY measurements for Layer 13 vs Layer 27
# =============================================================================
print(f"\n========================================================\n")
print(f"## 🎯 LAYER 13 vs LAYER 27 — OVERLAP CROSS-THRESHOLD SUMMARY\n")
print("| Thresh | Layer | Count Overlap | Prob Mass | TTHR 1% | TTHR 2% | TTHR 5% | TTHR 10% | TTHR 20% | TTHR 40% | TTHR 60% | TTHR 80% |")
print("|--------|-------|---------------|-----------|---------|---------|---------|----------|----------|----------|----------|----------|")

for thresh in THRESHOLDS:
    target_dir = os.path.join(output_dir_base, f"qwen-search-thresh-{thresh}")
    diag_files = glob.glob(os.path.join(target_dir, "*_diag.jsonl"))
    if not diag_files: continue

    merged = defaultdict(lambda: {"clusters": []})
    for fp in diag_files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                merged[str(d["layer"])]["clusters"].extend(d.get("clusters", []))

    for layer in ["13", "27"]:
        recs = merged.get(layer, {}).get("clusters", [])
        if not recs: continue
        print(f"| {thresh} | L{layer} | "
              f"{avg(recs, 'post_sum_overlap', 100):.2f}% | "
              f"{avg(recs, 'prob_mass_overlap', 100):.2f}% | "
              f"{avg(recs, 'tthr_10', 100):.2f}% | "
              f"{avg(recs, 'tthr_20', 100):.2f}% | "
              f"{avg(recs, 'tthr_40', 100):.2f}% | "
              f"{avg(recs, 'tthr_60', 100):.2f}% | "
              f"{avg(recs, 'tthr_80', 100):.2f}% |")

# In Bảng phong thần Layer 27 theo Dataset
print(f"\n========================================================")
print(f"🎯 LAYER 27 DATASET BREAKDOWN: DATASET NÀO GÁNH TEAM?")
print(f"========================================================\n")
print("| Dataset         | Thresh | TrueEnt (Mean) | Count Overlap | Prob Mass | Gap@Cutoff (Min) |")
print("|-----------------|--------|----------------|---------------|-----------|------------------|")

summary_records_dataset = sorted(summary_records_dataset, key=lambda x: (x["dataset"], x["thresh"]))

for rec in summary_records_dataset:
    print(f"| {rec['dataset']:<15} | {rec['thresh']:<6} | {rec['true_ent']:>14.4f} | {rec['count_ov']:>12.2f}% | {rec['mass_ov']:>8.2f}% | {rec['gap_min']:>16.2e} |")

print("\n(Note: Kiểm tra cột 'Prob Mass' ở bảng cuối. Nếu Count Overlap thấp nhưng Prob Mass cao -> Giả thuyết 'Core Tokens gánh team' là chính xác!)")