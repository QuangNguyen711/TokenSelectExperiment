# File: analyze_diagnostic_multilayer.py
"""
v2.2: Phiên bản Minimal - Chỉ phân tích Count Overlap và Prob Mass Overlap.
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


# --- Chỉ in Bảng Overlap ---
def print_overlap_stats(merged, target_layers):
    print("\n### Mean Query Overlap (Count vs Probability Mass)\n")
    print("| Layer | Count Overlap (Med) | Count Overlap (Mean) | Prob Mass (Med) | Prob Mass (Mean) |")
    print("|-------|---------------------|----------------------|-----------------|------------------|")
    for layer in target_layers:
        recs = merged.get(layer, {}).get("clusters", [])
        if not recs: continue
        print(f"| Layer {layer} | "
              f"{med(recs, 'post_sum_overlap', 100):.2f}% | "
              f"{avg(recs, 'post_sum_overlap', 100):.2f}% | "
              f"{med(recs, 'prob_mass_overlap', 100):.2f}% | "
              f"{avg(recs, 'prob_mass_overlap', 100):.2f}% |")


# =============================================================================
# MAIN
# =============================================================================
output_dir_base = "result_release_ttft/infinitbench"
THRESHOLDS = ["0.95", "0.97", "0.99", "0.999"]

for thresh in THRESHOLDS:
    target_dir = os.path.join(output_dir_base, f"qwen-search-thresh-{thresh}")
    diag_files = glob.glob(os.path.join(target_dir, "*_diag.jsonl"))

    if not diag_files:
        continue

    print(f"\n========================================================\n")
    print(f"## 🔬 OVERLAP DIAGNOSTIC FOR THRESHOLD: `{thresh}`\n")

    merged = defaultdict(lambda: {"clusters": []})

    for fpath in diag_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                layer_str = str(data["layer"])
                merged[layer_str]["clusters"].extend(data.get("clusters", []))

    if not merged: continue

    target_layers = get_target_layers(list(merged.keys()), num_points=5)
    print(f"*📌 Layers: {target_layers}*")

    print_overlap_stats(merged, target_layers)


# =============================================================================
# CROSS-THRESHOLD: KEY measurements for Layer 13 vs Layer 27
# =============================================================================
print(f"\n========================================================\n")
print(f"## 🎯 LAYER 13 vs LAYER 27 — OVERLAP CROSS-THRESHOLD SUMMARY\n")
print("| Thresh | Layer | Count Overlap (Mean) | Prob Mass (Mean) |")
print("|--------|-------|----------------------|------------------|")

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
              f"{avg(recs, 'prob_mass_overlap', 100):.2f}% |")