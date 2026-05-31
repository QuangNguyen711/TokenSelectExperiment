# File: analyze_overlap_multilayer.py
"""
Aggregate and report anchor-vs-mean overlap recorded by token_retrieval's
OVERLAP_RECORD_PATH mechanism.

Mirrors the structure of analyze_stats_multilayer.py:
  - Glob all *_overlap.jsonl files under qwen-search-thresh-{THRESHOLD}/
  - Merge by layer
  - Print one table per threshold for 5 target layers (chosen by linspace)
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


def print_overlap_table(data_by_layer, target_layers):
    """
    For each target layer, print median anchor/mean recalls and Δ.
    data_by_layer[layer_str] is a list of overlap-record dicts.
    """
    print("\n### Anchor vs Mean Recall (median across clusters)\n")
    print("| Layer | #Clusters | AvgClSz | GlobA (med) | GlobM (med) | Δ(A-M) | "
          "1stA | 1stM | LastA | LastM |")
    print("|-------|-----------|---------|-------------|-------------|--------|"
          "------|------|-------|-------|")

    for layer in target_layers:
        records = data_by_layer.get(layer, [])
        if not records:
            continue

        ga = np.array([r["global_recall_anchor"] for r in records]) * 100
        gm = np.array([r["global_recall_mean"] for r in records]) * 100
        fa = np.array([r["first_recall_anchor"] for r in records]) * 100
        fm = np.array([r["first_recall_mean"] for r in records]) * 100
        la = np.array([r["last_recall_anchor"] for r in records]) * 100
        lm = np.array([r["last_recall_mean"] for r in records]) * 100
        cs = np.array([r["cluster_size"] for r in records])

        ga_med = np.median(ga)
        gm_med = np.median(gm)
        diff = ga_med - gm_med

        print(f"| Layer {layer} | {len(records)} | {cs.mean():.2f} | "
              f"{ga_med:.2f} | {gm_med:.2f} | {diff:+.2f} | "
              f"{np.median(fa):.2f} | {np.median(fm):.2f} | "
              f"{np.median(la):.2f} | {np.median(lm):.2f} |")


def print_cluster_size_breakdown(data_by_layer, target_layers, max_size=8):
    """For each layer, show how many clusters per size — to confirm distribution."""
    print("\n### Cluster size distribution (only size >= 2 was logged)\n")

    header = "| Layer |"
    for s in range(2, max_size + 1):
        header += f" {s} Blks |"
    header += " Total |"
    print(header)

    sep = "|-------|"
    for s in range(2, max_size + 1):
        sep += "--------|"
    sep += "-------|"
    print(sep)

    for layer in target_layers:
        records = data_by_layer.get(layer, [])
        if not records:
            continue

        sizes = [r["cluster_size"] for r in records]
        total = len(sizes)
        row = f"| Layer {layer} |"
        for s in range(2, max_size + 1):
            if s < max_size:
                count = sum(1 for x in sizes if x == s)
            else:
                count = sum(1 for x in sizes if x >= s)
            pct = (count / total * 100) if total else 0
            row += f" {count} ({pct:.1f}%) |"
        row += f" **{total}** |"
        print(row)


# =============================================================================
# MAIN
# =============================================================================
output_dir_base = "result_release_ttft/infinitbench"
THRESHOLDS = ["0.95", "0.97", "0.99", "0.999"]

for thresh in THRESHOLDS:
    target_dir = os.path.join(output_dir_base, f"qwen-search-thresh-{thresh}")
    overlap_files = glob.glob(os.path.join(target_dir, "*_overlap.jsonl"))

    if not overlap_files:
        continue

    print(f"\n---\n")
    print(f"## 🚀 OVERLAP RESULTS FOR THRESHOLD: `{thresh}`\n")

    merged_records = defaultdict(list)

    for fpath in overlap_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                layer_str = str(data["layer"])
                # Each line has a list of overlap records (one per cluster of size>=2)
                for record in data.get("overlaps", []):
                    merged_records[layer_str].append(record)

    if not merged_records:
        continue

    target_layers = get_target_layers(list(merged_records.keys()), num_points=5)
    print(f"*📌 Tracking 5 layers: {target_layers}*")

    print_overlap_table(merged_records, target_layers)
    print_cluster_size_breakdown(merged_records, target_layers)


# =============================================================================
# Cross-threshold summary at the end
# =============================================================================
print(f"\n---\n## 📊 CROSS-THRESHOLD SUMMARY (Anchor vs Mean Δ, median)\n")
print("| Threshold | Layer | #Cl | AvgClSz | Anchor% | Mean% | Δ(A-M) |")
print("|-----------|-------|-----|---------|---------|-------|--------|")

for thresh in THRESHOLDS:
    target_dir = os.path.join(output_dir_base, f"qwen-search-thresh-{thresh}")
    overlap_files = glob.glob(os.path.join(target_dir, "*_overlap.jsonl"))
    if not overlap_files:
        continue

    merged = defaultdict(list)
    for fpath in overlap_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                for r in data.get("overlaps", []):
                    merged[str(data["layer"])].append(r)

    target_layers = get_target_layers(list(merged.keys()), num_points=5)
    for layer in target_layers:
        records = merged.get(layer, [])
        if not records:
            continue
        ga = np.array([r["global_recall_anchor"] for r in records]) * 100
        gm = np.array([r["global_recall_mean"] for r in records]) * 100
        cs = np.array([r["cluster_size"] for r in records])
        ga_med = np.median(ga)
        gm_med = np.median(gm)
        print(f"| {thresh} | Layer {layer} | {len(records)} | "
              f"{cs.mean():.2f} | {ga_med:.2f} | {gm_med:.2f} | "
              f"{ga_med - gm_med:+.2f} |")