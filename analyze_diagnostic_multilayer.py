# File: analyze_diagnostic_multilayer.py
"""
v2: phân tích sâu từng bước pipeline top-K để tìm nguyên nhân Layer 27 anomaly.

Mỗi bảng đo MỘT bước trong pipeline:
  Bước 4 — raw_score: paged_matmul output (bf16 quantization sensitivity)
  Bước 5a — per_head_softmax: entropy/sharpness của softmax per head
  Bước 5b — cross_head_divergence: các head có vote cùng token không?  <-- KEY
  Bước 6 — post_sum: entropy/max của scores SAU sum-over-heads          <-- KEY
  Bước 7 — cutoff/contested: vùng tranh chấp ở rìa top-K               <-- KEY
  Tautology — post_sum_overlap: phải khớp với recall (sanity)
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


def med(arr, key, scale=1.0):
    """Median của một field từ list of dicts."""
    vals = [r[key] * scale for r in arr if key in r and r[key] is not None]
    return float(np.median(vals)) if vals else float("nan")


def med_nested(arr, parent, child, scale=1.0):
    vals = [r[parent][child] * scale for r in arr
            if r.get(parent) and child in r[parent]]
    return float(np.median(vals)) if vals else float("nan")


def print_input_cosines(merged, target_layers):
    print("\n### 1. Query similarity (input to pipeline)\n")
    print("Sum-Cosine = cosine của hai vector scores_summed (anchor vs last)\n")
    print("| Layer | Cos-PRE | Cos-POST-RoPE | Sum-Cosine | Sum-Diff (mean abs) |")
    print("|-------|---------|----------------|------------|---------------------|")
    for layer in target_layers:
        recs = merged.get(layer, {}).get("clusters", [])
        if not recs:
            continue
        print(f"| Layer {layer} | "
              f"{med(recs, 'cos_pre_rope'):.4f} | "
              f"{med(recs, 'cos_post_rope'):.4f} | "
              f"{med(recs, 'sum_cosine'):.4f} | "
              f"{med(recs, 'sum_diff_norm'):.2e} |")


def print_raw_score_stats(merged, target_layers):
    print("\n### 2. Raw Score Magnitude (paged_matmul output, bfloat16)\n")
    print("Magnitude lớn => bf16 ULP lớn => khoảng cách quantization lớn\n")
    print("| Layer | max abs | mean abs | bf16 ULP at max |")
    print("|-------|---------|----------|-----------------|")
    for layer in target_layers:
        recs = merged.get(layer, {}).get("clusters", [])
        if not recs:
            continue
        print(f"| Layer {layer} | "
              f"{med_nested(recs, 'raw_score', 'max_abs'):.2f} | "
              f"{med_nested(recs, 'raw_score', 'mean_abs'):.4f} | "
              f"{med_nested(recs, 'raw_score', 'bf16_ulp_at_max'):.2e} |")


def print_per_head_softmax(merged, target_layers):
    print("\n### 3. Per-Head Softmax Sharpness (step 5a)\n")
    print("max_prob cao => head sharp/sink-dominated. Median qua 28 heads.\n")
    print("| Layer | PerHead Ent (med) | PerHead MaxProb (med) | PerHead MaxProb (MAX) |")
    print("|-------|-------------------|-----------------------|-----------------------|")
    for layer in target_layers:
        recs = merged.get(layer, {}).get("clusters", [])
        if not recs:
            continue
        print(f"| Layer {layer} | "
              f"{med(recs, 'per_head_entropy_norm_median'):.4f} | "
              f"{med(recs, 'per_head_max_prob_median'):.4f} | "
              f"{med(recs, 'per_head_max_prob_max'):.4f} |")


def print_cross_head_divergence(merged, target_layers):
    print("\n### 4. Cross-Head Divergence (step 5b) — KEY MEASUREMENT\n")
    print("Mỗi head vote 1 token argmax. Nếu các head vote KHÁC nhau\n"
          "-> sum lại thành PHẲNG. pct_aligned thấp = phân kỳ.\n")
    print("| Layer | UniqueArgmax /28 | HeadsVoteTop100 /28 | PctAligned |")
    print("|-------|-------------------|----------------------|------------|")
    for layer in target_layers:
        recs = merged.get(layer, {}).get("clusters", [])
        if not recs:
            continue
        print(f"| Layer {layer} | "
              f"{med_nested(recs, 'cross_head_divergence', 'unique_argmax_count'):.1f} | "
              f"{med_nested(recs, 'cross_head_divergence', 'heads_voting_top100'):.1f} | "
              f"{med_nested(recs, 'cross_head_divergence', 'pct_heads_aligned'):.4f} |")


def print_post_sum(merged, target_layers):
    print("\n### 5. POST-SUM Distribution (step 6) — KEY MEASUREMENT\n")
    print("Đây là vector THỰC SỰ quyết định top-K. Entropy CAO = phẳng = top-K nhạy.\n")
    print("| Layer | True Entropy (norm) | Post-Sum Max | Post-Sum Mean | Max/Mean Ratio |")
    print("|-------|---------------------|--------------|---------------|----------------|")
    for layer in target_layers:
        recs = merged.get(layer, {}).get("clusters", [])
        if not recs:
            continue
        max_med = med(recs, 'post_sum_max')
        mean_med = med(recs, 'post_sum_mean')
        ratio = max_med / mean_med if mean_med else 0
        print(f"| Layer {layer} | "
              f"{med(recs, 'true_entropy_norm'):.4f} | "
              f"{max_med:.4f} | "
              f"{mean_med:.4e} | "
              f"{ratio:.2f} |")


def print_cutoff_stats(merged, target_layers):
    print("\n### 6. Cutoff Gap & Contested Region (step 7) — KEY MEASUREMENT\n")
    print("Gap nhỏ + contested lớn => perturbation cực nhỏ flip hàng nghìn token.\n")
    print("| Layer | Gap at Cutoff | Cutoff Val | Score Range | Contested (#tokens) |")
    print("|-------|---------------|------------|-------------|---------------------|")
    for layer in target_layers:
        recs = merged.get(layer, {}).get("clusters", [])
        if not recs:
            continue
        print(f"| Layer {layer} | "
              f"{med(recs, 'gap_at_cutoff'):.2e} | "
              f"{med(recs, 'cutoff_val'):.4f} | "
              f"{med(recs, 'score_range'):.4f} | "
              f"{med(recs, 'contested_count'):.0f} |")


def print_post_sum_overlap(merged, target_layers):
    print("\n### 7. Post-Sum Overlap (= recall, sanity check)\n")
    print("Phải khớp với Last-Anchor Recall từ overlap_results.\n")
    print("| Layer | Post-Sum Overlap (median) |")
    print("|-------|---------------------------|")
    for layer in target_layers:
        recs = merged.get(layer, {}).get("clusters", [])
        if not recs:
            continue
        print(f"| Layer {layer} | {med(recs, 'post_sum_overlap', 100):.2f}% |")


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

    print(f"\n---\n")
    print(f"## 🔬 DIAGNOSTIC v2 FOR THRESHOLD: `{thresh}`\n")

    merged = defaultdict(lambda: {"chunk_cosines": [], "clusters": []})

    for fpath in diag_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                layer_str = str(data["layer"])
                merged[layer_str]["chunk_cosines"].extend(
                    data.get("chunk_cosines", [])
                )
                merged[layer_str]["clusters"].extend(data.get("clusters", []))

    if not merged:
        continue

    target_layers = get_target_layers(list(merged.keys()), num_points=5)
    print(f"*📌 Layers: {target_layers}*")

    print_input_cosines(merged, target_layers)
    print_raw_score_stats(merged, target_layers)
    print_per_head_softmax(merged, target_layers)
    print_cross_head_divergence(merged, target_layers)
    print_post_sum(merged, target_layers)
    print_cutoff_stats(merged, target_layers)
    print_post_sum_overlap(merged, target_layers)


# =============================================================================
# CROSS-THRESHOLD: KEY measurements for Layer 13 vs Layer 27
# =============================================================================
print(f"\n---\n## 🎯 LAYER 13 vs LAYER 27 — KEY measurements\n")
print("Cột nào lệch nhiều giữa L13 và L27 = nguyên nhân chính.\n")
print("| Thresh | Layer | PctHeadsAlign | True-Ent | Max/Mean | Gap@Cutoff | "
      "Contested | PostSum-Overlap |")
print("|--------|-------|---------------|----------|----------|------------|"
      "-----------|-----------------|")

for thresh in THRESHOLDS:
    target_dir = os.path.join(output_dir_base, f"qwen-search-thresh-{thresh}")
    diag_files = glob.glob(os.path.join(target_dir, "*_diag.jsonl"))
    if not diag_files:
        continue

    merged = defaultdict(lambda: {"clusters": []})
    for fp in diag_files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                merged[str(d["layer"])]["clusters"].extend(d.get("clusters", []))

    for layer in ["13", "27"]:
        recs = merged.get(layer, {}).get("clusters", [])
        if not recs:
            continue
        max_med = med(recs, 'post_sum_max')
        mean_med = med(recs, 'post_sum_mean')
        ratio = max_med / mean_med if mean_med else 0
        print(f"| {thresh} | L{layer} | "
              f"{med_nested(recs, 'cross_head_divergence', 'pct_heads_aligned'):.4f} | "
              f"{med(recs, 'true_entropy_norm'):.4f} | "
              f"{ratio:.1f} | "
              f"{med(recs, 'gap_at_cutoff'):.2e} | "
              f"{med(recs, 'contested_count'):.0f} | "
              f"{med(recs, 'post_sum_overlap', 100):.2f}% |")