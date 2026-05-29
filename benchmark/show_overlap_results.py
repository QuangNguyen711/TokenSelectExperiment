"""
Visualize Anchor-vs-Mean Profiling Results (v2).

Reads the JSON produced by topk_overlap_profile.py v2 (anchor-vs-mean schema)
and produces:
  - Console tables (aggregate + per-task + sanity check)
  - LaTeX tables (head-to-head comparison + per-layer breakdown)
  - Plots (if matplotlib available)

Usage:
    python benchmark/show_overlap_results.py --input overlap_results.json
    python benchmark/show_overlap_results.py --input overlap_results.json --layer 13
    python benchmark/show_overlap_results.py --input overlap_results.json --no_plots
"""
import os
import json
import argparse


# =============================================================================
# CONSOLE TABLES
# =============================================================================
def print_aggregate_table(summary, focus_layer=None):
    """Full aggregate, all (theta, L_max, layer) combinations."""
    print("\n" + "=" * 140)
    title = " ANCHOR vs MEAN — AGGREGATE"
    if focus_layer is not None:
        title += f" (Layer {focus_layer} only)"
    print(title)
    print("=" * 140)
    print(f"{'theta':<7}{'L_max':<7}{'layer':<7}{'#cl':<6}{'csz':<6}"
          f"{'GlobA':<8}{'GlobM':<8}{'Δ(A-M)':<10}"
          f"{'1stA':<8}{'1stM':<8}{'LastA':<8}{'LastM':<8}")
    print("-" * 140)
    for r in summary:
        if focus_layer is not None and r['layer'] != focus_layer:
            continue
        diff_str = f"{r['anchor_minus_mean_global']:+.2f}"
        print(f"{r['theta']:<7}{r['L_max']:<7}{r['layer']:<7}"
              f"{r['num_clusters']:<6}{r['avg_cluster_size']:<6.1f}"
              f"{r['global_anchor_median']:<8.2f}"
              f"{r['global_mean_median']:<8.2f}"
              f"{diff_str:<10}"
              f"{r['first_anchor_median']:<8.2f}"
              f"{r['first_mean_median']:<8.2f}"
              f"{r['last_anchor_median']:<8.2f}"
              f"{r['last_mean_median']:<8.2f}")


def print_sanity_check(summary):
    """First-Chunk Recall (anchor) must be ~100% — confirms pool consistency."""
    print("\n" + "=" * 80)
    print(" SANITY CHECK: First-Chunk Recall (Anchor) — must be ~100%")
    print(" Anchor uses the block-1 query on the SAME pool as the block-1 baseline.")
    print(" If <99%, pool is not actually consistent. Numbers cannot be trusted.")
    print("=" * 80)
    failures = []
    for r in summary:
        if r['first_anchor_median'] < 99.0:
            failures.append(r)
    if failures:
        print(" !!! SANITY FAILED !!!")
        for r in failures:
            print(f"   theta={r['theta']}, L_max={r['L_max']}, layer={r['layer']}: "
                  f"first_anchor={r['first_anchor_median']:.2f}%")
    else:
        print(" SANITY PASSED — first_anchor_recall ~= 100% for ALL configs.")
        # Show min just for confirmation
        min_first = min(r['first_anchor_median'] for r in summary)
        print(f" (min first_anchor_recall across configs: {min_first:.2f}%)")


def print_head_to_head_table(summary, focus_layer=20):
    """Focused console table for the main story: anchor vs mean."""
    print("\n" + "=" * 100)
    print(f" HEAD-TO-HEAD: Anchor vs Mean Global Recall (Layer {focus_layer})")
    print(" Δ > 0 = Anchor wins | Δ < 0 = Mean wins")
    print("=" * 100)
    print(f"{'theta':<8}{'L_max':<8}{'AvgClSz':<10}"
          f"{'Anchor%':<11}{'Mean%':<11}{'Δ(A-M)':<10}{'Winner':<12}")
    print("-" * 100)
    for r in sorted(summary, key=lambda x: (x['theta'], x['L_max'])):
        if r['layer'] != focus_layer:
            continue
        diff = r['anchor_minus_mean_global']
        if diff > 0.5:
            winner = "Anchor"
        elif diff < -0.5:
            winner = "Mean"
        else:
            winner = "Tie"
        print(f"{r['theta']:<8}{r['L_max']:<8}"
              f"{r['avg_cluster_size']:<10.2f}"
              f"{r['global_anchor_median']:<11.2f}"
              f"{r['global_mean_median']:<11.2f}"
              f"{diff:+.2f}{'':<5}"
              f"{winner:<12}")


def print_per_task_table(per_task_results, datasets, focus_layer=20, lmax=4096):
    """Per-task head-to-head at a fixed L_max."""
    print("\n" + "=" * 130)
    print(f" PER-TASK: Anchor vs Mean Global Recall (Layer {focus_layer}, L_max={lmax})")
    print("=" * 130)
    header = f"{'theta':<8}" + "".join(f"{d[:14]:<18}" for d in datasets)
    print(header)
    print("-" * len(header))

    thetas = sorted(set(
        r['theta'] for rows in per_task_results.values() for r in rows
    ))
    for theta in thetas:
        row = f"{theta:<8}"
        for ds in datasets:
            task_key = f"{ds}__theta{theta}__Lmax{lmax}"
            task_rows = per_task_results.get(task_key, [])
            match = next(
                (r for r in task_rows if r['layer'] == focus_layer), None
            )
            if match is None:
                row += f"{'--':<18}"
            else:
                a = match['global_anchor_median']
                m = match['global_mean_median']
                row += f"A:{a:.1f} M:{m:.1f}".ljust(18)
        print(row)


def print_verdict(summary):
    """Overall winner across all configs."""
    print("\n" + "=" * 80)
    print(" OVERALL VERDICT: Anchor vs Mean (based on Global Recall median)")
    print("=" * 80)
    anchor_wins = sum(1 for r in summary if r['anchor_minus_mean_global'] > 0.5)
    mean_wins = sum(1 for r in summary if r['anchor_minus_mean_global'] < -0.5)
    ties = len(summary) - anchor_wins - mean_wins
    total = len(summary)

    print(f"  Anchor wins (Δ > +0.5%):  {anchor_wins} / {total} configs")
    print(f"  Mean wins  (Δ < -0.5%):   {mean_wins} / {total} configs")
    print(f"  Comparable (|Δ| <= 0.5%): {ties} / {total} configs")

    avg_diff = sum(r['anchor_minus_mean_global'] for r in summary) / total
    print(f"  Average Δ(Anchor - Mean): {avg_diff:+.3f}%")

    print()
    if anchor_wins > 2 * mean_wins and anchor_wins > total * 0.4:
        print("  -> ANCHOR meaningfully better than MEAN")
    elif mean_wins > 2 * anchor_wins and mean_wins > total * 0.4:
        print("  -> MEAN meaningfully better than ANCHOR")
    else:
        print("  -> ANCHOR and MEAN perform comparably")


# =============================================================================
# LATEX
# =============================================================================
def gen_latex_main_table(summary, output_path, focus_layer=20):
    """Main head-to-head LaTeX table for paper."""
    rows = [r for r in summary if r['layer'] == focus_layer]
    rows.sort(key=lambda x: (x['theta'], x['L_max']))

    latex = r"""\begin{table}[t]
\centering
\caption{Head-to-head retrieval quality at Layer """ + str(focus_layer) + r""" 
under varying $\theta$ and $L_{\max}$. \textit{Global Recall} reports the 
average per-sub-chunk recall against the per-sub-chunk baseline (Definition B). 
$\Delta$ denotes Anchor $-$ Mean: positive values favor anchor-based retrieval. 
First-Chunk Recall (Anchor) is reported as a sanity check: anchor uses the 
first-block query on the same pool as the first-block baseline, so this 
value must be $\approx 100\%$ for the comparison to be valid.}
\label{tab:anchor-vs-mean}
\small
\begin{tabular}{cccccccc}
\toprule
$\theta$ & $L_{\max}$ & \#clust & \makecell{Avg Cluster\\Size} & \makecell{Global\\(Anchor)} & \makecell{Global\\(Mean)} & $\Delta$ & \makecell{First\\(Anchor)} \\
\midrule
"""
    prev_theta = None
    for r in rows:
        if prev_theta is not None and r['theta'] != prev_theta:
            latex += r"\midrule" + "\n"
        prev_theta = r['theta']
        latex += (
            f"${r['theta']}$ & {r['L_max']} & {r['num_clusters']} & "
            f"{r['avg_cluster_size']:.1f} & "
            f"{r['global_anchor_median']:.1f} & "
            f"{r['global_mean_median']:.1f} & "
            f"{r['anchor_minus_mean_global']:+.1f} & "
            f"{r['first_anchor_median']:.1f} \\\\\n"
        )
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"  LaTeX main table -> {output_path}")


def gen_latex_layerwise_table(summary, output_path, theta=0.95, lmax=4096):
    """Per-layer breakdown LaTeX table."""
    rows = [r for r in summary
            if r['theta'] == theta and r['L_max'] == lmax]
    rows.sort(key=lambda x: x['layer'])

    latex = r"""\begin{table}[t]
\centering
\caption{Per-layer head-to-head retrieval quality at $\theta = """ + str(theta) + r"""$, 
$L_{\max} = """ + str(lmax) + r"""$. Anchor and Mean queries are compared against 
per-sub-chunk baselines on the same candidate pool.}
\label{tab:anchor-vs-mean-layerwise}
\small
\begin{tabular}{lccccccc}
\toprule
Layer & \#clust & \makecell{Global\\(Anchor)} & \makecell{Global\\(Mean)} & $\Delta$ & \makecell{Last\\(Anchor)} & \makecell{Last\\(Mean)} \\
\midrule
"""
    for r in rows:
        latex += (
            f"Layer {r['layer']} & {r['num_clusters']} & "
            f"{r['global_anchor_median']:.1f} & "
            f"{r['global_mean_median']:.1f} & "
            f"{r['anchor_minus_mean_global']:+.1f} & "
            f"{r['last_anchor_median']:.1f} & "
            f"{r['last_mean_median']:.1f} \\\\\n"
        )
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"  LaTeX layerwise table -> {output_path}")


# =============================================================================
# PLOTS
# =============================================================================
def plot_anchor_vs_mean_bars(summary, output_path, focus_layer=20):
    """Bar chart: Anchor vs Mean Global Recall across (theta, L_max) configs."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [skip] matplotlib not installed")
        return

    rows = [r for r in summary if r['layer'] == focus_layer]
    rows.sort(key=lambda x: (x['theta'], x['L_max']))

    labels = [f"θ={r['theta']}\nL={r['L_max']}" for r in rows]
    anchor_vals = [r['global_anchor_median'] for r in rows]
    mean_vals = [r['global_mean_median'] for r in rows]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, anchor_vals, width, label='Anchor',
            color='#2196F3', alpha=0.85)
    ax.bar(x + width / 2, mean_vals, width, label='Mean',
            color='#FF9800', alpha=0.85)

    ax.set_ylabel('Global Recall (%)', fontsize=11)
    ax.set_title(f'Anchor vs Mean Global Recall (Layer {focus_layer})', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    for i, r in enumerate(rows):
        diff = r['anchor_minus_mean_global']
        mid_y = (anchor_vals[i] + mean_vals[i]) / 2
        ax.annotate(f'Δ{diff:+.1f}', xy=(i, mid_y), fontsize=7,
                     ha='center', va='center', color='gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot (anchor vs mean bars) -> {output_path}")


def plot_diff_vs_cluster_size(summary, output_path, focus_layer=20):
    """Scatter: Δ(Anchor - Mean) vs avg cluster size, colored by theta."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    rows = [r for r in summary if r['layer'] == focus_layer]
    theta_colors = {0.95: '#4CAF50', 0.97: '#FF9800', 0.99: '#F44336'}

    fig, ax = plt.subplots(figsize=(7, 5))
    for r in rows:
        color = theta_colors.get(r['theta'], 'gray')
        ax.scatter(r['avg_cluster_size'], r['anchor_minus_mean_global'],
                    s=100, color=color, edgecolors='black', linewidth=0.5,
                    label=f"θ={r['theta']}" if r['L_max'] == 1024 else "")
        ax.annotate(f"L={r['L_max']}", (r['avg_cluster_size'], r['anchor_minus_mean_global']),
                     fontsize=7, textcoords="offset points", xytext=(5, 5))

    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_xlabel('Avg Cluster Size (sub-chunks)', fontsize=11)
    ax.set_ylabel('Δ(Anchor − Mean) Global Recall (%)', fontsize=11)
    ax.set_title(f'Anchor advantage vs Cluster Size (Layer {focus_layer})', fontsize=12)
    ax.grid(True, alpha=0.3)

    handles, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot (diff vs cluster size) -> {output_path}")


def plot_layerwise(summary, output_path, theta=0.95, lmax=4096):
    """Bar chart: Anchor vs Mean across layers at fixed (theta, L_max)."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    rows = [r for r in summary if r['theta'] == theta and r['L_max'] == lmax]
    rows.sort(key=lambda x: x['layer'])

    layers = [f"L{r['layer']}" for r in rows]
    anchor_vals = [r['global_anchor_median'] for r in rows]
    mean_vals = [r['global_mean_median'] for r in rows]

    x = np.arange(len(layers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, anchor_vals, width, label='Anchor',
            color='#2196F3', alpha=0.85)
    ax.bar(x + width / 2, mean_vals, width, label='Mean',
            color='#FF9800', alpha=0.85)

    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Global Recall (%)', fontsize=11)
    ax.set_title(f'Anchor vs Mean by Layer (θ={theta}, L_max={lmax})', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot (layerwise) -> {output_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="overlap_viz")
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--layer", type=int, default=20,
                        help="Layer to focus on (default 20)")
    parser.add_argument("--theta", type=float, default=0.95)
    parser.add_argument("--lmax", type=int, default=4096)
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    aggregate = data["aggregate_results"]
    per_task = data.get("per_task_results", {})
    datasets = data["datasets"]

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n Loaded: {args.input}")
    print(f" Datasets: {datasets}")
    print(f" Samples per task: {data['num_samples_per_task']}")

    # Console
    print_aggregate_table(aggregate)
    print_sanity_check(aggregate)
    print_aggregate_table(aggregate, focus_layer=args.layer)
    print_head_to_head_table(aggregate, focus_layer=args.layer)
    print_per_task_table(per_task, datasets,
                          focus_layer=args.layer, lmax=args.lmax)
    print_verdict(aggregate)

    # LaTeX
    print("\n" + "=" * 80)
    print(" GENERATING LATEX")
    print("=" * 80)
    gen_latex_main_table(
        aggregate,
        os.path.join(args.output_dir, "table_anchor_vs_mean_main.tex"),
        focus_layer=args.layer
    )
    gen_latex_layerwise_table(
        aggregate,
        os.path.join(args.output_dir, "table_anchor_vs_mean_layerwise.tex"),
        theta=args.theta, lmax=args.lmax
    )

    # Plots
    if not args.no_plots:
        print("\n" + "=" * 80)
        print(" GENERATING PLOTS")
        print("=" * 80)
        plot_anchor_vs_mean_bars(
            aggregate,
            os.path.join(args.output_dir, "plot_anchor_vs_mean_bars.png"),
            focus_layer=args.layer
        )
        plot_diff_vs_cluster_size(
            aggregate,
            os.path.join(args.output_dir, "plot_diff_vs_cluster_size.png"),
            focus_layer=args.layer
        )
        plot_layerwise(
            aggregate,
            os.path.join(args.output_dir, "plot_layerwise.png"),
            theta=args.theta, lmax=args.lmax
        )

    print(f"\nAll outputs -> {args.output_dir}/")


if __name__ == "__main__":
    main()