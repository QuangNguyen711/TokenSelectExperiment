# File: benchmark/analyse_sink_logs.py  (v3 - sink + cumsum-k sweep + gap detection)
"""
Analyse sink logs (v2 logger format) with THREE blocks of metrics:

  A. SINK SHAPE (as before): front_mass, pos_hold, raw_gap (kept for context).

  B. CUMSUM-K SWEEP: for thresholds theta in {0.90, 0.95, 0.97, 0.99}, simulate
     cumsum selection (sort desc, keep until CDF>=theta) on the SAME post-soft-vote
     vector production uses (bf16-origin, we read it as logged). Report
     median/min/max of k across logged calls per (dataset, layer). This shows
     HOW MANY tokens cumsum actually keeps, and HOW MUCH IT VARIES (the variance
     is the instability signal).

  C. GAP DETECTION: on the FRONT full-res region (positions < front_full, where
     the important tokens live and the vector is not downsampled), sort scores
     desc and find the "natural cut":
       - gap_ratio_k : argmax of S[i]/S[i+1]  (largest multiplicative drop)
       - gap_diff_k  : argmax of S[i]-S[i+1]  (largest absolute drop)
       - gap_sharpness: (largest gap) / (median gap)  -> HIGH = clear natural
                        boundary; LOW = no boundary, every cut is arbitrary
                        (this is the task-stability signal both old analyses missed)
     We compare gap_k vs cumsum_k vs fixed top-k.

IMPORTANT measurement-tier note:
  v2 logger stores the FULL front region (0..front_full) and a STRIDED tail.
  - Cumsum (block B) uses stride-WEIGHTING so the CDF is an unbiased estimate of
    the full distribution; k reported is the implied full-resolution k.
  - Gap (block C) is computed ONLY on the front full-res region, because tail
    striding destroys small gaps. If the true cut is beyond front_full this will
    saturate at the front boundary (flagged via gap_at_boundary).

Run:
  python benchmark/analyse_sink_logs.py --log_dir result_release_ttft/infinitbench/qwen-token-retrieval-test/sink_logs --out_dir result_release_ttft/infinitbench/qwen-token-retrieval-test/sink_analysis
"""
import os, glob, csv, argparse
import numpy as np

THETAS = [0.90, 0.95, 0.97, 0.99, 0.999]


def load_records(log_dir):
    for f in sorted(glob.glob(os.path.join(log_dir, "*.npz"))):
        data = np.load(f, allow_pickle=True)
        for r in data["records"]:
            yield r


def stride_weights(ctx_pos, front_full, tail_stride):
    w = np.ones(len(ctx_pos), dtype=np.float64)
    w[ctx_pos >= front_full] = tail_stride
    return w


# ---------- block B: cumsum-k sweep (stride-weighted, full-dist estimate) ----------
def cumsum_k_sweep(scores, w, thetas=THETAS):
    """
    scores: kept post-soft-vote values; w: stride weights (tokens-per-point).
    We sort by per-TOKEN score (scores; the weight is multiplicity, not score),
    accumulate weighted mass, and report k as the IMPLIED number of full-res
    tokens needed to reach theta of total mass.
    """
    order = np.argsort(-scores)            # by score desc
    s = scores[order].astype(np.float64)
    ww = w[order]
    mass = s * ww                          # total mass contributed by each group
    total = mass.sum()
    if total <= 0:
        return {th: 0 for th in thetas}
    cdf = np.cumsum(mass) / total
    cum_tokens = np.cumsum(ww)             # implied full-res token count
    out = {}
    for th in thetas:
        idx = np.searchsorted(cdf, th)
        idx = min(idx, len(cum_tokens) - 1)
        out[th] = int(cum_tokens[idx])     # implied full-resolution k
    return out


# ---------- block C: gap detection on front full-res region ----------
def gap_detect(scores, ctx_pos, front_full):
    front_mask = ctx_pos < front_full
    s = scores[front_mask].astype(np.float64)
    if len(s) < 8:
        return None
    s = np.sort(s)[::-1]                    # desc
    s = s[s > 0]
    if len(s) < 8:
        return None
    # multiplicative gap S[i]/S[i+1]
    ratio = s[:-1] / (s[1:] + 1e-20)
    # additive gap
    diff = s[:-1] - s[1:]
    ratio_k = int(np.argmax(ratio)) + 1
    diff_k = int(np.argmax(diff)) + 1
    # sharpness: biggest additive gap vs median gap
    med_gap = np.median(diff) + 1e-20
    sharp = float(diff.max() / med_gap)
    n_front = len(s)
    return {
        "gap_ratio_k": ratio_k,
        "gap_diff_k": diff_k,
        "gap_sharpness": sharp,
        "gap_at_boundary": int(diff_k >= n_front - 1 or ratio_k >= n_front - 1),
        "n_front": n_front,
    }


# ---------- block A: sink shape (kept minimal) ----------
def front_mass(probs, ctx_pos, n_init, ex=128):
    return float(probs[ctx_pos < (n_init + ex)].sum())


def analyse(log_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    groups = {}
    n = 0
    for r in load_records(log_dir):
        n += 1
        ds, layer = r["dataset"], int(r["layer_id"])
        n_init = int(r["n_init"]) if r["n_init"] >= 0 else 128
        front_full = int(r["front_full"]); tail_stride = int(r["tail_stride"])
        ctx_pos = np.asarray(r["ctx_pos"]).astype(np.int64)
        scores = np.asarray(r["scores"]).astype(np.float64)
        if len(scores) < 8:
            continue
        w = stride_weights(ctx_pos, front_full, tail_stride)

        # block A
        m_tot = (scores * w)
        probs = m_tot / m_tot.sum() if m_tot.sum() > 0 else m_tot
        fm128 = front_mass(probs, ctx_pos, n_init, 128)

        # block B
        ksweep = cumsum_k_sweep(scores, w)

        # block C
        gap = gap_detect(scores, ctx_pos, front_full)

        rec = {"fm128": fm128, "ksweep": ksweep, "gap": gap, "T_full": int(r["T_full"])}
        # NEW: kéo k_effective nếu logger có ghi
        rec["keff"] = r.get("k_effective") if isinstance(r, dict) else None
        groups.setdefault((ds, layer), []).append(rec)

    rows = []
    for (ds, layer), ms in sorted(groups.items()):
        def stat(getter):
            v = [getter(x) for x in ms]
            v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
            if not v:
                return (float("nan"),) * 3
            return (float(np.median(v)), float(np.min(v)), float(np.max(v)))

        row = {"dataset": ds, "layer": layer, "n_calls": len(ms)}
        row["fm128_med"] = stat(lambda x: x["fm128"])[0]
        # cumsum k sweep: med/min/max per theta
        for th in THETAS:
            med, mn, mx = stat(lambda x: x["ksweep"][th])
            tag = f"k{int(th*100)}"
            row[f"{tag}_med"] = med
            row[f"{tag}_min"] = mn
            row[f"{tag}_max"] = mx
        # gap
        row["gap_ratio_k_med"] = stat(lambda x: x["gap"]["gap_ratio_k"] if x["gap"] else None)[0]
        row["gap_diff_k_med"] = stat(lambda x: x["gap"]["gap_diff_k"] if x["gap"] else None)[0]
        row["gap_sharp_med"] = stat(lambda x: x["gap"]["gap_sharpness"] if x["gap"] else None)[0]
        row["gap_boundary_frac"] = np.mean([x["gap"]["gap_at_boundary"] for x in ms if x["gap"]]) if any(x["gap"] for x in ms) else float("nan")
        # k_effective (chỉ có nếu logger v3 ghi)
        has_keff = any(x["keff"] is not None for x in ms)
        if has_keff:
            row["k_sum_ppl_med"]  = stat(lambda x: x["keff"]["k_sum_ppl"] if x["keff"] else None)[0]
            row["k_post_ppl_med"] = stat(lambda x: x["keff"]["k_post_ppl"] if x["keff"] else None)[0]
            row["ppl_head_med"]   = stat(lambda x: x["keff"]["ppl_head_med"] if x["keff"] else None)[0]
            row["E_norm_med"]     = stat(lambda x: x["keff"]["E_norm_med"] if x["keff"] else None)[0]
            # med_k_per_head và union theo theta (nếu có block C)
            for th in (0.90, 0.99):
                row[f"medkh{int(th*100)}"] = stat(
                    lambda x: x["keff"]["med_k_per_head"].get(th) if x["keff"] and x["keff"].get("med_k_per_head") else None
                )[0]
                row[f"union{int(th*100)}"] = stat(
                    lambda x: x["keff"]["k_union_cumsum"].get(th) if x["keff"] and x["keff"].get("k_union_cumsum") else None
                )[0]
        rows.append(row)

    if rows:
        with open(os.path.join(out_dir, "cumsum_gap_summary.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader()
            for r in rows: wr.writerow(r)

    print(f"Read {n} records, {len(rows)} (dataset,layer) groups.")
    
    # NEW: generate markdown formatted output
    print_markdown_keff_table(rows)
    print_markdown_tables(rows)


def print_markdown_keff_table(rows):
    rows = [r for r in rows if "k_sum_ppl_med" in r]
    if not rows:
        return
    print("### K_EFFECTIVE vs CUMSUM THẬT (đối chiếu công thức ước lượng với k thật)")
    print()
    print("| dataset | L | A:Σppl_h | B:ppl_post | ppl/head | medkh90 | union90 | union99 | k90_real | k99_real |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        ds = r['dataset'][:15]
        l = r['layer']
        a = f"{r.get('k_sum_ppl_med',float('nan')):.0f}"
        b = f"{r.get('k_post_ppl_med',float('nan')):.1f}"
        ph = f"{r.get('ppl_head_med',float('nan')):.2f}"
        mkh90 = f"{r.get('medkh90',float('nan')):.0f}"
        u90 = f"{r.get('union90',float('nan')):.0f}"
        u99 = f"{r.get('union99',float('nan')):.0f}"
        k90r = f"{r.get('k90_med', float('nan')):.0f}"
        k99r = f"{r.get('k99_med', float('nan')):.0f}"
        print(f"| {ds} | {l} | {a} | {b} | {ph} | {mkh90} | {u90} | {u99} | {k90r} | {k99r} |")
    print()
    print("**ĐỌC:**")
    print("- **A (Σ exp H_h)** ≈ `k90_real` -> Σperplexity per-head là estimator tốt cho mass-chính, RẺ.")
    print("- **B (ppl post-sum)** << `k thật` -> xác nhận công thức post-sum cũ SAI.")
    print("- **union99** ≈ `k99_real` -> head-wise cumsum θ=0.99 bắt đúng cả đuôi/needle.")
    print("- **ppl/head** ~1-2 -> head gần delta; k lớn là do UNION nhiều head, không phải head phẳng.")
    print()


def print_markdown_tables(rows):
    print("### CUMSUM-K SWEEP (implied full-res k to reach theta; med [min-max])")
    print()
    print("| dataset | L | fm128 | k90 med[min-max] | k95 | k99 med[min-max] |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        ds = r['dataset'][:15]
        l = r['layer']
        fm = f"{r['fm128_med']:.3f}"
        k90 = f"{r['k90_med']:.0f} [{r['k90_min']:.0f}-{r['k90_max']:.0f}]"
        k95 = f"{r['k95_med']:.0f}"
        k99 = f"{r['k99_med']:.0f} [{r['k99_min']:.0f}-{r['k99_max']:.0f}]"
        print(f"| {ds} | {l} | {fm} | {k90} | {k95} | {k99} |")
    print()
    
    print("### GAP DETECTION (front full-res region)")
    print()
    print("| dataset | L | gap_ratio_k | gap_diff_k | sharpness | @boundary |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        ds = r['dataset'][:15]
        l = r['layer']
        grk = f"{r['gap_ratio_k_med']:.0f}"
        gdk = f"{r['gap_diff_k_med']:.0f}"
        sharp = f"{r['gap_sharp_med']:.1f}"
        bound = f"{r['gap_boundary_frac']:.2f}"
        print(f"| {ds} | {l} | {grk} | {gdk} | {sharp} | {bound} |")
    print()
    print("**READ:**")
    print("- **CUMSUM:** `k99 med` = how many tokens cumsum(0.99) keeps. min-max spread")
    print("  - WIDE = unstable (k jumps call to call) -> unstable score.")
    print("- `k99` >> `k90` = long tail needed for last 9% mass (flat tail).")
    print("- **GAP:** sharpness HIGH = clear natural boundary (safe to cut there).")
    print("  - sharpness LOW + `@boundary~1` = no boundary in front; cut is arbitrary -> THIS is the unstable-task signature.")
    print("- Compare `gap_ratio_k` vs `k99`: if `gap_k` << `k99`, cumsum overshoots past the natural boundary (keeps rac); if `gap_k` >> `k99`, cumsum cuts into the important region (loses content).")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    analyse(args.log_dir, args.out_dir)