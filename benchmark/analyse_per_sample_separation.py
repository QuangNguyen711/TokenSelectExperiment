# File: benchmark/analyse_per_sample_separation.py
"""
Trả lời: k_eff / entropy per-SAMPLE có tách bạch giữa task không (median đã biết là khác),
hay phân bố per-sample CHỒNG nhau (median khác chỉ là ảo giác gộp).

- Mỗi record = 1 call (1 cluster). Ta gộp các call về MỖI (dataset, sample_id, layer)
  -> lấy k_eff đại diện cho sample đó (median các call trong sample).
- Rồi cho mỗi layer: vẽ phân bố k_eff per-sample theo task + tính overlap pairwise.

overlap_metric: với cặp task (A=retrieval, B=ngôn ngữ), tính
  P(k_eff của sample A  <  median k_eff của task B)
  = tỉ lệ sample task-khó bị "nhìn nhầm" thành dễ.
  ~0   -> tách bạch  -> adaptive proxy task tốt  (H_mạnh, adaptive sống)
  >0.2 -> chồng lấn  -> adaptive cấp k sai cho outlier (H_yếu, k-tĩnh an toàn hơn)

Chạy:
  python benchmark/analyse_per_sample_separation.py \
    --log_dir result_release_ttft/infinitbench/qwen-token-retrieval-test/sink_logs \
    --out_dir result_release_ttft/infinitbench/qwen-token-retrieval-test/sep_analysis
"""
import os, glob, csv, argparse
import numpy as np

THETAS = [0.90, 0.99]
# phân loại task thô để tính overlap retrieval-vs-language
RETRIEVAL = {"kv_retrieval", "passkey", "number_string", "math_find"}
LANGUAGE  = {"longdialogue_qa_eng", "code_debug"}


def load_records(log_dir):
    for f in sorted(glob.glob(os.path.join(log_dir, "*.npz"))):
        data = np.load(f, allow_pickle=True)
        for r in data["records"]:
            yield r


def stride_weights(ctx_pos, front_full, tail_stride):
    w = np.ones(len(ctx_pos), dtype=np.float64)
    w[ctx_pos >= front_full] = tail_stride
    return w


def cumsum_k(scores, w, theta):
    """k cumsum trên post-sum (stride-weighted), giống analyse_sink_logs."""
    order = np.argsort(-scores)
    s = scores[order].astype(np.float64)
    ww = w[order]
    mass = s * ww
    total = mass.sum()
    if total <= 0:
        return 0
    cdf = np.cumsum(mass) / total
    cum_tokens = np.cumsum(ww)
    idx = min(np.searchsorted(cdf, theta), len(cum_tokens) - 1)
    return int(cum_tokens[idx])


def post_entropy(scores, w):
    """entropy chuẩn hóa của post-sum (stride-weighted), in [0,1]."""
    m = (scores.astype(np.float64) * w)
    tot = m.sum()
    if tot <= 0:
        return float("nan")
    p = m / tot
    p = p[p > 0]
    H = -(p * np.log(p)).sum()
    return float(H / np.log(len(p))) if len(p) > 1 else 0.0


def analyse(log_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # gom theo (dataset, sample_id, layer): mỗi sample 1 đại diện
    per_call = {}   # key -> list of dict(k90,k99,Hpost)
    n = 0
    for r in load_records(log_dir):
        n += 1
        ds = r["dataset"]; layer = int(r["layer_id"]); sid = int(r["sample_id"])
        ctx = np.asarray(r["ctx_pos"]).astype(np.int64)
        sc  = np.asarray(r["scores"]).astype(np.float64)
        if len(sc) < 8:
            continue
        w = stride_weights(ctx, int(r["front_full"]), int(r["tail_stride"]))
        d = {
            "k90": cumsum_k(sc, w, 0.90),
            "k99": cumsum_k(sc, w, 0.99),
            "Hpost": post_entropy(sc, w),
        }
        per_call.setdefault((ds, sid, layer), []).append(d)

    # 1 sample = median các call trong sample đó
    per_sample = {}  # (ds, layer) -> list of sample-level dict
    for (ds, sid, layer), calls in per_call.items():
        agg = {
            "k90": float(np.median([c["k90"] for c in calls])),
            "k99": float(np.median([c["k99"] for c in calls])),
            "Hpost": float(np.median([c["Hpost"] for c in calls])),
        }
        per_sample.setdefault((ds, layer), []).append(agg)

    # ---- bảng 1: phân bố per-sample mỗi (task, layer): min / p25 / med / p75 / max ----
    rows = []
    for (ds, layer), samps in sorted(per_sample.items()):
        for metric in ("k99", "Hpost"):
            v = np.array([s[metric] for s in samps], dtype=np.float64)
            v = v[~np.isnan(v)]
            if len(v) == 0:
                continue
            rows.append({
                "dataset": ds, "layer": layer, "metric": metric, "n_samples": len(v),
                "min": float(v.min()), "p25": float(np.percentile(v, 25)),
                "med": float(np.median(v)), "p75": float(np.percentile(v, 75)),
                "max": float(v.max()),
            })

    # ---- bảng 2: overlap retrieval-vs-language per layer (dùng k99 per-sample) ----
    # với mỗi layer: gộp tất cả sample retrieval, tất cả sample language,
    # tính P(retrieval_sample_k < median(language_k))  và chiều ngược lại,
    # cùng "separation" = (med_retr - med_lang)/(spread).
    layers = sorted({layer for (_, layer) in per_sample.keys()})
    overlap_rows = []
    for layer in layers:
        retr = []; lang = []
        for (ds, lyr), samps in per_sample.items():
            if lyr != layer:
                continue
            ks = [s["k99"] for s in samps if not np.isnan(s["k99"])]
            if ds in RETRIEVAL:
                retr += ks
            elif ds in LANGUAGE:
                lang += ks
        if len(retr) < 2 or len(lang) < 2:
            continue
        retr = np.array(retr); lang = np.array(lang)
        med_lang = np.median(lang); med_retr = np.median(retr)
        # tỉ lệ sample retrieval bị nhìn như "dễ" (k thấp hơn median language)
        p_retr_below = float((retr < med_lang).mean())
        # tỉ lệ sample language bị nhìn như "khó" (k cao hơn median retrieval)
        p_lang_above = float((lang > med_retr).mean())
        # separation chuẩn hóa
        spread = (np.std(retr) + np.std(lang)) / 2 + 1e-9
        sep = float((med_retr - med_lang) / spread)
        overlap_rows.append({
            "layer": layer, "n_retr": len(retr), "n_lang": len(lang),
            "med_retr_k99": med_retr, "med_lang_k99": med_lang,
            "P(retr<med_lang)": p_retr_below, "P(lang>med_retr)": p_lang_above,
            "separation": sep,
        })

    # ---- ghi CSV ----
    if rows:
        with open(os.path.join(out_dir, "per_sample_dist.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader()
            for r in rows: wr.writerow(r)
    if overlap_rows:
        with open(os.path.join(out_dir, "task_overlap.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(overlap_rows[0].keys())); wr.writeheader()
            for r in overlap_rows: wr.writerow(r)

    print(f"Read {n} records.")
    _print(rows, overlap_rows)


def _print(rows, overlap_rows):
    print("="*100)
    print("PHÂN BỐ k99 PER-SAMPLE  (mỗi sample = median các cluster trong nó)")
    print("Nếu min-max RỘNG trong cùng task -> sample dao động mạnh -> median gộp che mất")
    print("="*100)
    print(f"{'dataset':<16}{'L':<4}{'metric':<7}{'n':<4}{'min':<9}{'p25':<9}{'med':<9}{'p75':<9}{'max':<9}")
    for r in rows:
        if r["metric"] != "k99":
            continue
        print(f"{r['dataset'][:15]:<16}{r['layer']:<4}{r['metric']:<7}{r['n_samples']:<4}"
              f"{r['min']:<9.0f}{r['p25']:<9.0f}{r['med']:<9.0f}{r['p75']:<9.0f}{r['max']:<9.0f}")
    print("="*100)
    print("CHỒNG LẤN TASK retrieval-vs-language theo layer (k99 per-sample)")
    print("="*100)
    print(f"{'L':<4}{'n_retr':<8}{'n_lang':<8}{'med_retr':<10}{'med_lang':<10}"
          f"{'P(retr<medL)':<14}{'P(lang>medR)':<14}{'separation':<11}")
    for r in overlap_rows:
        print(f"{r['layer']:<4}{r['n_retr']:<8}{r['n_lang']:<8}"
              f"{r['med_retr_k99']:<10.0f}{r['med_lang_k99']:<10.0f}"
              f"{r['P(retr<med_lang)']:<14.2f}{r['P(lang>med_retr)']:<14.2f}{r['separation']:<11.2f}")
    print("="*100)
    print("ĐỌC:")
    print(" P(retr<medL) ~0  & separation lớn (>2) -> TÁCH BẠCH per-sample")
    print("    -> entropy/k là proxy TASK tốt -> ADAPTIVE SỐNG (H_mạnh).")
    print(" P(retr<medL) >0.2 hoặc separation <1 -> CHỒNG LẤN")
    print("    -> sample retrieval khó có thể bị cấp k nhỏ -> trượt needle")
    print("    -> k-TĨNH-PER-LAYER an toàn hơn (H_yếu).")
    print(" Xem riêng layer GIỮA (6/13/20) — đó là vùng định bật adaptive.")
    print("="*100)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    analyse(args.log_dir, args.out_dir)