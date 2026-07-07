#!/usr/bin/env python3
"""
Phân tích log probe ngưỡng xác suất -> xuất MARKDOWN.

Đọc:  probe_<exp>_<dataset>_pid<...>.log
Dòng:  L13 T=15744 Hn=0.1234 ppl=312 REL(1e8=.. 1e7=.. 1e6=.. 1e5=.. 1e4=..) UNI(1=.. 2=.. 3=.. 5=.. 10=..)

Xuất 4 bảng markdown:
  1. Entropy & ppl theo (dataset, layer)
  2. So entropy giữa dataset (cùng layer)
  3. k-per-layer theo REL và UNI
  4. "Cùng Hn, khác k" — minh họa entropy nén mất thông tin đuôi

Dùng:
  python analyse_probe_md.py --log_dir ./hwa_logs > report.md
  python analyse_probe_md.py --log_dir ./hwa_logs --strip_prefix probe_normal_testing_2_ > report.md
"""
import os, re, glob, argparse
from collections import defaultdict
import statistics as st

LINE_RE = re.compile(
    r"L(?P<layer>\d+)\s+T=(?P<T>\d+)\s+"
    r"Hn=(?P<Hn>[-\d.]+)\s+ppl=(?P<ppl>[-\d.]+)\s+"
    r"REL\((?P<rel>[^)]*)\)\s+"
    r"UNI\((?P<uni>[^)]*)\)"
)

REL_ORDER = ["1e13", "1e12", "1e11", "1e10", "1e9", "1e8"]
UNI_ORDER = ["1", "2", "3", "5", "10"]

def parse_kv(s):
    out = {}
    for tok in s.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    pass
    return out

def dataset_from_path(p, strip_prefix):
    base = os.path.basename(p)
    if strip_prefix:
        base = re.sub(r"^" + re.escape(strip_prefix), "", base)
    else:
        base = re.sub(r"^probe_normal_testing_2_", "", base)
    base = re.sub(r"_pid\d+\.log$", "", base)
    return base

def load(log_dir, strip_prefix):
    recs = []
    files = sorted(glob.glob(os.path.join(log_dir, "probe_*.log")))
    for f in files:
        ds = dataset_from_path(f, strip_prefix)
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                m = LINE_RE.search(line)
                if not m:
                    continue
                recs.append((
                    ds, int(m.group("layer")), int(m.group("T")),
                    float(m.group("Hn")), float(m.group("ppl")),
                    parse_kv(m.group("rel")), parse_kv(m.group("uni")),
                ))
    return recs, files

def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else float("nan")

def summarize(recs):
    groups = defaultdict(list)
    for ds, layer, T, Hn, ppl, rel, uni in recs:
        groups[(ds, layer)].append((T, Hn, ppl, rel, uni))
    rows = []
    for (ds, layer), items in sorted(groups.items()):
        rel_keys = set().union(*[it[3].keys() for it in items])
        uni_keys = set().union(*[it[4].keys() for it in items])
        rows.append({
            "dataset": ds, "layer": layer, "n": len(items),
            "T_med": med([it[0] for it in items]),
            "Hn_med": med([it[1] for it in items]),
            "ppl_med": med([it[2] for it in items]),
            "rel": {k: med([it[3].get(k) for it in items]) for k in rel_keys},
            "uni": {k: med([it[4].get(k) for it in items]) for k in uni_keys},
        })
    return rows

def fmt(x, nd=0):
    if x is None or (isinstance(x, float) and x != x):  # nan
        return "—"
    return f"{x:.{nd}f}"

def md_entropy_table(rows):
    out = ["## 1. Entropy & perplexity theo (dataset, layer)",
           "",
           "`Hn` = entropy chuẩn hóa post-sum (cao = phẳng, thấp = nhọn). `ppl` = perplexity tuyệt đối.",
           "",
           "| dataset | L | n | T_med | Hn | ppl |",
           "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        out.append(f"| {r['dataset']} | {r['layer']} | {r['n']} | "
                   f"{fmt(r['T_med'])} | {fmt(r['Hn_med'],4)} | {fmt(r['ppl_med'])} |")
    return "\n".join(out)

def md_entropy_by_dataset(rows):
    by_layer = defaultdict(dict)
    for r in rows:
        by_layer[r["layer"]][r["dataset"]] = r["Hn_med"]
    datasets = sorted({r["dataset"] for r in rows})
    out = ["## 2. So entropy giữa dataset (cùng layer)",
           "",
           "Mỗi cột một dataset. Đọc theo hàng: cùng layer, entropy các task khác nhau thế nào.",
           "",
           "| layer | " + " | ".join(datasets) + " |",
           "|---:|" + "|".join(["---:"] * len(datasets)) + "|"]
    for layer in sorted(by_layer):
        cells = " | ".join(fmt(by_layer[layer].get(d), 4) for d in datasets)
        out.append(f"| {layer} | {cells} |")
    return "\n".join(out)

def md_k_table(rows, which):
    order = REL_ORDER if which == "rel" else UNI_ORDER
    label = "REL — `a ≥ θ`" if which == "rel" else "UNI — `a ≥ θ·(1/T)`"
    title = "3a" if which == "rel" else "3b"
    out = [f"## {title}. k-per-layer theo {label}",
           "",
           "Số token mỗi ngưỡng sẽ giữ (trung vị qua các call).",
           "",
           "| dataset | L | T_med | " + " | ".join(order) + " |",
           "|---|---:|---:|" + "|".join(["---:"] * len(order)) + "|"]
    for r in rows:
        cells = " | ".join(fmt(r[which].get(th)) for th in order)
        out.append(f"| {r['dataset']} | {r['layer']} | {fmt(r['T_med'])} | {cells} |")
    return "\n".join(out)

def md_same_entropy_diff_k(rows, hn_tol=0.02, ref_th="1e5"):
    """
    Minh họa luận điểm: cùng Hn (trong dải hn_tol) nhưng k (số token vượt ngưỡng ref_th)
    rất khác nhau -> entropy nén mất thông tin hình dạng đuôi.
    Gom các (dataset,layer) theo bucket Hn, trong mỗi bucket in min/max k để thấy spread.
    """
    # bucket theo Hn làm tròn về bội hn_tol
    buckets = defaultdict(list)
    for r in rows:
        k = r["rel"].get(ref_th)
        if k is None or r["Hn_med"] != r["Hn_med"]:
            continue
        b = round(r["Hn_med"] / hn_tol) * hn_tol
        buckets[b].append((r["dataset"], r["layer"], r["Hn_med"], k, r["T_med"]))
    out = [f"## 4. Cùng entropy, khác k — entropy nén mất thông tin đuôi",
           "",
           f"Gom (dataset, layer) theo bucket Hn rộng ±{hn_tol}. Trong mỗi bucket, "
           f"k = số token vượt ngưỡng `{ref_th}·max`. Nếu k chênh lớn trong cùng bucket "
           f"→ cùng entropy nhưng phân bố đuôi khác hẳn → entropy không phân biệt được, ngưỡng thì có.",
           "",
           "| Hn≈ | #điểm | k_min | k_max | tỉ lệ max/min | ví dụ k_min | ví dụ k_max |",
           "|---:|---:|---:|---:|---:|---|---|"]
    for b in sorted(buckets):
        items = buckets[b]
        if len(items) < 2:
            continue
        ks = [it[3] for it in items]
        kmin = min(items, key=lambda x: x[3])
        kmax = max(items, key=lambda x: x[3])
        ratio = (kmax[3] / kmin[3]) if kmin[3] > 0 else float("inf")
        ex_min = f"{kmin[0]} L{kmin[1]}"
        ex_max = f"{kmax[0]} L{kmax[1]}"
        out.append(f"| {b:.2f} | {len(items)} | {kmin[3]} | {kmax[3]} | "
                   f"{ratio:.1f}× | {ex_min} | {ex_max} |")
    out.append("")
    out.append("> Cột **tỉ lệ max/min** càng lớn càng chứng minh: ở cùng một mức entropy, "
               "số token \"đáng giữ\" theo ngưỡng xác suất biến thiên mạnh. Entropy (một scalar) "
               "không nắm được sự khác biệt này; ngưỡng đọc trực tiếp hình dạng phân bố nên nắm được.")
    return "\n".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default="./hwa_logs")
    ap.add_argument("--strip_prefix", default=None,
                    help="tiền tố file để bỏ, vd 'probe_normal_testing_2_' (để lại đúng tên dataset)")
    ap.add_argument("--hn_tol", type=float, default=0.02)
    ap.add_argument("--ref_th", default="1e5", choices=REL_ORDER)
    args = ap.parse_args()

    recs, files = load(args.log_dir, args.strip_prefix)
    print(f"<!-- Đọc {len(files)} file, {len(recs)} dòng hợp lệ. -->\n")
    if not recs:
        print("**Không parse được dòng nào.** Kiểm tra định dạng log / đường dẫn.")
        return
    rows = summarize(recs)

    print(md_entropy_table(rows));        print()
    print(md_entropy_by_dataset(rows));   print()
    print(md_k_table(rows, "rel"));       print()
    print(md_k_table(rows, "uni"));       print()
    print(md_same_entropy_diff_k(rows, args.hn_tol, args.ref_th))

if __name__ == "__main__":
    main()