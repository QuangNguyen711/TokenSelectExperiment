import sys, os, json, glob, statistics as st

# Doc *.gap.log (schema moi: k_perplex_max/med, k99_head_max/med).
# Doi so: thu muc goc HOAC danh sach file. Khong co -> tim tu thu muc hien tai.
args = sys.argv[1:] or ["."]
files = []
for a in args:
    if os.path.isdir(a):
        files += glob.glob(os.path.join(a, "**", "*.gap.log"), recursive=True)
    else:
        files.append(a)
files = sorted(set(files))
if not files:
    print("Khong tim thay *.gap.log"); sys.exit(1)

def dname(f): return os.path.basename(f).replace(".gap.log","")

allr, byds = {}, {}
for f in files:
    ds = dname(f)
    for line in open(f):
        line=line.strip()
        if not line: continue
        try: r=json.loads(line)
        except json.JSONDecodeError: continue
        if "k99_head_max" not in r: continue   # bo schema cu
        allr.setdefault(r["layer"],[]).append(r)
        byds.setdefault(ds,{}).setdefault(r["layer"],[]).append(r)

def med(recs,k): return st.median([r[k] for r in recs if k in r])

def table(title, by_layer):
    print("="*86)
    print(title)
    # k99_head_max = K an toan (head can nhieu nhat de du 99% mass) -> day la K cho "so thieu"
    print(f"{'Layer':>5} {'#cl':>5} {'perplex_med':>11} {'perplex_max':>11} {'k99head_med':>11} {'k99head_max':>11} {'maxK?':>6}")
    for L in sorted(by_layer):
        r=by_layer[L]
        km = med(r,'k99_head_max')
        flag = "FULL" if km >= 8000 else ""   # neu ~8192 thi cat vo nghia
        print(f"{L:>5} {len(r):>5} {med(r,'k_perplex_med'):>11.0f} {med(r,'k_perplex_max'):>11.0f} "
              f"{med(r,'k99_head_med'):>11.0f} {km:>11.0f} {flag:>6}")

print("Files:")
for f in files: print(f"  {dname(f):>20}: {f}")
print()
table("TONG HOP (gop tat ca dataset)", allr)
for ds in sorted(byds):
    print()
    table(f"Dataset: {ds}", byds[ds])

print("="*86)
print("Doc cot:")
print("  perplex_max  : K neu dung max-over-heads perplexity (y cua ban). Day la K 'so thieu'.")
print("  k99head_max  : so token de HEAD KHO NHAT dat 99% mass. Day la K an toan THAT (do thang, ko gia dinh).")
print("  maxK?=FULL   : k99head_max >= 8000 -> head kho nhat can gan het -> cat VO NGHIA o layer nay (giu 8192).")
print()
print("Quyet dinh:")
print("  - Layer co k99head_max nho (vai tram) -> cat duoc, max-over-heads an toan. BAT adaptive.")
print("  - Layer FULL -> giu 8192.")
print("  - So perplex_max vs k99head_max: neu perplex_max < k99head_max nhieu -> perplexity CAT QUA TAY (thieu mass).")