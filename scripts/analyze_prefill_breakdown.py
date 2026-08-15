"""
Tong hop breakdown prefill: mean +/- std tren tung sample, da tru overhead cua bo do.

Ba dieu chinh so voi ban do dau tien:
  1. Bo ban ghi warmup trong ttft.log (SGLang tu ban 1 request ~0.1s luc khoi tao)
     -> truoc day keo tut TTFT trung binh.
  2. Bao cao theo /sample thay vi cong don ca run.
  3. Boc toan bo attention forward ("0_total_attn_forward") -> phan chua duoc quy
     vao giai doan nao hien ra thanh muc "bookkeeping" thay vi bien mat.
Overhead cua chinh bo do duoc uoc luong tu chenh lech TTFT (on - off) chia cho
so lan bam gio, roi tru khoi tung giai doan theo so lan goi cua giai doan do.
"""
import glob
import json
import os
import statistics as st
import sys

STAGE_LABEL = {
    "1_chunk_plan":    "Chunk-plan construction",
    "2_store_kv":      "KV-cache store",
    "3_retrieval":     "Token retrieval",
    "4_wrapper_setup": "FlashInfer wrapper setup",
    "5_attention":     "Attention compute",
    "_bookkeeping":    "Loop bookkeeping",
}
SUBSTAGES = ["1_chunk_plan", "2_store_kv", "3_retrieval", "4_wrapper_setup", "5_attention"]
TOTAL = "0_total_attn_forward"


def ttft(path, n_keep):
    v = [float(x) for x in open(path) if x.strip()]
    return v[-n_keep:] if n_keep and len(v) > n_keep else v


def n_preds(d, ds):
    p = os.path.join(d, f"{ds}.jsonl")
    return sum(1 for _ in open(p)) if os.path.exists(p) else 0


def load_prof(d):
    f = glob.glob(os.path.join(d, "breakdown.pid*.json"))
    return json.load(open(f[0])) if f else None


def analyse(root, ds="kv_retrieval"):
    out = {}
    for name in ("baseline", "scr"):
        d_off, d_on = f"{root}/{name}-off", f"{root}/{name}-on"
        n = n_preds(d_off, ds)
        t_off = ttft(f"{d_off}/{ds}.ttft.log", n)
        t_on = ttft(f"{d_on}/{ds}.ttft.log", n_preds(d_on, ds))
        prof = load_prof(d_on)
        if prof is None:
            print(f"thieu breakdown cho {name}"); return

        ns = max(len(prof.get("per_sample") or []), 1)
        raw = {k: prof["stages"].get(k, 0.0) / ns for k in SUBSTAGES + [TOTAL]}
        per_call = 0.0
        corr = dict(raw)   # 'corr' = quy doi ve luot sach, tinh o duoi
        corr["_bookkeeping"] = corr[TOTAL] - sum(corr[s] for s in SUBSTAGES)
        raw["_bookkeeping"] = raw[TOTAL] - sum(raw[s] for s in SUBSTAGES)

        # mean +/- std cho tung giai doan tren cac sample
        spread = {}
        for s in SUBSTAGES + [TOTAL]:
            vals = [ps.get(s, 0.0) for ps in (prof.get("per_sample") or [])]
            spread[s] = (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0) if vals else (0, 0)

        out[name] = dict(n=n, ttft=t_off, ttft_on=t_on, raw=raw, corr=corr,
                         spread=spread, counts=prof["counts"], ns=ns,
                         per_call=per_call)
    return out


def main(root="result_prefill_breakdown", ds="kv_retrieval"):
    r = analyse(root, ds)
    if not r:
        return
    b, s = r["baseline"], r["scr"]
    n = min(len(b["ttft"]), len(s["ttft"]))

    print("=" * 82)
    print(f"BREAKDOWN PREFILL — {ds}, n = {n} sample")
    print("=" * 82)

    print("\nTTFT (khong bat profiler), giay/sample:")
    for k, v in (("TokenSelect goc", b), ("TokenSelect + SCR", s)):
        m = st.mean(v["ttft"]); sd = st.stdev(v["ttft"]) if len(v["ttft"]) > 1 else 0
        print(f"  {k:22} {m:7.3f} +/- {sd:5.3f}")
    mb, ms = st.mean(b["ttft"]), st.mean(s["ttft"])
    print(f"  {'chenh':22} {ms-mb:+7.3f} s/sample  ({(ms/mb-1)*100:+.1f}%)")

    print(f"\nOverhead con lai cua bo do (TTFT on - off): baseline +{st.mean(b['ttft_on'])-st.mean(b['ttft']):.3f}s/sample, "
          f"scr +{st.mean(s['ttft_on'])-st.mean(s['ttft']):.3f}s/sample")

    print("\nBreakdown do TRUC TIEP tren ca 28 layer, giay/sample (luot CO profiler):")
    print(f"  {'Giai doan':<28}{'goc':>10}{'SCR':>10}{'chenh':>11}{'% tiet kiem':>13}")
    print("  " + "-" * 70)
    saved = b["corr"][TOTAL] - s["corr"][TOTAL]
    for k in SUBSTAGES + ["_bookkeeping"]:
        x, y = b["corr"][k], s["corr"][k]
        print(f"  {STAGE_LABEL[k]:<28}{x:>9.3f}s{y:>9.3f}s{y-x:>+10.3f}s{-(y-x)/saved*100:>12.1f}%")
    print("  " + "-" * 70)
    print(f"  {'TONG attention forward':<28}{b['corr'][TOTAL]:>9.3f}s{s['corr'][TOTAL]:>9.3f}s"
          f"{s['corr'][TOTAL]-b['corr'][TOTAL]:>+10.3f}s{100.0:>12.1f}%")

    # doi chieu voi TTFT CUNG LUOT (co profiler) -> nhat quan noi tai
    mb_on, ms_on = st.mean(b["ttft_on"]), st.mean(s["ttft_on"])
    print(f"\nKiem tra nhat quan (cung luot co profiler):")
    print(f"  chenh TONG attention forward = {saved:.3f}s/sample")
    print(f"  chenh TTFT (luot co profiler) = {mb_on-ms_on:.3f}s/sample")
    print(f"  -> attention forward giai thich {saved/(mb_on-ms_on)*100:.1f}% muc giam "
          f"(phai <=100%; phan con lai la qkv_proj/MLP/LayerNorm)")
    print(f"  attention forward chiem {b['corr'][TOTAL]/mb_on*100:.1f}% TTFT o goc, "
          f"{s['corr'][TOTAL]/ms_on*100:.1f}% o SCR")

    k_scale = (mb-ms)/(mb_on-ms_on)
    print(f"\nQuy doi ve luot SACH (he so {k_scale:.4f} = chenh TTFT sach / chenh TTFT co profiler):")
    print(f"  {'Giai doan':<28}{'goc':>10}{'SCR':>10}{'chenh':>11}")
    print("  " + "-" * 59)
    for k in SUBSTAGES + ["_bookkeeping"]:
        x, y = b["corr"][k]*k_scale, s["corr"][k]*k_scale
        print(f"  {STAGE_LABEL[k]:<28}{x:>9.3f}s{y:>9.3f}s{y-x:>+10.3f}s")
    print("  " + "-" * 59)
    print(f"  {'TONG':<28}{b['corr'][TOTAL]*k_scale:>9.3f}s{s['corr'][TOTAL]*k_scale:>9.3f}s"
          f"{(s['corr'][TOTAL]-b['corr'][TOTAL])*k_scale:>+10.3f}s")

    print("\nDo tan mac giua cac sample (mean +/- std, giay/sample, CHUA tru overhead):")
    for k in SUBSTAGES + [TOTAL]:
        bm, bs_ = b["spread"][k]; sm, ss = s["spread"][k]
        print(f"  {STAGE_LABEL.get(k,k):<28}{bm:>7.3f}+/-{bs_:<6.3f}{sm:>8.3f}+/-{ss:<6.3f}")

    print("\nChi so cau truc:")
    for key, lab in (("n_subchunks", "So sub-chunk"),
                     ("n_retrieval_calls", "So lan retrieval"),
                     ("n_full_attn_fallback", "Lan bo qua selection"),
                     ("sum_kv_len", "Tong KV token")):
        x = b["counts"].get(key, 0) / b["ns"]; y = s["counts"].get(key, 0) / s["ns"]
        print(f"  {lab:<28}{x:>14,.0f}{y:>14,.0f}{y/x if x else 0:>9.2f}x")
    print("=" * 82)


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
