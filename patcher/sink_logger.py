# File: patcher/sink_logger.py  (v2 - lightweight)
"""
Sink-detection logger v2 for TokenSelect prefill.

WHY v2: v1 cloned the full [H, T] raw logit matrix and moved it to CPU EVERY
call. With T ~ 300k and ~585 chunks/sample x 5 layers, that is hundreds of GB
of PCIe traffic per sample -> the run hangs. v2 fixes this by:
  1. Logging only a SUBSET of calls (every N-th, capped per sample).
  2. NEVER storing [H, T]. Reduce over heads ON GPU to [T], keep FRONT region
     full-res (where sinks live) and DOWNSAMPLE the tail (every STRIDE-th).
  3. Reduce raw logits to a few scalars on GPU (top-2, min, max_abs) instead of
     shipping the matrix -> enough to separate true-sink vs bf16 collapse.
No-op unless ENABLED and prefill (chunk_len > 1).
"""
import os
import numpy as np
import torch

ENABLED = False
OUTPUT_DIR = None
DATASET = "unknown"
LAYERS_TO_LOG = {0, 6, 13, 20, 27}

LOG_EVERY_N_CALLS = 4
LOG_MAX_CALLS_PER_SAMPLE = 20
FRONT_FULL = 4096
TAIL_STRIDE = 32

_sample_id = -1
_call_seen = 0
_call_logged = 0
_buffer = []
_N_INIT = None
_N_LOCAL = None
_TOTAL_TOKENS = None
THETAS_KEFF = (0.90, 0.95, 0.99)

def _compute_k_effective(raw_2d):
    """
    raw_2d: [H, T] raw QK logits (pre-softmax), GPU tensor.
    Trả về dict các ước lượng k_effective theo nhiều công thức,
    để so với cumsum k90/k99 thật.
    TẤT CẢ tính trên fp32 để tránh bf16 collapse.
    """
    R = raw_2d.float()                       # [H, T]  ÉP fp32
    H, T = R.shape
    lnT = float(np.log(max(T, 2)))

    # --- per-head softmax ---
    P = torch.softmax(R, dim=-1)             # [H, T]
    eps = 1e-12

    # entropy thô từng head, và normalized
    H_raw = -(P * (P + eps).log()).sum(dim=-1)        # [H]  (nats)
    E_norm = H_raw / lnT                               # [H]  in [0,1]

    # perplexity per-head = exp(H_raw) = số token hiệu dụng của head đó
    ppl_head = torch.exp(H_raw)                        # [H]

    # --- A. tổng perplexity per-head (union-size estimate, upper bound nếu head rời nhau) ---
    k_sum_ppl = float(ppl_head.sum().item())

    # --- B. perplexity của post-sum (cái code cũ hay dùng) ---
    S = P.sum(dim=0)                                   # [T] post-sum
    S = S / (S.sum() + eps)
    H_post = -(S * (S + eps).log()).sum()
    k_post_ppl = float(torch.exp(H_post).item())

    # --- C. cumsum per-head tới theta rồi UNION (đúng head-wise nhất) ---
    #   với mỗi head sort desc, đếm tới khi CDF>=theta, lấy chỉ số token, union
    out_union = {}
    Psort, Pidx = torch.sort(P, dim=-1, descending=True)   # [H,T]
    cdf = torch.cumsum(Psort, dim=-1)                       # [H,T]
    for th in THETAS_KEFF:
        # k_h = số token tới khi đạt theta cho từng head
        reach = (cdf >= th).float()
        k_h = torch.argmax(reach, dim=-1) + 1              # [H]
        k_h = torch.where(reach.sum(dim=-1) > 0, k_h, torch.full_like(k_h, T))
        # union size: gom token id thật từng head rồi đếm unique
        mask = torch.zeros(T, dtype=torch.bool, device=R.device)
        for h in range(H):
            mask[Pidx[h, :k_h[h]]] = True
        out_union[th] = int(mask.sum().item())

    # --- D. median k_h per-head (không union) để biết mỗi head cần bao nhiêu ---
    med_k_h = {}
    for th in THETAS_KEFF:
        reach = (cdf >= th).float()
        k_h = torch.argmax(reach, dim=-1) + 1
        k_h = torch.where(reach.sum(dim=-1) > 0, k_h, torch.full_like(k_h, T))
        med_k_h[th] = float(k_h.float().median().item())

    return {
        "k_sum_ppl": k_sum_ppl,                # A: Σ exp(H_h)
        "k_post_ppl": k_post_ppl,              # B: exp(H_postsum)
        "k_union_cumsum": out_union,           # C: |union of per-head topθ|
        "med_k_per_head": med_k_h,             # D
        "E_norm_med": float(E_norm.median().item()),
        "E_norm_min": float(E_norm.min().item()),
        "ppl_head_med": float(ppl_head.median().item()),
        "n_heads": int(H), "T": int(T),
    }

def set_output_dir(path):
    global OUTPUT_DIR
    OUTPUT_DIR = path
    if path:
        os.makedirs(path, exist_ok=True)


def set_dataset(name):
    global DATASET
    DATASET = name


def enable(flag=True):
    global ENABLED
    ENABLED = flag


def set_geometry(n_init, n_local, total_tokens):
    global _N_INIT, _N_LOCAL, _TOTAL_TOKENS
    _N_INIT = int(n_init)
    _N_LOCAL = int(n_local)
    _TOTAL_TOKENS = int(total_tokens)


def next_sample():
    global _sample_id, _call_seen, _call_logged
    flush()
    _sample_id += 1
    _call_seen = 0
    _call_logged = 0


def _np(t):
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        t = t.detach()
        if t.is_cuda:
            t = t.cpu()
        return t.numpy()
    return np.asarray(t)


def _downsample_vector_gpu(vec, n_init):
    T = vec.shape[0]
    front_count = max(0, min(T, FRONT_FULL - n_init))
    front_vals = vec[:front_count]
    front_pos = torch.arange(n_init, n_init + front_count, device=vec.device, dtype=torch.int64)
    if T > front_count:
        tail_vals = vec[front_count::TAIL_STRIDE]
        tail_idx = torch.arange(front_count, T, TAIL_STRIDE, device=vec.device, dtype=torch.int64)
        tail_pos = tail_idx + n_init
        pos = torch.cat([front_pos, tail_pos])
        vals = torch.cat([front_vals, tail_vals])
    else:
        pos, vals = front_pos, front_vals
    return _np(pos.to(torch.int32)), _np(vals.to(torch.float32))


def log_call(*, layer_id, relevant_indices, scores_postvote,
             raw_scores_2d=None, key_norms=None, anchor_qnorm=None, chunk_len=None):
    global _call_seen, _call_logged
    if not ENABLED or OUTPUT_DIR is None:
        return
    if LAYERS_TO_LOG is not None and layer_id not in LAYERS_TO_LOG:
        return
    if _sample_id < 0:
        return
    if chunk_len is not None and chunk_len <= 1:
        return

    _call_seen += 1
    if (_call_seen % LOG_EVERY_N_CALLS) != 1:
        return
    if _call_logged >= LOG_MAX_CALLS_PER_SAMPLE:
        return

    n_init = _N_INIT if _N_INIT is not None else 128
    with torch.no_grad():
        sc = scores_postvote.float()
        pos_np, scores_np = _downsample_vector_gpu(sc, n_init)
        kn_np = None
        if key_norms is not None:
            _, kn_np = _downsample_vector_gpu(key_norms.float(), n_init)
        raw_stats = None
        keff = None
        if raw_scores_2d is not None:
            keff = _compute_k_effective(raw_scores_2d)
            R = raw_scores_2d.float()
            r = R.sum(dim=0)
            top2 = torch.topk(r, k=min(2, r.shape[0])).values
            max_raw = top2[0]
            second = top2[1] if top2.shape[0] > 1 else top2[0]
            min_raw = r.min()
            max_abs = R.abs().max()
            raw_stats = {
                "max_raw": float(max_raw.item()),
                "second_raw": float(second.item()),
                "min_raw": float(min_raw.item()),
                "max_abs": float(max_abs.item()),
            }

    rec = {
        "dataset": DATASET, "sample_id": _sample_id, "call_id": _call_seen,
        "layer_id": int(layer_id), "n_init": n_init,
        "n_local": _N_LOCAL if _N_LOCAL is not None else -1,
        "total_tokens": _TOTAL_TOKENS if _TOTAL_TOKENS is not None else -1,
        "chunk_len": int(chunk_len) if chunk_len is not None else -1,
        "T_full": int(scores_postvote.shape[0]),
        "front_full": FRONT_FULL, "tail_stride": TAIL_STRIDE,
        "ctx_pos": pos_np, "scores": scores_np, "key_norms": kn_np,
        "anchor_qnorm": float(anchor_qnorm) if anchor_qnorm is not None else float("nan"),
        "raw_stats": raw_stats,
        "k_effective": keff
    }
    _buffer.append(rec)
    _call_logged += 1


def flush():
    global _buffer
    if not _buffer or OUTPUT_DIR is None:
        _buffer = []
        return
    ds = _buffer[0]["dataset"]
    sid = _buffer[0]["sample_id"]
    out = os.path.join(OUTPUT_DIR, f"{ds}__sample{sid:03d}.npz")
    arr = np.empty(len(_buffer), dtype=object)
    for i, r in enumerate(_buffer):
        arr[i] = r
    np.savez_compressed(out, records=arr)
    _buffer = []


def finalize():
    flush()