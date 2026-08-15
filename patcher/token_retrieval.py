# File: patcher/token_retrieval.py

from contextlib import contextmanager
from math import ceil
from typing import Optional
# import patcher.sink_logger as slog
# # ---- value-norm probe logging ----
# import os as _os
# import atexit as _atexit
# _VLOG_FH = None
# _VLOG_COUNTERS = {}
# _VLOG_MAX_PER_LAYER = 500
# _VLOG_LAYERS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27}

# def _vlog_should(layer_id):
#     ds = _os.environ.get("SINK_DATASET", "unknown")
#     if getattr(_vlog_should, "_ds", None) != ds:
#         _VLOG_COUNTERS.clear()              # reset quota mỗi dataset
#         _vlog_should._ds = ds
#     if layer_id not in _VLOG_LAYERS:
#         return False
#     c = _VLOG_COUNTERS.get(layer_id, 0)
#     if c >= _VLOG_MAX_PER_LAYER:
#         return False
#     _VLOG_COUNTERS[layer_id] = c + 1
#     return True

# def _vlog_write(line):
#     global _VLOG_FH
#     ds = _os.environ.get("SINK_DATASET", "unknown")
#     if _VLOG_FH is None or getattr(_vlog_write, "_ds", None) != ds:
#         # đổi file khi sang dataset mới
#         if _VLOG_FH is not None:
#             _VLOG_FH.close()
#         p = _os.path.join(_os.environ.get("HWLOG_DIR", "./hwa_logs"),
#                           f"probe_{_os.environ.get('CURRENT_EXP','exp')}_{ds}_pid{_os.getpid()}.log")
#         _os.makedirs(_os.path.dirname(p), exist_ok=True)
#         _VLOG_FH = open(p, "a", encoding="utf-8")
#         _vlog_write._ds = ds
#         print(f"[VLOG] -> {_os.path.abspath(p)}", flush=True)
#     _VLOG_FH.write(line + "\n")
#     _VLOG_FH.flush()

import sglang
import torch
import torch.distributed as dist
import triton
import triton.language as tl
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode, InputMetadata
from vllm.config import (
    DeviceConfig,
    ModelConfig,
    LoRAConfig,
    MultiModalConfig,
    ParallelConfig,
    SchedulerConfig,
    CacheConfig,
)
from vllm.model_executor.layers import rotary_embedding
from vllm.model_executor.model_loader.loader import (
    DefaultModelLoader,
    _initialize_model,
    device_loading_context,
)
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
import time

_TRACKER_PREFILL_START = 0.0
GLOBAL_TOTAL_TTFT = 0.0
TTFT_RECORD_PATH = None
GAP_RECORD_PATH = None
SKIP_RECORD_PATH = None


def record_ttft(ttft: float):
    global GLOBAL_TOTAL_TTFT
    GLOBAL_TOTAL_TTFT += ttft

    if TTFT_RECORD_PATH:
        with open(TTFT_RECORD_PATH, "a", encoding="utf-8") as f:
            f.write(f"{ttft}\n")

ROPE_BASE = -1
ROPE_SCALE = -1
ROPE_MODE = ""
MAX_N_TOKENS = -1
TOP_K = -1
N_INIT = -1
N_Local = -1
PREFILL_CHUNK_SIZE = -1
QUERY_ROTATE = False
QUERY_CACHE = False
QUERY_CACHE_SIM_THRESHOLD = 0.9  # ngưỡng cosine similarity riêng cho Selection Cache (decode), tách khỏi SIM_THRESHOLD (Dynamic Chunking prefill)
KERNEL_SIZE=-1
ADAPTIVE_TOPK = False
ATTENTION_THRESHOLD = 0.9
WEIGHTED_SOFT_VOTE = False
UNION_OF_SETS = False
L2_NORM_POOLING = False
DYNAMIC_CAPACITY_UNION = False
HEAD_WISE_ADAPTIVE = False
DCU_ENERGY_MODE = "both"
SIM_THRESHOLD = 0.95
MAX_DYNAMIC_CHUNK = 1024
USE_DYNAMIC_CHUNKING = False
DYNAMIC_BUDGET_BALANCING = True
USE_CUMSUM_ADAPTIVE = False
USE_HYBRID_ADAPTIVE = False
CUMSUM_THRESHOLD = 0.95
N_TAIL = 2048
PPL_MODE = "sum"     # "sum" = Σexp(H_h);  "post" = exp(H_postsum)

@contextmanager
def cuda_timer(timer_name="Operation"):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    yield

    end.record()
    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(end)
    print(f"{timer_name} time (ms): {elapsed_time:.4f}")


class RotaryEmbedding(torch.nn.Module):
    def __init__(
            self,
            dim: int,
            base: float,
            distance_scale: float = 1.0,
            device: torch.device = "cuda",
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.distance_scale = distance_scale
        self.device = device
        self._seq_len_cached = -1
        self._cos_table = None
        self._sin_table = None

    def _init_inv_freq(self):
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (
                self.base
                ** (
                        torch.arange(0, self.dim, 2, device=self.device, dtype=torch.float32)
                        / self.dim
                )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _update_cos_sin_tables(self, seq_len: int):
        """
        _cos_table and _sin_table are 2-D tensors, but adapts to 2-4D input via broadcast.
        """
        if not hasattr(self, "inv_freq"):
            self._init_inv_freq()
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=self.device)
            freqs = torch.outer(t * self.distance_scale, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_table = emb.cos().unsqueeze(1)
            self._sin_table = emb.sin().unsqueeze(1)

    def apply_rotary_pos_emb(
            self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        self._update_cos_sin_tables(int(position_ids.max()) + 1)
        cos = self._cos_table[position_ids, :]
        sin = self._sin_table[position_ids, :]
        rotated_x = ((x.float() * cos) + (self.rotate_half(x).float() * sin)).to(
            x.dtype
        )
        return rotated_x

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return self.apply_rotary_pos_emb(x, position_ids)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class ReqToTokenRetriever:
    def __init__(
            self,
            num_layers,
            head_dim,
            num_heads,
            num_kv_heads,
            fingerprint_dim,
            max_num_tokens,
            token_to_kv_pool,
            dtype,
            device,
    ):
        # now we only support one running request
        self.rope_embedding = RotaryEmbedding(head_dim, ROPE_BASE, ROPE_SCALE, device)
        self.kwargs = {
            "num_layers": num_layers,
            "head_dim": head_dim,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "fingerprint_dim": fingerprint_dim,
            "max_num_tokens": max_num_tokens,
            "dtype": dtype,
            "device": device,
            "token_to_kv_pool": token_to_kv_pool,
            "rotary_embedding": self.rope_embedding,
        }
        self.current_req_id = None
        self.current_token_retriever = None

    def get_token_retriever(self, req_id):
        if self.current_req_id != req_id:
            self.current_req_id = req_id
            # # === SINK: rid mới = sample mới ===
            # slog.next_sample()
            if (
                    self.current_token_retriever is not None
                    and QUERY_CACHE
                    and self.current_token_retriever.retrieval_count > 0
            ):
                skip = self.current_token_retriever.skip_count
                total = self.current_token_retriever.retrieval_count
                print("skip_count:", skip)
                print("retrieval_count:", total)
                print("skip_rate:", skip / total)
                # Server chạy trong tiến trình con nên stdout thường không được thu lại;
                # ghi thêm ra file để còn kiểm chứng cache có hoạt động thật hay không.
                if SKIP_RECORD_PATH:
                    with open(SKIP_RECORD_PATH, "a", encoding="utf-8") as f:
                        f.write(f"{skip} {total}\n")
            self.current_token_retriever = TokenRetriever(**self.kwargs)
        return self.current_token_retriever


@triton.jit
def paged_matmul_kernel(
        # Pointers to input and output tensors
        query_ptr,  # [num_heads, head_dim]
        token_ptr,  # [max_num_tokens, num_kv_heads, head_dim]
        indices_ptr,  # [num_relevant_tokens]
        scores_ptr,  # [num_heads, num_relevant_tokens]
        # Variables
        num_relevant_tokens,
        # Constants
        NUM_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE_TOKENS: tl.constexpr = 128,
):
    """
    Triton kernel to compute softmax scores for queries against keys per head with GQA.

    Parameters:
    - query_ptr: Pointer to query_fingerprints [num_heads, head_dim]
    - token_ptr: Pointer to token_fingerprints [max_num_tokens, num_kv_heads, head_dim]
    - indices_ptr: Pointer to relevant_indices [num_relevant_tokens]
    - scores_ptr: Pointer to output scores [num_heads, num_relevant_tokens]
    - num_heads: Number of attention heads
    - num_relevant_tokens: Number of relevant tokens
    - num_kv_heads: Number of KV heads (num_heads % num_kv_heads == 0)
    - head_dim: Dimension of each head
    """

    head_id = tl.program_id(0)  # Head index
    token_block_id = tl.program_id(1)  # Token block index

    token_start = token_block_id * BLOCK_SIZE_TOKENS
    token_indices = token_start + tl.arange(0, BLOCK_SIZE_TOKENS)
    mask_tokens = token_indices < num_relevant_tokens

    kv_head = head_id % NUM_KV_HEADS  # KV head index for the current head

    # Shape: [head_dim]
    query_offset = head_id * HEAD_DIM
    query = tl.load(
        query_ptr + query_offset + tl.arange(0, HEAD_DIM),
        mask=head_id < NUM_HEADS,
        other=0.0,
    )

    # Shape: [BLOCK_SIZE_TOKENS]
    indices = tl.load(indices_ptr + token_indices, mask=mask_tokens, other=0)

    # Shape: [BLOCK_SIZE_TOKENS, head_dim]
    token_offsets = (
            (indices[:, None] * NUM_KV_HEADS * HEAD_DIM)  # [num_relevant_tokens, 1]
            + (kv_head * HEAD_DIM)  # scalar
            + tl.arange(0, HEAD_DIM)  # [head_dim]
    )
    tokens = tl.load(token_ptr + token_offsets, mask=mask_tokens[:, None], other=0.0)

    # Shape: [BLOCK_SIZE_TOKENS]
    scores = tl.sum(query[None, :] * tokens, axis=1)

    # Shape: [num_heads, num_relevant_tokens]
    scores_offset = (
            head_id * num_relevant_tokens + token_start + tl.arange(0, BLOCK_SIZE_TOKENS)
    )

    tl.store(scores_ptr + scores_offset, scores, mask=mask_tokens)


def paged_matmul(
        query,  # torch.Tensor of shape [num_heads, head_dim]
        token,  # torch.Tensor of shape [max_num_tokens, num_kv_heads, head_dim]
        indices,  # torch.Tensor of shape [num_relevant_tokens]
        scores,  # torch.Tensor of shape [num_heads, num_relevant_tokens]
        num_relevant_tokens,
        num_heads,
        num_kv_heads,
        head_dim,
        BLOCK_SIZE_TOKENS=128,
):
    num_token_blocks = (
                               num_relevant_tokens + BLOCK_SIZE_TOKENS - 1
                       ) // BLOCK_SIZE_TOKENS
    grid = (num_heads, num_token_blocks)

    paged_matmul_kernel[grid](
        query_ptr=query,
        token_ptr=token,
        indices_ptr=indices,
        scores_ptr=scores,
        num_relevant_tokens=num_relevant_tokens,
        NUM_HEADS=num_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE_TOKENS=BLOCK_SIZE_TOKENS,
        num_warps=4,
    )


class TokenRetriever:
    def __init__(
            self,
            num_layers,
            head_dim,
            num_heads,
            num_kv_heads,
            fingerprint_dim,
            max_num_tokens,
            dtype,
            device,
            token_to_kv_pool=None,
            rotary_embedding: Optional["RotaryEmbedding"] = None,
    ):

        self.num_layers = num_layers
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.fingerprint_dim = fingerprint_dim
        self.max_num_tokens = max_num_tokens
        self.dtype = dtype
        self.device = device

        self.token_fingerprints = [
            token_to_kv_pool.get_key_buffer(layer_id)
            for layer_id in range(self.num_layers)
        ]

        self.token_indices = torch.empty(
            (self.num_layers, self.max_num_tokens),
            device=self.device,
            dtype=torch.int32,
        )

        self.rope_embedding = rotary_embedding

        self.query_fingerprints_cache = torch.empty(
            (self.num_layers, self.num_heads * self.head_dim),
            device=self.device,
            dtype=self.dtype,
        )

        # Selection Cache (chỉ dùng ở decode). Lưu dạng list vì số token được chọn
        # thay đổi theo layer khi bật adaptive top-k -> không thể dùng buffer cố định [L, TOP_K].
        self.topk_indices_cache = [None for _ in range(self.num_layers)]

        self.similarity_threshold = torch.tensor(
            [QUERY_CACHE_SIM_THRESHOLD for _ in range(self.num_layers)], device=self.device, dtype=self.dtype
        )

        self.skip_count = 0
        self.retrieval_count = 0

        self.is_first_query = [True for _ in range(self.num_layers)]

        self.clear()

    def clear(self):
        self.num_tokens = [0 for _ in range(self.num_layers)]
        self.is_first_query = [True for _ in range(self.num_layers)]
        self.skip_count = 0
        self.phase1_events = []
        self.phase2_events = []
        self.prefill_reported = False

    def get_all_tokens(self, layer_id):
        return self.token_indices[layer_id, : self.num_tokens[layer_id]]

    def add_k_cache(self, indices, layer_id):
        tail_idx = self.num_tokens[layer_id] + indices.shape[0]
        assert tail_idx <= self.max_num_tokens
        self.token_indices[layer_id, self.num_tokens[layer_id]: tail_idx] = indices
        self.num_tokens[layer_id] = tail_idx

    def get_topk_tokens(self, query_fingerprints, token_fingerprints, topk, indices, layer_id=-1, chunk_size=-1):
        num_q_heads = query_fingerprints.shape[-1] // self.head_dim
        query_fingerprints = query_fingerprints.view(num_q_heads, self.head_dim)

        num_heads = num_q_heads
        num_tokens = indices.shape[0]

        scores = torch.empty(
            (num_heads, num_tokens), device=self.device, dtype=torch.bfloat16
        )

        # Launch the Triton kernel
        paged_matmul(
            query_fingerprints,
            token_fingerprints,
            indices,
            scores,
            num_tokens,
            num_heads,
            self.num_kv_heads,
            self.head_dim,
        )

        # # === SINK v2: giữ tham chiếu raw (KHÔNG copy) để reduce sau ===
        # _raw_2d = scores if (slog.ENABLED and (slog.LAYERS_TO_LOG is None or layer_id in slog.LAYERS_TO_LOG)) else None

        actual_topk = min(topk, num_tokens)

        # ---------------------------------------------------------
        # CHOOSE POOLING/VOTING STRATEGY (THE SWITCH)
        # ---------------------------------------------------------
        if DYNAMIC_CAPACITY_UNION:
            # --- PHƯƠNG PHÁP A: DYNAMIC CAPACITY UNION (DCU CÓ SWITCH - UNMAPPED) ---
            
            # Tính toán có chọn lọc để tiết kiệm thời gian tối đa
            if DCU_ENERGY_MODE == "l2_only":
                head_energy = torch.norm(query_fingerprints, p=2, dim=-1)
            elif DCU_ENERGY_MODE == "max_only":
                head_energy = scores.max(dim=-1).values
            else: # "both"
                norms = torch.norm(query_fingerprints, p=2, dim=-1)
                max_vals = scores.max(dim=-1).values
                head_energy = norms * max_vals
            
            # Thêm hằng số chống lỗi chia 0
            tau = head_energy.mean() + 1e-5
            head_weights = torch.softmax(head_energy / tau, dim=0) 
            
            k_per_head = (head_weights * actual_topk).to(torch.int32)
            
            # [ĐÃ THÁO BỎ KẸP TRẦN]: Cho phép 1 Head ôm trọn ngân sách nếu Softmax dồn quyền lực!
            k_per_head = torch.clamp(k_per_head, min=1, max=actual_topk)
            
            max_k = k_per_head.max().item()
            
            # Khúc này max_k có thể lên tới 8192, cứ để GPU cày Global Sort xem tốc độ rớt cỡ nào
            _, batched_idx = torch.topk(scores, max_k, dim=-1) 
            
            seq_arange = torch.arange(max_k, device=scores.device).unsqueeze(0)
            valid_mask = seq_arange < k_per_head.unsqueeze(1)
            
            final_indices = torch.unique(batched_idx[valid_mask])
            
            if dist.is_initialized():
                mask = torch.zeros(num_tokens, device=scores.device, dtype=torch.int32)
                mask[final_indices] = 1
                dist.all_reduce(mask, op=dist.ReduceOp.MAX)
                final_indices = torch.nonzero(mask).squeeze(-1)
            
            sorted_topk_tokens = final_indices

        elif UNION_OF_SETS:
            k_per_head = max(1, actual_topk // num_heads)
            _, topk_indices_per_head = torch.topk(scores, k_per_head, dim=-1)
            all_indices_flat = topk_indices_per_head.flatten()
            final_indices = torch.unique(all_indices_flat)
            sorted_topk_tokens = final_indices
        elif USE_CUMSUM_ADAPTIVE:
            # =========================================================
            # HEAD-WISE CUMSUM UNION (θ = CUMSUM_THRESHOLD, vd 0.9 / 0.99)
            # Mỗi head tự lấy token tới khi CDF >= θ, rồi UNION các tập.
            # Khớp union90/union99 trong phân tích sink.
            # scores ở đây vẫn là [H, T] (chưa reduce).
            # =========================================================
            KEEP_FULL_LAYERS = {}   # L0 phẳng thật, L27 bf16-collapse -> để sau
            head_probs = torch.softmax(scores.float(), dim=-1)            # [H, T] fp32

            if layer_id in KEEP_FULL_LAYERS:
                # hành xử y hệt TokenSelect gốc: head-soft-vote post-sum top-k
                post = head_probs.sum(dim=0)                              # [T]
                if dist.is_initialized():
                    dist.all_reduce(post, op=dist.ReduceOp.SUM)
                _, final_indices = torch.topk(post, actual_topk, dim=-1)
                sorted_topk_tokens = torch.sort(final_indices).values
            else:
                # top-actual_topk per head để chặn chi phí sort (đã chứa ~toàn bộ mass)
                topk_probs, topk_idx = torch.topk(head_probs, actual_topk, dim=-1)  # [H, k]
                cdf = torch.cumsum(topk_probs, dim=-1)                    # cumsum TUYỆT ĐỐI (không renormalize)
                reach = cdf >= CUMSUM_THRESHOLD                          # [H, k]
                has = reach.any(dim=-1)
                k_h = torch.argmax(reach.int(), dim=-1) + 1              # [H] số token tới ngưỡng
                k_h = torch.where(has, k_h, torch.full_like(k_h, actual_topk))
                max_k = int(k_h.max().item())
                sel = topk_idx[:, :max_k]                                # [H, max_k]
                col = torch.arange(max_k, device=scores.device).unsqueeze(0)
                valid = col < k_h.unsqueeze(1)                          # [H, max_k]
                final_indices = torch.unique(sel[valid])                # union (unique đã sort)

                if dist.is_initialized():  # TP: hợp tập chọn giữa các rank
                    mask = torch.zeros(num_tokens, device=scores.device, dtype=torch.int32)
                    mask[final_indices] = 1
                    dist.all_reduce(mask, op=dist.ReduceOp.MAX)
                    final_indices = torch.nonzero(mask).squeeze(-1)

                # rào an toàn: union phình quá trần -> lùi về post-sum top-k
                if final_indices.shape[0] > actual_topk:
                    post = head_probs.sum(dim=0)
                    if dist.is_initialized():
                        dist.all_reduce(post, op=dist.ReduceOp.SUM)
                    _, final_indices = torch.topk(post, actual_topk, dim=-1)

                sorted_topk_tokens = torch.sort(final_indices).values
        elif HEAD_WISE_ADAPTIVE:
            # --- PERPLEXITY-PER-HEAD + N TAIL, UNION TOKEN THẬT ---
            # k_h = round(exp(H_h)) + N_TAIL  (đỉnh theo perplexity + N token đuôi kế tiếp)
            # mỗi head lấy top-k_h theo mass-ranking, rồi union indices.
            head_probs = torch.softmax(scores.float(), dim=-1)          # [H, T] fp32
            eps = 1e-12

            KEEP_FULL_LAYERS = {}   # L0 phẳng thật, L27 bf16-collapse
            if layer_id in KEEP_FULL_LAYERS:
                post = head_probs.sum(dim=0)
                if dist.is_initialized():
                    dist.all_reduce(post, op=dist.ReduceOp.SUM)
                _, final_indices = torch.topk(post, actual_topk, dim=-1)
                sorted_topk_tokens = torch.sort(final_indices).values
            else:
                H_h = -(head_probs * (head_probs + eps).log()).sum(dim=-1)   # [H] entropy
                ppl_h = torch.exp(H_h)                                       # [H] đỉnh perplexity
                k_h = (ppl_h.round().long() + N_TAIL)                        # đỉnh + N đuôi
                k_h = torch.clamp(k_h, min=1, max=actual_topk)              # [H]

                max_k = int(k_h.max().item())
                # top-max_k mỗi head theo mass, rồi mask theo k_h từng head
                _, topk_idx = torch.topk(head_probs, max_k, dim=-1)         # [H, max_k]
                col = torch.arange(max_k, device=scores.device).unsqueeze(0)
                valid = col < k_h.unsqueeze(1)                             # [H, max_k]
                final_indices = torch.unique(topk_idx[valid])             # union token thật

                if dist.is_initialized():  # TP: hợp tập giữa các rank
                    mask = torch.zeros(num_tokens, device=scores.device, dtype=torch.int32)
                    mask[final_indices] = 1
                    dist.all_reduce(mask, op=dist.ReduceOp.MAX)
                    final_indices = torch.nonzero(mask).squeeze(-1)

                # rào an toàn: union phình quá trần -> lùi post-sum top-k
                if final_indices.shape[0] > actual_topk:
                    post = head_probs.sum(dim=0)
                    if dist.is_initialized():
                        dist.all_reduce(post, op=dist.ReduceOp.SUM)
                    _, final_indices = torch.topk(post, actual_topk, dim=-1)

                sorted_topk_tokens = torch.sort(final_indices).values
        else:
            k_eff_sum_ppl = None
            k_eff_raw = None
            if ADAPTIVE_TOPK:
                _Ph  = torch.softmax(scores.float(), dim=-1)        # [H, T]
                _eps = 1e-12
                if PPL_MODE == "fixed" or PPL_MODE == "thresh":
                    k_eff_raw = 0                       # thresh tính k sau, ở post-sum
                elif PPL_MODE == "post":
                    k_eff_raw = 0        # pplpost tính ở TẦNG HAI (trên 8192, có scale)
                
                else:  # "sum"
                    _Hh = -(_Ph * (_Ph + _eps).log()).sum(dim=-1)  # [H]
                    k_eff_raw = int(torch.exp(_Hh).sum().item())
                k_eff_sum_ppl = k_eff_raw + N_TAIL                 # tail từ config

            if WEIGHTED_SOFT_VOTE:
                head_probs = torch.softmax(scores, dim=-1)
                head_energy = scores.sum(dim=-1)
                head_weights = torch.softmax(head_energy, dim=0).unsqueeze(1)
                scores = torch.sum(head_probs * head_weights, dim=0)
            else:
                scores_raw = scores
                scores = torch.softmax(scores, dim=-1).sum(dim=0)

            if dist.is_initialized():
                dist.all_reduce(scores, op=dist.ReduceOp.SUM)

            if KERNEL_SIZE > 0:
                total_pad = KERNEL_SIZE - 1
                pad_left = total_pad // 2
                pad_right = total_pad - pad_left
                x = scores.unsqueeze(0).unsqueeze(0)
                x = torch.nn.functional.pad(x, (pad_left, pad_right), value=float("-inf"))
                scores = torch.nn.functional.max_pool1d(
                    x, kernel_size=KERNEL_SIZE, stride=1
                ).squeeze(0).squeeze(0)

            if ADAPTIVE_TOPK or USE_HYBRID_ADAPTIVE:
                available_tokens = scores.shape[-1]
                KEEP_FULL_LAYERS = {}
                
                if PPL_MODE == "fixed":
                    KEEP_FULL_LAYERS = {0, 1, 2, 3, 27}
                else:
                    KEEP_FULL_LAYERS = {}

                if ADAPTIVE_TOPK and PPL_MODE == "unioncoll":
                    # ===== TẦNG 1 GIỮ NGUYÊN, chỉ thay TẦNG 2 =====
                    # Tầng 1: _sel = top-k1 theo soft-vote (y hệt pplpost).
                    # Tầng 2: mỗi head lấy k_h = N_alpha (Hill/Rényi) token của RIÊNG nó, rồi UNION.
                    #   N_alpha = (sum p^alpha)^(1/(1-alpha)),  alpha -> 1 cho exp(H) Shannon.
                    #   alpha lay tu CUMSUM_THRESHOLD (arg 17 cua run_experiment).
                    _k1 = min(actual_topk, scores.shape[-1])
                    _, _sel = torch.topk(scores, _k1, dim=-1)                        # [k1] index trong [0,T)
                    _scp = 1.0 / (self.head_dim ** 0.5)
                    _p = torch.softmax(scores_raw[:, _sel].float() * _scp, dim=-1)   # [H,k1]

                    _a = float(CUMSUM_THRESHOLD)
                    if abs(_a - 1.0) < 1e-6:
                        # Shannon. xlogy cho 0*log0 = 0 dung dinh nghia -> khong can hack +1e-12
                        # (hack do lam log(p) bi chan o -27.6 voi p < 1e-12 => entropy hut).
                        _kh = torch.exp(-torch.xlogy(_p, _p).sum(dim=-1))
                    else:
                        # Log-domain: p^alpha underflow khi alpha lon, pow(1/(1-a)) khuech dai sai so.
                        _kh = torch.exp(
                            torch.logsumexp(_a * torch.log(_p.clamp_min(1e-30)), dim=-1) / (1.0 - _a)
                        )
                    _kh = torch.nan_to_num(_kh, nan=1.0, posinf=float(_k1), neginf=1.0)
                    _kh = _kh.round().long().clamp_(1, _k1)                          # [H]

                    _mk = int(_kh.max().item())
                    _idx = torch.topk(_p, _mk, dim=-1).indices                       # [H, _mk]
                    _col = torch.arange(_mk, device=_p.device).unsqueeze(0)
                    _u = torch.unique(_idx[_col < _kh.unsqueeze(1)])                 # index trong [0,k1)
                    final_indices = _sel[_u]                                         # map ve [0,T)
                    k_dynamic = -1                                                   # da set final_indices
                elif ADAPTIVE_TOPK and PPL_MODE == "thresh":
                    if layer_id in KEEP_FULL_LAYERS:
                        k_dynamic = actual_topk
                    else:
                        a = scores.float()
                        a = a / (a.sum() + 1e-12)          # post-sum chuẩn hóa [T]
                        thr = CUMSUM_THRESHOLD              # ngưỡng tuyệt đối trên a (BỎ * a.max())
                        cnt = int((a >= thr).sum().item())
                        k_dynamic = max(256, cnt)   # N_TAIL làm SÀN
                elif ADAPTIVE_TOPK and PPL_MODE == "post":
                    if layer_id in KEEP_FULL_LAYERS:
                        k_dynamic = actual_topk
                    else:
                        # BƯỚC 1: top-8192 thô (không scale) từ post-sum đã có
                        _k1 = min(actual_topk, scores.shape[-1])
                        _, _sel = torch.topk(scores, _k1, dim=-1)          # [k1] index [0,T)
                        # TẦNG HAI: pplpost CÓ scale, chỉ trên 8192
                        _scp = 1.0 / (self.head_dim ** 0.5)
                        _s_sel = scores_raw[:, _sel].float()               # [H, k1]
                        _P = torch.softmax(_s_sel * _scp, dim=-1).sum(dim=0)   # [k1]
                        _P = _P / (_P.sum() + 1e-12)
                        _Hpost = -(_P * (_P + 1e-12).log()).sum()
                        _kfp = torch.exp(_Hpost).item()
                        # hệ số an toàn = CUMSUM_THRESHOLD (dùng làm alpha, vd 1.5)
                        _k2 = int(round(CUMSUM_THRESHOLD * _kfp))
                        _k2 = max(256, min(_k2, _k1))
                        # giữ _k2 token P cao nhất, map về [0,T)
                        _keep = torch.topk(_P, _k2).indices
                        final_indices = _sel[_keep]
                        k_dynamic = -1     # đánh dấu: đã set final_indices, bỏ qua topk cuối
                elif ADAPTIVE_TOPK:
                    # --- Σ exp(H_h) (đã tính ở trên), chọn top-k trên post-sum ---
                    k_dynamic = actual_topk if layer_id in KEEP_FULL_LAYERS else k_eff_sum_ppl
                else:  # USE_HYBRID_ADAPTIVE — giữ nguyên logic cũ trên post-sum
                    if layer_id in KEEP_FULL_LAYERS:
                        k_dynamic = actual_topk
                    else:
                        scores_float = scores.float()
                        probs = scores_float / (scores_float.sum(dim=-1, keepdim=True) + 1e-12)
                        eps = 1e-12
                        entropy = -(probs * (probs + eps).log()).sum(dim=-1)
                        k_guess = int(torch.exp(entropy).item() * 1.5)
                        k_guess = max(min(64, available_tokens), min(k_guess, min(8192, available_tokens)))
                        topk_probs, _ = torch.topk(probs, k_guess, dim=-1)
                        if topk_probs.sum().item() >= CUMSUM_THRESHOLD:
                            k_dynamic = k_guess
                        else:
                            sorted_probs, _ = torch.sort(probs, descending=True)
                            cum = torch.cumsum(sorted_probs, dim=-1)
                            m = cum >= CUMSUM_THRESHOLD
                            k_dynamic = int(m.int().argmax().item()) + 1 if m.any() else available_tokens

                if k_dynamic != -1:          # chỉ kẹp khi k_dynamic là SỐ TOKEN thật
                    max_k_limit = min(8192, available_tokens)
                    min_k_limit = min(64, available_tokens)
                    k_dynamic = max(min_k_limit, min(k_dynamic, max_k_limit))
                    _, final_indices = torch.topk(scores, k_dynamic, dim=-1)
            else:
                _, final_indices = torch.topk(scores, actual_topk, dim=-1)

            sorted_topk_tokens = torch.sort(final_indices).values

        return sorted_topk_tokens

    def retrieval_indices(self, query, layer_id, n_init, n_local, topk, is_decode=False):
        current_num_tokens = self.num_tokens[layer_id]
        if n_init + topk + n_local >= current_num_tokens:
            return None

        if QUERY_ROTATE:
            position_ids = torch.arange(
                n_local,
                n_local + query.shape[0],
                device=self.device,
            )
            query = self.rope_embedding(
                query.view(query.shape[0], -1, self.head_dim), position_ids
            ).view(query.shape[0], -1)
        else:
            query = query.view(query.shape[0], -1)
        
        anchor_len = min(PREFILL_CHUNK_SIZE, query.shape[0])
        anchor_query = query[:anchor_len]

        if L2_NORM_POOLING:
            # --- GENERALIZED FINGERPRINT: L2-Norm Weighted Mean ---
            norms = torch.norm(query, p=2, dim=-1, keepdim=True) # Shape: [L, num_heads * head_dim]
            weights = norms / (norms.sum(dim=0, keepdim=True) + 1e-6) 
            query_fingerprints = torch.sum(weights * query, dim=0)
        else:
            # # --- ORIGINAL PAPER: Mean Pooling --- Tạm thời bỏ vì hiệu quả không tốt, thay bằng phương pháp mới bên dưới
            # query_fingerprints = torch.mean(query, dim=0)
            # ====================================================================
            # --- ANCHOR POOLING (Lấy Mean của riêng phần Anchor) ---
            query_fingerprints = torch.mean(anchor_query, dim=0)

            num_blocks = query.shape[0] // PREFILL_CHUNK_SIZE
            # if num_blocks >= 2 and ADAPTIVE_TOPK:
            #     # fingerprint (mean) của từng block-512 trong cluster
            #     q_blocks = query[:num_blocks * PREFILL_CHUNK_SIZE].view(
            #         num_blocks, PREFILL_CHUNK_SIZE, -1
            #     ).mean(dim=1)                              # [num_blocks, D]
            #     block_norms = q_blocks.norm(dim=-1)        # [num_blocks]
            #     min_norm = block_norms.min()
            #     anchor_norm = query_fingerprints.norm() + 1e-8
            #     # giữ hướng anchor, ép độ dài về norm nhỏ nhất
            #     query_fingerprints = query_fingerprints * (min_norm / anchor_norm)

        # Selection Cache CHỈ áp cho decode. Prefill đã có Dynamic Chunking lo việc
        # tái sử dụng kết quả chọn token, bật cache ở đó vừa chậm vừa nhiễu thí nghiệm.
        use_query_cache = QUERY_CACHE and is_decode

        cache_hit = False
        if use_query_cache and not self.is_first_query[layer_id]:
            cache_hit = bool(
                torch.cosine_similarity(
                    self.query_fingerprints_cache[layer_id], query_fingerprints, dim=0
                )
                >= self.similarity_threshold[layer_id]
            )

        if cache_hit:
            topk_tokens = self.topk_indices_cache[layer_id]
            self.skip_count += 1
        else:
            relevant_indices = self.token_indices[
                               layer_id, n_init: current_num_tokens - n_local
                               ]

            token_fingerprints = self.token_fingerprints[layer_id]
            # # === SINK_LOG_HOOK C: tell logger the absolute frame ===
            # slog.set_geometry(n_init, n_local, current_num_tokens)
            topk_tokens = (
                    self.get_topk_tokens(
                        query_fingerprints, token_fingerprints, topk, relevant_indices, layer_id=layer_id, chunk_size=query.shape[0]
                    )
                    + n_init
            )

            if use_query_cache:
                self.topk_indices_cache[layer_id] = topk_tokens
                self.query_fingerprints_cache[layer_id] = query_fingerprints
                self.is_first_query[layer_id] = False

        if use_query_cache:
            self.retrieval_count += 1

        retrieved_tokens = torch.cat(
            [
                torch.arange(0, n_init, device=self.device),
                topk_tokens,
                torch.arange(
                    current_num_tokens - n_local,
                    current_num_tokens,
                    device=self.device,
                ),
            ]
        )

        final_indices = self.token_indices[layer_id, retrieved_tokens]
        return final_indices


def patch_model_runner():
    from sglang.srt.model_executor.model_runner import ModelRunner

    class PatchedModelRunner(ModelRunner):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.req_to_token_retriever = ReqToTokenRetriever(
                self.model_config.num_hidden_layers,
                self.model_config.head_dim,
                self.model_config.num_attention_heads,
                self.model_config.get_num_kv_heads(self.tp_size),
                self.model_config.get_num_kv_heads(self.tp_size)
                * self.model_config.head_dim,
                MAX_N_TOKENS,
                self.token_to_kv_pool,
                self.dtype,
                "cuda",
            )
            print("max_total_num_tokens:", self.max_total_num_tokens)

    sglang.srt.model_executor.model_runner.ModelRunner = PatchedModelRunner


def patch_input_metadata():
    class PatchedInputMetadata(InputMetadata):

        token_retriever: TokenRetriever

        def __init__(self, **kwargs):
            self.token_retriever = kwargs.pop("token_retriever")
            super().__init__(**kwargs)

        def init_flashinfer_handlers(
                self,
                model_runner,
                prefix_lens,
                flashinfer_use_ragged,
        ):
            flashinfer_use_ragged = False
            patched_forward_batch_info_update_flashinfer_indices(
                self.forward_mode,
                model_runner,
                self.req_pool_indices,
                self.seq_lens,
                prefix_lens,
                flashinfer_use_ragged=flashinfer_use_ragged,
            )

            (
                self.flashinfer_prefill_wrapper_ragged,
                self.flashinfer_prefill_wrapper_paged,
                self.flashinfer_decode_wrapper,
                self.flashinfer_use_ragged,
            ) = (
                model_runner.flashinfer_prefill_wrapper_ragged,
                model_runner.flashinfer_prefill_wrapper_paged,
                model_runner.flashinfer_decode_wrapper,
                flashinfer_use_ragged,
            )

        @classmethod
        def from_schedule_batch(
                cls,
                model_runner,
                batch: ScheduleBatch,
                forward_mode: ForwardMode,
        ):
            ret = cls(
                forward_mode=forward_mode,
                batch_size=batch.batch_size(),
                req_pool_indices=batch.req_pool_indices,
                seq_lens=batch.seq_lens,
                req_to_token_pool=model_runner.req_to_token_pool,
                token_to_kv_pool=model_runner.token_to_kv_pool,
                token_retriever=model_runner.req_to_token_retriever.get_token_retriever(
                    batch.reqs[0].rid
                ),
                out_cache_loc=batch.out_cache_loc,
                return_logprob=batch.return_logprob,
                top_logprobs_nums=batch.top_logprobs_nums,
            )

            ret.compute_positions(batch)

            ret.compute_extend_infos(batch)

            if (
                    forward_mode != ForwardMode.DECODE
                    or model_runner.server_args.disable_flashinfer
            ):
                ret.total_num_tokens = int(torch.sum(ret.seq_lens))

            if forward_mode != ForwardMode.DECODE:
                ret.init_multimuldal_info(batch)

            prefix_lens = None
            if forward_mode != ForwardMode.DECODE:
                prefix_lens = torch.tensor(
                    [len(r.prefix_indices) for r in batch.reqs], device="cuda"
                )

            if model_runner.server_args.disable_flashinfer:
                ret.init_triton_args(batch, prefix_lens)

            flashinfer_use_ragged = False
            if not model_runner.server_args.disable_flashinfer:
                if (
                        forward_mode != ForwardMode.DECODE
                        and int(torch.sum(ret.seq_lens)) > 4096
                        and model_runner.sliding_window_size is None
                ):
                    flashinfer_use_ragged = True
                ret.init_flashinfer_handlers(
                    model_runner, prefix_lens, flashinfer_use_ragged
                )

            return ret

    def patched_forward_batch_info_update_flashinfer_indices(
            forward_mode,
            model_runner,
            req_pool_indices,
            seq_lens,
            prefix_lens,
            flashinfer_decode_wrapper=None,
            flashinfer_use_ragged=False,
    ):
        """Init auxiliary variables for FlashInfer attention backend."""
        num_qo_heads = (
                model_runner.model_config.num_attention_heads // model_runner.tp_size
        )
        num_kv_heads = model_runner.model_config.get_num_kv_heads(model_runner.tp_size)
        head_dim = model_runner.model_config.head_dim
        batch_size = len(req_pool_indices)

        if model_runner.sliding_window_size is None:
            if flashinfer_use_ragged:
                raise NotImplementedError("Ragged attention not supported yet")
            else:
                paged_kernel_lens = seq_lens

            kv_indptr = torch.zeros((batch_size + 1,), dtype=torch.int32, device="cuda")
            kv_indptr[1:] = torch.cumsum(paged_kernel_lens, dim=0)
            req_pool_indices_cpu = req_pool_indices.cpu().numpy()
            paged_kernel_lens_cpu = paged_kernel_lens.cpu().numpy()
            kv_indices = torch.cat(
                [
                    model_runner.req_to_token_pool.req_to_token[
                    req_pool_indices_cpu[i], : paged_kernel_lens_cpu[i]
                    ]
                    for i in range(batch_size)
                ],
                dim=0,
            ).contiguous()

            kv_last_page_len = torch.ones(
                (batch_size,), dtype=torch.int32, device="cuda"
            )

            if forward_mode == ForwardMode.DECODE:
                # CUDA graph uses different flashinfer_decode_wrapper
                if flashinfer_decode_wrapper is None:
                    flashinfer_decode_wrapper = model_runner.flashinfer_decode_wrapper

                flashinfer_decode_wrapper.end_forward()
                flashinfer_decode_wrapper.begin_forward(
                    kv_indptr,
                    kv_indices,
                    kv_last_page_len,
                    num_qo_heads,
                    num_kv_heads,
                    head_dim,
                    1,
                )
            else:

                # extend part
                qo_indptr = torch.zeros(
                    (batch_size + 1,), dtype=torch.int32, device="cuda"
                )
                qo_indptr[1:] = torch.cumsum(seq_lens - prefix_lens, dim=0)

                if flashinfer_use_ragged:
                    raise NotImplementedError("Ragged attention not supported yet")

                # cached part
                model_runner.flashinfer_prefill_wrapper_paged.end_forward()
                model_runner.flashinfer_prefill_wrapper_paged.begin_forward(
                    qo_indptr,
                    kv_indptr,
                    kv_indices,
                    kv_last_page_len,
                    num_qo_heads,
                    num_kv_heads,
                    head_dim,
                    1,
                )
        else:
            raise NotImplementedError("Sliding window not supported yet")

    sglang.srt.model_executor.forward_batch_info.InputMetadata = PatchedInputMetadata
    sglang.srt.model_executor.forward_batch_info.update_flashinfer_indices = (
        patched_forward_batch_info_update_flashinfer_indices
    )


def patch_model():
    def patched_default_model_loader_load_model(
            self,
            *,
            model_config: ModelConfig,
            device_config: DeviceConfig,
            lora_config: Optional[LoRAConfig],
            multimodal_config: Optional[MultiModalConfig],
            parallel_config: ParallelConfig,
            scheduler_config: SchedulerConfig,
            cache_config: CacheConfig,
    ) -> torch.nn.Module:
        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = _initialize_model(
                    model_config,
                    self.load_config,
                    lora_config,
                    multimodal_config,
                    cache_config,
                    scheduler_config,
                )
            model.load_weights(
                self._get_weights_iterator(
                    model_config.model,
                    model_config.revision,
                    fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
                ),
            )

            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    # When quant methods need to process weights after loading
                    # (for repacking, quantizing, etc), they expect parameters
                    # to be on the global target device. This scope is for the
                    # case where cpu offloading is used, where we will move the
                    # parameters onto device for processing and back off after.
                    with device_loading_context(module, target_device):
                        quant_method.process_weights_after_loading(module)
        return patch_attention(model.eval())

    def patch_attention(model):

        def patched_radix_attention_extend_forward_flashinfer(
                self, q, k, v, input_metadata: InputMetadata
        ):
            # --- CÀI BẤM GIỜ BẮT ĐẦU PREFILL ---
            global _TRACKER_PREFILL_START
            # Chỉ bấm giờ ở layer 0 và lúc _TRACKER_PREFILL_START đang = 0 
            # (Để đề phòng trường hợp Chunked Prefill nó gọi hàm này nhiều lần, ta chỉ lấy mốc thời gian của chunk đầu tiên)
            if self.layer_id == 0 and _TRACKER_PREFILL_START == 0.0:
                _TRACKER_PREFILL_START = time.time()
            # -----------------------------------
            prefill_wrapper_paged = input_metadata.flashinfer_prefill_wrapper_paged
            if self.sliding_window_size != -1:
                prefill_wrapper_paged = prefill_wrapper_paged[0]
            else:
                if isinstance(prefill_wrapper_paged, list):
                    prefill_wrapper_paged = prefill_wrapper_paged[1]

            assert not input_metadata.flashinfer_use_ragged

            seq_len = q.shape[0]
            outputs = torch.empty_like(q)
            kv_last_page_len = prefill_wrapper_paged._paged_kv_last_page_len_buf.clone()
            qo_indptr = prefill_wrapper_paged._qo_indptr_buf.clone()

            # =================================================================
            # LÕI V2.1: DYNAMIC CHUNKING SỬ DỤNG ANCHOR
            # =================================================================
            BASE_CHUNK = PREFILL_CHUNK_SIZE
            chunk_plan = []

            if USE_DYNAMIC_CHUNKING and MAX_DYNAMIC_CHUNK > BASE_CHUNK:
                # BƯỚC 1: Tính Vector trung bình cho các block
                with torch.no_grad():
                    num_blocks = seq_len // BASE_CHUNK
                    if num_blocks > 0:
                        q_trunc = q[:num_blocks * BASE_CHUNK].float()
                        q_blocks_mean = q_trunc.view(num_blocks, BASE_CHUNK, -1).mean(dim=1)
                        # Normalize sẵn để tính cosine similarity bằng tích vô hướng (dot product) cho nhanh
                        q_blocks_mean = torch.nn.functional.normalize(q_blocks_mean, p=2, dim=-1)
                    else:
                        q_blocks_mean = None

                current_pos = 0
                block_idx = 0

                while current_pos < seq_len:
                    step = BASE_CHUNK
                    anchor_idx = block_idx
                    
                    # Cuộn tuyết bằng Anchor: So sánh block tiếp theo trực tiếp với Anchor
                    while (block_idx + 1) < num_blocks and step < MAX_DYNAMIC_CHUNK:
                        next_idx = block_idx + 1
                        # Tính cosine similarity (đã normalize nên chỉ cần dot product)
                        sim = torch.dot(q_blocks_mean[anchor_idx], q_blocks_mean[next_idx]).item()
                        
                        if sim >= SIM_THRESHOLD:
                            step += BASE_CHUNK
                            block_idx += 1
                        else:
                            break # Khác Anchor -> Ngắt Chunk lập tức để tránh Topic Drift

                    start = current_pos
                    end = min(current_pos + step, seq_len)
                    chunk_plan.append((start, end))

                    current_pos = end
                    block_idx += 1
            else:
                # Fallback về chạy cơ bản nếu ko bật Dynamic
                num_chunks = ceil(seq_len / BASE_CHUNK)
                for chunk_idx in range(num_chunks):
                    start = chunk_idx * BASE_CHUNK
                    end = min((chunk_idx + 1) * BASE_CHUNK, seq_len)
                    chunk_plan.append((start, end))

            # BƯỚC 2: Thực thi theo Plan
            for start, end in chunk_plan:
                actual_step = end - start

                if k is not None:
                    assert v is not None
                    self.store_kv_cache(k, v, input_metadata, start=start, end=end)

                # --- Cân bằng Local và Top-K ---
                current_n_local = max(N_Local, actual_step)
                
                if DYNAMIC_BUDGET_BALANCING:
                    # Bật bù trừ: Giữ nguyên tổng ngân sách (Tăng Local -> Giảm TopK)
                    current_top_k = max(1, TOP_K - (current_n_local - N_Local))
                else:
                    # Tắt bù trừ: Giữ nguyên Top-K, cho phép tổng ngân sách phình to ra
                    current_top_k = TOP_K

                retrieved_indices = input_metadata.token_retriever.retrieval_indices(
                    q[start:end].contiguous(), self.layer_id, N_INIT, current_n_local, current_top_k,
                    is_decode=False,
                )
                
                retrieved_indptr = prefill_wrapper_paged._paged_kv_indptr_buf.clone()

                if retrieved_indices is None:
                    retrieved_indices = input_metadata.token_retriever.get_all_tokens(self.layer_id)

                retrieved_indptr[1] = len(retrieved_indices)
                qo_indptr[1] = actual_step
                
                prefill_wrapper_paged.end_forward()
                prefill_wrapper_paged.begin_forward(
                    qo_indptr, retrieved_indptr, retrieved_indices, kv_last_page_len,
                    self.tp_q_head_num, self.tp_k_head_num, self.head_dim, 1,
                )

                o = prefill_wrapper_paged.forward(
                    q[start:end].contiguous().view(-1, self.tp_q_head_num, self.head_dim),
                    input_metadata.token_to_kv_pool.get_kv_buffer(self.layer_id),
                    causal=True, sm_scale=self.scaling, window_left=self.sliding_window_size,
                    logits_soft_cap=self.logit_cap, rope_scale=ROPE_SCALE, 
                    rope_theta=ROPE_BASE, pos_encoding_mode=ROPE_MODE,
                )

                outputs[start:end] = o.view(-1, self.tp_q_head_num * self.head_dim)

            return outputs

        def patched_radix_attention_decode_forward_flashinfer(
                self, q, k, v, input_metadata: InputMetadata
        ):
            # --- CHỐT GIỜ TTFT Ở TOKEN ĐẦU TIÊN ---
            global _TRACKER_PREFILL_START, GLOBAL_TOTAL_TTFT
            if self.layer_id == 0 and _TRACKER_PREFILL_START > 0.0:
                ttft = time.time() - _TRACKER_PREFILL_START
                record_ttft(ttft)
                _TRACKER_PREFILL_START = 0.0  # Reset để chuẩn bị đo cho câu hỏi (sample) tiếp theo trong dataset
            # --------------------------------------
            decode_wrapper = input_metadata.flashinfer_decode_wrapper
            if self.sliding_window_size != -1:
                decode_wrapper = decode_wrapper[0]
            else:
                if isinstance(decode_wrapper, list):
                    decode_wrapper = decode_wrapper[1]

            if k is not None:
                assert v is not None
                self.store_kv_cache(k, v, input_metadata)

            retrieved_indices = input_metadata.token_retriever.retrieval_indices(q.contiguous(), self.layer_id, N_INIT,
                                                                                 N_Local, TOP_K, is_decode=True)
            if retrieved_indices is not None:
                # Adaptive Top-K làm số lượng indices thay đổi theo layer. 
                # Ta cần reset FlashInfer wrapper nếu độ dài bị lệch so với buffer hiện tại.
                if (
                        self.layer_id == 0
                        or len(retrieved_indices) != len(decode_wrapper._paged_kv_indices_buf)
                ):  
                    retrieved_indptr = decode_wrapper._paged_kv_indptr_buf.clone()
                    retrieved_indptr[1] = len(retrieved_indices)
                    kv_last_page_len = (
                        decode_wrapper._paged_kv_last_page_len_buf.clone()
                    )
                    decode_wrapper.end_forward()
                    decode_wrapper.begin_forward(
                        retrieved_indptr,
                        retrieved_indices,
                        kv_last_page_len,
                        self.tp_q_head_num,
                        self.tp_k_head_num,
                        self.head_dim,
                        1,
                    )
                else:
                    decode_wrapper._paged_kv_indices_buf.copy_(retrieved_indices)

            o = decode_wrapper.forward(
                q.contiguous().view(-1, self.tp_q_head_num, self.head_dim),
                input_metadata.token_to_kv_pool.get_kv_buffer(self.layer_id),
                sm_scale=self.scaling,
                logits_soft_cap=self.logit_cap,
                rope_scale=ROPE_SCALE,
                rope_theta=ROPE_BASE,
                pos_encoding_mode=ROPE_MODE,
            )

            return o.view(-1, self.tp_q_head_num * self.head_dim)

        def patched_radix_attention_forward(
                self, q, k, v, input_metadata: InputMetadata
        ):
            if k is not None:
                assert v is not None
                k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
                v = v.view(-1, self.tp_v_head_num, self.v_head_dim)

            if input_metadata.forward_mode == ForwardMode.EXTEND:
                return patched_radix_attention_extend_forward_flashinfer(
                    self, q, k, v, input_metadata
                )
            elif input_metadata.forward_mode == ForwardMode.DECODE:
                return patched_radix_attention_decode_forward_flashinfer(
                    self, q, k, v, input_metadata
                )

        def patched_radix_attention_store_kv_cache(
                self, cache_k, cache_v, input_metadata: InputMetadata, start=0, end=None
        ):
            k_cache = input_metadata.token_to_kv_pool.get_key_buffer(self.layer_id)
            v_cache = input_metadata.token_to_kv_pool.get_value_buffer(self.layer_id)
            k_cache[input_metadata.out_cache_loc[start:end]] = cache_k[start:end]
            v_cache[input_metadata.out_cache_loc[start:end]] = cache_v[start:end]
            input_metadata.token_retriever.add_k_cache(
                input_metadata.out_cache_loc[start:end],
                self.layer_id,
            )

        def patched_meta_attention_forward(
                self,
                positions: torch.Tensor,
                hidden_states: torch.Tensor,
                input_metadata: InputMetadata,
        ) -> torch.Tensor:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            attn_output = self.attn(q, k, v, input_metadata)
            output, _ = self.o_proj(attn_output)
            return output

        for layer in model.model.layers:
            layer.self_attn.__class__.forward = (
                patched_meta_attention_forward  # support for Llama, Qwen2, Mistral
            )
            layer.self_attn.attn.__class__.forward = patched_radix_attention_forward
            layer.self_attn.attn.__class__.store_kv_cache = (
                patched_radix_attention_store_kv_cache
            )
        return model

    def patched_rotary_embedding_get_rope(*args, **kwargs) -> None:
        return None

    rotary_embedding.get_rope = patched_rotary_embedding_get_rope
    DefaultModelLoader.load_model = patched_default_model_loader_load_model


def patch(
        rope_base=1e6,
        rope_scale=1,
        rope_model="ROPE_LLAMA",
        max_n_tokens=1024,
        top_k=16,
        n_init=1,
        n_local=16,
        kernel_size=-1,
        adaptive_topk=False,
        attention_threshold=0.9,
        weighted_soft_vote=False,
        union_of_sets=False,
        l2_norm_pooling=False,
        dynamic_capacity_union=False,
        head_wise_adaptive=False,
        dcu_energy_mode="both",
        prefill_chunk_size=512,
        sim_threshold=0.95,
        max_dynamic_chunk=1024,
        use_dynamic_chunking=False,
        dynamic_budget_balancing=True,
        use_cumsum_adaptive=False,
        use_hybrid_adaptive=False,
        cumsum_threshold=0.95,
        n_tail=2048,
        ppl_mode="sum",
        query_cache=False,
        query_cache_sim_threshold=0.9,
):
    global ROPE_BASE
    global ROPE_SCALE
    global ROPE_MODE
    global MAX_N_TOKENS
    global TOP_K
    global N_INIT
    global N_Local
    global QUERY_ROTATE
    global QUERY_CACHE
    global QUERY_CACHE_SIM_THRESHOLD
    global PREFILL_CHUNK_SIZE
    global KERNEL_SIZE
    global ADAPTIVE_TOPK
    global ATTENTION_THRESHOLD
    global WEIGHTED_SOFT_VOTE
    global UNION_OF_SETS
    global L2_NORM_POOLING
    global DYNAMIC_CAPACITY_UNION
    global HEAD_WISE_ADAPTIVE
    global DCU_ENERGY_MODE
    global PREFILL_CHUNK_SIZE
    global SIM_THRESHOLD
    global MAX_DYNAMIC_CHUNK
    global USE_DYNAMIC_CHUNKING
    global DYNAMIC_BUDGET_BALANCING
    global USE_CUMSUM_ADAPTIVE
    global USE_HYBRID_ADAPTIVE
    global CUMSUM_THRESHOLD
    global N_TAIL
    global PPL_MODE

    ROPE_BASE = rope_base
    ROPE_SCALE = rope_scale
    ROPE_MODE = rope_model
    MAX_N_TOKENS = max_n_tokens
    TOP_K = top_k
    N_INIT = n_init
    N_Local = n_local
    KERNEL_SIZE=kernel_size

    ADAPTIVE_TOPK = adaptive_topk
    ATTENTION_THRESHOLD = attention_threshold
    WEIGHTED_SOFT_VOTE = weighted_soft_vote

    UNION_OF_SETS = union_of_sets

    L2_NORM_POOLING = l2_norm_pooling

    DYNAMIC_CAPACITY_UNION = dynamic_capacity_union

    HEAD_WISE_ADAPTIVE = head_wise_adaptive

    DCU_ENERGY_MODE = dcu_energy_mode

    QUERY_ROTATE = True
    PREFILL_CHUNK_SIZE = prefill_chunk_size
    QUERY_CACHE = query_cache
    QUERY_CACHE_SIM_THRESHOLD = query_cache_sim_threshold

    SIM_THRESHOLD = sim_threshold
    MAX_DYNAMIC_CHUNK = max_dynamic_chunk
    USE_DYNAMIC_CHUNKING = use_dynamic_chunking
    DYNAMIC_BUDGET_BALANCING = dynamic_budget_balancing

    USE_CUMSUM_ADAPTIVE = use_cumsum_adaptive
    USE_HYBRID_ADAPTIVE = use_hybrid_adaptive
    CUMSUM_THRESHOLD = cumsum_threshold
    N_TAIL = n_tail
    PPL_MODE = ppl_mode
    # print(f"[PATCH-ARGS] ADAPTIVE_TOPK={ADAPTIVE_TOPK} PPL_MODE={PPL_MODE!r} "
    #       f"N_TAIL={N_TAIL} CUMSUM_THRESHOLD={CUMSUM_THRESHOLD}", flush=True)

    # # === SINK LOGGER: configure inside SERVER process ===
    # import os as _os
    # slog.enable(True)
    # import atexit as _atexit
    # _atexit.register(slog.finalize)
    # slog.set_dataset(_os.environ.get("SINK_DATASET", "unknown"))
    # slog.set_output_dir(_os.environ.get("SINK_LOG_DIR", "./sink_logs"))
    # print(f"[SINK] logger ON in server pid={_os.getpid()} "
    #       f"ds={slog.DATASET} dir={slog.OUTPUT_DIR}", flush=True)

    patch_input_metadata()
    patch_model_runner()
    patch_model()


def patch_rope_only(
        rope_base=1e6,
        rope_scale=1,
        rope_model="ROPE_LLAMA",
        max_n_tokens=1048576,
):
    """Apply only RoPE scaling for extended context without TokenSelect attention."""
    global ROPE_BASE
    global ROPE_SCALE
    global ROPE_MODE
    global MAX_N_TOKENS

    ROPE_BASE = rope_base
    ROPE_SCALE = rope_scale
    ROPE_MODE = rope_model
    MAX_N_TOKENS = max_n_tokens

    # Lấy hàm get_rope gốc của vLLM/SGLang
    from vllm.model_executor.layers import rotary_embedding
    original_get_rope = rotary_embedding.get_rope

    def patched_get_rope(*args, **kwargs):
        # Chuyển args thành list để dễ can thiệp nếu tham số truyền dạng positional
        new_args = list(args)
        
        # 1. Ép max_position (vLLM thường để ở args[2])
        if "max_position" in kwargs:
            kwargs["max_position"] = int(MAX_N_TOKENS)
        elif len(new_args) > 2:
            new_args[2] = int(MAX_N_TOKENS)
        else:
            kwargs["max_position"] = int(MAX_N_TOKENS)
            
        # 2. Ép base (vLLM thường để ở args[3])
        if "base" in kwargs:
            kwargs["base"] = float(ROPE_BASE)
        elif len(new_args) > 3:
            new_args[3] = float(ROPE_BASE)
        else:
            kwargs["base"] = float(ROPE_BASE)
            
        # 3. Ép rope_scaling nếu có thay đổi (vLLM thường để ở args[5])
        if float(ROPE_SCALE) != 1.0:
            scale_dict = {"type": "linear", "factor": float(ROPE_SCALE)}
            if "rope_scaling" in kwargs:
                kwargs["rope_scaling"] = scale_dict
            elif len(new_args) > 5:
                new_args[5] = scale_dict
            else:
                kwargs["rope_scaling"] = scale_dict

        # Trả quyền khởi tạo lại cho engine gốc với tham số đã được tiêm
        return original_get_rope(*new_args, **kwargs)

    # Đánh chặn lúc khởi tạo
    rotary_embedding.get_rope = patched_get_rope
    
    print(f"✓ Clean SDPA patch applied: Native SGLang attention preserved with injected RoPE (base={ROPE_BASE}, scale={ROPE_SCALE}, max_tokens={MAX_N_TOKENS})")

