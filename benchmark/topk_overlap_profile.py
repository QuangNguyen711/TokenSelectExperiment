"""
Profile Anchor-vs-Mean Retrieval Quality (v2).

Key fixes from v1:
1. FIXED POOL per cluster = token_indices[n_init : cluster_start].
   Anchor, mean, and per-sub-chunk baselines all retrieve on this SAME pool.
   This makes the comparison apples-to-apples and ensures First-Chunk
   Recall = 100% as a sanity check.

2. CALL get_topk_tokens DIRECTLY (not retrieval_indices) so we control
   the candidate pool ourselves — no local-window pollution.

3. ANCHOR vs MEAN side by side. For each cluster, retrieve with three
   query types on the fixed pool:
     - anchor query  = q[anchor_block_start : anchor_block_end]
     - mean query    = q[cluster_start : cluster_end]
     - per-sub-chunk = q[block_j_start : block_j_end] for each j (baselines)

4. METRICS (Definition B — average per-sub-chunk recall):
     - Global Recall (anchor) = mean_j |I_anchor ∩ I_j| / |I_j|
     - Global Recall (mean)   = mean_j |I_mean   ∩ I_j| / |I_j|
     - First-Chunk Recall (anchor) = |I_anchor ∩ I_1| / |I_1|   ==> sanity, must be 100%
     - Last-Chunk Recall  (anchor) = |I_anchor ∩ I_n| / |I_n|
     - Same metrics for mean

5. WORKER DEBUG. Prints theta, L_max, PREFILL_CHUNK_SIZE at layer 0 of
   the first sample of each sweep, to confirm IPC works.

Usage:
    CUDA_VISIBLE_DEVICES=1 python benchmark/topk_overlap_profile.py \\
        --config_path config/qwen-overlap-profile.yaml \\
        --datasets kv_retrieval,passkey \\
        --num_samples 5 \\
        --output_path overlap_anchor_vs_mean.json
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
import random
import argparse
from collections import defaultdict
from math import ceil

import torch
import requests
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])
import patcher.token_retrieval as tr


# =============================================================================
# CONFIGURATION
# =============================================================================
SWEEP_CONFIGS = [
    (0.95, 1024), (0.95, 2048), (0.95, 4096),
    (0.97, 1024), (0.97, 2048), (0.97, 4096),
    (0.99, 1024), (0.99, 2048), (0.99, 4096),
    (0.999, 1024), (0.999, 2048), (0.999, 4096),
]
TARGET_LAYERS = [0, 6, 13, 20, 27]

SWEEP_PARAM_FILE = "sweep_params.json"
OVERLAP_LOG_FILE = "overlap_log.jsonl"

_DEBUG_PRINTED_FOR_SWEEP = set()


def compute_cluster_plan(q_blocks_normalized, num_blocks, theta, L_max, base_chunk):
    max_dynamic_blocks = L_max // base_chunk
    clusters = []
    block_idx = 0
    while block_idx < num_blocks:
        anchor_idx = block_idx
        cluster_blocks = [block_idx]
        while (len(cluster_blocks) < max_dynamic_blocks
               and block_idx + 1 < num_blocks):
            next_idx = block_idx + 1
            sim = torch.dot(
                q_blocks_normalized[anchor_idx],
                q_blocks_normalized[next_idx]
            ).item()
            if sim >= theta:
                cluster_blocks.append(next_idx)
                block_idx += 1
            else:
                break
        clusters.append(cluster_blocks)
        block_idx += 1
    return clusters


def build_query_fingerprint(retriever, q_slice, head_dim, n_local):
    """Mirror retrieval_indices' fingerprint construction (mean-pool + RoPE)."""
    if tr.QUERY_ROTATE:
        position_ids = torch.arange(
            n_local, n_local + q_slice.shape[0], device=q_slice.device,
        )
        q_rotated = retriever.rope_embedding(
            q_slice.view(q_slice.shape[0], -1, head_dim), position_ids
        ).view(q_slice.shape[0], -1)
    else:
        q_rotated = q_slice.view(q_slice.shape[0], -1)
    return torch.mean(q_rotated, dim=0)


def retrieve_topk_on_pool(retriever, layer_id, query_fingerprint, pool_indices, topk):
    """Call get_topk_tokens directly on our custom pool."""
    if pool_indices.shape[0] == 0:
        return set()

    token_fingerprints = retriever.token_fingerprints[layer_id]
    topk_positions = retriever.get_topk_tokens(
        query_fingerprint, token_fingerprints, topk, pool_indices
    )
    # topk_positions are POSITIONS into pool_indices, not actual cache indices.
    # Map back through pool_indices to get the real cache slot ids.
    actual_indices = pool_indices[topk_positions.long()]
    return set(actual_indices.cpu().tolist())


def profile_cluster(self_attn, q, retriever, head_dim,
                    cluster_blocks, base_chunk, seq_len,
                    theta, lmax, dataset):
    """Run anchor-vs-mean head-to-head on one cluster."""
    cluster_size = len(cluster_blocks)
    if cluster_size < 2:
        return

    layer_id = self_attn.layer_id
    num_tokens_before_cluster = retriever.num_tokens[layer_id]

    if num_tokens_before_cluster <= tr.N_INIT:
        return

    pool_indices = retriever.token_indices[
        layer_id, tr.N_INIT : num_tokens_before_cluster
    ].contiguous()

    if pool_indices.shape[0] < tr.TOP_K:
        return

    cluster_q_start = cluster_blocks[0] * base_chunk
    cluster_q_end = min((cluster_blocks[-1] + 1) * base_chunk, seq_len)

    # Anchor query = block 0 of cluster
    anchor_start = cluster_blocks[0] * base_chunk
    anchor_end = min((cluster_blocks[0] + 1) * base_chunk, seq_len)
    anchor_fp = build_query_fingerprint(
        retriever, q[anchor_start:anchor_end].contiguous(), head_dim, tr.N_Local
    )

    # Mean query = mean over whole cluster
    mean_fp = build_query_fingerprint(
        retriever, q[cluster_q_start:cluster_q_end].contiguous(), head_dim, tr.N_Local
    )

    # Per-sub-chunk baselines on the SAME pool
    per_subchunk_topk = []
    for block_idx in cluster_blocks:
        b_start = block_idx * base_chunk
        b_end = min((block_idx + 1) * base_chunk, seq_len)
        sub_fp = build_query_fingerprint(
            retriever, q[b_start:b_end].contiguous(), head_dim, tr.N_Local
        )
        I_j = retrieve_topk_on_pool(retriever, layer_id, sub_fp,
                                     pool_indices, tr.TOP_K)
        per_subchunk_topk.append(I_j)

    I_anchor = retrieve_topk_on_pool(retriever, layer_id, anchor_fp,
                                      pool_indices, tr.TOP_K)
    I_mean = retrieve_topk_on_pool(retriever, layer_id, mean_fp,
                                    pool_indices, tr.TOP_K)

    # Definition B: per-sub-chunk recall, then average
    per_recall_anchor = []
    per_recall_mean = []
    for I_j in per_subchunk_topk:
        if len(I_j) == 0:
            continue
        per_recall_anchor.append(len(I_anchor & I_j) / len(I_j))
        per_recall_mean.append(len(I_mean & I_j) / len(I_j))

    if not per_recall_anchor:
        return

    with open(OVERLAP_LOG_FILE, "a") as f:
        f.write(json.dumps({
            "dataset": dataset, "theta": theta, "L_max": lmax,
            "layer": layer_id, "cluster_size": cluster_size,
            "pool_size": int(pool_indices.shape[0]),
            "global_recall_anchor": sum(per_recall_anchor) / len(per_recall_anchor),
            "global_recall_mean": sum(per_recall_mean) / len(per_recall_mean),
            "first_recall_anchor": per_recall_anchor[0],
            "first_recall_mean": per_recall_mean[0],
            "last_recall_anchor": per_recall_anchor[-1],
            "last_recall_mean": per_recall_mean[-1],
        }) + "\n")


def patched_extend_forward_with_overlap(self, q, k, v, input_metadata):
    """
    Path A (drives generation): per-sub-chunk prefill exactly like production.
    Path B (profiling): BEFORE storing each cluster's first block, run the
                        anchor-vs-mean profiling on the pool that exists at
                        that moment (= context strictly before cluster).
    """
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

    BASE_CHUNK = tr.PREFILL_CHUNK_SIZE
    num_blocks = seq_len // BASE_CHUNK

    # IPC: active sweep config
    try:
        with open(SWEEP_PARAM_FILE, "r") as f:
            params = json.load(f)
            current_theta = params.get("theta", 0.95)
            current_lmax = params.get("lmax", 4096)
            current_dataset = params.get("dataset", "unknown")
    except Exception:
        current_theta, current_lmax, current_dataset = 0.95, 4096, "unknown"

    # Debug: confirm worker sees correct config
    sweep_key = (current_dataset, current_theta, current_lmax)
    if self.layer_id == 0 and sweep_key not in _DEBUG_PRINTED_FOR_SWEEP:
        print(f"[WORKER DEBUG] dataset={current_dataset} "
              f"theta={current_theta} L_max={current_lmax} "
              f"BASE_CHUNK={BASE_CHUNK} seq_len={seq_len} "
              f"num_blocks={num_blocks} "
              f"N_INIT={tr.N_INIT} N_Local={tr.N_Local} TOP_K={tr.TOP_K}",
              flush=True)
        _DEBUG_PRINTED_FOR_SWEEP.add(sweep_key)

    # Build cluster plan
    cluster_plan = []
    if num_blocks >= 2 and self.layer_id in TARGET_LAYERS:
        with torch.no_grad():
            q_trunc = q[:num_blocks * BASE_CHUNK].float()
            q_blocks_mean = q_trunc.view(num_blocks, BASE_CHUNK, -1).mean(dim=1)
            q_blocks_normalized = torch.nn.functional.normalize(
                q_blocks_mean, p=2, dim=-1
            )
        cluster_plan = compute_cluster_plan(
            q_blocks_normalized, num_blocks,
            current_theta, current_lmax, BASE_CHUNK
        )

    retriever = input_metadata.token_retriever
    head_dim = self.head_dim

    cluster_start_blocks = {c[0]: c for c in cluster_plan}

    num_chunks = ceil(seq_len / BASE_CHUNK)
    for chunk_idx in range(num_chunks):
        start = chunk_idx * BASE_CHUNK
        end = min((chunk_idx + 1) * BASE_CHUNK, seq_len)
        actual_step = end - start

        # PROFILING: at the START of a cluster, BEFORE storing the cluster
        if (self.layer_id in TARGET_LAYERS
                and chunk_idx in cluster_start_blocks):
            profile_cluster(
                self, q, retriever, head_dim,
                cluster_start_blocks[chunk_idx], BASE_CHUNK, seq_len,
                current_theta, current_lmax, current_dataset
            )

        if k is not None:
            assert v is not None
            self.store_kv_cache(k, v, input_metadata, start=start, end=end)

        retrieved_indices = retriever.retrieval_indices(
            q[start:end].contiguous(), self.layer_id, tr.N_INIT,
            tr.N_Local, tr.TOP_K
        )

        retrieved_indptr = prefill_wrapper_paged._paged_kv_indptr_buf.clone()
        if retrieved_indices is None:
            retrieved_indices = retriever.get_all_tokens(self.layer_id)

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
            causal=True, sm_scale=self.scaling,
            window_left=self.sliding_window_size,
            logits_soft_cap=self.logit_cap, rope_scale=tr.ROPE_SCALE,
            rope_theta=tr.ROPE_BASE, pos_encoding_mode=tr.ROPE_MODE,
        )
        outputs[start:end] = o.view(-1, self.tp_q_head_num * self.head_dim)

    return outputs


# =============================================================================
# PATCH INSTALLATION
# =============================================================================
def install_overlap_patch():
    tr.patch_input_metadata()
    tr.patch_model_runner()

    def patched_model_load():
        from vllm.model_executor.model_loader.loader import (
            DefaultModelLoader, _initialize_model, device_loading_context,
        )
        from vllm.model_executor.model_loader.utils import set_default_torch_dtype
        from vllm.model_executor.layers import rotary_embedding
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        def patched_default_model_loader_load_model(
            self, *, model_config, device_config, lora_config,
            multimodal_config, parallel_config, scheduler_config, cache_config,
        ):
            target_device = torch.device(device_config.device)
            with set_default_torch_dtype(model_config.dtype):
                with target_device:
                    model = _initialize_model(
                        model_config, self.load_config, lora_config,
                        multimodal_config, cache_config, scheduler_config,
                    )
                model.load_weights(
                    self._get_weights_iterator(
                        model_config.model, model_config.revision,
                        fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
                    ),
                )
                for _, module in model.named_modules():
                    quant_method = getattr(module, "quant_method", None)
                    if quant_method is not None:
                        with device_loading_context(module, target_device):
                            quant_method.process_weights_after_loading(module)

            def patched_radix_attention_forward(self_attn, q, k, v, input_metadata):
                if k is not None:
                    assert v is not None
                    k = k.view(-1, self_attn.tp_k_head_num, self_attn.qk_head_dim)
                    v = v.view(-1, self_attn.tp_v_head_num, self_attn.v_head_dim)
                if input_metadata.forward_mode == ForwardMode.EXTEND:
                    return patched_extend_forward_with_overlap(
                        self_attn, q, k, v, input_metadata
                    )
                elif input_metadata.forward_mode == ForwardMode.DECODE:
                    return _original_decode_forward(self_attn, q, k, v, input_metadata)

            global _original_decode_forward
            def _original_decode_forward(self_attn, q, k, v, input_metadata):
                decode_wrapper = input_metadata.flashinfer_decode_wrapper
                if self_attn.sliding_window_size != -1:
                    decode_wrapper = decode_wrapper[0]
                else:
                    if isinstance(decode_wrapper, list):
                        decode_wrapper = decode_wrapper[1]
                if k is not None:
                    self_attn.store_kv_cache(k, v, input_metadata)
                retrieved = input_metadata.token_retriever.retrieval_indices(
                    q.contiguous(), self_attn.layer_id, tr.N_INIT, tr.N_Local, tr.TOP_K
                )
                if retrieved is not None:
                    if (self_attn.layer_id == 0
                            or len(retrieved) != len(decode_wrapper._paged_kv_indices_buf)):
                        retrieved_indptr = decode_wrapper._paged_kv_indptr_buf.clone()
                        retrieved_indptr[1] = len(retrieved)
                        klp = decode_wrapper._paged_kv_last_page_len_buf.clone()
                        decode_wrapper.end_forward()
                        decode_wrapper.begin_forward(
                            retrieved_indptr, retrieved, klp,
                            self_attn.tp_q_head_num, self_attn.tp_k_head_num,
                            self_attn.head_dim, 1,
                        )
                    else:
                        decode_wrapper._paged_kv_indices_buf.copy_(retrieved)
                o = decode_wrapper.forward(
                    q.contiguous().view(-1, self_attn.tp_q_head_num, self_attn.head_dim),
                    input_metadata.token_to_kv_pool.get_kv_buffer(self_attn.layer_id),
                    sm_scale=self_attn.scaling, logits_soft_cap=self_attn.logit_cap,
                    rope_scale=tr.ROPE_SCALE, rope_theta=tr.ROPE_BASE,
                    pos_encoding_mode=tr.ROPE_MODE,
                )
                return o.view(-1, self_attn.tp_q_head_num * self_attn.head_dim)

            def patched_store_kv_cache(self_attn, cache_k, cache_v, input_metadata, start=0, end=None):
                k_cache = input_metadata.token_to_kv_pool.get_key_buffer(self_attn.layer_id)
                v_cache = input_metadata.token_to_kv_pool.get_value_buffer(self_attn.layer_id)
                k_cache[input_metadata.out_cache_loc[start:end]] = cache_k[start:end]
                v_cache[input_metadata.out_cache_loc[start:end]] = cache_v[start:end]
                input_metadata.token_retriever.add_k_cache(
                    input_metadata.out_cache_loc[start:end], self_attn.layer_id,
                )

            def patched_meta_attention_forward(self_meta, positions, hidden_states, input_metadata):
                qkv, _ = self_meta.qkv_proj(hidden_states)
                q, k, v = qkv.split(
                    [self_meta.q_size, self_meta.kv_size, self_meta.kv_size], dim=-1
                )
                attn_output = self_meta.attn(q, k, v, input_metadata)
                output, _ = self_meta.o_proj(attn_output)
                return output

            for layer in model.model.layers:
                layer.self_attn.__class__.forward = patched_meta_attention_forward
                layer.self_attn.attn.__class__.forward = patched_radix_attention_forward
                layer.self_attn.attn.__class__.store_kv_cache = patched_store_kv_cache

            return model.eval()

        rotary_embedding.get_rope = lambda *args, **kwargs: None
        DefaultModelLoader.load_model = patched_default_model_loader_load_model

    patched_model_load()


# =============================================================================
# DATA LOADING & SWEEPS
# =============================================================================
def load_infinite_bench_sample(path, data_name, num_samples, seed=42):
    fin = open(os.path.join(path, data_name + ".jsonl"), "r")
    data = [json.loads(line) for line in fin.readlines()]
    fin.close()
    random.seed(seed)
    return random.sample(data, min(num_samples, len(data)))


def build_prompt(eg, data_name, prompt_template):
    instance = {
        "context": eg.get("context", eg.get("content", "")),
        "input": eg.get("input", ""),
    }
    if data_name == "kv_retrieval":
        instance["key"] = eg["input"][6:44]
    elif data_name == "code_debug":
        instance.update({
            "OPTION_A": eg["options"][0], "OPTION_B": eg["options"][1],
            "OPTION_C": eg["options"][2], "OPTION_D": eg["options"][3],
        })
    elif data_name == "math_find":
        import re
        prompt = eg["input"]
        find_result = re.findall(r"The .+ of", prompt)
        target_number = find_result[0].lower()[:-3]
        instance["prefix"] = f"What is {target_number} in the following list?"
        instance["input"] = prompt
    return prompt_template.format(**instance)


def run_one_sweep(theta, lmax, model, tokenizer, samples,
                  prompt_template, dataset_name, max_gen):
    print(f"\n>>> Sweep: theta={theta}, L_max={lmax}")
    with open(SWEEP_PARAM_FILE, "w") as f:
        json.dump({"theta": theta, "lmax": lmax, "dataset": dataset_name}, f)
    _DEBUG_PRINTED_FOR_SWEEP.clear()

    for eg in tqdm(samples, desc=f"theta={theta}, L_max={lmax}"):
        prompt = build_prompt(eg, dataset_name, prompt_template)
        prompt_formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True
        )
        tokenized = tokenizer(
            prompt_formatted, truncation=False, return_tensors="pt",
            add_special_tokens=True,
        ).input_ids[0]
        sampling_params = {
            "max_new_tokens": max_gen,
            "stop_token_ids": [tokenizer.eos_token_id],
            "temperature": 0,
        }
        requests.post(
            model.url + "/generate",
            json={"input_ids": tokenized.tolist(), "sampling_params": sampling_params},
        )


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--datasets", type=str, default="kv_retrieval")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--output_path", default="overlap_anchor_vs_mean.json")
    parser.add_argument("--tp_size", type=int, default=1)
    args = parser.parse_args()

    for f in [OVERLAP_LOG_FILE, SWEEP_PARAM_FILE]:
        if os.path.exists(f):
            os.remove(f)

    datasets_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    print(f"Will profile on {len(datasets_list)} dataset(s): {datasets_list}")

    config = OmegaConf.load(args.config_path)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_path if hasattr(config.model, 'tokenizer_path')
        else config.model.path
    )

    tr.ROPE_BASE = config.model.rope_base
    tr.ROPE_SCALE = config.model.rope_scale
    tr.ROPE_MODE = "ROPE_LLAMA"
    tr.MAX_N_TOKENS = config.model.max_n_tokens
    tr.TOP_K = config.model.top_k
    tr.N_INIT = config.model.n_init
    tr.N_Local = config.model.n_local
    tr.PREFILL_CHUNK_SIZE = 512
    tr.QUERY_ROTATE = True
    tr.QUERY_CACHE = False
    tr.USE_DYNAMIC_CHUNKING = False
    tr.DYNAMIC_BUDGET_BALANCING = False
    tr.SIM_THRESHOLD = 0.95
    tr.MAX_DYNAMIC_CHUNK = 4096
    tr.L2_NORM_POOLING = False

    install_overlap_patch()

    from sglang.srt.server import Runtime
    print("Launching model server...")
    model = Runtime(
        model_path=config.model.path,
        dtype=config.dtype,
        chunked_prefill_size=config.chunk_size,
        max_prefill_tokens=config.max_len,
        mem_fraction_static=0.5,
        disable_cuda_graph=True,
        disable_regex_jump_forward=True,
        disable_radix_cache=True,
        disable_disk_cache=True,
        disable_flashinfer_sampling=True,
        max_running_requests=1,
        tp_size=args.tp_size,
        port=60010,
        additional_ports=list(range(60011, 60100)),
        context_length=config.max_len,
    )

    dataset2prompt = json.load(open("benchmark/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("benchmark/config/dataset2maxlen.json", "r"))

    for dataset_name in datasets_list:
        print("\n" + "=" * 80)
        print(f" DATASET: {dataset_name}")
        print("=" * 80)
        prompt_template = dataset2prompt[dataset_name]
        max_gen = dataset2maxlen[dataset_name]
        samples = load_infinite_bench_sample(
            "benchmark/data/infinite-bench", dataset_name, args.num_samples
        )
        print(f"Loaded {len(samples)} samples from {dataset_name}")
        for theta, lmax in SWEEP_CONFIGS:
            run_one_sweep(theta, lmax, model, tokenizer, samples,
                          prompt_template, dataset_name, max_gen)

    # =========================================================================
    # Aggregate and report
    # =========================================================================
    import numpy as np

    per_task_records = defaultdict(lambda: defaultdict(list))
    aggregate_records = defaultdict(list)

    if os.path.exists(OVERLAP_LOG_FILE):
        with open(OVERLAP_LOG_FILE, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                key = (d["theta"], d["L_max"], d["layer"])
                task_key = f"{d['dataset']}__theta{d['theta']}__Lmax{d['L_max']}"
                per_task_records[task_key][key].append(d)
                aggregate_records[key].append(d)

    def summarize(records_dict):
        summary = []
        for (theta, lmax, layer), recs in sorted(records_dict.items()):
            if not recs:
                continue
            ga = np.array([r["global_recall_anchor"] for r in recs]) * 100
            gm = np.array([r["global_recall_mean"] for r in recs]) * 100
            fa = np.array([r["first_recall_anchor"] for r in recs]) * 100
            fm = np.array([r["first_recall_mean"] for r in recs]) * 100
            la = np.array([r["last_recall_anchor"] for r in recs]) * 100
            lm = np.array([r["last_recall_mean"] for r in recs]) * 100
            cs = np.array([r["cluster_size"] for r in recs])
            ps = np.array([r["pool_size"] for r in recs])
            summary.append({
                "theta": theta, "L_max": lmax, "layer": layer,
                "num_clusters": len(recs),
                "avg_cluster_size": float(cs.mean()),
                "avg_pool_size": float(ps.mean()),
                "global_anchor_median": float(np.median(ga)),
                "global_mean_median": float(np.median(gm)),
                "first_anchor_median": float(np.median(fa)),
                "first_mean_median": float(np.median(fm)),
                "last_anchor_median": float(np.median(la)),
                "last_mean_median": float(np.median(lm)),
                "anchor_minus_mean_global": float(np.median(ga) - np.median(gm)),
            })
        return summary

    per_task_summary = {tk: summarize(r) for tk, r in per_task_records.items()}
    aggregate_summary = summarize(aggregate_records)

    output = {
        "datasets": datasets_list,
        "num_samples_per_task": args.num_samples,
        "sweep_configs": SWEEP_CONFIGS,
        "target_layers": TARGET_LAYERS,
        "per_task_results": per_task_summary,
        "aggregate_results": aggregate_summary,
    }
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)

    # =========================================================================
    # Console report
    # =========================================================================
    print("\n" + "=" * 140)
    print(" ANCHOR vs MEAN — AGGREGATE")
    print("=" * 140)
    print(f"{'theta':<7}{'L_max':<7}{'layer':<7}{'#cl':<6}{'csz':<6}"
          f"{'GlobA':<8}{'GlobM':<8}{'Δ(A-M)':<10}"
          f"{'1stA':<8}{'1stM':<8}{'LastA':<8}{'LastM':<8}")
    print("-" * 140)

    sanity_failures = []
    for r in aggregate_summary:
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
        if r['first_anchor_median'] < 99.0:
            sanity_failures.append((r['theta'], r['L_max'], r['layer'],
                                     r['first_anchor_median']))

    print("\n" + "=" * 80)
    if sanity_failures:
        print(" !!! SANITY CHECK FAILED !!!")
        print(" First-Chunk Recall (Anchor) must be ~100% — anchor uses block-1 query")
        print(" on the SAME pool as the block-1 baseline. Failures:")
        for theta, lmax, layer, val in sanity_failures:
            print(f"   theta={theta}, L_max={lmax}, layer={layer}: "
                  f"first_anchor_recall={val:.2f}%")
        print(" Investigate before trusting any numbers.")
    else:
        print(" SANITY CHECK PASSED")
        print(" First-Chunk Recall (Anchor) ~ 100% across all configs.")
        print(" Pool consistency confirmed. Numbers are trustworthy.")
    print("=" * 80)

    # Verdict
    print("\n" + "=" * 80)
    print(" VERDICT: anchor vs mean (based on Global Recall median)")
    print("=" * 80)
    anchor_wins = sum(1 for r in aggregate_summary
                       if r['anchor_minus_mean_global'] > 0.5)
    mean_wins = sum(1 for r in aggregate_summary
                     if r['anchor_minus_mean_global'] < -0.5)
    ties = len(aggregate_summary) - anchor_wins - mean_wins
    print(f"  Anchor wins (Δ > +0.5%):  {anchor_wins} / {len(aggregate_summary)} configs")
    print(f"  Mean wins  (Δ < -0.5%):   {mean_wins} / {len(aggregate_summary)} configs")
    print(f"  Comparable (|Δ| <= 0.5%): {ties} / {len(aggregate_summary)} configs")

    if anchor_wins > 2 * mean_wins:
        print("\n  -> ANCHOR meaningfully better than MEAN")
    elif mean_wins > 2 * anchor_wins:
        print("\n  -> MEAN meaningfully better than ANCHOR")
    else:
        print("\n  -> ANCHOR and MEAN perform comparably")

    print(f"\nResults saved to {args.output_path}")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()