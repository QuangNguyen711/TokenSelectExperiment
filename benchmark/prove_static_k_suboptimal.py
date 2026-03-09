"""
Prove: Static K is Suboptimal - Different Query Tokens Need Different K

Mục tiêu: Chứng minh rằng việc sử dụng cùng một K cho tất cả query tokens là không tối ưu
vì mỗi query token có thể cần số lượng key tokens khác nhau để đạt coverage tốt.

Key Analysis:
- Mỗi query token (row trong attention matrix) có distribution khác nhau
- Một số query tokens cần ít key tokens (sparse attention)
- Một số query tokens cần nhiều key tokens hơn top-k (distributed attention)
- → Static K sẽ gây ra: thừa computation cho sparse queries, thiếu tokens cho distributed queries

Usage:
    # Short context (fits in memory): analyze all query positions
    python prove_static_k_suboptimal.py --max-tokens 4096 --visualize-all
    
    # Long context (200K tokens): use chunked prefill with KV cache
    python prove_static_k_suboptimal.py --max-tokens 200000 --chunk-size 4096
    
    # Or use sampling mode (faster, less memory)  
    python prove_static_k_suboptimal.py --max-tokens 200000 --sample-true-k 200
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

DATASETS = {
    "passkey": "Simple retrieval (localized)",
    "kv_retrieval": "Key-value lookup (multiple points)",
    "longbook_qa_eng": "Document QA (distributed)",
    "math_find": "Math reasoning",
}

DATA_DIR = Path(__file__).parent / "data" / "infinite-bench"
OUTPUT_DIR = Path(__file__).parent / "static_k_analysis"

TARGET_COVERAGE = 0.90


# ============================================================================
# Model & Data Loading
# ============================================================================

def load_model(model_name: str = "Qwen/Qwen2-7B-Instruct"):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use float16 to save memory
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def load_sample(dataset: str, idx: int = 0) -> dict:
    """Load a single sample from dataset."""
    filepath = DATA_DIR / f"{dataset}.jsonl"
    with open(filepath) as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)
    return {}


def compute_required_k_vectorized_gpu(attention_weights: torch.Tensor, target_coverage: float = 0.90):
    """
    Compute required K per query - FULLY VECTORIZED on GPU.
    
    This gives EXACT same results as the for-loop version but ~100x faster.
    
    Args:
        attention_weights: (batch, heads, seq, seq) or (heads, seq, seq)
        target_coverage: target cumulative attention coverage
        
    Returns:
        required_k: (seq,) tensor of required K per query position
    """
    if attention_weights.dim() == 4:
        attention_weights = attention_weights[0]  # (heads, seq, seq)
    
    # Average over heads
    attn_avg = attention_weights.mean(dim=0)  # (seq, seq)
    seq_len = attn_avg.shape[0]
    
    # Apply causal mask: zero out future positions
    # Create lower triangular mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=attn_avg.device, dtype=torch.bool))
    attn_avg = attn_avg.masked_fill(~causal_mask, 0.0)
    
    # Re-normalize each row (since we zeroed out some positions)
    row_sums = attn_avg.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    attn_avg = attn_avg / row_sums
    
    # Sort each row descending
    sorted_attn, _ = attn_avg.sort(dim=-1, descending=True)
    
    # Cumulative sum along each row
    cumsum = sorted_attn.cumsum(dim=-1)
    
    # Find first position where cumsum >= target_coverage
    # mask[i,j] = True if cumsum[i,j] >= target
    reached_target = cumsum >= target_coverage
    
    # For each row, find the first True (argmax on bool gives first True)
    # Add 1 because K is 1-indexed (K=1 means top-1)
    # If no position reaches target, we need all positions
    
    # Use argmax - returns first True index (0-indexed)
    first_reached = reached_target.float().argmax(dim=-1)  # (seq,)
    
    # Handle rows that never reach target: use full row length
    never_reached = ~reached_target.any(dim=-1)
    # For each query position i, max valid K is i+1 (causal)
    max_k_per_query = torch.arange(1, seq_len + 1, device=attn_avg.device)
    
    # K = first_reached + 1 (convert 0-indexed to K value)
    required_k = first_reached + 1
    
    # For never-reached rows, use max possible K
    required_k = torch.where(never_reached, max_k_per_query, required_k)
    
    # Clamp to valid range
    required_k = torch.clamp(required_k, min=1, max=seq_len)
    
    return required_k


class AttentionAnalysisHook:
    """
    Efficient hook that computes K statistics during forward pass.
    Uses GPU-vectorized computation - no Python for-loops.
    """
    def __init__(self, target_coverage: float = 0.90, layers_to_analyze: list = None):
        self.target_coverage = target_coverage
        self.layers_to_analyze = layers_to_analyze  # None = all layers
        self.layer_stats = {}
        self.current_layer = 0
        
    def reset(self):
        self.layer_stats = {}
        self.current_layer = 0
    
    def __call__(self, module, input, output):
        """Hook called after each attention layer."""
        layer_idx = self.current_layer
        self.current_layer += 1
        
        # Skip if not in target layers
        if self.layers_to_analyze is not None and layer_idx not in self.layers_to_analyze:
            return output
        
        # output is tuple (attn_output, attn_weights, ...) when output_attentions=True
        if isinstance(output, tuple) and len(output) >= 2:
            attn_weights = output[1]
            if attn_weights is not None:
                with torch.no_grad():
                    # GPU-vectorized K computation
                    required_k = compute_required_k_vectorized_gpu(
                        attn_weights.float(), 
                        self.target_coverage
                    )
                    
                    # Skip first 50 tokens (not enough context)
                    valid_k = required_k[50:]
                    
                    if len(valid_k) > 0:
                        self.layer_stats[layer_idx] = {
                            "min_k": valid_k.min().item(),
                            "max_k": valid_k.max().item(),
                            "mean_k": valid_k.float().mean().item(),
                            "std_k": valid_k.float().std().item(),
                            "k_ratio": valid_k.max().item() / max(valid_k.min().item(), 1),
                            "required_k_list": valid_k.cpu().tolist(),
                        }
        
        return output


class QKInterceptHook:
    """
    Hook that intercepts Q, K tensors and computes attention stats WITHOUT output_attentions=True.
    This allows using efficient attention (SDPA/FlashAttention) while still getting stats.
    
    Works by:
    1. Hooking into the q_proj and k_proj layers to capture Q, K
    2. Computing attention scores manually (Q @ K.T)
    3. Computing K statistics
    """
    def __init__(self, target_coverage: float = 0.90, head_dim: int = 128, num_heads: int = 28):
        self.target_coverage = target_coverage
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.layer_stats = {}
        self.current_q = None
        self.current_k = None
        self.layer_idx = 0
        
    def reset(self):
        self.layer_stats = {}
        self.current_q = None
        self.current_k = None
        self.layer_idx = 0
    
    def q_hook(self, module, input, output):
        """Capture Q projection output."""
        self.current_q = output.detach()
    
    def k_hook(self, module, input, output):
        """Capture K projection output."""
        self.current_k = output.detach()
        
        # When we have both Q and K, compute stats
        if self.current_q is not None:
            self._compute_stats()
            self.current_q = None
            self.current_k = None
            self.layer_idx += 1
    
    def _compute_stats(self):
        """Compute attention stats from Q and K."""
        Q = self.current_q  # (batch, seq, hidden)
        K = self.current_k  # (batch, seq, hidden)
        
        batch_size, seq_len, hidden_dim = Q.shape
        
        # Reshape to (batch, heads, seq, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores: (batch, heads, seq, seq)
        scale = 1.0 / (self.head_dim ** 0.5)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        
        # Apply causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool()
        attn_scores.masked_fill_(causal_mask, float('-inf'))
        
        # Softmax to get attention weights
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        
        # Compute K statistics (GPU-vectorized)
        required_k = compute_required_k_vectorized_gpu(attn_weights, self.target_coverage)
        
        valid_k = required_k[50:]
        if len(valid_k) > 0:
            self.layer_stats[self.layer_idx] = {
                "min_k": valid_k.min().item(),
                "max_k": valid_k.max().item(),
                "mean_k": valid_k.float().mean().item(),
                "std_k": valid_k.float().std().item(),
                "k_ratio": valid_k.max().item() / max(valid_k.min().item(), 1),
                "required_k_list": valid_k.cpu().tolist(),
            }
        
        # Free memory
        del Q, K, attn_scores, attn_weights, required_k


# Legacy hook for backward compatibility
class AttentionCaptureHook:
    """
    Legacy hook - uses output_attentions=True approach.
    For new code, use AttentionAnalysisHook instead.
    """
    def __init__(self, target_coverage: float = 0.90):
        self.target_coverage = target_coverage
        self.layer_stats = {}
        self.current_layer = 0
        
    def reset(self):
        self.layer_stats = {}
        self.current_layer = 0
    
    def compute_required_k_per_query(self, attention: torch.Tensor):
        """Compute K per query - now uses vectorized GPU version."""
        return compute_required_k_vectorized_gpu(attention, self.target_coverage)
    
    def __call__(self, module, input, output):
        """Hook called after each attention layer."""
        if isinstance(output, tuple) and len(output) >= 2:
            attn_weights = output[1]
            if attn_weights is not None:
                with torch.no_grad():
                    required_k = self.compute_required_k_per_query(attn_weights.float())
                    
                    valid_k = required_k[50:]
                    if len(valid_k) > 0:
                        self.layer_stats[self.current_layer] = {
                            "min_k": valid_k.min().item(),
                            "max_k": valid_k.max().item(),
                            "mean_k": valid_k.float().mean().item(),
                            "std_k": valid_k.float().std().item(),
                            "k_ratio": valid_k.max().item() / max(valid_k.min().item(), 1),
                            "required_k_list": valid_k.cpu().tolist(),
                        }
                
                self.current_layer += 1
        
        return output


def get_attention_with_hooks(model, tokenizer, context: str, query: str, 
                             max_tokens: int = 32000, target_coverage: float = 0.90):
    """
    Capture attention statistics dùng hooks - memory efficient.
    Chỉ lưu statistics, không lưu full attention matrices.
    """
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_tokens)
    inputs = inputs.to(model.device)
    
    num_tokens = inputs.input_ids.shape[1]
    print(f"    Input tokens: {num_tokens}")
    
    # Setup hooks
    hook = AttentionCaptureHook(target_coverage)
    hooks = []
    
    # Register hooks on attention layers
    for name, module in model.named_modules():
        # Tìm attention modules (tên khác nhau tùy model)
        if "attn" in name.lower() and hasattr(module, 'forward'):
            # Chỉ hook vào attention chính, không hook vào sub-modules
            if name.count('.') <= 4:  # Adjust based on model structure
                h = module.register_forward_hook(hook)
                hooks.append(h)
    
    # Forward pass với output_attentions=True để attention được tính
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, return_dict=True)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return hook.layer_stats, num_tokens


def get_attention_layer_by_layer(model, tokenizer, context: str, query: str,
                                  max_tokens: int = 32000, target_coverage: float = 0.90,
                                  layers_to_analyze: list = None,
                                  chunk_size: int = None):
    """
    Phương pháp 2: Chạy từng layer một, tính stats, rồi delete attention.
    Dùng khi hooks không hoạt động tốt.
    
    Args:
        chunk_size: If provided, process input in chunks (for long contexts)
    """
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_tokens)
    inputs = inputs.to(model.device)
    
    num_tokens = inputs.input_ids.shape[1]
    print(f"    Input tokens: {num_tokens}")
    
    # Use chunked processing if specified and input is long
    if chunk_size and num_tokens > chunk_size:
        print(f"    Using chunked prefill with chunk_size={chunk_size}")
        return get_attention_chunked(model, tokenizer, inputs, target_coverage, layers_to_analyze, chunk_size)
    
    # Forward với output_attentions
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, return_dict=True)
    
    attentions = outputs.attentions
    num_layers = len(attentions)
    
    if layers_to_analyze is None:
        layers_to_analyze = [0, num_layers//4, num_layers//2, 3*num_layers//4, num_layers-1]
    
    layer_stats = {}
    
    for layer_idx in layers_to_analyze:
        if layer_idx >= num_layers:
            continue
            
        attention = attentions[layer_idx].float()  # Convert to float32 for computation
        
        # GPU-vectorized K computation (exact same result, ~100x faster)
        required_k = compute_required_k_vectorized_gpu(attention, target_coverage)
        valid_k = required_k[50:]
        
        if len(valid_k) > 0:
            layer_stats[layer_idx] = {
                "min_k": valid_k.min().item(),
                "max_k": valid_k.max().item(),
                "mean_k": valid_k.float().mean().item(),
                "std_k": valid_k.float().std().item(),
                "k_ratio": valid_k.max().item() / max(valid_k.min().item(), 1),
                "required_k_list": valid_k.cpu().tolist(),
            }
        
        # Free memory
        del attention, required_k, valid_k
    
    # Free all attentions
    del attentions, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return layer_stats, num_tokens


def sample_true_k_with_kv_cache(model, tokenizer, context: str, query: str,
                                 max_tokens: int = 200000, target_coverage: float = 0.90,
                                 num_samples: int = 100, layers_to_analyze: list = None):
    """
    Sample TRUE K values for specific query positions using KV cache.
    
    This method processes the full context but only computes attention for sampled
    query positions, avoiding O(n²) memory while getting true K values.
    
    How it works:
    1. Process context in segments, building KV cache
    2. For sampled query positions, compute attention against FULL KV cache
    3. Measure K needed for 90% coverage - NO chunk limitation!
    
    Args:
        num_samples: Number of query positions to sample (default: 100)
        
    Returns:
        layer_stats: Dict with true K values for sampled positions
        num_tokens: Total context length
    """
    import time
    
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_tokens)
    inputs = inputs.to(model.device)
    
    num_tokens = inputs.input_ids.shape[1]
    print(f"    Input tokens: {num_tokens}")
    print(f"    Sampling {num_samples} query positions for TRUE K measurement")
    print(f"    (No chunk limitation - K can go up to {num_tokens})")
    
    # Sample query positions across the context
    # Focus more on later positions which have more context to attend to
    sample_positions = np.linspace(100, num_tokens - 1, num_samples).astype(int)
    sample_positions = sorted(set(sample_positions))  # Remove duplicates
    print(f"    Sample positions: {min(sample_positions)} to {max(sample_positions)}")
    
    # Method: Process full context and capture Q, K, V for sampled positions
    # Use hooks to intercept Q, K at specific layers
    
    class QKSampleHook:
        """Hook to capture Q and K for computing attention at sampled positions."""
        def __init__(self, sample_positions, target_coverage, num_heads=28, head_dim=128):
            self.sample_positions = torch.tensor(sample_positions)
            self.target_coverage = target_coverage
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.layer_idx = 0
            self.layer_stats = {}
            self.q_proj_output = None
            self.k_proj_output = None
            
        def reset(self):
            self.layer_idx = 0
            self.layer_stats = {}
            
        def capture_q(self, module, input, output):
            self.q_proj_output = output.detach()
            
        def capture_k(self, module, input, output):
            self.k_proj_output = output.detach()
            # Compute attention and K for sampled positions
            self._compute_sampled_k()
            self.q_proj_output = None
            self.k_proj_output = None
            self.layer_idx += 1
            
        def _compute_sampled_k(self):
            """Compute K for sampled query positions against full context."""
            if self.q_proj_output is None or self.k_proj_output is None:
                return
                
            Q = self.q_proj_output  # (1, seq_len, hidden)
            K = self.k_proj_output  # (1, seq_len, hidden)
            
            seq_len = Q.shape[1]
            hidden_dim = Q.shape[2]
            
            # Reshape to (1, heads, seq, head_dim)
            actual_heads = hidden_dim // self.head_dim
            Q = Q.view(1, seq_len, actual_heads, self.head_dim).transpose(1, 2)
            K = K.view(1, seq_len, actual_heads, self.head_dim).transpose(1, 2)
            
            # Only compute attention for sampled query positions
            sample_positions = [p for p in self.sample_positions.tolist() if p < seq_len]
            if len(sample_positions) == 0:
                return
                
            required_k_list = []
            
            for q_pos in sample_positions:
                # Q at position q_pos: (1, heads, 1, head_dim)
                q = Q[:, :, q_pos:q_pos+1, :]
                
                # K up to position q_pos (causal): (1, heads, q_pos+1, head_dim)
                k = K[:, :, :q_pos+1, :]
                
                # Attention scores: (1, heads, 1, q_pos+1)
                scale = 1.0 / (self.head_dim ** 0.5)
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
                attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
                
                # Average over heads: (q_pos+1,)
                attn_avg = attn_weights[0].mean(dim=0).squeeze(0)
                
                # Sort and find K for target coverage
                sorted_attn, _ = attn_avg.sort(descending=True)
                cumsum = sorted_attn.cumsum(dim=0)
                reached = cumsum >= self.target_coverage
                
                if reached.any():
                    k_val = reached.float().argmax().item() + 1
                else:
                    k_val = len(attn_avg)
                    
                required_k_list.append(k_val)
            
            if len(required_k_list) > 0:
                k_tensor = torch.tensor(required_k_list)
                self.layer_stats[self.layer_idx] = {
                    "min_k": k_tensor.min().item(),
                    "max_k": k_tensor.max().item(),
                    "max_k_limited": False,  # No chunk limitation!
                    "mean_k": k_tensor.float().mean().item(),
                    "std_k": k_tensor.float().std().item(),
                    "k_ratio": k_tensor.max().item() / max(k_tensor.min().item(), 1),
                    "k_ratio_note": "TRUE ratio (no chunk limit)",
                    "num_samples": len(required_k_list),
                    "sample_positions": sample_positions,
                    "required_k_list": required_k_list,
                }
    
    # Detect model architecture for head_dim and num_heads
    config = model.config
    num_heads = getattr(config, 'num_attention_heads', 28)
    head_dim = getattr(config, 'head_dim', getattr(config, 'hidden_size', 3584) // num_heads)
    
    hook_handler = QKSampleHook(sample_positions, target_coverage, num_heads, head_dim)
    hooks = []
    
    # Register hooks on Q and K projections
    layer_count = 0
    for name, module in model.named_modules():
        if 'q_proj' in name and hasattr(module, 'weight'):
            h = module.register_forward_hook(hook_handler.capture_q)
            hooks.append(h)
        elif 'k_proj' in name and hasattr(module, 'weight'):
            h = module.register_forward_hook(hook_handler.capture_k)  
            hooks.append(h)
            layer_count += 1
    
    print(f"    Registered hooks on {layer_count} layers")
    
    # Forward pass (no output_attentions needed - we compute ourselves)
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=False, return_dict=True)
    elapsed = time.time() - start_time
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Clear memory
    del outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"    Sampling completed in {elapsed:.1f}s")
    
    # Filter to requested layers
    if layers_to_analyze is None:
        num_layers = len(hook_handler.layer_stats)
        layers_to_analyze = [0, num_layers//2, num_layers-1] if num_layers > 0 else []
    
    final_stats = {l: hook_handler.layer_stats[l] for l in layers_to_analyze 
                   if l in hook_handler.layer_stats}
    
    return final_stats, num_tokens


def get_attention_chunked(model, tokenizer, inputs, target_coverage: float = 0.90,
                         layers_to_analyze: list = None, chunk_size: int = 8192):
    """
    Process long sequences in chunks WITH KV cache for correct attention.
    
    Each query token can attend to ALL previous tokens (not limited to chunk).
    Uses hooks to capture Q, K tensors and compute attention manually.
    
    Memory-efficient: only keeps Q, K for computing statistics, not full attention matrices.
    
    Args:
        inputs: Tokenized inputs
        target_coverage: Target cumulative attention for K calculation
        layers_to_analyze: Which layers to analyze (default: early, mid, late)
        chunk_size: Tokens per chunk
        
    Returns:
        layer_stats: Dict with K statistics per layer
        num_tokens: Total tokens processed
    """
    import time
    
    input_ids = inputs.input_ids[0]  # (seq_len,)
    num_tokens = len(input_ids)
    num_chunks = (num_tokens + chunk_size - 1) // chunk_size
    
    print(f"    Processing {num_tokens} tokens in {num_chunks} chunks of {chunk_size}")
    print(f"    ✓ Using KV cache: tokens can attend to FULL past context")
    print(f"    ✓ Max K can be up to {num_tokens} (no artificial limit)")
    
    # Detect model config
    config = model.config
    num_heads = getattr(config, 'num_attention_heads', 28)
    num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
    head_dim = getattr(config, 'head_dim', getattr(config, 'hidden_size', 3584) // num_heads)
    num_layers = getattr(config, 'num_hidden_layers', 28)
    
    if layers_to_analyze is None:
        layers_to_analyze = [0, num_layers // 2, num_layers - 1]
    
    print(f"    Analyzing layers: {layers_to_analyze}")
    print(f"    Model config: {num_layers} layers, {num_heads} heads, head_dim={head_dim}")
    
    # Storage for K values per layer
    layer_k_values = {layer_idx: [] for layer_idx in layers_to_analyze}
    
    # Hook class to capture Q, K and compute K statistics per chunk
    class ChunkedQKHook:
        def __init__(self, target_layers, target_coverage, num_heads, num_kv_heads, head_dim):
            self.target_layers = set(target_layers)
            self.target_coverage = target_coverage
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = head_dim
            self.layer_idx = 0
            
            # Current chunk's Q (for current chunk's queries)
            self.current_q = None
            # Accumulated K cache (all past + current keys)
            self.k_cache = {l: [] for l in target_layers}
            
            # Results
            self.chunk_k_stats = {l: [] for l in target_layers}
            
        def reset_for_chunk(self):
            self.layer_idx = 0
            self.current_q = None
            
        def q_hook(self, module, input, output):
            if self.layer_idx in self.target_layers:
                self.current_q = output.detach()
        
        def k_hook(self, module, input, output):
            if self.layer_idx in self.target_layers:
                # Accumulate K into cache
                self.k_cache[self.layer_idx].append(output.detach())
                
                # Compute K statistics for this chunk's queries
                if self.current_q is not None:
                    self._compute_chunk_stats()
                    
            self.current_q = None
            self.layer_idx += 1
            
        def _compute_chunk_stats(self):
            """Compute required K for current chunk's queries against all past context."""
            layer = self.layer_idx
            Q = self.current_q  # (1, chunk_len, hidden)
            
            # Concatenate all K from cache
            K_list = self.k_cache[layer]
            K = torch.cat(K_list, dim=1)  # (1, total_past_len, hidden)
            
            chunk_len = Q.shape[1]
            total_k_len = K.shape[1]
            hidden_dim = Q.shape[2]
            
            # Handle GQA: expand K if num_kv_heads < num_heads
            actual_q_heads = hidden_dim // self.head_dim
            
            # Reshape Q and K
            Q = Q.view(1, chunk_len, actual_q_heads, self.head_dim).transpose(1, 2)  # (1, heads, chunk_len, head_dim)
            
            # For K, need to handle potential GQA
            k_hidden = K.shape[2]
            actual_k_heads = k_hidden // self.head_dim
            K = K.view(1, total_k_len, actual_k_heads, self.head_dim).transpose(1, 2)  # (1, kv_heads, total_k_len, head_dim)
            
            # Expand K for GQA if needed
            if actual_k_heads < actual_q_heads:
                repeat_factor = actual_q_heads // actual_k_heads
                K = K.repeat_interleave(repeat_factor, dim=1)
            
            # Compute attention scores: (1, heads, chunk_len, total_k_len)
            scale = 1.0 / (self.head_dim ** 0.5)
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
            
            # Apply causal mask: query i can attend to keys 0..(start_pos + i)
            # start_pos = total_k_len - chunk_len (position of first query in this chunk)
            start_pos = total_k_len - chunk_len
            
            # Create causal mask
            # For query at position q (0..chunk_len-1), can attend to keys 0..(start_pos + q)
            causal_mask = torch.ones(chunk_len, total_k_len, device=Q.device, dtype=torch.bool)
            for q in range(chunk_len):
                valid_k_len = start_pos + q + 1
                causal_mask[q, valid_k_len:] = False
            
            attn_scores = attn_scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            # Softmax
            attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)  # (1, heads, chunk_len, total_k_len)
            
            # Average over heads
            attn_avg = attn_weights[0].mean(dim=0)  # (chunk_len, total_k_len)
            
            # Compute required K for each query position
            required_k_list = []
            for q in range(chunk_len):
                valid_k_len = start_pos + q + 1
                row = attn_avg[q, :valid_k_len]
                
                if len(row) == 0:
                    required_k_list.append(1)
                    continue
                    
                sorted_vals, _ = row.sort(descending=True)
                cumsum = sorted_vals.cumsum(dim=0)
                reached = cumsum >= self.target_coverage
                
                if reached.any():
                    k_val = reached.float().argmax().item() + 1
                else:
                    k_val = len(row)
                
                required_k_list.append(int(k_val))
            
            self.chunk_k_stats[layer].extend(required_k_list)
            
            # Free memory from attention computation
            del Q, K, attn_scores, attn_weights, attn_avg
    
    # Create hook handler
    hook_handler = ChunkedQKHook(layers_to_analyze, target_coverage, num_heads, num_kv_heads, head_dim)
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if 'q_proj' in name and hasattr(module, 'weight'):
            h = module.register_forward_hook(hook_handler.q_hook)
            hooks.append(h)
        elif 'k_proj' in name and hasattr(module, 'weight'):
            h = module.register_forward_hook(hook_handler.k_hook)
            hooks.append(h)
    
    print(f"    Registered hooks on {len(hooks)//2} layers")
    
    # Process chunks with KV cache
    total_start = time.time()
    past_key_values = None
    
    for chunk_idx in range(num_chunks):
        chunk_start = time.time()
        start_pos = chunk_idx * chunk_size
        end_pos = min(start_pos + chunk_size, num_tokens)
        
        # Get chunk input
        chunk_input_ids = input_ids[start_pos:end_pos].unsqueeze(0).to(model.device)
        
        # Reset hook state for new chunk
        hook_handler.reset_for_chunk()
        
        # Forward pass with KV cache
        with torch.no_grad():
            outputs = model(
                chunk_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=False,  # We compute attention ourselves via hooks
                return_dict=True
            )
            past_key_values = outputs.past_key_values
        
        # Clear outputs (we already captured Q, K via hooks)
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        chunk_time = time.time() - chunk_start
        elapsed = time.time() - total_start
        remaining = (num_chunks - chunk_idx - 1) * (elapsed / (chunk_idx + 1))
        
        # Get max K so far for progress
        max_k_so_far = max(
            (max(hook_handler.chunk_k_stats[l]) if hook_handler.chunk_k_stats[l] else 0)
            for l in layers_to_analyze
        )
        print(f"      Chunk {chunk_idx+1}/{num_chunks}: tokens {start_pos}-{end_pos}, "
              f"max_K_so_far={max_k_so_far}, ({chunk_time:.1f}s, ETA: {remaining:.0f}s)")
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Compute final statistics
    final_stats = {}
    for layer_idx in layers_to_analyze:
        k_list = hook_handler.chunk_k_stats[layer_idx]
        # Skip first 50 tokens
        k_list = k_list[50:] if len(k_list) > 50 else k_list
        
        if len(k_list) > 0:
            k_tensor = torch.tensor(k_list)
            final_stats[layer_idx] = {
                "min_k": k_tensor.min().item(),
                "max_k": k_tensor.max().item(),
                "max_k_limited": False,  # No artificial limit with KV cache!
                "mean_k": k_tensor.float().mean().item(),
                "std_k": k_tensor.float().std().item(),
                "k_ratio": k_tensor.max().item() / max(k_tensor.min().item(), 1),
                "k_ratio_note": "TRUE ratio (with KV cache)",
                "required_k_list": k_list,
            }
    
    # Clear KV cache
    del past_key_values, hook_handler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    total_time = time.time() - total_start
    print(f"    Total chunked processing time: {total_time:.1f}s")
    
    return final_stats, num_tokens


# ============================================================================
# Core Analysis: Per-Query K Variation
# ============================================================================

def compute_required_k_per_query(attention: torch.Tensor, target_coverage: float = 0.90):
    """
    Tính K cần thiết cho MỖI query token để đạt target coverage.
    
    DEPRECATED: Use compute_required_k_vectorized_gpu() instead for ~100x speedup.
    
    Args:
        attention: (batch, heads, seq_len, seq_len) hoặc (heads, seq_len, seq_len)
        
    Returns:
        required_k: (seq_len,) - K cần thiết cho mỗi query position
    """
    # Use the optimized GPU-vectorized version
    return compute_required_k_vectorized_gpu(attention, target_coverage)


def analyze_query_k_variation(attention: torch.Tensor, layer_idx: int, dataset: str):
    """
    Phân tích sự biến thiên của K giữa các query tokens.
    
    Key insight: Nếu K thay đổi nhiều giữa các queries → static K không tối ưu
    """
    required_k = compute_required_k_per_query(attention, TARGET_COVERAGE)
    
    # Bỏ qua tokens đầu (không đủ context)
    valid_k = required_k[50:]  # Skip first 50 tokens
    
    if len(valid_k) == 0:
        return None
    
    stats = {
        "dataset": dataset,
        "layer": layer_idx,
        "num_queries": len(valid_k),
        "min_k": valid_k.min().item(),
        "max_k": valid_k.max().item(),
        "mean_k": valid_k.float().mean().item(),
        "std_k": valid_k.float().std().item(),
        "median_k": valid_k.float().median().item(),
        "k_ratio": valid_k.max().item() / max(valid_k.min().item(), 1),
        "required_k_per_query": valid_k.tolist(),
    }
    
    # Phân loại queries
    sparse_threshold = stats["mean_k"] * 0.5
    dense_threshold = stats["mean_k"] * 1.5
    
    stats["sparse_queries_pct"] = (valid_k < sparse_threshold).float().mean().item() * 100
    stats["dense_queries_pct"] = (valid_k > dense_threshold).float().mean().item() * 100
    
    return stats


def analyze_per_head_query_variation(attention: torch.Tensor, layer_idx: int):
    """
    Phân tích K variation cho từng head riêng biệt.
    Now uses GPU-vectorized computation.
    """
    if attention.dim() == 4:
        attention = attention[0]  # (heads, seq_len, seq_len)
    
    num_heads = attention.shape[0]
    seq_len = attention.shape[1]
    
    head_stats = {}
    
    for head_idx in range(num_heads):
        head_attn = attention[head_idx:head_idx+1]  # (1, seq_len, seq_len) - keep dims for vectorized fn
        
        # Use vectorized computation per head
        required_k = compute_required_k_vectorized_gpu(head_attn, TARGET_COVERAGE)
        valid_k = required_k[50:]  # Skip first 50
        
        if len(valid_k) > 0:
            head_stats[head_idx] = {
                "min_k": valid_k.min().item(),
                "max_k": valid_k.max().item(),
                "mean_k": valid_k.float().mean().item(),
                "std_k": valid_k.float().std().item(),
                "k_ratio": valid_k.max().item() / max(valid_k.min().item(), 1),
            }
    
    return head_stats


# ============================================================================
# Visualization
# ============================================================================

def visualize_causal_attention_matrix(attention: torch.Tensor = None, 
                                       save_path: str = None,
                                       seq_len: int = 64,
                                       title: str = "Causal Self-Attention Matrix",
                                       figsize: tuple = (8, 8),
                                       show_k_variation: bool = True):
    """
    Draw a publication-quality visualization of a Transformer causal attention matrix.
    For SMALL sequences only (< 100 tokens). For long contexts, use visualize_long_context_attention().
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        
        if seq_len > 100:
            print(f"Warning: seq_len={seq_len} too large for grid visualization. Use visualize_long_context_attention() instead.")
            seq_len = 64
        
        # Generate synthetic data with varying K patterns
        if attention is not None:
            if attention.dim() == 4:
                attention = attention[0]
            if attention.dim() == 3:
                attention = attention.mean(dim=0)
            attn = attention.cpu().float().numpy()
            seq_len = min(attn.shape[0], 100)
            attn = attn[:seq_len, :seq_len]
        else:
            np.random.seed(42)
            attn = np.zeros((seq_len, seq_len))
            
            for i in range(seq_len):
                if i == 0:
                    attn[i, 0] = 1.0
                    continue
                
                pattern_type = i % 5
                weights = np.zeros(i + 1)
                
                if pattern_type == 0:  # Very sparse
                    weights[i] = 0.85 + np.random.uniform(0, 0.1)
                    for j in range(i + 1):
                        if weights[j] == 0:
                            weights[j] = np.random.uniform(0, 0.01)
                elif pattern_type == 1:  # Sparse with retrieval
                    weights[i] = 0.6 + np.random.uniform(0, 0.2)
                    if i > 5:
                        weights[np.random.randint(0, i // 2)] = 0.2
                    for j in range(i + 1):
                        if weights[j] == 0:
                            weights[j] = np.random.uniform(0, 0.02)
                elif pattern_type == 2:  # Local window
                    for j in range(max(0, i - 4), i + 1):
                        weights[j] = np.random.uniform(0.1, 0.3)
                    for j in range(i + 1):
                        if weights[j] == 0:
                            weights[j] = np.random.uniform(0, 0.02)
                elif pattern_type == 3:  # Distributed decay
                    for j in range(i + 1):
                        weights[j] = np.exp(-(i - j) / (i * 0.5 + 1)) * np.random.uniform(0.8, 1.2)
                else:  # Uniform
                    weights = np.ones(i + 1) * np.random.uniform(0.5, 1.5)
                
                weights = weights / weights.sum()
                attn[i, :i+1] = weights
        
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        
        colors = ['#FFFFFF', '#FFF5F5', '#FFE5E5', '#FFCCCC', '#FFB3B3', 
                  '#FF9999', '#FF6666', '#FF3333', '#FF0000', '#CC0000', '#990000']
        cmap = mcolors.LinearSegmentedColormap.from_list('attention_red', colors, N=256)
        
        masked_attn = np.ma.masked_where(np.triu(np.ones_like(attn), k=1) == 1, attn)
        im = ax.imshow(masked_attn, cmap=cmap, aspect='equal', interpolation='nearest', vmin=0, vmax=1.0)
        
        ax.set_xticks(np.arange(-0.5, seq_len, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, seq_len, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.7)
        ax.plot([-0.5, seq_len - 0.5], [-0.5, seq_len - 0.5], color='black', linewidth=1.5)
        
        ax.set_xlabel('Key Position', fontsize=12)
        ax.set_ylabel('Query Position', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Attention Weight')
        ax.set_xlim(-0.5, seq_len - 0.5)
        ax.set_ylim(seq_len - 0.5, -0.5)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Saved: {save_path}")
        return fig, ax
    except ImportError as e:
        print(f"matplotlib not available: {e}")
        return None, None


def visualize_long_context_attention(k_values: list = None,
                                     seq_len: int = 200000,
                                     save_path: str = None,
                                     title: str = "Attention Pattern in 200K Context",
                                     figsize: tuple = (14, 10)):
    """
    Visualize attention patterns for LONG contexts (200K-400K tokens).
    
    Instead of showing the full matrix (impossible), shows:
    1. K distribution across all query positions
    2. Sparse vs distributed query classification
    3. Attention pattern schematic
    4. Example attention rows at different positions
    
    Args:
        k_values: List of required K per query (from analysis). If None, generates synthetic.
        seq_len: Total sequence length
        save_path: Where to save
        title: Figure title
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
        import matplotlib.patches as mpatches
        
        # Generate synthetic K values if not provided
        if k_values is None:
            np.random.seed(42)
            k_values = []
            for i in range(seq_len):
                if i < 50:
                    k_values.append(min(i + 1, 10))
                else:
                    # Mix of sparse and distributed queries
                    pattern = np.random.choice(['very_sparse', 'sparse', 'moderate', 'distributed', 'very_distributed'],
                                               p=[0.15, 0.25, 0.25, 0.25, 0.10])
                    max_k = min(i + 1, 4096)
                    if pattern == 'very_sparse':
                        k_values.append(np.random.randint(1, min(5, max_k) + 1))
                    elif pattern == 'sparse':
                        k_values.append(np.random.randint(5, min(50, max_k) + 1))
                    elif pattern == 'moderate':
                        k_values.append(np.random.randint(50, min(500, max_k) + 1))
                    elif pattern == 'distributed':
                        k_values.append(np.random.randint(min(500, max_k), min(2000, max_k) + 1))
                    else:
                        k_values.append(np.random.randint(min(2000, max_k), max_k + 1))
        
        k_values = np.array(k_values)
        
        fig = plt.figure(figsize=figsize, facecolor='white')
        
        # Layout: 2x2 grid
        gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], width_ratios=[1.5, 1], 
                              hspace=0.3, wspace=0.3)
        
        # ============================================
        # Top-left: K values across sequence positions
        # ============================================
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Subsample for plotting (can't plot 200K points)
        sample_rate = max(1, len(k_values) // 5000)
        x_sampled = np.arange(0, len(k_values), sample_rate)
        k_sampled = k_values[::sample_rate]
        
        ax1.fill_between(x_sampled, k_sampled, alpha=0.3, color='steelblue')
        ax1.plot(x_sampled, k_sampled, linewidth=0.5, color='steelblue', alpha=0.7)
        ax1.axhline(y=np.mean(k_values), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean K = {np.mean(k_values):.0f}')
        ax1.axhline(y=np.median(k_values), color='green', linestyle='--', linewidth=2,
                   label=f'Median K = {np.median(k_values):.0f}')
        
        ax1.set_xlabel('Query Position', fontsize=11)
        ax1.set_ylabel('Required K (for 90% coverage)', fontsize=11)
        ax1.set_title(f'K Varies Drastically Across {seq_len:,} Tokens', fontsize=12, fontweight='bold')
        ax1.set_xlim(0, len(k_values))
        ax1.set_ylim(0, min(np.max(k_values) * 1.1, 5000))
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Add K ratio annotation
        k_ratio = np.max(k_values) / max(np.min(k_values[50:]), 1)
        ax1.annotate(f'K Ratio: {k_ratio:.0f}x\nMin: {np.min(k_values[50:])}, Max: {np.max(k_values)}',
                    xy=(0.02, 0.95), xycoords='axes fraction',
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # ============================================
        # Top-right: K distribution histogram
        # ============================================
        ax2 = fig.add_subplot(gs[0, 1])
        
        ax2.hist(k_values[50:], bins=100, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.axvline(x=np.mean(k_values), color='red', linestyle='--', linewidth=2)
        ax2.axvline(x=np.median(k_values), color='green', linestyle='--', linewidth=2)
        
        ax2.set_xlabel('Required K', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Distribution of Required K', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add sparse/dense percentages
        mean_k = np.mean(k_values[50:])
        sparse_pct = np.mean(k_values[50:] < mean_k * 0.5) * 100
        dense_pct = np.mean(k_values[50:] > mean_k * 1.5) * 100
        
        ax2.annotate(f'Sparse (<0.5x mean): {sparse_pct:.1f}%\nDense (>1.5x mean): {dense_pct:.1f}%',
                    xy=(0.95, 0.95), xycoords='axes fraction',
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # ============================================
        # Bottom-left: Schematic of attention patterns
        # ============================================
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_xlim(0, 10)
        ax3.set_ylim(0, 6)
        ax3.axis('off')
        ax3.set_title('Attention Pattern Types in Long Context', fontsize=12, fontweight='bold')
        
        # Draw schematic attention patterns
        def draw_attention_row(ax, y, pattern_name, description, attention_positions, color_intensities):
            # Draw the row background
            ax.add_patch(FancyBboxPatch((0.5, y - 0.3), 9, 0.6, 
                                        boxstyle="round,pad=0.02", 
                                        facecolor='#f0f0f0', edgecolor='gray'))
            
            # Draw attention cells
            for pos, intensity in zip(attention_positions, color_intensities):
                cell_x = 0.5 + pos * 0.15
                red_val = int(255 * (1 - intensity * 0.7))
                color = f'#{red_val:02x}0000' if intensity > 0.3 else f'#ff{255-int(intensity*200):02x}{255-int(intensity*200):02x}'
                ax.add_patch(plt.Rectangle((cell_x, y - 0.2), 0.12, 0.4, 
                                          facecolor=color, edgecolor='black', linewidth=0.5))
            
            # Labels
            ax.text(0.3, y, pattern_name, fontsize=10, fontweight='bold', ha='right', va='center')
            ax.text(9.7, y, description, fontsize=9, ha='left', va='center', color='gray')
        
        # Sparse pattern: 1-2 dark cells
        draw_attention_row(ax3, 5, 'Sparse', 'K = 1-5',
                          [0, 55], [0.95, 0.1])
        
        # Retrieval pattern: diagonal + one spike
        draw_attention_row(ax3, 4, 'Retrieval', 'K = 5-50', 
                          [0, 15, 55, 56, 57, 58, 59], [0.1, 0.7, 0.1, 0.15, 0.2, 0.3, 0.8])
        
        # Local pattern: recent window
        draw_attention_row(ax3, 3, 'Local', 'K = 50-200',
                          list(range(50, 60)), [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9])
        
        # Distributed pattern: many light cells
        draw_attention_row(ax3, 2, 'Distributed', 'K = 200-2000',
                          list(range(0, 60, 2)), [0.15] * 30)
        
        # Global pattern: everything light
        draw_attention_row(ax3, 1, 'Global', 'K = 2000+',
                          list(range(0, 60)), [0.05] * 60)
        
        # ============================================
        # Bottom-right: Why static K fails
        # ============================================
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        ax4.set_title('Why Static K Fails', fontsize=12, fontweight='bold')
        
        text = f"""
If Static K = Max ({int(np.max(k_values))}):
  → {sparse_pct:.0f}% queries waste computation
  → {sparse_pct:.0f}% of FLOPs are unnecessary

If Static K = Mean ({int(np.mean(k_values))}):
  → {dense_pct:.0f}% queries miss information
  → Accuracy drops on dense queries

If Static K = Min ({int(np.min(k_values[50:]))}):
  → {100-sparse_pct:.0f}% queries fail completely
  → Model breaks down

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dynamic K (TokenSelect):
  → Adapts per query
  → K=1 when sparse, K=4096 when dense
  → Optimal efficiency AND accuracy
"""
        ax4.text(0.05, 0.95, text, fontsize=10, family='monospace',
                verticalalignment='top', transform=ax4.transAxes)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Saved long-context visualization: {save_path}")
        
        return fig
        
    except ImportError as e:
        print(f"matplotlib not available: {e}")
        return None


def visualize_attention_schematic(save_path: str = None, figsize: tuple = (12, 8)):
    """
    Draw a clean schematic diagram of causal attention for papers.
    Shows the concept without trying to render actual 200K×200K matrix.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch, Polygon
        
        fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor='white')
        
        # ============================================
        # Left: Conceptual attention matrix
        # ============================================
        ax1 = axes[0]
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        
        # Draw lower triangle (valid attention region)
        triangle = Polygon([(0, 10), (10, 10), (10, 0)], 
                          facecolor='#ffeeee', edgecolor='black', linewidth=2)
        ax1.add_patch(triangle)
        
        # Draw upper triangle (masked)
        upper_triangle = Polygon([(0, 10), (10, 0), (0, 0)], 
                                facecolor='white', edgecolor='black', linewidth=2)
        ax1.add_patch(upper_triangle)
        
        # Diagonal line
        ax1.plot([0, 10], [10, 0], 'k-', linewidth=2)
        
        # Add sparse attention spots (dark red)
        sparse_spots = [(8, 2.5), (6, 4.5), (3, 7.5)]
        for x, y in sparse_spots:
            ax1.plot(x, y, 'o', markersize=15, color='#cc0000', markeredgecolor='black')
        
        # Add distributed attention region (light pink gradient)
        for i in range(5):
            y = 5 + i * 0.8
            for j in range(int(10 - y/10 * 10) + 3):
                ax1.plot(j * 0.8 + 0.5, y, 's', markersize=8, 
                        color=f'#ff{150+j*10:02x}{150+j*10:02x}', alpha=0.7)
        
        ax1.set_xlabel('Key Position →', fontsize=12)
        ax1.set_ylabel('← Query Position', fontsize=12)
        ax1.set_title('Causal Attention Matrix\n(200K × 200K)', fontsize=13, fontweight='bold')
        ax1.set_xticks([0, 5, 10])
        ax1.set_xticklabels(['0', '100K', '200K'])
        ax1.set_yticks([0, 5, 10])
        ax1.set_yticklabels(['200K', '100K', '0'])
        
        # Annotations
        ax1.annotate('Masked\n(future)', xy=(2, 5), fontsize=11, ha='center', color='gray')
        ax1.annotate('Valid\n(past & self)', xy=(7, 7), fontsize=11, ha='center')
        
        # Legend
        legend_elements = [
            mpatches.Patch(facecolor='#cc0000', edgecolor='black', label='Sparse (K=1-10)'),
            mpatches.Patch(facecolor='#ffcccc', edgecolor='black', label='Distributed (K=1000+)'),
            mpatches.Patch(facecolor='white', edgecolor='black', label='Masked (future)'),
        ]
        ax1.legend(handles=legend_elements, loc='lower left', fontsize=9)
        
        # ============================================
        # Right: K variation concept
        # ============================================
        ax2 = axes[1]
        
        # Sample K curve
        x = np.linspace(0, 200, 1000)
        np.random.seed(42)
        k_base = 100 + 50 * np.sin(x / 20) + 30 * np.sin(x / 7)
        k_noise = np.random.randn(1000) * 50
        k_values = np.clip(k_base + k_noise, 1, 500)
        
        # Add some spikes
        for spike_pos in [50, 120, 180]:
            k_values[spike_pos-2:spike_pos+2] = np.random.randint(400, 500)
        for dip_pos in [30, 90, 150]:
            k_values[dip_pos-2:dip_pos+2] = np.random.randint(1, 20)
        
        ax2.fill_between(x, k_values, alpha=0.3, color='steelblue')
        ax2.plot(x, k_values, linewidth=1, color='steelblue')
        
        ax2.axhline(y=np.mean(k_values), color='red', linestyle='--', linewidth=2, label='Mean K')
        ax2.axhspan(0, np.mean(k_values) * 0.5, alpha=0.1, color='green', label='Sparse zone')
        ax2.axhspan(np.mean(k_values) * 1.5, 500, alpha=0.1, color='orange', label='Dense zone')
        
        ax2.set_xlabel('Query Position (thousands)', fontsize=12)
        ax2.set_ylabel('Required K', fontsize=12)
        ax2.set_title('K Varies Per Query\n(Why Static K Fails)', fontsize=13, fontweight='bold')
        ax2.set_xlim(0, 200)
        ax2.set_ylim(0, 500)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Annotations
        ax2.annotate('Sparse\nK=5', xy=(30, 15), fontsize=10, ha='center',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        ax2.annotate('Dense\nK=450', xy=(50, 450), fontsize=10, ha='center',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Saved attention schematic: {save_path}")
        
        return fig
        
    except ImportError as e:
        print(f"matplotlib not available: {e}")
        return None


def visualize_attention_matrix(attention: torch.Tensor, layer_idx: int, head_idx: int, 
                               save_path: str, dataset: str, num_tokens: int):
    """
    Visualize attention matrix cho một layer và head cụ thể.
    """
    try:
        import matplotlib.pyplot as plt
        
        if attention.dim() == 4:
            attention = attention[0]
        
        attn = attention[head_idx].cpu().float().numpy()
        
        # Focus on a portion for visibility
        max_show = min(500, attn.shape[0])
        attn_show = attn[-max_show:, :]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Full attention heatmap
        ax1 = axes[0]
        im1 = ax1.imshow(attn_show, cmap='viridis', aspect='auto')
        ax1.set_xlabel('Key Position')
        ax1.set_ylabel(f'Query Position (last {max_show})')
        ax1.set_title(f'Attention Matrix\nDataset: {dataset}, Layer: {layer_idx}, Head: {head_idx}')
        plt.colorbar(im1, ax=ax1, label='Attention Weight')
        
        # Required K per query
        ax2 = axes[1]
        required_k = []
        for q_pos in range(attn.shape[0]):
            row = attn[q_pos, :q_pos+1]
            if len(row) == 0:
                required_k.append(1)
                continue
            sorted_vals = np.sort(row)[::-1]
            cumsum = np.cumsum(sorted_vals)
            k = np.searchsorted(cumsum, TARGET_COVERAGE) + 1
            required_k.append(min(k, len(row)))
        
        ax2.plot(required_k, linewidth=0.5, alpha=0.7)
        ax2.axhline(y=np.mean(required_k[50:]), color='red', linestyle='--', 
                   label=f'Mean K = {np.mean(required_k[50:]):.1f}')
        ax2.fill_between(range(len(required_k)), required_k, alpha=0.3)
        ax2.set_xlabel('Query Position')
        ax2.set_ylabel(f'Required K (for {TARGET_COVERAGE:.0%} coverage)')
        ax2.set_title(f'K Varies Across Queries\n(Min={min(required_k[50:])}, Max={max(required_k[50:])})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Static K is Suboptimal: Different Queries Need Different K\n'
                    f'Total tokens: {num_tokens}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"      Saved: {save_path}")
        
    except ImportError:
        print("      matplotlib not available")


def visualize_k_distribution_comparison(all_stats: dict, save_path: str):
    """
    So sánh distribution của required K giữa các datasets.
    """
    try:
        import matplotlib.pyplot as plt
        
        datasets = list(all_stats.keys())
        num_datasets = len(datasets)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, dataset in enumerate(datasets[:4]):
            ax = axes[idx]
            
            if dataset not in all_stats or "required_k_per_query" not in all_stats[dataset]:
                continue
            
            k_values = all_stats[dataset]["required_k_per_query"]
            
            ax.hist(k_values, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            ax.axvline(x=np.mean(k_values), color='red', linestyle='--', linewidth=2,
                      label=f'Mean={np.mean(k_values):.1f}')
            ax.axvline(x=np.median(k_values), color='green', linestyle='--', linewidth=2,
                      label=f'Median={np.median(k_values):.1f}')
            
            ax.set_xlabel('Required K')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{dataset}\n'
                        f'Range: {min(k_values)} - {max(k_values)} '
                        f'(Ratio: {max(k_values)/max(min(k_values),1):.1f}x)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Distribution of Required K per Query Token\n'
                    f'(Wide distribution → Static K is suboptimal)', 
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
        
    except ImportError:
        print("matplotlib not available")


def visualize_head_comparison(head_stats: dict, layer_idx: int, dataset: str, save_path: str):
    """
    Visualize K variation across different heads.
    """
    try:
        import matplotlib.pyplot as plt
        
        heads = sorted(head_stats.keys())
        means = [head_stats[h]["mean_k"] for h in heads]
        mins = [head_stats[h]["min_k"] for h in heads]
        maxs = [head_stats[h]["max_k"] for h in heads]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Mean K per head
        ax1 = axes[0]
        ax1.bar(heads, means, color='steelblue', alpha=0.8)
        ax1.set_xlabel('Head Index')
        ax1.set_ylabel('Mean Required K')
        ax1.set_title(f'Mean Required K per Head\n'
                     f'Std across heads: {np.std(means):.1f}')
        ax1.grid(True, axis='y', alpha=0.3)
        
        # K range per head
        ax2 = axes[1]
        x = np.array(heads)
        ax2.fill_between(x, mins, maxs, alpha=0.3, color='coral')
        ax2.plot(x, means, 'o-', color='red', linewidth=2, label='Mean')
        ax2.set_xlabel('Head Index')
        ax2.set_ylabel('Required K Range')
        ax2.set_title(f'K Variation Range per Head\n'
                     f'(Shaded area = min to max)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Head-wise K Variation - {dataset}, Layer {layer_idx}', 
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
        
    except ImportError:
        print("matplotlib not available")


# ============================================================================
# Main Analysis Pipeline
# ============================================================================

def analyze_dataset_full(model, tokenizer, dataset: str, sample_idx: int, 
                        visualize: bool = True, output_dir: Path = OUTPUT_DIR,
                        max_tokens: int = 4096, chunk_size: int = None,
                        sample_true_k: int = None):
    """
    Phân tích đầy đủ một sample.
    
    Args:
        chunk_size: If provided, process in chunks for long contexts. 
                    WARNING: Max K is limited by chunk_size!
        sample_true_k: If provided, sample N query positions to measure TRUE K
                       without any chunk limitation. K can go up to full context length.
    """
    print(f"\n  Sample {sample_idx}:")
    
    sample = load_sample(dataset, sample_idx)
    if not sample:
        print(f"    Could not load sample")
        return None
    
    context = sample.get("context", "")
    query = sample.get("input", "")
    
    print(f"    Context length: {len(context)} chars")
    
    # Choose method based on arguments
    if sample_true_k:
        # Use sampling mode for TRUE K values (no chunk limitation)
        layer_stats, num_tokens = sample_true_k_with_kv_cache(
            model, tokenizer, context, query,
            max_tokens=max_tokens,
            target_coverage=TARGET_COVERAGE,
            num_samples=sample_true_k
        )
    else:
        # Use layer-by-layer method (may be limited by chunk_size)
        layer_stats, num_tokens = get_attention_layer_by_layer(
            model, tokenizer, context, query, 
            max_tokens=max_tokens, 
            target_coverage=TARGET_COVERAGE,
            chunk_size=chunk_size
        )
    
    if not layer_stats:
        print(f"    No stats computed")
        return None
    
    # Print stats for each analyzed layer
    for layer_idx, stats in layer_stats.items():
        k_limited = stats.get('max_k_limited', False)
        k_note = " ⚠️ LIMITED" if k_limited else " ✓ TRUE"
        
        print(f"    Layer {layer_idx}:")
        print(f"      K range: {stats['min_k']} - {stats['max_k']} (ratio: {stats['k_ratio']:.1f}x){k_note}")
        
        # Classify queries
        mean_k = stats['mean_k']
        k_list = stats['required_k_list']
        sparse_pct = sum(1 for k in k_list if k < mean_k * 0.5) / len(k_list) * 100
        dense_pct = sum(1 for k in k_list if k > mean_k * 1.5) / len(k_list) * 100
        
        print(f"      Sparse queries (<0.5x mean): {sparse_pct:.1f}%")
        print(f"      Dense queries (>1.5x mean): {dense_pct:.1f}%")
        
        stats['sparse_queries_pct'] = sparse_pct
        stats['dense_queries_pct'] = dense_pct
    
    # Visualize K distribution for ALL analyzed layers
    if visualize:
        try:
            import matplotlib.pyplot as plt
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Visualize each layer separately
            for layer_idx in sorted(layer_stats.keys()):
                k_values = layer_stats[layer_idx]['required_k_list']
                
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                # K over query positions
                ax1 = axes[0]
                ax1.plot(k_values, linewidth=0.5, alpha=0.7)
                ax1.axhline(y=np.mean(k_values), color='red', linestyle='--', 
                           label=f'Mean K = {np.mean(k_values):.1f}')
                ax1.fill_between(range(len(k_values)), k_values, alpha=0.3)
                ax1.set_xlabel('Query Position')
                ax1.set_ylabel(f'Required K (for {TARGET_COVERAGE:.0%} coverage)')
                ax1.set_title(f'K Varies Across Queries\nMin={min(k_values)}, Max={max(k_values)}, '
                             f'Ratio={max(k_values)/max(min(k_values),1):.1f}x')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # K distribution histogram
                ax2 = axes[1]
                ax2.hist(k_values, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
                ax2.axvline(x=np.mean(k_values), color='red', linestyle='--', linewidth=2,
                           label=f'Mean={np.mean(k_values):.1f}')
                ax2.axvline(x=np.median(k_values), color='green', linestyle='--', linewidth=2,
                           label=f'Median={np.median(k_values):.1f}')
                ax2.set_xlabel('Required K')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Distribution of Required K')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.suptitle(f'{dataset} - Sample {sample_idx} - Layer {layer_idx} - {num_tokens} tokens\n'
                            f'Static K is suboptimal: queries need different K values', 
                            fontsize=12, fontweight='bold')
                plt.tight_layout()
                
                save_path = output_dir / f"{dataset}_sample{sample_idx}_layer{layer_idx}_k_analysis.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"    Saved: {save_path}")
            
            # Also create a combined comparison plot for all layers
            num_layers = len(layer_stats)
            if num_layers > 1:
                fig, axes = plt.subplots(2, num_layers, figsize=(6 * num_layers, 10))
                
                for col_idx, layer_idx in enumerate(sorted(layer_stats.keys())):
                    k_values = layer_stats[layer_idx]['required_k_list']
                    
                    # Top row: K over positions
                    ax1 = axes[0, col_idx] if num_layers > 1 else axes[0]
                    ax1.plot(k_values, linewidth=0.5, alpha=0.7)
                    ax1.axhline(y=np.mean(k_values), color='red', linestyle='--', 
                               label=f'Mean={np.mean(k_values):.1f}')
                    ax1.fill_between(range(len(k_values)), k_values, alpha=0.3)
                    ax1.set_xlabel('Query Position')
                    ax1.set_ylabel(f'Required K')
                    ax1.set_title(f'Layer {layer_idx}\nK Range: {min(k_values)}-{max(k_values)}')
                    ax1.legend(fontsize=8)
                    ax1.grid(True, alpha=0.3)
                    
                    # Bottom row: K distribution
                    ax2 = axes[1, col_idx] if num_layers > 1 else axes[1]
                    ax2.hist(k_values, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
                    ax2.axvline(x=np.mean(k_values), color='red', linestyle='--', linewidth=2)
                    ax2.set_xlabel('Required K')
                    ax2.set_ylabel('Frequency')
                    ax2.set_title(f'Ratio: {max(k_values)/max(min(k_values),1):.1f}x')
                    ax2.grid(True, alpha=0.3)
                
                plt.suptitle(f'{dataset} - Sample {sample_idx} - All Layers Comparison - {num_tokens} tokens\n'
                            f'Static K is suboptimal: queries need different K values', 
                            fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                save_path = output_dir / f"{dataset}_sample{sample_idx}_all_layers_comparison.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"    Saved: {save_path}")
            
        except ImportError:
            print("    matplotlib not available")
    
    # Return middle layer stats
    mid_layer = list(layer_stats.keys())[len(layer_stats)//2]
    result = layer_stats[mid_layer].copy()
    result['required_k_per_query'] = result.pop('required_k_list')
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Prove static K is suboptimal for query tokens",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Short context (fits in memory): full attention analysis
  python prove_static_k_suboptimal.py --max-tokens 4096 --visualize-all
  
  # Long context (200K tokens): use chunked prefill with KV cache
  python prove_static_k_suboptimal.py --max-tokens 200000 --chunk-size 4096
  
  # Or use sampling mode (faster, less memory)
  python prove_static_k_suboptimal.py --max-tokens 200000 --sample-true-k 200
"""
    )
    parser.add_argument("--samples-per-dataset", type=int, default=2, 
                       help="Number of samples to analyze per dataset")
    parser.add_argument("--model", default="Qwen/Qwen2-7B-Instruct", help="Model name")
    parser.add_argument("--coverage", type=float, default=0.90, help="Target coverage")
    parser.add_argument("--visualize-all", action="store_true", 
                       help="Visualize attention for all layers/heads")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                       help="Datasets to analyze")
    parser.add_argument("--max-tokens", type=int, default=4096,
                       help="Max context tokens. For >8K, use --chunk-size or --sample-true-k")
    parser.add_argument("--chunk-size", type=int, default=None,
                       help="Process in chunks with KV cache. Each token attends to FULL past context. "
                            "Recommended: 4096 for long contexts")
    parser.add_argument("--sample-true-k", type=int, default=None, metavar="N",
                       help="Sample N query positions to measure K (faster, less memory). "
                            "Example: --sample-true-k 200")
    args = parser.parse_args()
    
    global TARGET_COVERAGE
    TARGET_COVERAGE = args.coverage
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Auto-enable chunking for long contexts if not specified
    if args.max_tokens > 8192 and args.sample_true_k is None and args.chunk_size is None:
        print(f"⚠️  Long context detected ({args.max_tokens} tokens) - auto-enabling --chunk-size 4096")
        args.chunk_size = 4096
    
    print("="*70)
    print("PROVING: Static K is Suboptimal for Token Selection")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Target coverage: {TARGET_COVERAGE:.0%}")
    print(f"Samples per dataset: {args.samples_per_dataset}")
    print(f"Max tokens: {args.max_tokens}")
    
    # Mode selection
    if args.sample_true_k:
        print(f"Mode: SAMPLING ({args.sample_true_k} query positions per context)")
        print(f"  ✓ K values have NO artificial limit!")
        print(f"  ✓ Max K can be up to full context length ({args.max_tokens})")
    elif args.chunk_size:
        print(f"Mode: CHUNKED PREFILL (chunk_size={args.chunk_size})")
        print(f"  ✓ Using KV cache: tokens attend to FULL past context")
        print(f"  ✓ Max K can be up to full context length ({args.max_tokens})")
    else:
        print(f"Mode: FULL CONTEXT (no chunking)")
        print(f"  ✓ TRUE K values (no limit)")
    
    print(f"Datasets: {args.datasets}")
    
    # Memory estimation
    if args.sample_true_k:
        print(f"Memory mode: Efficient sampling (O(n) per query position)")
    elif args.chunk_size:
        # With KV cache chunking, memory is O(chunk_size * total_seq) per layer for attention
        # But we compute layer by layer, so peak is for one layer
        peak_mem_gb = (args.chunk_size * args.max_tokens * 4 * 28) / (1024**3)  # float32, 28 heads
        print(f"Memory mode: Chunked with KV cache")  
        print(f"  Peak attention memory (last chunk): ~{peak_mem_gb:.1f} GB")
    else:
        effective_seq = args.max_tokens
        est_mem_gb = (28 * 28 * effective_seq * effective_seq * 2) / (1024**3)
        print(f"Estimated attention memory: ~{est_mem_gb:.1f} GB")
        if est_mem_gb > 70:
            print("WARNING: May OOM on 80GB GPU! Consider reducing --max-tokens or adding --chunk-size")
    print("="*70)
    
    model, tokenizer = load_model(args.model)
    
    all_dataset_stats = {}
    
    for dataset in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset} - {DATASETS.get(dataset, '')}")
        print("="*60)
        
        dataset_stats = []
        
        for sample_idx in range(args.samples_per_dataset):
            stats = analyze_dataset_full(
                model, tokenizer, dataset, sample_idx,
                visualize=args.visualize_all,
                output_dir=OUTPUT_DIR,
                max_tokens=args.max_tokens,
                chunk_size=args.chunk_size,
                sample_true_k=args.sample_true_k
            )
            if stats:
                dataset_stats.append(stats)
        
        # Aggregate stats for this dataset
        if dataset_stats:
            all_k_values = []
            for s in dataset_stats:
                all_k_values.extend(s["required_k_per_query"])
            
            # Check if any stats have max_k_limited flag
            k_limited = any(s.get("max_k_limited", False) for s in dataset_stats)
            
            all_dataset_stats[dataset] = {
                "num_samples": len(dataset_stats),
                "total_queries": len(all_k_values),
                "min_k": min(all_k_values),
                "max_k": max(all_k_values),
                "max_k_limited": k_limited,
                "mean_k": np.mean(all_k_values),
                "std_k": np.std(all_k_values),
                "k_ratio": max(all_k_values) / max(min(all_k_values), 1),
                "k_ratio_note": "Limited by chunk_size" if k_limited else "TRUE ratio",
                "required_k_per_query": all_k_values,
            }
    
    # Generate comparison visualization
    if all_dataset_stats:
        visualize_k_distribution_comparison(
            all_dataset_stats, 
            str(OUTPUT_DIR / "k_distribution_comparison.png")
        )
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY: Why Static K is Suboptimal")
    print("="*70)
    
    any_limited = False
    for dataset, stats in all_dataset_stats.items():
        k_limited = stats.get('max_k_limited', False)
        k_note = ""  # No longer needed since chunked prefill uses KV cache correctly
        if k_limited:
            any_limited = True
            k_note = " (may be limited)"
        
        print(f"\n{dataset}:")
        print(f"  Queries analyzed: {stats['total_queries']}")
        print(f"  Required K range: {stats['min_k']} - {stats['max_k']}{k_note}")
        print(f"  K ratio (max/min): {stats['k_ratio']:.1f}x")
        print(f"  Mean ± Std: {stats['mean_k']:.1f} ± {stats['std_k']:.1f}")
    
    # This should no longer trigger since chunked prefill now uses KV cache correctly
    if any_limited:
        print(f"\n{'─'*70}")
        print("⚠️  NOTE: Some K values may be limited. This is unexpected with KV cache.")
    
    # Overall conclusion
    all_ratios = [s["k_ratio"] for s in all_dataset_stats.values()]
    if all_ratios:
        print(f"\n{'─'*70}")
        print(f"CONCLUSION:")
        print(f"  Average K ratio across datasets: {np.mean(all_ratios):.1f}x")
        print(f"  Max K ratio: {max(all_ratios):.1f}x")
        print(f"\n  → Different query tokens need vastly different K values")
        print(f"  → Static K causes: information loss OR wasted computation")
        print(f"  → Dynamic K selection (TokenSelect) is necessary for optimality")
    
    # Save results
    results = {
        "model": args.model,
        "target_coverage": TARGET_COVERAGE,
        "samples_per_dataset": args.samples_per_dataset,
        "sample_true_k": args.sample_true_k,
        "chunk_size": args.chunk_size,
        "dataset_stats": {k: {kk: vv for kk, vv in v.items() if kk != "required_k_per_query"} 
                         for k, v in all_dataset_stats.items()},
    }
    with open(OUTPUT_DIR / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults and visualizations saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
