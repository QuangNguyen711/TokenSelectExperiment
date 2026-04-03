# File: benchmark/serve.py

"""Launch the inference server."""

import argparse

from omegaconf import OmegaConf
from sglang.srt.server_args import ServerArgs


def patch_model(config):
    import sys
    sys.path.append(".")
    from patcher.token_retrieval import patch

    patch(
        rope_base=config.model.rope_base,
        rope_scale=config.model.rope_scale,
        rope_model="ROPE_LLAMA",
        max_n_tokens=config.model.max_n_tokens,
        n_init=config.model.n_init,
        n_local=config.model.n_local,
        top_k=config.model.top_k,
        adaptive_topk=getattr(config.model, 'adaptive_topk', False),
        attention_threshold=getattr(config.model, 'attention_threshold', 0.9),
        weighted_soft_vote=getattr(config.model, 'weighted_soft_vote', False),
        union_of_sets=getattr(config.model, 'union_of_sets', False),
        l2_norm_pooling=getattr(config.model, 'l2_norm_pooling', False),
        dynamic_capacity_union=getattr(config.model, 'dynamic_capacity_union', False),
        head_wise_adaptive=getattr(config.model, 'head_wise_adaptive', False),
    )


def patch_rope_only(config):
    """Apply only RoPE scaling without TokenSelect attention modifications."""
    import sys
    sys.path.append(".")
    from patcher.token_retrieval import patch_rope_only as rope_patch
    
    rope_patch(
        rope_base=config.model.rope_base,
        rope_scale=config.model.rope_scale,
        rope_model="ROPE_LLAMA",
        max_n_tokens=config.model.max_n_tokens,
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sgl-conf-file", type=str, default="")
    parser.add_argument("--use-spda", action="store_true", 
                        help="Use standard PyTorch SDPA instead of TokenSelect")
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)

    if args.sgl_conf_file:
        sgl_conf = OmegaConf.load(open(args.sgl_conf_file))
        
        if args.use_spda:
            # Apply only RoPE settings without TokenSelect
            patch_rope_only(sgl_conf)
            # Override context length from config
            if not server_args.context_length and hasattr(sgl_conf.model, 'max_n_tokens'):
                server_args.context_length = sgl_conf.model.max_n_tokens
            print(f"✓ Using standard PyTorch SDPA with extended RoPE (base={sgl_conf.model.rope_base}, context={server_args.context_length})")
        else:
            # Full TokenSelect with all patches
            patch_model(sgl_conf)
            print("✓ TokenSelect enabled with config:", args.sgl_conf_file)
    elif args.use_spda:
        print("✓ Using standard PyTorch SDPA (TokenSelect disabled)")
    else:
        print("⚠ No TokenSelect config provided, using default attention")

    from sglang.srt.server import launch_server

    launch_server(server_args)
