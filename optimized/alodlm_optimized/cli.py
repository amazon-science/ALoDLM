"""Generate with the bundled ALoDLM engine on a CUDA GPU.

Install: python -m pip install ./optimized
Run: alodlm-generate-optimized --model amazon/ALoDLM-8B --prompt "Hello."
"""

import argparse
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import types


def install_attention():
    """Use FlashAttention 2, including the compatible vLLM kernel bundle."""
    if importlib.util.find_spec("flash_attn") is not None:
        return
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    def compatible_varlen(*args, **kwargs):
        if kwargs.get("block_table") is not None:
            cumulative = kwargs.pop("cu_seqlens_k")
            kwargs["seqused_k"] = cumulative[1:] - cumulative[:-1]
        kwargs["fa_version"] = 2
        return flash_attn_varlen_func(*args, **kwargs)

    shim = types.ModuleType("flash_attn")
    shim.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    shim.flash_attn_varlen_func = compatible_varlen
    sys.modules["flash_attn"] = shim


def install_rope_compatibility():
    """Accept an explicit default-RoPE configuration from newer Transformers."""
    from wedlm.models import wedlm as model_module
    from wedlm.layers.rotary_embedding import get_rope

    def compatible_rope(head_size, rotary_dim, max_position, base, rope_scaling=None):
        if isinstance(rope_scaling, dict):
            if (rope_scaling.get("rope_type", "default") != "default"
                    or set(rope_scaling) - {"rope_type", "rope_theta"}):
                raise ValueError("This example supports the models' default RoPE only")
            base = float(rope_scaling.get("rope_theta", base))
            rope_scaling = None
        return get_rope(head_size, rotary_dim, max_position, base, rope_scaling)

    model_module.get_rope = compatible_rope


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="amazon/ALoDLM-8B")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--mode", choices=("entropy", "left1"), default="entropy")
    parser.add_argument("--q", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--nccl-port", type=int, default=0,
                        help="Local distributed port; 0 chooses an available port")
    parser.add_argument("--json", action="store_true", help="Print text, token IDs, and graph status as JSON")
    return parser


def read_model_metadata(model_path):
    """Reject incomplete weight directories before constructing a GPU engine."""
    for name in ("config.json", "alodlm_config.json", "exit_gate.pt"):
        if not (model_path / name).is_file():
            raise ValueError(f"Model directory is missing {name}")
    if not any(model_path.glob("*.safetensors")):
        raise ValueError("Model directory must contain safetensors weights")
    recurrence = json.loads((model_path / "alodlm_config.json").read_text())
    backbone = json.loads((model_path / "config.json").read_text())
    start, end, depth = (recurrence[k] for k in ("loop_start", "loop_end", "max_depth"))
    if (any(type(value) is not int for value in (start, end, depth))
            or not (0 <= start < end <= backbone["num_hidden_layers"] and depth > 1)):
        raise ValueError("Invalid recurrent layer range or depth")
    if backbone.get("model_type") != "qwen3":
        raise ValueError("The optimized engine supports Qwen3 backbones")
    if backbone.get("tie_word_embeddings", False):
        raise ValueError("ALoDLM weights must retain their untied output projection")
    return recurrence, backbone


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0 <= args.q <= 1 or not 0 <= args.tau < float("inf"):
        parser.error("q must be in [0, 1] and tau must be finite and nonnegative")
    if not 1 <= args.max_new_tokens <= 4096:
        parser.error("max-new-tokens must be in [1, 4096]")
    if args.max_model_len < 1 or not 0 < args.gpu_memory_utilization < 1:
        parser.error("max-model-len must be positive and gpu-memory-utilization must be in (0, 1)")
    if not 0 <= args.nccl_port <= 65535:
        parser.error("nccl-port must be in [0, 65535]")

    import torch
    from huggingface_hub import snapshot_download

    if not torch.cuda.is_available():
        parser.error("The optimized engine requires an NVIDIA CUDA GPU")
    model_path = Path(args.model).expanduser()
    if not model_path.is_dir():
        model_path = Path(snapshot_download(repo_id=args.model))
    model_path = model_path.resolve()
    try:
        recurrence, backbone = read_model_metadata(model_path)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    start, end, depth = (recurrence[k] for k in ("loop_start", "loop_end", "max_depth"))
    context_limit = min(args.max_model_len, backbone["max_position_embeddings"])

    # Keep unrelated research-mode overrides out of this single-request example.
    for key in list(os.environ):
        if key.startswith("WEDLM_"):
            del os.environ[key]
    os.environ.update(
        WEDLM_GATE=str(model_path), WEDLM_MODEL_PATH=str(model_path),
        WEDLM_ADAPTIVE_DEPTH="1", WEDLM_COMMIT_ON_AGREE="1",
        WEDLM_GATE_STOP="gate_depth" if args.mode == "entropy" else "gate_depth_left1",
        WEDLM_QEXIT=str(args.q),
        WEDLM_LOOP_START=str(start), WEDLM_LOOP_END=str(end),
        WEDLM_LOOP_K=str(depth), WEDLM_LOOP_CARRY_NORM="1",
        WEDLM_PASS_GRAPH="1", WEDLM_CHECK_GRAPH="1", WEDLM_CHAT_PROMPT="1",
    )
    install_attention()
    from .adaptive_cache_repair import install
    install()
    install_rope_compatibility()
    from wedlm import LLM, SamplingParams

    torch.set_num_threads(1)
    port = args.nccl_port
    if not port:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
    engine = LLM(
        str(model_path), tensor_parallel_size=1, max_num_seqs=1,
        # Adaptive pass/decision graphs are enabled above; full-model capture
        # cannot represent adaptive depth and must remain disabled.
        enforce_eager=True, max_model_len=context_limit,
        max_num_batched_tokens=context_limit, kvcache_block_size=256,
        gpu_memory_utilization=args.gpu_memory_utilization, wedlm_window_size=16,
        nccl_port=port,
        mask_token_id=recurrence["mask_token_id"],
    )
    ids = engine.tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True, return_dict=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    if len(ids) + args.max_new_tokens + 32 > context_limit:
        parser.error("Prompt and output budget exceed the context/window capacity")
    stops = sorted({
        token for token in (
            engine.tokenizer.eos_token_id,
            engine.tokenizer.get_vocab().get("<|im_end|>"),
            engine.tokenizer.get_vocab().get("<|endoftext|>"),
        ) if token is not None
    })
    params = SamplingParams(
        temperature=0, max_tokens=args.max_new_tokens, stop_token_ids=stops,
        wedlm_entropy_threshold=args.tau, wedlm_pos_penalty_factor=0.02,
    )
    with torch.inference_mode():
        result = engine.generate([ids], params, use_tqdm=False)[0]
    graphs = getattr(engine.model_runner, "_passgraphs", {})
    if not graphs:
        raise RuntimeError("The engine did not activate adaptive pass CUDA graphs")
    if end < backbone["num_hidden_layers"] and not any(
            isinstance(entry, dict) and entry.get("check_graphs")
            for entry in graphs.values()):
        raise RuntimeError("The engine did not activate the 8B decision CUDA graphs")
    if args.json:
        print(json.dumps({
            "text": result["text"], "token_ids": result["token_ids"],
            "mode": args.mode, "pass_graphs": bool(graphs),
            "decision_graphs": any(isinstance(entry, dict) and bool(entry.get("check_graphs"))
                                   for entry in graphs.values()),
        }))
    else:
        print(result["text"])


if __name__ == "__main__":
    main()
