"""Generate from local weights or a Hugging Face model ID. Run: python -m alodlm.generate --help."""

import argparse
import json
from pathlib import Path

import torch
import yaml

from .config import ModelConfig
from .data import load_tokenizer
from .inference import DecodeConfig, Decoder
from .hub import resolve_model
from .model import ALoDLM


def model_arguments(parser, required=True):
    parser.add_argument("--model", required=required, help="Local model directory or Hugging Face model ID")
    parser.add_argument("--model-config", help="YAML model section for a checkpoint without recurrence metadata")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--q", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.4)
    parser.add_argument("--mode", choices=("entropy", "left1"), default="entropy")
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--position-penalty", type=float)


def load_decoder(args):
    if not args.model:
        raise ValueError("--model is required for generation")
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    config = None
    if args.model_config:
        config = ModelConfig(**yaml.safe_load(Path(args.model_config).read_text())["model"])
    directory = resolve_model(args.model)
    model = ALoDLM.from_pretrained(directory, config=config, dtype=dtype).to(args.device)
    return Decoder(model, load_tokenizer(directory))


def decode_config(args):
    return DecodeConfig(q=args.q, tau=args.tau, mode=args.mode, window_size=args.window_size,
                        max_new_tokens=args.max_new_tokens, position_penalty=args.position_penalty)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model_arguments(parser)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()
    decoder = load_decoder(args)
    ids = decoder.tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=args.thinking)
    print(json.dumps(decoder.generate(ids, decode_config(args))))


if __name__ == "__main__":
    main()
