"""Synthetic model fixtures. Run: python -m unittest discover -s tests."""

from dataclasses import asdict
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from alodlm.config import ModelConfig
from alodlm.model import ALoDLM


def tiny_model():
    torch.manual_seed(42)
    backbone = Qwen3ForCausalLM(Qwen3Config(
        hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        vocab_size=64, max_position_embeddings=128,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    ))
    config = ModelConfig(loop_start=1, loop_end=2, mask_token_id=3, block_size=4,
                         attention_backend="sdpa", compile_layers=False)
    return ALoDLM(backbone, config)


def tiny_tokenizer():
    vocab = {"<pad>": 0, "<unk>": 1, "</s>": 2, "<mask>": 3,
             "user": 4, "assistant": 5, "system": 6,
             "Return": 7, "value": 8, "one": 9, "two": 10, "three": 11, ".": 12}
    vocab.update({f"t{i}": i for i in range(len(vocab), 64)})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    result = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", pad_token="<pad>",
        eos_token="</s>", mask_token="<mask>",
    )
    result.chat_template = (
        "{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' </s> ' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
    )
    return result


def write_fixture(directory):
    root = Path(directory)
    model_dir = root / "base"
    model = tiny_model()
    model.backbone.save_pretrained(model_dir)
    tiny_tokenizer().save_pretrained(model_dir)
    data = root / "training.jsonl"
    data.write_text("".join(json.dumps([
        {"role": "user", "content": "Return value " + word},
        {"role": "assistant", "content": word + " ."},
    ]) + "\n" for word in ("one", "two", "three") * 4))
    return {
        "model_dir": str(model_dir), "train_file": str(data),
        "output_dir": str(root / "run"), "cache_file": str(root / "packed.pt"),
        "model": asdict(model.config), "max_seq_length": 24,
        "epochs": 2, "max_steps": 4, "precision": "fp32",
        "logging_steps": 1, "save_steps": 2, "seed": 17,
        "use_deepspeed": False,
    }
