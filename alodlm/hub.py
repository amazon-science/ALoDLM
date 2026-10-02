"""Resolve inference artifacts from a local directory or Hugging Face model ID."""

from pathlib import Path


INFERENCE_FILES = [
    "config.json", "generation_config.json", "alodlm_config.json", "exit_gate.pt",
    "*.safetensors", "*.safetensors.index.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "tokenizer.model", "vocab.json", "vocab.txt", "merges.txt",
    "*.jinja", "chat_templates/*.jinja",
]


def resolve_model(model):
    path = Path(model).expanduser()
    if path.is_dir():
        return path.resolve()
    if path.is_absolute() or str(model).startswith((".", "~")):
        raise ValueError("The local model directory does not exist")
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=str(model), allow_patterns=INFERENCE_FILES))
