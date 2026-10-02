"""Pack locally supplied messages. Run: python -m alodlm.data --help."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .config import local_dir


def load_tokenizer(directory):
    tokenizer = AutoTokenizer.from_pretrained(
        str(local_dir(directory)), local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError("A local tokenizer with a chat template is required")
    if "<|im_end|>" in tokenizer.get_vocab():
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    elif tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_identity(train_file, tokenizer, max_length, seed):
    specification = {
        "format_version": 3, "data_sha256": digest_file(train_file),
        "vocabulary": sorted(tokenizer.get_vocab().items()),
        "tokenizer": tokenizer.backend_tokenizer.to_str(),
        "chat_template": tokenizer.chat_template,
        "special_tokens": tokenizer.special_tokens_map,
        "max_length": max_length, "seed": seed,
    }
    return hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()


def tokenize_messages(messages, tokenizer, max_length):
    if not messages:
        return None
    if not isinstance(messages, list):
        raise ValueError("Each record must contain a message list")
    if any(not isinstance(m, dict) or not isinstance(m.get("content"), str) for m in messages):
        raise ValueError("Message content must be text")
    if len(messages) > 1 and messages[-1].get("role") != "assistant":
        raise ValueError("Multi-message conversations must end with an assistant response")
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    ids = tokenizer.encode(full, add_special_tokens=False)
    if not ids or len(ids) > max_length:
        return None
    if len(messages) > 1:
        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        prompt_length = min(len(tokenizer.encode(prompt, add_special_tokens=False)), len(ids))
    else:
        prompt_length = len(ids)
    labels = [-100] * prompt_length + ids[prompt_length:]
    return torch.tensor(ids, dtype=torch.int32), torch.tensor(labels, dtype=torch.int32)


def prepare_cache(train_file, tokenizer, cache_file, max_length=4096, seed=42):
    if max_length < 2:
        raise ValueError("max_length must be at least two")
    identity = cache_identity(train_file, tokenizer, max_length, seed)
    target = Path(cache_file)
    if target.exists():
        cached = torch.load(target, mmap=True, weights_only=True)
        if cached.get("identity") != identity:
            raise ValueError("Packed cache does not match the local input and tokenizer")
        return target
    samples, dropped, invalid = [], 0, 0
    with Path(train_file).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                sample = tokenize_messages(record, tokenizer, max_length)
            except (TypeError, ValueError, KeyError, AttributeError):
                invalid += 1
                continue
            if sample is None:
                dropped += 1
            else:
                samples.append(sample)
    if not samples:
        raise ValueError("No usable training conversations")
    random.Random(seed).shuffle(samples)
    id_parts, label_parts, batch_offsets, cum_flat, cum_offsets = [], [], [0], [], [0]
    used, total, boundaries = 0, 0, [0]
    for ids, labels in samples:
        offset = 0
        while offset < len(ids):
            take = min(max_length - used, len(ids) - offset)
            id_parts.append(ids[offset:offset + take])
            label_parts.append(labels[offset:offset + take])
            offset += take
            used += take
            total += take
            boundaries.append(used)
            if used == max_length:
                batch_offsets.append(total)
                cum_flat.extend(boundaries)
                cum_offsets.append(len(cum_flat))
                used, boundaries = 0, [0]
    if used:
        batch_offsets.append(total)
        cum_flat.extend(boundaries)
        cum_offsets.append(len(cum_flat))
    packed = {
        "identity": identity,
        "input_ids": torch.cat(id_parts), "labels": torch.cat(label_parts),
        "batch_offsets": torch.tensor(batch_offsets),
        "cum_flat": torch.tensor(cum_flat), "cum_offsets": torch.tensor(cum_offsets),
        "filtered_records": dropped + invalid,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(packed, temporary)
    temporary.replace(target)
    return target


class PackedDataset(Dataset):
    def __init__(self, filename):
        self.cache = torch.load(filename, mmap=True, weights_only=True)

    def __len__(self):
        return len(self.cache["batch_offsets"]) - 1

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError("Pack index out of range")
        c = self.cache
        left, right = c["batch_offsets"][index:index + 2].tolist()
        a, b = c["cum_offsets"][index:index + 2].tolist()
        return {"ids": c["input_ids"][left:right].long(),
                "labels": c["labels"][left:right].long(),
                "boundaries": c["cum_flat"][a:b].clone()}


def one_pack(items):
    if len(items) != 1:
        raise ValueError("Each microbatch contains one packed sequence")
    return items[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    path = prepare_cache(args.input, load_tokenizer(args.model), args.output,
                         args.max_length, args.seed)
    print(json.dumps({"packs": len(PackedDataset(path))}))


if __name__ == "__main__":
    main()
