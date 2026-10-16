"""Tiny Hugging Face checkpoints written from scratch, so no test downloads one.

Everything here goes through public, long-stable transformers API
(`LlamaConfig`, `PretrainedConfig`, `PreTrainedModel`, `PreTrainedTokenizerFast`),
so the same files load on the 4.x and 5.x releases the CI matrix pins.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import textwrap
from pathlib import Path

import pytest

requires_transformers = pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None, reason="needs transformers"
)

TINY_VOCAB = 1000
TINY_HIDDEN = 64


def write_tiny_llama(dest: Path, **overrides) -> Path:
    from transformers import LlamaConfig

    values = {
        "vocab_size": TINY_VOCAB,
        "hidden_size": TINY_HIDDEN,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 512,
    }
    values.update(overrides)
    dest.mkdir(parents=True, exist_ok=True)
    LlamaConfig(**values).save_pretrained(dest)
    return dest


_REMOTE_CONFIG = """
from transformers import PretrainedConfig


class TinyRemoteConfig(PretrainedConfig):
    model_type = "tiny_remote"

    def __init__(self, vocab_size=1000, hidden_size=64, num_hidden_layers=2, **kwargs):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        super().__init__(**kwargs)
"""

_REMOTE_MODEL = """
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_tiny_remote import TinyRemoteConfig


class TinyRemoteForCausalLM(PreTrainedModel):
    config_class = TinyRemoteConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            nn.Linear(config.hidden_size, config.hidden_size)
            for _ in range(config.num_hidden_layers)
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, use_cache=None, **kwargs):
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = torch.relu(layer(hidden))
        return CausalLMOutputWithPast(logits=self.lm_head(hidden))
"""


def write_remote_checkpoint(dest: Path) -> Path:
    """A checkpoint that needs `trust_remote_code`, like the motivating K2 model."""
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "configuration_tiny_remote.py").write_text(_REMOTE_CONFIG)
    (dest / "modeling_tiny_remote.py").write_text(_REMOTE_MODEL)
    (dest / "config.json").write_text(
        json.dumps(
            {
                "model_type": "tiny_remote",
                "architectures": ["TinyRemoteForCausalLM"],
                "auto_map": {
                    "AutoConfig": "configuration_tiny_remote.TinyRemoteConfig",
                    "AutoModelForCausalLM": "modeling_tiny_remote.TinyRemoteForCausalLM",
                },
                "vocab_size": TINY_VOCAB,
                "hidden_size": TINY_HIDDEN,
                "num_hidden_layers": 2,
            }
        )
    )
    return dest


def write_tokenizer(dest: Path) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "[PAD]": 1, **{f"w{i}": i + 2 for i in range(50)}}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        model_input_names=["input_ids", "attention_mask"],
    )
    dest.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(dest)
    return dest


def write_inputs(path: Path, lengths: dict[str, int]) -> Path:
    """A safetensors file of int64 token ids, written by hand so it needs no torch."""
    header: dict[str, dict] = {}
    offset = 0
    for name, length in lengths.items():
        size = length * 8
        header[name] = {"dtype": "I64", "shape": [length], "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(offset))
    return path


def dedent(text: str) -> str:
    return textwrap.dedent(text).lstrip("\n")
