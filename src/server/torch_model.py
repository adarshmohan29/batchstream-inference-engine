import json
import struct
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    rope_theta: float
    rms_norm_eps: float

    @classmethod
    def from_file(cls, path: Path) -> "ModelConfig":
        config = json.loads(path.read_text())
        return cls(
            num_layers=config["num_hidden_layers"],
            hidden_size=config["hidden_size"],
            num_heads=config["num_attention_heads"],
            num_kv_heads=config["num_key_value_heads"],
            rope_theta=config["rope_theta"],
            rms_norm_eps=config["rms_norm_eps"],
        )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


class ContiguousKVCache:
    def __init__(
        self, num_layers: int, num_kv_heads: int, head_dim: int, capacity: int, dtype: torch.dtype
    ):
        shape = (num_layers, num_kv_heads, capacity, head_dim)
        self._keys = torch.zeros(shape, dtype=dtype)
        self._values = torch.zeros(shape, dtype=dtype)
        self._capacity = capacity
        self.length = 0

    def check_room(self, count: int) -> None:
        if self.length + count > self._capacity:
            raise ValueError(f"KV cache holds {self._capacity} tokens, needs {self.length + count}")

    def append(
        self, layer: int, keys: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store one layer's new K/V and return everything cached for that layer."""
        end = self.length + keys.shape[1]
        self._keys[layer, :, self.length : end] = keys
        self._values[layer, :, self.length : end] = values
        return self._keys[layer, :, :end], self._values[layer, :, :end]

    def advance(self, count: int) -> None:
        self.length += count


def _load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    data = bytearray(path.read_bytes())
    header_size = struct.unpack_from("<Q", data)[0]
    header = json.loads(data[8 : 8 + header_size])
    header.pop("__metadata__")
    weights = {}
    for name, entry in header.items():
        if entry["dtype"] != "BF16":
            raise ValueError(f"{name} has unsupported dtype {entry['dtype']}")
        start, end = entry["data_offsets"]
        tensor = torch.frombuffer(
            data, dtype=torch.bfloat16, count=(end - start) // 2, offset=8 + header_size + start
        )
        weights[name] = tensor.reshape(entry["shape"])
    return weights


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class TorchModel:
    def __init__(self, config: ModelConfig, weights: dict[str, torch.Tensor]):
        self.config = config
        self.weights = weights
        steps = torch.arange(0, config.head_dim, 2, dtype=torch.float)
        self._inverse_frequencies = 1.0 / (config.rope_theta ** (steps / config.head_dim))

    @classmethod
    def load(cls, model_dir: Path) -> "TorchModel":
        config = ModelConfig.from_file(model_dir / "config.json")
        return cls(config, _load_safetensors(model_dir / "model.safetensors"))

    def new_cache(self, capacity: int) -> ContiguousKVCache:
        return ContiguousKVCache(
            self.config.num_layers,
            self.config.num_kv_heads,
            self.config.head_dim,
            capacity,
            torch.bfloat16,
        )

    @torch.inference_mode()
    def forward(self, tokens: list[int], cache: ContiguousKVCache) -> torch.Tensor:
        """Run new tokens through the model and return logits with shape (tokens, vocab)."""
        count = len(tokens)
        cache.check_room(count)
        embeddings = self.weights["model.embed_tokens.weight"]
        rotary = self._rotary(torch.arange(cache.length, cache.length + count))
        mask = self._causal_mask(count, cache.length)
        hidden = embeddings[tokens]
        for layer in range(self.config.num_layers):
            hidden = self._layer(layer, hidden, rotary, mask, cache)
        cache.advance(count)
        hidden = self._rms_norm(hidden, self.weights["model.norm.weight"])
        # The checkpoint ties the output projection to the input embeddings.
        return F.linear(hidden, embeddings)

    def _layer(self, layer, hidden, rotary, mask, cache) -> torch.Tensor:
        prefix = f"model.layers.{layer}."
        normed = self._rms_norm(hidden, self.weights[prefix + "input_layernorm.weight"])
        hidden = hidden + self._attention(layer, prefix + "self_attn.", normed, rotary, mask, cache)
        normed = self._rms_norm(hidden, self.weights[prefix + "post_attention_layernorm.weight"])
        return hidden + self._mlp(prefix + "mlp.", normed)

    def _attention(self, layer, prefix, hidden, rotary, mask, cache) -> torch.Tensor:
        config = self.config
        queries = self._project(prefix + "q_proj", hidden, config.num_heads)
        keys = self._project(prefix + "k_proj", hidden, config.num_kv_heads)
        values = self._project(prefix + "v_proj", hidden, config.num_kv_heads)
        cos, sin = rotary
        queries = queries * cos + _rotate_half(queries) * sin
        keys = keys * cos + _rotate_half(keys) * sin
        all_keys, all_values = cache.append(layer, keys, values)
        groups = config.num_heads // config.num_kv_heads
        all_keys = all_keys.repeat_interleave(groups, dim=0)
        all_values = all_values.repeat_interleave(groups, dim=0)
        scores = queries @ all_keys.transpose(1, 2) * config.head_dim**-0.5
        # Softmax in float32, as the reference does, so results stay bit-identical.
        attention = F.softmax(scores.masked_fill(mask, -torch.inf), dim=-1, dtype=torch.float32)
        output = attention.to(queries.dtype) @ all_values
        output = output.transpose(0, 1).reshape(hidden.shape[0], -1)
        return F.linear(output, self.weights[prefix + "o_proj.weight"])

    def _project(self, name: str, hidden: torch.Tensor, heads: int) -> torch.Tensor:
        projected = F.linear(hidden, self.weights[name + ".weight"], self.weights[name + ".bias"])
        return projected.view(hidden.shape[0], heads, -1).transpose(0, 1)

    def _mlp(self, prefix: str, hidden: torch.Tensor) -> torch.Tensor:
        gate = F.silu(F.linear(hidden, self.weights[prefix + "gate_proj.weight"]))
        up = F.linear(hidden, self.weights[prefix + "up_proj.weight"])
        return F.linear(gate * up, self.weights[prefix + "down_proj.weight"])

    def _rms_norm(self, hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        wide = hidden.to(torch.float32)
        variance = wide.pow(2).mean(-1, keepdim=True)
        normalized = wide * torch.rsqrt(variance + self.config.rms_norm_eps)
        return weight * normalized.to(hidden.dtype)

    def _rotary(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = positions[:, None].float() * self._inverse_frequencies
        angles = torch.cat((frequencies, frequencies), dim=-1)
        return angles.cos().to(torch.bfloat16), angles.sin().to(torch.bfloat16)

    @staticmethod
    def _causal_mask(count: int, past: int) -> torch.Tensor:
        hidden_future = torch.ones(count, count, dtype=torch.bool).triu(diagonal=1)
        return torch.cat((torch.zeros(count, past, dtype=torch.bool), hidden_future), dim=1)
