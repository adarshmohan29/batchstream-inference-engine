import json
import struct
from dataclasses import dataclass
from itertools import accumulate, pairwise
from pathlib import Path

import torch
import torch.nn.functional as F

from server.block_allocator import BlockAllocator


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


def attend(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor | None,
    groups: int,
    scaling: float,
) -> torch.Tensor:
    """Attention over contiguous K/V, returned as (heads, tokens, head_dim)."""
    all_keys = keys.repeat_interleave(groups, dim=0)
    all_values = values.repeat_interleave(groups, dim=0)
    scores = queries @ all_keys.transpose(1, 2) * scaling
    if mask is not None:
        scores = scores.masked_fill(mask, -torch.inf)
    # Softmax in float32, as the reference does, so results stay bit-identical.
    weights = F.softmax(scores, dim=-1, dtype=torch.float32)
    return weights.to(queries.dtype) @ all_values


def _read_blocks(
    store: torch.Tensor, block_table: list[int], length: int, block_size: int
) -> torch.Tensor:
    pieces = []
    for index, block in enumerate(block_table):
        pieces.append(store[:, block, : min(block_size, length - index * block_size)])
    return torch.cat(pieces, dim=1)


def paged_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    block_table: list[int],
    length: int,
    block_size: int,
    groups: int,
    scaling: float,
) -> torch.Tensor:
    """Attention for one new token, reading K/V of shape (kv_heads, blocks, block_size, head_dim)."""
    cached_keys = _read_blocks(keys, block_table, length, block_size)
    cached_values = _read_blocks(values, block_table, length, block_size)
    return attend(query[:, None], cached_keys, cached_values, None, groups, scaling)[:, 0]


class ContiguousKVCache:
    def __init__(
        self, num_layers: int, num_kv_heads: int, head_dim: int, capacity: int, dtype: torch.dtype
    ):
        shape = (num_layers, num_kv_heads, capacity, head_dim)
        self._keys = torch.zeros(shape, dtype=dtype)
        self._values = torch.zeros(shape, dtype=dtype)
        self._capacity = capacity
        self.length = 0

    def reserve(self, count: int) -> None:
        if self.length + count > self._capacity:
            raise ValueError(f"KV cache holds {self._capacity} tokens, needs {self.length + count}")

    def append(self, layer: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        end = self.length + keys.shape[1]
        self._keys[layer, :, self.length : end] = keys
        self._values[layer, :, self.length : end] = values

    def read(self, layer: int, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return everything cached for a layer, including count not yet advanced tokens."""
        end = self.length + count
        return self._keys[layer, :, :end], self._values[layer, :, :end]

    def attend(self, layer, queries, mask, groups, scaling) -> torch.Tensor:
        keys, values = self.read(layer, queries.shape[1])
        return attend(queries, keys, values, mask, groups, scaling)

    def advance(self, count: int) -> None:
        self.length += count


class PagedKVPool:
    def __init__(self, config: "ModelConfig", num_blocks: int, block_size: int):
        shape = (config.num_layers, config.num_kv_heads, num_blocks, block_size, config.head_dim)
        self.keys = torch.zeros(shape, dtype=torch.bfloat16)
        self.values = torch.zeros(shape, dtype=torch.bfloat16)
        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks, block_size)

    @property
    def free_blocks(self) -> int:
        return self.allocator.free_blocks

    def block_table(self, seq_id: int) -> list[int]:
        return self.allocator.block_table(seq_id)

    def new_cache(self, seq_id: int) -> "PagedKVCache":
        return PagedKVCache(self, seq_id)

    def release(self, cache: "PagedKVCache") -> None:
        self.allocator.free(cache.seq_id)


class PagedKVCache:
    def __init__(self, pool: PagedKVPool, seq_id: int):
        self._pool = pool
        self.seq_id = seq_id
        self.length = 0

    def reserve(self, count: int) -> None:
        self._pool.allocator.allocate(self.seq_id, self.length + count)

    def append(self, layer: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        for index in range(keys.shape[1]):
            block, offset = self._pool.allocator.locate(self.seq_id, self.length + index)
            self._pool.keys[layer, :, block, offset] = keys[:, index]
            self._pool.values[layer, :, block, offset] = values[:, index]

    def attend(self, layer, queries, mask, groups, scaling) -> torch.Tensor:
        pool = self._pool
        table = pool.block_table(self.seq_id)
        length = self.length + queries.shape[1]
        if queries.shape[1] == 1:
            output = paged_attention(
                queries[:, 0], pool.keys[layer], pool.values[layer], table, length,
                pool.block_size, groups, scaling,
            )
            return output[:, None]
        keys = _read_blocks(pool.keys[layer], table, length, pool.block_size)
        values = _read_blocks(pool.values[layer], table, length, pool.block_size)
        return attend(queries, keys, values, mask, groups, scaling)

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
    def forward(self, tokens: list[int], cache) -> torch.Tensor:
        """Run new tokens through the model and return logits with shape (tokens, vocab)."""
        hidden = self._hidden_states([(tokens, cache)])
        # The checkpoint ties the output projection to the input embeddings.
        return F.linear(hidden, self.weights["model.embed_tokens.weight"])

    @torch.inference_mode()
    def forward_batch(self, groups: list[tuple[list[int], object]]) -> torch.Tensor:
        """Run several (tokens, cache) pairs together and return each one's last-token logits."""
        hidden = self._hidden_states(groups)
        ends = list(accumulate(len(tokens) for tokens, _ in groups))
        last = hidden[[end - 1 for end in ends]]
        return F.linear(last, self.weights["model.embed_tokens.weight"])

    def _hidden_states(self, groups: list[tuple[list[int], object]]) -> torch.Tensor:
        for tokens, cache in groups:
            cache.reserve(len(tokens))
        positions = [
            torch.arange(cache.length, cache.length + len(tokens)) for tokens, cache in groups
        ]
        rotary = self._rotary(torch.cat(positions))
        masks = [self._causal_mask(len(tokens), cache.length) for tokens, cache in groups]
        slices = list(pairwise(accumulate((len(tokens) for tokens, _ in groups), initial=0)))
        caches = [cache for _, cache in groups]
        flat_tokens = [token for tokens, _ in groups for token in tokens]
        hidden = self.weights["model.embed_tokens.weight"][flat_tokens]
        for layer in range(self.config.num_layers):
            hidden = self._layer(layer, hidden, rotary, masks, slices, caches)
        for tokens, cache in groups:
            cache.advance(len(tokens))
        return self._rms_norm(hidden, self.weights["model.norm.weight"])

    def _layer(self, layer, hidden, rotary, masks, slices, caches) -> torch.Tensor:
        prefix = f"model.layers.{layer}."
        normed = self._rms_norm(hidden, self.weights[prefix + "input_layernorm.weight"])
        attended = self._attention(layer, prefix + "self_attn.", normed, rotary, masks, slices, caches)
        hidden = hidden + attended
        normed = self._rms_norm(hidden, self.weights[prefix + "post_attention_layernorm.weight"])
        return hidden + self._mlp(prefix + "mlp.", normed)

    def _attention(self, layer, prefix, hidden, rotary, masks, slices, caches) -> torch.Tensor:
        config = self.config
        queries = self._project(prefix + "q_proj", hidden, config.num_heads)
        keys = self._project(prefix + "k_proj", hidden, config.num_kv_heads)
        values = self._project(prefix + "v_proj", hidden, config.num_kv_heads)
        cos, sin = rotary
        queries = queries * cos + _rotate_half(queries) * sin
        keys = keys * cos + _rotate_half(keys) * sin
        groups = config.num_heads // config.num_kv_heads
        scaling = config.head_dim**-0.5
        outputs = []
        for (start, end), mask, cache in zip(slices, masks, caches):
            cache.append(layer, keys[:, start:end], values[:, start:end])
            outputs.append(cache.attend(layer, queries[:, start:end], mask, groups, scaling))
        output = torch.cat(outputs, dim=1).transpose(0, 1).reshape(hidden.shape[0], -1)
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
