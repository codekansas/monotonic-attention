"""Defines the monotonic attention PyTorch module."""

import functools
import math
from typing import Callable, Literal, TypeVar

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from monotonic_attention.one_to_many.cpu_log import monotonic_attention as monotonic_attention_cpu
from monotonic_attention.utils import supports_triton

Mode = Literal["many_keys_one_query", "one_key_many_queries"]


Tq = TypeVar("Tq", Tensor, None)
Tk = TypeVar("Tk", Tensor, None)
Tv = TypeVar("Tv", Tensor, None)


def phi_to_pos_prob(phi: float) -> float:
    return 1 / (math.exp(-phi) + 1)


def phi_to_neg_prob(phi: float) -> float:
    return 1 / (math.exp(phi) + 1)


def pos_prob_to_phi(prob: float) -> float:
    return math.log((1 / prob) - 1)


def neg_prob_to_phi(prob: float) -> float:
    return -math.log((1 / prob) - 1)


@functools.lru_cache(None)
def get_monotonic_attention_fn(device_type: str) -> Callable[[Tensor], Tensor]:
    if device_type != "cuda" or not supports_triton():
        return monotonic_attention_cpu

    from monotonic_attention.one_to_many.triton.gpu_log import monotonic_attention as monotonic_attention_gpu

    return monotonic_attention_gpu


class OneToManyMultiheadMonotonicAttention(nn.Module):
    """Defines a one-to-many multihead monotonic attention layer.

    Parameters:
        mode: Specifies either many keys for each query, or many queries for
            each key. In the former case, the key and value sequences should
            be longer than the query sequence; vice versa in the latter case.
        embed_dim: The input and output embedding dimension.
        num_heads: The number of attention heads.
        bias: Whether to include a bias term in the projection layers.
        kdim: The dimension of the key projection. Defaults to ``embed_dim``.
        vdim: The dimension of the value projection. Defaults to ``embed_dim``.
        gqa_factor: The GQA factor to use, meaning the ratio of the number of
            queries to the number of keys. Higher values will result in more
            queries than keys, which can speed up inference.
        clamp_prob: The maximum transition probability to clamp to.
        soft_clamp: If set, soft-clamp the transition probabailities using the
            tanh function, otherwise hard-clamp them.

    Inputs:
        query: The query tensor, of shape ``(B, T, C)``.
        key: The key tensor, of shape ``(B, T, C)``.
        value: The value tensor, of shape ``(B, T, C)``.
        state: The previous key and value tensors, of shape
            ``(B * H, T', C // H)``, where ``T'`` is the number of previous
            timesteps and ``H`` is the number of attention heads. This is
            only supported if ``is_causal=True``.
        is_causal: Whether to apply a causal mask to the attention matrix.
            Note that the "mask" is only applied implicitly and isn't actually
            instantiated as a tensor.

    Outputs:
        output: The output tensor, of shape ``(B, T, C)``, along with the
            key and value state for the next timestep.
    """

    __constants__ = [
        "num_heads",
        "gqa_factor",
        "kv_num_heads",
        "head_dim",
        "max_kv_cache_len",
        "embed_dim",
        "kv_embed_dim",
        "kdim",
        "vdim",
        "_qkv_same_embed_dim",
        "clamp_prob",
        "soft_clamp",
        "norm_fact",
    ]

    def __init__(
        self,
        mode: Mode,
        embed_dim: int,
        num_heads: int = 1,
        bias: bool = True,
        kdim: int | None = None,
        vdim: int | None = None,
        gqa_factor: int = 1,
        max_kv_cache_len: int | None = None,
        clamp_prob: float = 0.98,
        soft_clamp: bool = True,
    ) -> None:
        super().__init__()

        assert embed_dim % num_heads == 0, f"`{embed_dim=}` must be divisible by `{num_heads=}`"
        assert num_heads % gqa_factor == 0, f"`{num_heads=}` must be divisible by `{gqa_factor=}`"
        assert clamp_prob > 0.5 and clamp_prob < 1.0, f"`{clamp_prob=}` must be in the range (0.5, 1.0]"

        # Stores some constant values.
        self.mode = mode
        self.num_heads = num_heads
        self.gqa_factor = gqa_factor
        self.kv_num_heads = num_heads // gqa_factor
        self.head_dim = embed_dim // num_heads
        self.max_kv_cache_len = max_kv_cache_len
        self.clamp_value = neg_prob_to_phi(clamp_prob)
        self.soft_clamp = soft_clamp
        self.norm_fact = math.sqrt(self.head_dim)

        self.embed_dim = embed_dim
        self.kv_embed_dim = self.kv_num_heads * self.head_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        if not self._qkv_same_embed_dim:
            self.qa_proj_weight = nn.Parameter(torch.empty((embed_dim, embed_dim)))
            self.q_proj_weight = nn.Parameter(torch.empty((embed_dim, embed_dim)))
            self.k_proj_weight = nn.Parameter(torch.empty((self.kv_embed_dim, self.kdim)))
            self.v_proj_weight = nn.Parameter(torch.empty((self.kv_embed_dim, self.vdim)))
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty((embed_dim * 2 + self.kv_embed_dim * 2, embed_dim)))
            self.register_parameter("qa_proj_weight", None)
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(embed_dim * 2 + self.kv_embed_dim * 2))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.qa_proj_weight)
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

    def _clamp_attn(self, attn: Tensor) -> Tensor:
        if not self.soft_clamp:
            return attn.clamp(-self.clamp_value, self.clamp_value)
        return attn.tanh() * self.clamp_value

    def _compute_attn(self, xq: Tensor, xk: Tensor, mask: Tensor | None = None) -> Tensor:
        bsz, gqa, num_heads = xq.shape[:3]
        monotonic_attention = get_monotonic_attention_fn(xq.device.type)

        if self.mode == "one_key_many_queries":
            attn = torch.einsum("bghqc,bghkc->bghqk", xq, xk)
            if mask is not None:
                attn = attn + mask[:, None, None]
            attn = self._clamp_attn(attn)
            return monotonic_attention(attn.flatten(0, 2)).unflatten(0, (bsz, gqa, num_heads))

        if self.mode == "many_keys_one_query":
            attn = torch.einsum("bghqc,bghkc->bghkq", xq, xk)
            if mask is not None:
                attn = attn + mask.transpose(-2, -1)[:, None, None]
            attn = self._clamp_attn(attn)
            output = monotonic_attention(attn.flatten(0, 2)).unflatten(0, (bsz, gqa, num_heads))
            return output.transpose(-2, -1)

        raise NotImplementedError(f"Unknown mode: {self.mode}")

    def forward_matmuls(self, query: Tq, key: Tk, value: Tv) -> tuple[Tq, Tq, Tk, Tv]:
        # Gets the weights for the matmuls.
        qkw_splits = (self.embed_dim, self.embed_dim, self.kv_embed_dim, self.kv_embed_dim)
        if self._qkv_same_embed_dim:
            qwa, qw, kw, vw = self.in_proj_weight.split(qkw_splits, dim=0)
        else:
            qwa, qw, kw, vw = self.qa_proj_weight, self.q_proj_weight, self.k_proj_weight, self.v_proj_weight
        if self.in_proj_bias is None:
            qba, qb, kb, vb = None, None, None, None
        else:
            qba, qb, kb, vb = self.in_proj_bias.split(qkw_splits, dim=0)

        # Computes the query projection.
        if query is None:
            xqa, xq = None, None
        else:
            assert query.dim() == 3
            xqa = F.linear(query, qwa, qba)
            xqa = xqa.unflatten(-1, (self.gqa_factor, self.kv_num_heads, self.head_dim)).permute(0, 2, 3, 1, 4)
            xq = F.linear(query, qw, qb)
            xq = xq.unflatten(-1, (self.gqa_factor, self.kv_num_heads, self.head_dim)).permute(0, 2, 3, 1, 4)
            xq = torch.stack((xqa, xq), dim=0)

        # Computes the key projection.
        if key is None:
            xk = None
        else:
            assert key.dim() == 3
            xk = F.linear(key, kw, kb)
            xk = xk.unflatten(-1, (1, self.kv_num_heads, self.head_dim)).permute(0, 2, 3, 1, 4)

        # Computes the value projection.
        if value is None:
            xv = None
        else:
            assert value.dim() == 3
            xv = F.linear(value, vw, vb)
            xv = xv.unflatten(-1, (1, self.kv_num_heads, self.head_dim)).permute(0, 2, 3, 1, 4)

        return xq, xk, xv

    def get_attn_matrix(self, query: Tensor, key: Tensor, mask: Tensor | None = None) -> Tensor:
        """Computes the attention matrix for a given query and key.

        This function can be used for visualization purposes.

        Args:
            query: The query vector, with shape ``(B, Tq, C)``
            key: The key vector, with shape ``(B, Tk, C)``
            mask: The attention mask, of shape ``(B, Tq, Tk)``. If ``None``,
                don't apply an attention mask.

        Returns:
            The attention matrix, of shape ``(B, G, H, Tq, Tk)``.
        """
        xq, xk, _ = self.forward_matmuls(query, key, None)
        xqa, _ = xq.unbind(0)
        return self._compute_attn(xqa, xk, mask)

    def forward_attn(self, xq: Tensor, xk: Tensor, xv: Tensor, mask: Tensor | None = None) -> Tensor:
        xqa, xq = xq.unbind(0)
        monotonic_attn = self._compute_attn(xqa, xk, mask)
        regular_attn = torch.einsum("bghqc,bghkc->bghqk", xq, xk) / self.norm_fact
        attn = (regular_attn + monotonic_attn).softmax(dim=-1)
        xo = torch.einsum("bghqk,bghkc->bghqc", attn, xv)
        xo = xo.permute(0, 3, 1, 2, 4).flatten(2)
        xo = self.out_proj(xo)
        return xo

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None) -> Tensor:
        xq, xk, xv = self.forward_matmuls(query, key, value)
        xo = self.forward_attn(xq, xk, xv, mask)
        return xo
