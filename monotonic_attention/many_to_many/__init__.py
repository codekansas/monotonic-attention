"""Defines the many-to-many monotonic attention PyTorch module."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from monotonic_attention.many_to_many.cpu_log_faster import monotonic_attention


class ManyToManyMultiheadMonotonicAttention(nn.Module):
    """Defines a many-to-many multihead monotonic attention layer.

    Parameters:
        embed_dim: The input and output embedding dimension.
        num_heads: The number of attention heads.
        bias: Whether to include a bias term in the projection layers.
        kdim: The dimension of the key projection. Defaults to ``embed_dim``.
        vdim: The dimension of the value projection. Defaults to ``embed_dim``.
        gqa_factor: The GQA factor to use, meaning the ratio of the number of
            queries to the number of keys. Higher values will result in more
            queries than keys, which can speed up inference.

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
        "epsilon",
    ]

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        kdim: int | None = None,
        vdim: int | None = None,
        gqa_factor: int = 1,
        max_kv_cache_len: int | None = None,
        epsilon: float = 1e-3,
    ) -> None:
        super().__init__()

        assert embed_dim % num_heads == 0, f"`{embed_dim=}` must be divisible by `{num_heads=}`"
        assert num_heads % gqa_factor == 0, f"`{num_heads=}` must be divisible by `{gqa_factor=}`"

        # Stores some constant values.
        self.num_heads = num_heads
        self.gqa_factor = gqa_factor
        self.kv_num_heads = num_heads // gqa_factor
        self.head_dim = embed_dim // num_heads
        self.max_kv_cache_len = max_kv_cache_len
        self.epsilon = epsilon

        self.embed_dim = embed_dim
        self.kv_embed_dim = self.kv_num_heads * self.head_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = nn.Parameter(torch.empty((embed_dim, embed_dim)))
            self.k_proj_weight = nn.Parameter(torch.empty((self.kv_embed_dim, self.kdim)))
            self.v_proj_weight = nn.Parameter(torch.empty((self.kv_embed_dim, self.vdim)))
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty((embed_dim + 2 * self.kv_embed_dim, embed_dim)))
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(embed_dim + 2 * self.kv_embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

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
        assert query.dim() == 3 and key.dim() == 3

        # Computes query, key, and value projections
        qkw_splits = (self.embed_dim, self.kv_embed_dim, self.kv_embed_dim)
        if self._qkv_same_embed_dim:
            qw, kw, _ = self.in_proj_weight.split(qkw_splits, dim=0)
        else:
            qw, kw = self.q_proj_weight, self.k_proj_weight
        qb, kb, _ = (None, None, None) if self.in_proj_bias is None else self.in_proj_bias.split(qkw_splits, dim=0)

        xq = F.linear(query, qw, qb)
        xk = F.linear(key, kw, kb)

        # Permutes (B, T, G * H * C) -> (B, G, H, T, C)
        xq = xq.unflatten(-1, (self.gqa_factor, self.kv_num_heads, self.head_dim)).permute(0, 2, 3, 1, 4)
        xk = xk.unflatten(-1, (1, self.kv_num_heads, self.head_dim)).permute(0, 2, 3, 1, 4)

        # Computes the unnormalized attention scores.
        attn = torch.einsum("bghqc,bghkc->bghqk", xq, xk)

        # Applies the additional attention mask, if provided.
        if mask is not None:
            attn = attn + mask[:, None, None]

        # Applies the monotonic attention normalization.
        attn = monotonic_attention(attn.sigmoid())

        return attn

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None) -> Tensor:
        assert query.dim() == 3 and key.dim() == 3 and value.dim() == 3

        # Computes query, key, and value projections
        qkw_splits = (self.embed_dim, self.kv_embed_dim, self.kv_embed_dim)
        if self._qkv_same_embed_dim:
            qw, kw, vw = self.in_proj_weight.split(qkw_splits, dim=0)
        else:
            qw, kw, vw = self.q_proj_weight, self.k_proj_weight, self.v_proj_weight
        qb, kb, vb = (None, None, None) if self.in_proj_bias is None else self.in_proj_bias.split(qkw_splits, dim=0)

        xq = F.linear(query, qw, qb)
        xk = F.linear(key, kw, kb)
        xv = F.linear(value, vw, vb)

        # Permutes (B, T, G * H * C) -> (B, G, H, T, C)
        xq = xq.unflatten(-1, (self.gqa_factor, self.kv_num_heads, self.head_dim)).permute(0, 2, 3, 1, 4)
        xk = xk.unflatten(-1, (1, self.kv_num_heads, self.head_dim)).permute(0, 2, 3, 1, 4)
        xv = xv.unflatten(-1, (1, self.kv_num_heads, self.head_dim)).permute(0, 2, 3, 1, 4)

        # Computes the unnormalized attention scores.
        attn = torch.einsum("bghqc,bghkc->bghqk", xq, xk)

        # Applies the additional attention mask, if provided.
        if mask is not None:
            attn = attn + mask[:, None, None]

        # Applies the monotonic attention normalization.
        attn = monotonic_attention(attn.sigmoid())

        # Computes the weighted average of the values.
        xo = torch.einsum("bghqk,bghkc->bghqc", attn, xv)

        # Flattens (B, G, H, T, C) -> (B, T, G * H * C)
        xo = xo.permute(0, 3, 1, 2, 4).flatten(2)

        # Applies output projection
        xo = self.out_proj(xo)

        return xo
