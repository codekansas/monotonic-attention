# monotonic-attention

Monotonic attention as a probabilistic graphical model

[Write-up explaining how this works](https://ben.bolte.cc/monotonic-attention)

Check out the `examples/` directory for more information.

## Getting Started

Install from PyPi:

```bash
pip install monotonic-attention
```

Install from source:

```bash
pip install git+https://github.com/codekansas/monotonic-attention.git
```

You should also install Triton if you plan to use the GPU kernels (highly recommended):

```bash
pip install triton
```

## Usage

```python
from monotonic_attention import OneToManyMultiheadMonotonicAttention

# Many keys mapped to a single query.
attn = OneToManyMultiheadMonotonicAttention(
  mode="many_keys_one_query",
  embed_dim=1024,
  num_heads=16,
)

output = attn(query, key, value)

# Many queries mapped to a single key.
attn = OneToManyMultiheadMonotonicAttention(
  mode="many_queries_one_key",
  embed_dim=1024,
  num_heads=16,
)

output = attn(query, key, value)
```
