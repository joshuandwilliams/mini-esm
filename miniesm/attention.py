"""Scaled dot-product attention (the core operation of every transformer).

The recipe, in one line:

    Output = softmax( (Q @ K^T) / sqrt(d) ) @ V

Intuition (see your Obsidian "Attention Step by Step" note):
    Q = queries  ("what am I looking for?")
    K = keys     ("what do I contain?")
    V = values   ("what I'll hand over if attended to")
Score every query against every key (dot product), scale, softmax into weights
that sum to 1, then take the weighted sum of values.

Shape convention (memorise this — it fixes every transpose):
    Q : (n, d)      n tokens, each a d-dim query
    K : (n, d)      SAME d as Q (they meet in a dot product)
    V : (n, d_v)    d_v may differ from d; here it's the same
    scores  : (n, n)    scores[i, j] = query i . key j
    weights : (n, n)    each ROW sums to 1 (a distribution over keys)
    Output  : (n, d_v)  one new, context-aware vector per token

New tool today: PyTorch autograd. We write only this forward pass; autograd
computes the backward pass for us, and `tests/test_attention.py` proves it
correct with `torch.autograd.gradcheck` (yesterday's finite-difference oracle,
now one call). Use float64 tensors when gradchecking.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor) -> tuple[Tensor, Tensor]:
    """Return (output, weights) for one scaled dot-product attention layer.

    Args:
        Q: queries, shape (n, d)
        K: keys,    shape (n, d)
        V: values,  shape (n, d_v)

    Returns:
        output:  shape (n, d_v) — the weighted sum of values per token
        weights: shape (n, n)   — attention weights, each row sums to 1
    """
    # TODO(you): four lines of algebra. Watch the slip points flagged below.

    # 1. d = the key/query dimension (the LAST axis of Q). You divide by sqrt(d).
    #    d = ...

    # 2. scores = Q @ K^T / sqrt(d)          -> (n, n)
    #    SLIP: transpose the KEYS, not the queries. Use K.transpose(-2, -1)
    #    (the batch-safe .T that swaps only the last two axes), NOT plain K.
    #    Use math.sqrt(d) for the scaling.
    #    scores = ...

    # 3. weights = softmax over the KEYS so each query's weights sum to 1.
    #    SLIP: dim=-1 (the last axis = keys). Use torch.softmax(scores, dim=...).
    #    weights = ...

    # 4. output = weighted sum of values = weights @ V   -> (n, d_v)
    #    output = ...

    # return output, weights
    d = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d)
    weights = torch.softmax(scores, dim=-1)  # Across the keys - along the queries
    output = weights @ V

    return output, weights


if __name__ == "__main__":
    # Quick smoke test you can run by hand: `python -m miniesm.attention`
    # (Uses the river/bank/money example from your Obsidian note.)
    Q = torch.tensor([[2.0, 1.0], [3.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
    K = torch.tensor([[3.0, 0.0], [1.0, 1.0], [0.0, 3.0]], dtype=torch.float64)
    V = torch.tensor([[10.0, 0.0], [5.0, 5.0], [0.0, 10.0]], dtype=torch.float64)
    out, w = scaled_dot_product_attention(Q, K, V)
    print("weights (rows should sum to 1):\n", w)
    print("row sums:", w.sum(dim=-1))
    print("output (bank row should be ~[9.72, 0.28]):\n", out)
