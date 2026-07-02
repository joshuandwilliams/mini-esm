"""Tests for scaled dot-product attention.

Four tests, in rising order of interest:
    1. shape                — output has the right shape          (sanity)
    2. rows sum to 1        — a PROPERTY test (always true)
    3. identity-value       — a KNOWN-ANSWER test (we rig the inputs)
    4. gradcheck            — autograd's backward matches finite differences

Use float64 everywhere: gradcheck needs the precision (float32 fails spuriously,
the same catastrophic-cancellation issue as Day 1's too-small epsilon).
"""

from __future__ import annotations

import torch

from miniesm.attention import scaled_dot_product_attention


def _example() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Small fixed inputs to reuse across tests (n=3, d=2, float64)."""
    Q = torch.tensor([[2.0, 1.0], [3.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
    K = torch.tensor([[3.0, 0.0], [1.0, 1.0], [0.0, 3.0]], dtype=torch.float64)
    V = torch.tensor([[10.0, 0.0], [5.0, 5.0], [0.0, 10.0]], dtype=torch.float64)
    return Q, K, V


def test_output_shape() -> None:
    """output should have the same shape as V: (n, d_v)."""
    # TODO(you):
    #   1. Q, K, V = _example()
    #   2. output, _ = scaled_dot_product_attention(Q, K, V)
    #   3. assert output.shape == V.shape
    Q, K, V = _example()
    output, _ = scaled_dot_product_attention(Q, K, V)
    assert output.shape == V.shape


def test_weights_rows_sum_to_one() -> None:
    """Each query's attention weights form a distribution: every row sums to 1."""
    # TODO(you):
    #   1. get weights from the function (the second return value)
    #   2. row_sums = weights.sum(dim=-1)          # shape (n,)
    #   3. assert torch.allclose(row_sums, torch.ones_like(row_sums))
    Q, K, V = _example()
    _, weights = scaled_dot_product_attention(Q, K, V)
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums))


def test_identity_value_recovery() -> None:
    """If query i attends almost entirely to key i, output ~= V (unchanged).

    Construction (your idea): make the keys orthogonal one-hot directions and the
    queries the SAME directions but scaled up big, so each query points straight
    at its own key and is perpendicular to the others. Then weights ~= identity
    and output ~= V.
    """
    # TODO(you):
    #   n, d = 3, 3
    #   K = torch.eye(n, dtype=torch.float64)          # each key is a distinct axis
    #   Q = 50.0 * torch.eye(n, dtype=torch.float64)   # big => softmax ~ one-hot
    #   V = torch.tensor([[1., 2., 3.],
    #                     [4., 5., 6.],
    #                     [7., 8., 9.]], dtype=torch.float64)
    #   output, _ = scaled_dot_product_attention(Q, K, V)
    #   assert torch.allclose(output, V, atol=1e-6)    # recovered V
    n, _ = 3, 3
    K = torch.eye(n, dtype=torch.float64)
    Q = 50.0 * torch.eye(n, dtype=torch.float64)
    V = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=torch.float64)
    output, _ = scaled_dot_product_attention(Q, K, V)
    assert torch.allclose(output, V, atol=1e-6)


def test_gradcheck_matches_autograd() -> None:
    """autograd's backward pass matches finite differences (Day 1's oracle, one call).

    torch.autograd.gradcheck perturbs each input entry, recomputes the forward,
    and checks the numeric gradient against autograd's. It needs:
      - float64 inputs
      - requires_grad=True on the inputs it should differentiate
      - a function returning a single tensor (not the (output, weights) tuple)
    """
    # TODO(you):
    #   1. Make small random float64 inputs with requires_grad=True, e.g.:
    #        torch.manual_seed(0)
    #        Q = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    #        K = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    #        V = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    #   2. gradcheck wants a fn that returns ONE tensor. Wrap so it returns only
    #      the output (drop the weights):
    #        fn = lambda q, k, v: scaled_dot_product_attention(q, k, v)[0]
    #   3. assert torch.autograd.gradcheck(fn, (Q, K, V))
    torch.manual_seed(0)
    Q = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    K = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    V = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)

    def fn(q, k, v):
        return scaled_dot_product_attention(q, k, v)[0]

    assert torch.autograd.gradcheck(fn, (Q, K, V))
