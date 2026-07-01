"""Test that the reference MLP's hand-derived gradient matches finite differences.

This is the grad-check from reference/mlp_numpy.py, promoted to a real pytest test
so CI re-proves the oracle correct on every push.
"""

from __future__ import annotations

import numpy as np

from reference.mlp_numpy import finite_diff_check, init_params


def test_reference_gradient_matches_finite_difference() -> None:
    """Analytic gradient should agree with the numeric one to < 1e-5."""
    # TODO(you):
    #   1. Make a *seeded* RNG so the test is deterministic:  np.random.default_rng(0)
    #   2. Build params with init_params(rng, n_in=4, n_hid=5, n_out=3)
    #   3. Make random x (shape 4) and y (shape 3) from the same rng.
    #   4. Call finite_diff_check(...) to get the max error.
    #   5. assert it is < 1e-5   (optionally with a message explaining the failure)
    rng = np.random.default_rng(0)
    params = init_params(rng, n_in=4, n_hid=5, n_out=3)
    x = rng.standard_normal(4)
    y = rng.standard_normal(3)
    max_err = finite_diff_check(params, x, y)
    assert max_err < 1e-5, f"grad check failed: {max_err:.2e}"
