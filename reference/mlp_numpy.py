"""Reference 2-layer MLP in pure NumPy: forward + hand-derived backward.

This is the *oracle* for the whole project: a gradient we compute by hand and
prove correct with finite differences. Later days check torch/autograd against
patterns established here.

Network (single example, no batch):

    x  --Linear(W1,b1)-->  z1  --tanh-->  a1  --Linear(W2,b2)-->  yhat (also called z2)
    L = 0.5 * sum((yhat - y)**2)

Convention (memorise this — it fixes every transpose in the backward pass):
    x    : shape (n_in,)          a column of inputs
    W1   : shape (n_hid, n_in)    so  z1 = W1 @ x + b1   has shape (n_hid,)
    b1   : shape (n_hid,)
    W2   : shape (n_out, n_hid)   so  z2 = W2 @ a1 + b2  has shape (n_out,)
    b2   : shape (n_out,)
    yhat : shape (n_out,)         no activation on the output layer

We use tanh (not ReLU): tanh is smooth everywhere, so finite differences don't
trip over a kink at 0. Useful fact you'll need in backward:
    d/dz tanh(z) = 1 - tanh(z)**2
"""

from __future__ import annotations

import numpy as np


def init_params(rng: np.random.Generator, n_in: int, n_hid: int, n_out: int) -> dict:
    """Return a dict of small random parameters: {"W1","b1","W2","b2"}.

    Use rng.standard_normal(shape) * 0.1 for weights so activations stay in
    tanh's linear-ish region; zeros for biases is fine.
    """
    # TODO(you): build and return the params dict with the shapes above.
    return {
        "W1": rng.standard_normal((n_hid, n_in)) * 0.1,
        "b1": np.zeros(n_hid),
        "W2": rng.standard_normal((n_out, n_hid)) * 0.1,
        "b2": np.zeros(n_out),
    }


def forward(params: dict, x: np.ndarray, y: np.ndarray) -> tuple[float, dict]:
    """Run the forward pass. Return (L, cache).

    `cache` should stash every intermediate the backward pass needs so you
    don't recompute it: at least z1, a1, z2, yhat. (Caching activations for
    the backward pass is exactly what torch does under the hood.)
    """
    # TODO(you): compute z1, a1, z2, yhat, then the scalar loss L.
    #   z1 = ...        # (n_hid,)
    #   a1 = ...        # (n_hid,)  tanh
    #   z2 = ...        # (n_out,)
    #   yhat = ...      # (n_out,)
    #   L  = ...        # scalar
    # cache = {...}
    # return L, cache
    z1 = params["W1"] @ x + params["b1"]
    a1 = np.tanh(z1)
    z2 = params["W2"] @ a1 + params["b2"]
    yhat = z2
    L = 0.5 * np.sum((yhat - y) ** 2)
    cache = {"z1": z1, "a1": a1, "z2": z2, "yhat": yhat}
    return L, cache


def backward(params: dict, cache: dict, x: np.ndarray, y: np.ndarray) -> dict:
    """Return grads {"W1","b1","W2","b2"}, each the same shape as its param.

    We'll DERIVE these together (matrix chain rule is the new skill today).
    The order you'll compute them, walking the graph backwards:
        dyhat -> dz2 -> {dW2, db2} -> da1 -> dz1 -> {dW1, db1}
    Leave the formulas blank until we've derived them on paper.
    """
    # TODO(you, after we derive it): fill in each line.
    dyhat = cache["yhat"] - y  # (n_out,)
    dz2 = dyhat  # (n_out,)
    db2 = dz2  # (n_out,)
    dW2 = np.outer(dz2, cache["a1"])  # (n_out, n_hid)
    da1 = params["W2"].T @ dz2  # (n_hid,)
    dz1 = da1 * (1 - cache["a1"] ** 2)  # (n_hid,) — upstream × tanh' local slope
    db1 = dz1  # (n_hid,)
    dW1 = np.outer(dz1, x)  # (n_hid, n_in)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def finite_diff_check(params: dict, x: np.ndarray, y: np.ndarray, eps: float = 1e-5) -> float:
    """Return the max abs difference between analytic and numeric gradients.

    For every parameter, for every entry in it: nudge that ONE entry by +eps and
    -eps, run forward each time, and estimate the slope with the central
    difference   (L(+eps) - L(-eps)) / (2*eps).  Compare to backward()'s value.
    Return the largest absolute mismatch across all entries.
    """
    # TODO(you): implement the loop. Hint for iterating entries of an array:
    #   np.ndindex(arr.shape) yields every index tuple.
    _, cache = forward(params, x, y)
    grads = backward(params, cache, x, y)

    max_diff = 0
    for name in params:
        for idx in np.ndindex(params[name].shape):
            orig = params[name][idx]

            params[name][idx] = orig + eps
            L_plus, _ = forward(params, x, y)
            params[name][idx] = orig - eps
            L_minus, _ = forward(params, x, y)
            params[name][idx] = orig

            numeric = (L_plus - L_minus) / (2 * eps)
            diff = abs(numeric - grads[name][idx])
            if diff > max_diff:
                max_diff = diff

    return max_diff


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    params = init_params(rng, n_in=4, n_hid=5, n_out=3)
    x = rng.standard_normal(4)
    y = rng.standard_normal(3)
    max_err = finite_diff_check(params, x, y)
    print(f"max |analytic - numeric| = {max_err:.2e}")
    assert max_err < 1e-5, "gradient check FAILED"
    print("gradient check passed ✓")
