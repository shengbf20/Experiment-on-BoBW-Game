"""Invariants for the separation-example game. Does not overwrite paper runs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from games import _vec  # noqa: E402
from separation import (  # noqa: E402
    DELTA0,
    E0,
    ETA,
    G_LOWER,
    G_UPPER,
    R_STAR,
    SeparationGame,
    THEORY_LATE_DG2,
    U_STAR,
    XI_STAR,
    comparator_local_variation,
    generate_opponent_sequence,
    grad_x_scalar,
    logcosh,
    q_prime,
    replay_frozen_sequence,
    run_part_a,
)

_DTYPE = np.float64


def _load_cfg():
    import yaml

    with (ROOT / "configs" / "default.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _assert_close(a, b, tag: str, atol: float = 1e-10, rtol: float = 0.0):
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(f"{tag}: {a} vs {b}")


def test_constants():
    if not (R_STAR < 0.0 and abs(R_STAR) < 1.0):
        raise AssertionError(f"r_star should lie in (-1,0), got {R_STAR}")
    _assert_close(float(np.tanh(XI_STAR)), R_STAR, "tanh xi_star", atol=1e-12)
    if DELTA0 <= 0.0:
        raise AssertionError(f"delta0 must be positive, got {DELTA0}")
    _assert_close(DELTA0, 0.0411083, "delta0 paper value", atol=5e-7)
    if ETA <= 0.0 or E0 <= 0.0:
        raise AssertionError("eta and e0 must be positive")
    _assert_close(E0, ETA / 2.0, "e0 = eta/2")
    if THEORY_LATE_DG2 <= 0.0:
        raise AssertionError("theory late jump must be positive")


def test_logcosh():
    xs = np.linspace(-20.0, 20.0, 41)
    naive = np.log(np.cosh(np.clip(xs, -20.0, 20.0)))
    _assert_close(logcosh(xs), naive, "logcosh", atol=1e-10)


def test_gradients_fd():
    rng = np.random.default_rng(0)
    game = SeparationGame()
    for _ in range(12):
        x = rng.normal()
        y = rng.normal(size=2)
        gx = game.grad_x_phi(x, y)
        gy = game.grad_y_phi(x, y)
        eps = 1e-6
        fd_x = (game.phi(x + eps, y) - game.phi(x - eps, y)) / (2.0 * eps)
        _assert_close(float(gx[0]), fd_x, "fd dphi/dx", atol=1e-7)
        fd_y = np.zeros(2, dtype=_DTYPE)
        for j in range(2):
            yp, ym = y.copy(), y.copy()
            yp[j] += eps
            ym[j] -= eps
            fd_y[j] = (game.phi(x, yp) - game.phi(x, ym)) / (2.0 * eps)
        _assert_close(gy, fd_y, "fd dphi/dy", atol=1e-7)
    half = float(np.arctanh(0.5))
    x0, y0 = 0.0, np.array([half, half], dtype=_DTYPE)
    g0 = float(game.grad_x_phi(x0, y0)[0])
    p = float(np.tanh(-1.0))
    expected = grad_x_scalar(x0, 0.5, 0.5)
    _assert_close(g0, expected, "g(0, +1/2, +1/2)")
    if abs(g0) <= G_LOWER:
        raise AssertionError(f"|g1| should exceed {G_LOWER}, got {abs(g0)}")
    if abs(g0) > G_UPPER:
        raise AssertionError(f"|g| exceeded analytic bound {G_UPPER}")


def test_strict_convex_concave():
    rng = np.random.default_rng(1)
    game = SeparationGame()
    eps = 1e-5
    for _ in range(8):
        x = rng.normal()
        y = rng.normal(size=2)
        d2x = (
            game.phi(x + eps, y) - 2.0 * game.phi(x, y) + game.phi(x - eps, y)
        ) / (eps * eps)
        if d2x <= 0.0:
            raise AssertionError(f"phi not strictly convex in x: {d2x}")
        for j in range(2):
            yp, ym = y.copy(), y.copy()
            yp[j] += eps
            ym[j] -= eps
            d2y = (game.phi(x, yp) - 2.0 * game.phi(x, y) + game.phi(x, ym)) / (eps * eps)
            if d2y >= 0.0:
                raise AssertionError(f"phi not strictly concave in y[{j}]: {d2y}")


def test_comparator_variation_and_best_fixed():
    T = 20
    cfg = _load_cfg()
    generated = generate_opponent_sequence(T, cfg)
    game = generated["game"]
    V = comparator_local_variation(game, generated["y"])
    _assert_close(V, 1.0, "V_T(u*)", atol=1e-12)
    n = T // 2
    v1 = np.array([0.5] * n + [-0.5] * n)
    _assert_close(float(np.sum(v1)), 0.0, "sum v1")
    u = np.array([U_STAR], dtype=_DTYPE)
    dsum = 0.0
    for y in generated["y"]:
        dsum += float(game.grad_x_phi(u, y)[0])
    _assert_close(dsum, 0.0, "first-order condition at u*", atol=1e-12)
    # Unique minimizer of the path-sum: nearby points have larger cumulative loss.
    total_u = sum(game.phi(u, y) for y in generated["y"])
    for dx in (-0.2, -0.05, 0.05, 0.2):
        total = sum(game.phi(u + dx, y) for y in generated["y"])
        if total <= total_u:
            raise AssertionError(f"u* is not a strict minimizer at dx={dx}")


def test_generate_replay_and_jump():
    cfg = _load_cfg()
    out = run_part_a(16, cfg)
    if not out["replay_match"]:
        raise AssertionError("replay_match is false")
    if abs(out["x1"]) > 1e-15:
        raise AssertionError(f"x1 must be 0, got {out['x1']}")
    if out["G_T"] > G_UPPER + 1e-12:
        raise AssertionError(f"G_T={out['G_T']} > {G_UPPER}")
    if out["G_T"] <= G_LOWER:
        raise AssertionError(f"G_T={out['G_T']} <= {G_LOWER}")
    _assert_close(out["V_T"], 1.0, "run_part_a V_T", atol=1e-12)
    game = SeparationGame()
    generated = generate_opponent_sequence(16, cfg, game=game)
    replayed = replay_frozen_sequence(generated["y"], cfg, game=game)
    _assert_close(generated["x"], replayed["x"], "x generate vs replay", atol=1e-12)
    _assert_close(generated["g"], replayed["g"], "g generate vs replay", atol=1e-12)
    for t in range(1, 16):
        floor = E0 * abs(q_prime(float(replayed["x"][t]) - 1.0))
        jump = abs(float(replayed["g"][t] - replayed["g"][t - 1]))
        if jump + 1e-12 < floor:
            raise AssertionError(f"jump inequality failed at t={t+1}")
    # Frozen y is open-loop: coordinates only take two tanh values.
    v = np.tanh(generated["y"])
    if not np.all(np.isclose(np.abs(v), 0.5, atol=1e-12)):
        raise AssertionError("opponent tanh coordinates must be ±1/2")


def test_feedback_sign():
    game = SeparationGame()
    x = np.array([0.3], dtype=_DTYPE)
    y = np.array([0.2, -0.4], dtype=_DTYPE)
    gx, gy = game.feedback(x, y)
    _assert_close(gx, game.grad_x_phi(x, y), "feedback gx")
    _assert_close(gy, -game.grad_y_phi(x, y), "feedback gy = -grad_y phi")
    _assert_close(_vec(game.comparator_x(), 1), np.array([1.0]), "comparator")


def main():
    test_constants()
    test_logcosh()
    test_gradients_fd()
    test_strict_convex_concave()
    test_comparator_variation_and_best_fixed()
    test_generate_replay_and_jump()
    test_feedback_sign()
    print("separation checks passed")


if __name__ == "__main__":
    main()
