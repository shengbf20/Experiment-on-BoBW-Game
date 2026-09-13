"""Assertions for games.py and metrics.py. No learner."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from games import (  # noqa: E402
    BilinearGame,
    QuadraticGame,
    paper_saddle,
)
from metrics import RunningMetrics  # noqa: E402


def _assert_close(a, b, tag: str, atol: float = 1e-10):
    if not np.allclose(a, b, atol=atol, rtol=0.0):
        raise AssertionError(f"{tag}: {a} vs {b}")


def test_saddles():
    g1 = BilinearGame(dim=10)
    g2 = QuadraticGame(dim=10, mu=0.2)
    for game in (g1, g2):
        xs, ys = game.saddle()
        gx, gy = game.feedback(xs, ys)
        _assert_close(gx, 0.0, f"{game.name} F^x(saddle)")
        _assert_close(gy, 0.0, f"{game.name} F^y(saddle)")


def test_incremental_matches_batch():
    rng = np.random.default_rng(1)
    game = QuadraticGame(dim=4, mu=0.2)
    xs = rng.normal(size=(15, 4))
    ys = rng.normal(size=(15, 4))
    m = RunningMetrics(game, radius=1.0)
    for x, y in zip(xs, ys):
        m.step(x, y)
    u, v = m.u_x, m.u_y
    reg_x = sum(game.phi(x, y) - game.phi(u, y) for x, y in zip(xs, ys))
    reg_y = sum(game.phi(x, v) - game.phi(x, y) for x, y in zip(xs, ys))
    Q = sum(np.sum(np.concatenate([xs[t] - xs[t - 1], ys[t] - ys[t - 1]]) ** 2) for t in range(1, 15))
    Vx = sum(
        np.sum((game.grad_x_phi(u, ys[t]) - game.grad_x_phi(u, ys[t - 1])) ** 2)
        for t in range(1, 15)
    )
    _assert_close(m.reg_x, reg_x, "batch reg_x", atol=1e-9)
    _assert_close(m.reg_y, reg_y, "batch reg_y", atol=1e-9)
    _assert_close(m.Q, Q, "batch Q", atol=1e-9)
    _assert_close(m.V_x, Vx, "batch V_x", atol=1e-9)
    if m.reg_x > m.lin_x + 1e-8:
        raise AssertionError("convexity: Reg^x should be <= LinReg^x")


def test_g1_gap_and_linreg_identity():
    game = BilinearGame(dim=1, A=np.array([[2.0]]))
    x, y = np.array([0.3]), np.array([-0.4])
    gap = game.restricted_gap(x, y, radius=1.0)
    _assert_close(gap, 1.0 * (abs(2.0 * 0.3) + abs(2.0 * (-0.4))), "G1 1d gap")
    xs = [np.array([0.2]), np.array([-0.1]), np.array([0.4])]
    ys = [np.array([0.5]), np.array([0.0]), np.array([-0.3])]
    m = RunningMetrics(game)
    for x, y in zip(xs, ys):
        m.step(x, y)
    _assert_close(m.reg_x, m.lin_x, "G1 bilinear Reg=LinReg x")
    _assert_close(m.reg_y, m.lin_y, "G1 bilinear Reg=LinReg y")


def test_g2_gap_unconstrained_and_boundary():
    game = QuadraticGame(dim=1, mu=1.0, A=np.array([[1.0]]))
    # Unconstrained v* = x/μ = 0.1 is inside R=1.
    x, y = np.array([0.1]), np.array([-0.2])
    gap = game.restricted_gap(x, y, radius=1.0)
    # Φ = 0.5 x² + x y − 0.5 y²; max_v on ℝ: v=x, value 0.5 x² + 0.5 x² = x²
    # wait: max_v (xy_term): x v - 0.5 v² at v=x gives 0.5 x², plus 0.5 x² from first term = x²
    # inf_u: 0.5 u² + u y − 0.5 y² at u=-y gives -0.5 y² − 0.5 y² = -y²
    # Gap = x² - (-y²) = x² + y²
    _assert_close(gap, float(x[0] ** 2 + y[0] ** 2), "G2 interior gap")
    # Boundary: x=3, μ=1, R=1 ⇒ unconstrained v*=3 outside.
    x2, y2 = np.array([3.0]), np.array([0.0])
    gap_b = game.restricted_gap(x2, y2, radius=1.0)
    # sup: 0.5*9 + (1*3 - 0.5*1) = 4.5 + 3 - 0.5 = 7.0
    # inf at y=0: unconstrained u=0 inside, inf = 0
    _assert_close(gap_b, 7.0, "G2 boundary gap")


def test_shifted_saddle_and_origin_gradient():
    dim = 10
    sx, sy = paper_saddle(dim)
    for game in (
        BilinearGame(dim=dim, saddle_x=sx, saddle_y=sy),
        QuadraticGame(dim=dim, mu=0.2, saddle_x=sx, saddle_y=sy),
    ):
        ax, ay = game.saddle()
        _assert_close(ax, sx, f"{game.name} saddle x")
        _assert_close(ay, sy, f"{game.name} saddle y")
        gx, gy = game.feedback(ax, ay)
        _assert_close(gx, 0.0, f"{game.name} F^x(shifted saddle)")
        _assert_close(gy, 0.0, f"{game.name} F^y(shifted saddle)")
        z = np.zeros(dim)
        gx0, gy0 = game.feedback(z, z)
        if np.allclose(gx0, 0.0) and np.allclose(gy0, 0.0):
            raise AssertionError(f"{game.name}: origin is still a rest point")
        m = RunningMetrics(game, radius=1.0)
        _assert_close(m.u_x, sx, f"{game.name} metrics u_x")
        _assert_close(m.u_y, sy, f"{game.name} metrics u_y")


def test_g2_induced_minimizer_vs_const():
    dim = 10
    sx, sy = paper_saddle(dim)
    game = QuadraticGame(dim=dim, mu=0.2, saddle_x=sx, saddle_y=sy)
    e1 = np.zeros(dim)
    e1[0] = 1.0
    x_star = game.induced_minimizer_x(e1)
    expected = sx - (game.A @ (e1 - sy)) / game.mu
    _assert_close(x_star, expected, "x* formula")
    gx = game.grad_x_phi(x_star, e1)
    _assert_close(gx, 0.0, "∇xΦ(x*, e1)")
    m = RunningMetrics(game, u_x=x_star, radius=1.0)
    for _ in range(20):
        m.step(np.zeros(dim), e1)
    if m.V_x != 0.0:
        raise AssertionError(f"V_x(x*) must be 0 against y≡e1, got {m.V_x}")
    _assert_close(m.u_x, x_star, "metrics u_x is x*")


def test_shifted_g1_gap_and_gaussian_norm():
    sx, sy = paper_saddle(2)
    game = BilinearGame(dim=2, A=np.eye(2), saddle_x=sx, saddle_y=sy)
    x = sx + np.array([0.3, 0.0])
    y = sy + np.array([0.0, -0.4])
    gap = game.restricted_gap(x, y, radius=1.0)
    _assert_close(gap, abs(0.3) + abs(-0.4), "shifted G1 gap")
    g = BilinearGame.gaussian(dim=4, seed=0, normalize=True)
    op = float(np.linalg.norm(g.A, 2))
    if abs(op - 1.0) > 1e-12:
        raise AssertionError(f"gaussian A should have ||A||_2=1, got {op}")


def main():
    test_saddles()
    test_incremental_matches_batch()
    test_g1_gap_and_linreg_identity()
    test_g2_gap_unconstrained_and_boundary()
    test_shifted_saddle_and_origin_gradient()
    test_g2_induced_minimizer_vs_const()
    test_shifted_g1_gap_and_gaussian_norm()
    print("games + metrics checks passed")


if __name__ == "__main__":
    main()
