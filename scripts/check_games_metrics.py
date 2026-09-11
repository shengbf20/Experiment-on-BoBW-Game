"""Assertions for games.py and metrics.py. No learner."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from games import BilinearGame, QuadraticGame, SeparationGame  # noqa: E402
from metrics import RunningMetrics  # noqa: E402


def _assert_close(a, b, tag: str, atol: float = 1e-10):
    if not np.allclose(a, b, atol=atol, rtol=0.0):
        raise AssertionError(f"{tag}: {a} vs {b}")


def test_saddles_and_g3_stationarity():
    g1 = BilinearGame(dim=10)
    g2 = QuadraticGame(dim=10, mu=0.2)
    g3 = SeparationGame()
    for game in (g1, g2, g3):
        xs, ys = game.saddle()
        gx, gy = game.feedback(xs, ys)
        _assert_close(gx, 0.0, f"{game.name} F^x(saddle)")
        _assert_close(gy, 0.0, f"{game.name} F^y(saddle)")
    u = g3.comparator_x()
    gx_star = g3.grad_x_phi(u, np.array([1.0]))
    _assert_close(gx_star, 0.0, "G3 ∇xΦ(u*, 1)")


def test_g3_constant_opponent_metrics():
    game = SeparationGame()
    m = RunningMetrics(game, radius=1.0)
    x = np.zeros(1)
    y = np.array([1.0])
    for _ in range(50):
        gx, gy = game.feedback(x, y)
        m.step(x, y, gx, gy)
    if m.V_x != 0.0:
        raise AssertionError(f"G3 V_x(u*) should be 0, got {m.V_x}")
    if not (1.0 - 1e-12 <= m.G_x <= 3.0):
        raise AssertionError(f"G3 G_x on x=0,y=1 should be 1, got {m.G_x}")
    rng = np.random.default_rng(0)
    for _ in range(20):
        xx = rng.normal(size=1)
        gx, _ = game.feedback(xx, y)
        if float(np.abs(gx[0])) > 3.0 + 1e-12:
            raise AssertionError(f"|g^x| > 3 at x={xx}")


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


def main():
    test_saddles_and_g3_stationarity()
    test_g3_constant_opponent_metrics()
    test_incremental_matches_batch()
    test_g1_gap_and_linreg_identity()
    test_g2_gap_unconstrained_and_boundary()
    print("games + metrics checks passed")


if __name__ == "__main__":
    main()
