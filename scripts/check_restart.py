"""Cold-restart checks. Does not run the T=2e4 Exp.3 figure."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from games import QuadraticGame  # noqa: E402
from learner import ClosedFormPlayer, run_loop  # noqa: E402

_DTYPE = np.float64


def test_doubling_wipes_state():
    p = ClosedFormPlayer(dim=1, adaptive=True, ell1=1.0e-6, beta0=1.0, restart=True)
    p.observe(np.array([1.0]), np.array([1.0, 0.0]))
    if p.n_restarts != 0:
        raise AssertionError("t=1 must not restart")
    p.observe(np.array([2.0]), np.array([2.0, 0.0]))
    if p.J != 1 or p.n_restarts != 1:
        raise AssertionError(f"expected one reset, J={p.J}, n={p.n_restarts}")
    if float(np.linalg.norm(p.action)) != 0.0:
        raise AssertionError("restart must return to the origin")
    if float(np.linalg.norm(p.G_cum)) != 0.0:
        raise AssertionError("G_cum must be cleared")
    if abs(p.gamma - p.epsilon * p.beta) > 1e-12:
        raise AssertionError("restart gamma must equal epsilon * beta_+")
    if p.g_prev is not None or p.z_prev is not None:
        raise AssertionError("epoch must skip chi on the next round")


def test_warm_does_not_wipe():
    p = ClosedFormPlayer(dim=1, adaptive=True, ell1=1.0e-6, beta0=1.0, restart=False)
    p.observe(np.array([1.0]), np.array([1.0, 0.0]))
    p.observe(np.array([2.0]), np.array([2.0, 0.0]))
    if p.n_restarts != 0:
        raise AssertionError("warm restarted")
    if abs(p.gamma - p.gamma_init) > 0.0:
        raise AssertionError("warm gamma moved")
    if float(np.linalg.norm(p.G_cum)) == 0.0:
        raise AssertionError("warm G_cum must keep the prefix")


def test_origin_jump_vs_const():
    dim = 4
    T = 40
    game = QuadraticGame(dim=dim, mu=0.2)
    e1 = np.zeros(dim, dtype=_DTYPE)
    e1[0] = 1.0

    def y_policy(_t, _x):
        return e1.copy()

    kwargs = dict(epsilon=1.0, beta0=1.0, ell1=1.0e-3, adaptive=True)
    pw = ClosedFormPlayer(dim, restart=False, **kwargs)
    pr = ClosedFormPlayer(dim, restart=True, **kwargs)
    _, hw, _ = run_loop(game, pw, T, y_policy=y_policy, observe_y=lambda _t: False)
    _, hr, _ = run_loop(game, pr, T, y_policy=y_policy, observe_y=lambda _t: False)
    if hw["J_x"][-1] < 1 or hr["J_x"][-1] < 1:
        raise AssertionError("expected a doubling against y=e1")
    if pr.n_restarts != hr["J_x"][-1]:
        raise AssertionError("each doubling should trigger one cold reset")
    if pw.n_restarts != 0:
        raise AssertionError("warm restarted")
    # t=3 is the first action after the t=2 doubling.
    if abs(hr["x_norm"][2]) > 1e-15:
        raise AssertionError("restart action at t=3 must be the origin")
    if hw["x_norm"][2] <= 0.0:
        raise AssertionError("warm should not jump to the origin at t=3")
    if abs(pw.gamma - pw.gamma_init) > 0.0:
        raise AssertionError("warm gamma moved")


def main():
    test_doubling_wipes_state()
    test_warm_does_not_wipe()
    test_origin_jump_vs_const()
    print("restart checks passed")


if __name__ == "__main__":
    main()
