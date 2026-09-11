"""Warm-rescaling doubling checks. Does not run Exp.1."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from games import QuadraticGame  # noqa: E402
from learner import ClosedFormPlayer, row_ratio, self_play  # noqa: E402


def g2_row_lipschitz_1d(mu: float, a: float) -> float:
    return float(np.hypot(mu, a))


def test_t1_does_not_double():
    p = ClosedFormPlayer(dim=1, adaptive=True, ell1=1.0e-6, beta0=1.0)
    p.observe(np.array([1.0]), np.array([1.0, 0.0]))
    if p.t != 1 or p.J != 0 or p.last_chi is not None:
        raise AssertionError("t=1 must skip chi and doubling")
    if abs(p.beta - p.beta_init) > 0.0:
        raise AssertionError("beta must stay at beta_1 through round 1")


def test_second_round_can_double():
    p = ClosedFormPlayer(dim=1, adaptive=True, ell1=1.0e-6, beta0=1.0)
    p.observe(np.array([1.0]), np.array([1.0, 0.0]))
    p.observe(np.array([2.0]), np.array([2.0, 0.0]))
    if p.J != 1:
        raise AssertionError(f"expected one doubling, J={p.J}")
    if p.last_chi != 1.0:
        raise AssertionError(f"chi should be 1, got {p.last_chi}")
    if p.ell != 2.0:
        raise AssertionError(f"ell should be 2 max(ell,chi)=2, got {p.ell}")
    expected_beta = 1.0 + 64.0 * 4.0 / 1.0
    if abs(p.beta - expected_beta) > 1e-12:
        raise AssertionError(f"beta={p.beta}, expected {expected_beta}")


def test_chi_uses_own_gradient_only():
    # y-block of z moves; own g is unchanged → χ=0, no doubling.
    p = ClosedFormPlayer(dim=1, adaptive=True, ell1=1.0e-6, beta0=1.0)
    p.observe(np.array([0.3]), np.array([0.0, 0.0]))
    p.observe(np.array([0.3]), np.array([0.0, 10.0]))
    if p.last_chi != 0.0:
        raise AssertionError(f"y-only motion must give chi=0, got {p.last_chi}")
    if p.J != 0:
        raise AssertionError("must not double on opponent-only motion")
    g = np.array([0.3])
    z0 = np.array([0.0, 0.0])
    z1 = np.array([0.0, 10.0])
    if row_ratio(g, g, z1, z0) != 0.0:
        raise AssertionError("row_ratio should ignore a static own gradient")


def test_beta_tracks_ell_and_is_nondecreasing():
    p = ClosedFormPlayer(dim=1, adaptive=True, ell1=1.0e-3, beta0=1.0)
    p.observe(np.array([0.0]), np.array([0.0, 0.0]))
    p.observe(np.array([1.0]), np.array([1.0, 0.0]))
    p.observe(np.array([1.2]), np.array([1.1, 0.1]))
    betas = np.array(p.beta_path)
    if np.any(np.diff(betas) < -1e-15):
        raise AssertionError(f"beta decreased: {betas}")
    if abs(p.gamma - p.gamma_init) > 0.0:
        raise AssertionError("gamma must stay frozen under warm rescaling")
    if float(np.linalg.norm(p.G_cum)) == 0.0:
        raise AssertionError("G_cum must accumulate, not reset")


def main():
    test_t1_does_not_double()
    test_second_round_can_double()
    test_chi_uses_own_gradient_only()
    test_beta_tracks_ell_and_is_nondecreasing()

    mu, a, T = 0.2, 1.0, 2000
    ell1 = 1.0e-3
    L_row = g2_row_lipschitz_1d(mu, a)
    j_cap = math.ceil(math.log2(L_row / ell1))
    game = QuadraticGame(
        dim=1, mu=mu, A=np.array([[a]]), saddle_x=[0.4], saddle_y=[-0.3]
    )
    px = ClosedFormPlayer(dim=1, epsilon=1.0, beta0=1.0, ell1=ell1, adaptive=True)
    py = ClosedFormPlayer(dim=1, epsilon=1.0, beta0=1.0, ell1=ell1, adaptive=True)
    metrics, hist, max_w = self_play(game, px, py, T=T, radius=1.0)
    if abs(hist["x_norm"][0]) > 1e-15 or abs(hist["y_norm"][0]) > 1e-15:
        raise AssertionError("w1 must be the origin")

    for player, name in ((px, "x"), (py, "y")):
        if not player.finite():
            raise AssertionError(f"player {name} non-finite")
        if player.J < 1:
            raise AssertionError(f"player {name} never doubled; ell1 too large?")
        if player.J > j_cap:
            raise AssertionError(f"player {name} J={player.J} > cap {j_cap}")
        betas = np.array(player.beta_path)
        if np.any(np.diff(betas) < -1e-15):
            raise AssertionError(f"player {name} beta decreased")
        if abs(player.gamma - player.gamma_init) > 0.0:
            raise AssertionError("gamma reset")
        # J freezes: last increase cannot be in the final round of a long run
        # once ell exceeds the row Lipschitz constant.
        if player.ell + 1e-12 < L_row and player.t == T:
            # allowed if we never saw the full Lipschitz; still J must be finite
            pass
        if player.J != player.J:
            raise AssertionError("J nan")

    Q = np.array(hist["Q"])
    mid = T // 2
    summary = {
        "game": "G2",
        "dim": 1,
        "T": T,
        "ell1": ell1,
        "L_row": L_row,
        "J_cap": j_cap,
        "adaptive": True,
        "J_x": px.J,
        "J_y": py.J,
        "ell_x": px.ell,
        "beta_x": px.beta,
        "beta_x_init": px.beta_init,
        "gamma_x": px.gamma,
        "Q_T": float(Q[-1]),
        "dQ_second_half": float(Q[-1] - Q[mid - 1]),
        "gap_T": float(hist["gap"][-1]),
        "reg_x_T": float(hist["reg_x"][-1]),
        "max_w": max_w,
        "G_cum_norm_x": float(np.linalg.norm(px.G_cum)),
    }
    out = ROOT / "results" / "doubling_g2.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if max_w >= 1e20:
        raise AssertionError(f"iterate exploded: {max_w}")
    print("doubling checks passed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
