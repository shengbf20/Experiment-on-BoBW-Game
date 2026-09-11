"""Frozen-β closed-form self-play health check (no doubling)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from games import QuadraticGame  # noqa: E402
from learner import ClosedFormPlayer, radial_q, self_play  # noqa: E402


def g2_lipschitz_1d(mu: float, a: float) -> float:
    M = np.array([[mu, a], [-a, mu]], dtype=np.float64)
    return float(np.linalg.norm(M, 2))


def test_radial_q_stable():
    q = radial_q(1.0e8, a=2.0, beta=3.0, alpha=0.1)
    if not np.isfinite(q):
        raise AssertionError("radial_q overflowed on large s")
    q0 = radial_q(0.0, a=2.0, beta=3.0, alpha=0.1)
    if abs(q0) > 1e-12:
        raise AssertionError(f"q(0) should be 0, got {q0}")


def test_w1_origin():
    p = ClosedFormPlayer(dim=1, adaptive=False, beta=5.0)
    if p.t != 0 or np.any(p.action != 0):
        raise AssertionError("t=1 action must be the origin")
    if p.gamma != p.epsilon * p.beta:
        raise AssertionError("gamma = epsilon * beta_1")


def test_origin_is_rest_point(mu: float, a: float, beta: float):
    game = QuadraticGame(dim=1, mu=mu, A=np.array([[a]]))
    px = ClosedFormPlayer(dim=1, epsilon=1.0, adaptive=False, beta=beta)
    py = ClosedFormPlayer(dim=1, epsilon=1.0, adaptive=False, beta=beta)
    metrics, _, _ = self_play(game, px, py, T=8, radius=1.0)
    if metrics.Q != 0.0 or metrics.G_x != 0.0:
        raise AssertionError("saddle at 0 is a rest point of w_1=0 self-play")


def main():
    test_radial_q_stable()
    test_w1_origin()

    mu, a, T = 0.2, 1.0, 2000
    L_F = g2_lipschitz_1d(mu, a)
    beta = 5.0 * L_F
    test_origin_is_rest_point(mu, a, beta)

    sx, sy = 0.4, -0.3
    game = QuadraticGame(dim=1, mu=mu, A=np.array([[a]]), saddle_x=[sx], saddle_y=[sy])
    px = ClosedFormPlayer(dim=1, epsilon=1.0, adaptive=False, beta=beta)
    py = ClosedFormPlayer(dim=1, epsilon=1.0, adaptive=False, beta=beta)
    metrics, hist, max_w = self_play(game, px, py, T=T, radius=1.0)
    if abs(hist["x_norm"][0]) > 1e-15 or abs(hist["y_norm"][0]) > 1e-15:
        raise AssertionError("w1 must be the origin")
    if metrics.Q == 0.0:
        raise AssertionError("shifted saddle must move from w1=0")

    for player, name in ((px, "x"), (py, "y")):
        if not player.finite():
            raise AssertionError(f"player {name} non-finite")
        if player.adaptive or player.J != 0:
            raise AssertionError("frozen-β must not double")
        if abs(player.beta - player.beta_init) > 0.0:
            raise AssertionError("beta moved while adaptive=False")
        if abs(player.gamma - player.gamma_init) > 0.0:
            raise AssertionError("gamma must stay frozen")
        if player.B < 4.0 or player.Vbar + 1e-15 < 4.0 * player.Mhat**2:
            raise AssertionError("clipping invariants")
        if player.t != T:
            raise AssertionError("t should equal T")

    if max_w >= 1e20 or not np.isfinite(max_w):
        raise AssertionError(f"iterate exploded: max||w||={max_w}")

    reg = np.array(hist["reg_x"])
    Q = np.array(hist["Q"])
    gap = np.array(hist["gap"])
    mid = T // 2
    dQ1 = Q[mid - 1] - Q[0]
    dQ2 = Q[-1] - Q[mid - 1]
    dR1 = abs(reg[mid - 1] - reg[0])
    dR2 = abs(reg[-1] - reg[mid - 1])

    summary = {
        "game": "G2",
        "dim": 1,
        "T": T,
        "mu": mu,
        "L_F": L_F,
        "beta": beta,
        "beta_over_LF": beta / L_F,
        "adaptive": False,
        "reg_x_T": float(reg[-1]),
        "reg_y_T": float(hist["reg_y"][-1]),
        "Q_T": float(Q[-1]),
        "gap_T": float(gap[-1]),
        "T_gap_T": float(T * gap[-1]),
        "G_x": metrics.G_x,
        "max_w": max_w,
        "dQ_first_half": float(dQ1),
        "dQ_second_half": float(dQ2),
        "dReg_first_half": float(dR1),
        "dReg_second_half": float(dR2),
        "J_x": px.J,
        "beta_x": px.beta,
        "gamma_x": px.gamma,
        "x_norm_T": float(hist["x_norm"][-1]),
        "y_norm_T": float(hist["y_norm"][-1]),
        "saddle": [sx, sy],
        "init": "origin",
    }
    out = ROOT / "results" / "frozen_beta_g2.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Saturation: second-half movement and regret increments should not dominate.
    if dQ2 > max(1e-9, 2.0 * dQ1) and dQ2 > 1e-6:
        raise AssertionError(f"Q still accelerating: dQ1={dQ1}, dQ2={dQ2}")
    if dR2 > max(1e-9, 2.0 * dR1) and abs(reg[-1]) > 10.0 * np.sqrt(T):
        raise AssertionError(
            f"regret still growing fast: Reg_T={reg[-1]}, dR1={dR1}, dR2={dR2}"
        )
    if gap[-1] > gap[min(49, T - 1)] and T * gap[-1] > 1e3:
        raise AssertionError(f"restricted gap not shrinking: gap_T={gap[-1]}")
    if hist["x_norm"][-1] < 1e-4 and hist["y_norm"][-1] < 1e-4:
        raise AssertionError("iterates stayed at the origin")

    print("frozen-beta checks passed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
