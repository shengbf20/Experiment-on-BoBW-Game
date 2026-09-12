"""Hsieh OptDA algebraic identities and G2 self-play platform smoke.

The T=4000 self-play check only rejects a large second-half drift
(|Reg|>20 and late movement comparable to the first half). A mild
sqrt(T) trend would pass. It is not an independent rate proof; the
Go evidence is the T=2e4 json (this instance freezes by t≈70).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from games import QuadraticGame, assert_saddle_comparator, paper_saddle  # noqa: E402
from hsieh import HsiehOptDA  # noqa: E402
from learner import self_play  # noqa: E402

_DTYPE = np.float64


def test_origin_and_closed_form_step():
    p = HsiehOptDA(dim=2, tau=1.0)
    if np.any(p.action):
        raise AssertionError("w1 must be the origin")
    g1 = np.array([3.0, 4.0], dtype=_DTYPE)
    p.observe(g1, np.zeros(4, dtype=_DTYPE))
    lam2 = float(np.sqrt(1.0 + float(g1 @ g1)))
    want = -2.0 * g1 / lam2
    if not np.allclose(p.action, want):
        raise AssertionError(f"x2 mismatch: {p.action} vs {want}")
    g2 = np.array([0.0, -1.0], dtype=_DTYPE)
    p.observe(g2, np.zeros(4, dtype=_DTYPE))
    lam3 = float(np.sqrt(1.0 + float(g1 @ g1) + float((g2 - g1) @ (g2 - g1))))
    want3 = -(g1 + 2.0 * g2) / lam3
    if not np.allclose(p.action, want3):
        raise AssertionError(f"x3 mismatch: {p.action} vs {want3}")
    if abs(p.lam - lam3) > 1e-12:
        raise AssertionError("lambda off Adapt")


def test_g2_selfplay_platforms():
    dim = 10
    T = 4000
    sx, sy = paper_saddle(dim)
    game = QuadraticGame(dim=dim, mu=0.2, saddle_x=sx, saddle_y=sy)
    px, py = HsiehOptDA(dim, tau=1.0), HsiehOptDA(dim, tau=1.0)
    if np.any(px.action) or np.any(py.action):
        raise AssertionError("must not overwrite w1=0")
    metrics, hist, max_w = self_play(game, px, py, T)
    assert_saddle_comparator(game, metrics)
    if max_w >= 1e20 or not np.isfinite(max_w):
        raise AssertionError(f"exploded max_w={max_w}")
    if abs(hist["x_norm"][0]) > 1e-15 or abs(hist["y_norm"][0]) > 1e-15:
        raise AssertionError("w1 must be the origin")
    if hist["x_norm"][1] == 0.0 and hist["y_norm"][1] == 0.0:
        raise AssertionError("stayed at the origin after round 1")
    half = T // 2
    q = np.asarray(hist["Q"], dtype=float)
    reg = np.asarray(hist["reg_x"], dtype=float)
    dQ_late = float(q[-1] - q[half - 1])
    dQ_early = float(q[half - 1] - q[0])
    if dQ_late > 0.25 * max(dQ_early, 1e-6) and q[-1] > 0.5:
        raise AssertionError(
            f"Q did not freeze: early ΔQ={dQ_early:.4g}, late ΔQ={dQ_late:.4g}, Q_T={q[-1]:.4g}"
        )
    dreg = abs(float(reg[-1] - reg[half - 1]))
    # sqrt(T) growth from T/2 to T is about 0.41 * sqrt(T) in the scale of the
    # first-half rise. Fail if the second half still looks like that.
    # This threshold is intentionally loose (a mild sqrt(T) trend would pass).
    first = abs(float(reg[half - 1] - reg[0]))
    if dreg > max(10.0, 0.5 * first) and abs(reg[-1]) > 20.0:
        raise AssertionError(
            f"regret still moving like a rate: half={reg[half-1]:.4g}, T={reg[-1]:.4g}"
        )
    print(
        f"G2 self-play smoke T={T}: Reg^x(a)={reg[-1]:.4g}, "
        f"Q_T={q[-1]:.4g}, late ΔQ={dQ_late:.4g}, max_w={max_w:.4g} "
        "(smoke only; Go evidence is T=2e4 json)"
    )


if __name__ == "__main__":
    test_origin_and_closed_form_step()
    test_g2_selfplay_platforms()
    print("check_hsieh ok")
