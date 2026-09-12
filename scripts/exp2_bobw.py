"""Exp.2: same-run switch (2a) and G3 constant-opponent separation (2b)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from games import (  # noqa: E402
    QuadraticGame,
    SeparationGame,
    assert_saddle_comparator,
    paper_saddle,
)
from learner import ClosedFormPlayer, run_loop  # noqa: E402
from io_results import dump_compact  # noqa: E402

_DTYPE = np.float64
PERIOD = 200
ANCHOR_TOL = 0.05


def _load_cfg():
    with (ROOT / "configs" / "default.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _player(dim: int, cfg: dict) -> ClosedFormPlayer:
    return ClosedFormPlayer(
        dim,
        epsilon=float(cfg["epsilon"]),
        beta0=float(cfg["beta0"]),
        ell1=float(cfg["ell1"]),
        adaptive=True,
    )


def _dump_switch(tag: str, payload: dict, hist: dict) -> None:
    dump_compact(
        f"exp2_{tag}",
        payload,
        hist,
        keys=("dist_x", "dist_y", "V_x", "reg_x", "J_x"),
    )


def run_2b(cfg: dict, T: int, radius: float) -> dict:
    game = SeparationGame()
    px = _player(1, cfg)
    # Paper initialization: x_1 = 0. Constant opponent y_t = 1 is not a rest point.

    def y_policy(_t, _x):
        return np.array([1.0], dtype=_DTYPE)

    metrics, hist, max_w = run_loop(
        game, px, T, y_policy=y_policy, player_y=None, observe_y=lambda _t: False, radius=radius
    )
    v = np.asarray(hist["V_x"], dtype=float)
    if np.any(v != 0.0):
        raise AssertionError(f"G3 V_x(u*) must be exactly 0, got max {v.max()}")
    gT = hist["G_x"][-1]
    # Lower bound 1 comes from the origin init: g_1 = 2*0/sqrt(1+0) + 1 = 1
    # exactly. Changing the init makes this assertion spuriously fail.
    if not (1.0 - 1e-12 <= gT <= 3.0):
        raise AssertionError(f"G3 G_x should lie in [1, 3], got {gT}")
    if max_w >= 1e20:
        raise AssertionError(f"G3 exploded max_w={max_w}")
    if px.t != T:
        raise AssertionError("player x must run the full horizon")
    payload = {
        "meta": {
            "tag": "G3_const",
            "game": "G3",
            "opponent": "const y=1",
            "T": T,
            "adaptive": True,
            "epsilon": float(cfg["epsilon"]),
            "beta0": float(cfg["beta0"]),
            "ell1": float(cfg["ell1"]),
            "comparator": "u_star",
            "init": "origin",
        },
        "summary": {
            "max_w": max_w,
            "reg_x_T": hist["reg_x"][-1],
            "V_x_T": hist["V_x"][-1],
            "G_x": metrics.G_x,
            "J_x": hist["J_x"][-1],
            "beta_x": hist["beta_x"][-1],
            "gamma_x": px.gamma,
        },
    }
    dump_compact(
        "exp2_G3_const",
        payload,
        hist,
        keys=("reg_x", "V_x", "G_x", "x_norm", "J_x"),
    )
    print(payload["summary"])
    return payload


def run_2a(cfg: dict, T: int, radius: float, dim: int = 10) -> dict:
    sx, sy = paper_saddle(dim)
    game = QuadraticGame(dim=dim, mu=0.2, saddle_x=sx, saddle_y=sy)
    px, py = _player(dim, cfg), _player(dim, cfg)
    if np.any(px.action) or np.any(py.action):
        raise AssertionError("must not overwrite w1=0")
    half = T // 2
    e1 = np.zeros(dim, dtype=_DTYPE)
    e1[0] = 1.0
    y_played: dict[int, np.ndarray] = {}

    def y_policy(t, _x):
        if t <= half:
            y = py.action.copy()
        else:
            y = y_played[half] + np.sin(2.0 * np.pi * (t - half) / PERIOD) * e1
        y_played[t] = y.copy()
        return y

    metrics, hist, max_w = run_loop(
        game,
        px,
        T,
        y_policy=y_policy,
        player_y=py,
        observe_y=lambda t: t <= half,
        radius=radius,
    )
    assert_saddle_comparator(game, metrics)
    if abs(hist["x_norm"][0]) > 1e-15 or abs(hist["y_norm"][0]) > 1e-15:
        raise AssertionError("w1 must be the origin")
    if px.t != T or py.t != half:
        raise AssertionError(f"observe counts: x.t={px.t} y.t={py.t}, expected {T} and {half}")
    if abs(px.gamma - px.gamma_init) > 0.0:
        raise AssertionError("gamma reset at switch")
    if max_w >= 1e20:
        raise AssertionError(f"switch exploded max_w={max_w}")
    y_anchor = y_played[half]
    y_next = y_played[half + 1]
    step = float(np.linalg.norm(y_next - y_anchor))
    expected_step = abs(np.sin(2.0 * np.pi / PERIOD))
    if abs(step - expected_step) > 1e-12:
        raise AssertionError(f"switch jump {step} != sine step {expected_step}")
    dist_b = float(np.linalg.norm(y_anchor - sy))
    if dist_b > ANCHOR_TOL:
        raise AssertionError(f"y_anchor not near b: ||y-b||={dist_b}")
    v_half, v_T = hist["V_x"][half - 1], hist["V_x"][-1]
    if v_T <= v_half:
        raise AssertionError(f"V_x(a) should rise after switch: half={v_half}, T={v_T}")
    dx_post = np.asarray(hist["dist_x"][half:], dtype=float)
    dy_post = np.asarray(hist["dist_y"][half:], dtype=float)
    if float(dx_post.max()) > 5.0 or float(dy_post.max()) > 2.5:
        raise AssertionError(
            f"post-switch distances exploded: max||x-a||={dx_post.max()}, max||y-b||={dy_post.max()}"
        )
    # J_x=1 after the switch is this instance: self-play already raised ell
    # past post-switch chi (t=2: chi≈0.721, ell←1.442; at the switch
    # chi≈1<1.442). A faster or larger sine can still double. Do not treat
    # a frozen J as a general post-switch guarantee. gamma must not reset
    # (checked above).
    payload = {
        "meta": {
            "tag": "G2_switch",
            "game": "G2",
            "A": "identity",
            "mu": 0.2,
            "dim": dim,
            "T": T,
            "half": half,
            "period": PERIOD,
            "adaptive": True,
            "epsilon": float(cfg["epsilon"]),
            "beta0": float(cfg["beta0"]),
            "ell1": float(cfg["ell1"]),
            "init": "origin (paper w1=0); slow sine after T/2, continuous at b",
            "saddle_x": sx.tolist(),
            "saddle_y": sy.tolist(),
        },
        "summary": {
            "max_w": max_w,
            "reg_x_half": hist["reg_x"][half - 1],
            "reg_x_T": hist["reg_x"][-1],
            "V_x_half": v_half,
            "V_x_T": v_T,
            "dist_x_half": hist["dist_x"][half - 1],
            "dist_x_T": hist["dist_x"][-1],
            "dist_y_half": hist["dist_y"][half - 1],
            "dist_y_T": hist["dist_y"][-1],
            "dist_x_post_max": float(dx_post.max()),
            "dist_y_post_max": float(dy_post.max()),
            "J_x": hist["J_x"][-1],
            "J_y": hist["J_y"][-1],
            "t_y": hist["t_y"],
            "gamma_x": px.gamma,
            "gamma_init": px.gamma_init,
            "G_x": metrics.G_x,
            "y_anchor_dist_b": dist_b,
            "switch_step": step,
        },
    }
    _dump_switch("G2_switch", payload, hist)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--only", choices=["2a", "2b", "both"], default="both")
    args = parser.parse_args()
    cfg = _load_cfg()
    T = int(args.T if args.T is not None else cfg["T"])
    radius = float(cfg["gap_radius"])
    if args.only in ("2b", "both"):
        run_2b(cfg, T, radius)
    if args.only in ("2a", "both"):
        run_2a(cfg, T, radius)


if __name__ == "__main__":
    main()
