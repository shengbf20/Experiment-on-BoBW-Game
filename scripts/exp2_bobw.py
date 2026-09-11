"""Exp.2: same-run switch (2a) and G3 constant-opponent separation (2b)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from games import QuadraticGame, SeparationGame  # noqa: E402
from learner import ClosedFormPlayer, run_loop  # noqa: E402

_DTYPE = np.float64
PERIOD = 200


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


def _dump(tag: str, payload: dict) -> None:
    path = ROOT / "results" / f"exp2_{tag}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    s = payload["summary"]
    print(f"wrote {path}  {s}")


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
        "hist": hist,
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
    _dump("G3_const", payload)
    return payload


def run_2a(cfg: dict, T: int, radius: float, dim: int = 10) -> dict:
    game = QuadraticGame(dim=dim, mu=0.2)
    px, py = _player(dim, cfg), _player(dim, cfg)
    px.action[0] = 1.0
    py.action[0] = -1.0
    half = T // 2
    e1 = np.zeros(dim, dtype=_DTYPE)
    e1[0] = 1.0

    def y_policy(t, _x):
        if t <= half:
            return py.action.copy()
        return np.sin(2.0 * np.pi * t / PERIOD) * e1

    metrics, hist, max_w = run_loop(
        game,
        px,
        T,
        y_policy=y_policy,
        player_y=py,
        observe_y=lambda t: t <= half,
        radius=radius,
    )
    if px.t != T or py.t != half:
        raise AssertionError(f"observe counts: x.t={px.t} y.t={py.t}, expected {T} and {half}")
    if abs(px.gamma - px.gamma_init) > 0.0:
        raise AssertionError("gamma reset at switch")
    if max_w >= 1e20:
        raise AssertionError(f"switch exploded max_w={max_w}")
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
            "init": "e1 / -e1 then slow after T/2",
        },
        "hist": hist,
        "summary": {
            "max_w": max_w,
            "reg_x_half": hist["reg_x"][half - 1],
            "reg_x_T": hist["reg_x"][-1],
            "V_x_half": hist["V_x"][half - 1],
            "V_x_T": hist["V_x"][-1],
            "J_x": hist["J_x"][-1],
            "J_y": hist["J_y"][-1],
            "t_y": hist["t_y"],
            "gamma_x": px.gamma,
            "G_x": metrics.G_x,
        },
    }
    _dump("G2_switch", payload)
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
