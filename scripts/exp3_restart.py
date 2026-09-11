"""Exp.3: warm rescaling vs cold restart on G2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from games import QuadraticGame  # noqa: E402
from learner import ClosedFormPlayer, run_loop, self_play  # noqa: E402

_DTYPE = np.float64


def _load_cfg():
    with (ROOT / "configs" / "default.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _player(dim: int, cfg: dict, restart: bool) -> ClosedFormPlayer:
    return ClosedFormPlayer(
        dim,
        epsilon=float(cfg["epsilon"]),
        beta0=float(cfg["beta0"]),
        ell1=float(cfg["ell1"]),
        adaptive=True,
        restart=restart,
    )


def _dump(tag: str, payload: dict) -> None:
    path = ROOT / "results" / f"exp3_{tag}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {path}  {payload['summary']}")


def _smoke_selfplay(tag: str, hist: dict, px: ClosedFormPlayer, T: int) -> None:
    if hist["J_x"][-1] < 1:
        raise AssertionError(f"{tag}: expected J>=1")
    mid = T // 2
    if hist["J_x"][-1] != hist["J_x"][mid]:
        raise AssertionError(f"{tag}: J did not freeze in the second half")
    if px.restart:
        if px.n_restarts < 1:
            raise AssertionError(f"{tag}: restart player never reset")
        if px.n_restarts != hist["J_x"][-1]:
            raise AssertionError("each doubling should trigger one cold reset")
    else:
        if px.n_restarts != 0:
            raise AssertionError("warm player restarted")
        if abs(px.gamma - px.gamma_init) > 0.0:
            raise AssertionError("warm gamma moved")


def run_selfplay(cfg: dict, T: int, radius: float, dim: int, restart: bool) -> dict:
    tag = "selfplay_restart" if restart else "selfplay_warm"
    game = QuadraticGame(dim=dim, mu=0.2)
    px, py = _player(dim, cfg, restart), _player(dim, cfg, restart)
    px.action[0] = 1.0
    py.action[0] = -1.0
    metrics, hist, max_w = self_play(game, px, py, T=T, radius=radius)
    _smoke_selfplay(tag, hist, px, T)
    payload = {
        "meta": {
            "tag": tag,
            "game": "G2",
            "A": "identity",
            "mode": "restart" if restart else "warm",
            "protocol": "selfplay",
            "dim": dim,
            "T": T,
            "init": "e1 / -e1",
        },
        "hist": hist,
        "summary": {
            "max_w": max_w,
            "reg_x_T": hist["reg_x"][-1],
            "Q_T": hist["Q"][-1],
            "J_x": hist["J_x"][-1],
            "n_restarts_x": px.n_restarts,
            "gamma_x": px.gamma,
            "G_x": metrics.G_x,
        },
    }
    _dump(tag, payload)
    return payload


def run_vs_const(cfg: dict, T: int, radius: float, dim: int, restart: bool) -> dict:
    tag = "const_restart" if restart else "const_warm"
    game = QuadraticGame(dim=dim, mu=0.2)
    px = _player(dim, cfg, restart)
    e1 = np.zeros(dim, dtype=_DTYPE)
    e1[0] = 1.0

    def y_policy(_t, _x):
        return e1.copy()

    metrics, hist, max_w = run_loop(
        game, px, T, y_policy=y_policy, player_y=None, observe_y=lambda _t: False, radius=radius
    )
    if hist["J_x"][-1] < 1:
        raise AssertionError(f"{tag}: expected J>=1 against const opponent")
    if restart:
        if px.n_restarts < 1:
            raise AssertionError(f"{tag}: never reset")
        if px.n_restarts != hist["J_x"][-1]:
            raise AssertionError("each doubling should trigger one cold reset")
        if abs(hist["x_norm"][2]) > 1e-15:
            raise AssertionError(f"{tag}: t=3 action should be the origin")
        if abs(px.gamma - px.epsilon * px.beta) > 1e-12:
            raise AssertionError("restart gamma must equal epsilon * beta")
    else:
        if px.n_restarts != 0:
            raise AssertionError("warm player restarted")
        if abs(px.gamma - px.gamma_init) > 0.0:
            raise AssertionError("warm gamma moved")
        if hist["x_norm"][2] <= 0.0:
            raise AssertionError(f"{tag}: warm should not jump to the origin at t=3")
    payload = {
        "meta": {
            "tag": tag,
            "game": "G2",
            "A": "identity",
            "mode": "restart" if restart else "warm",
            "protocol": "const e1",
            "dim": dim,
            "T": T,
            "init": "origin",
        },
        "hist": hist,
        "summary": {
            "max_w": max_w,
            "reg_x_T": hist["reg_x"][-1],
            "J_x": hist["J_x"][-1],
            "n_restarts_x": px.n_restarts,
            "gamma_x": px.gamma,
            "G_x": metrics.G_x,
        },
    }
    _dump(tag, payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--dim", type=int, default=10)
    args = parser.parse_args()
    cfg = _load_cfg()
    T = int(args.T if args.T is not None else cfg["T"])
    radius = float(cfg["gap_radius"])
    dim = int(args.dim)
    run_selfplay(cfg, T, radius, dim, restart=False)
    run_selfplay(cfg, T, radius, dim, restart=True)
    run_vs_const(cfg, T, radius, dim, restart=False)
    run_vs_const(cfg, T, radius, dim, restart=True)


if __name__ == "__main__":
    main()
