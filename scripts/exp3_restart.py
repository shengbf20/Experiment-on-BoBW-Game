"""Warm rescaling vs cold restart on one representative G2 setting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from games import QuadraticGame, paper_saddle  # noqa: E402
from learner import ClosedFormPlayer, run_loop  # noqa: E402
from io_results import dump_compact  # noqa: E402

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


def _dump_const(tag: str, payload: dict, hist: dict) -> None:
    dump_compact(
        f"exp3_{tag}",
        payload,
        hist,
        keys=("J_x", "x_norm", "reg_x", "restart_x", "V_x"),
    )


def _paper_game(dim: int) -> QuadraticGame:
    sx, sy = paper_saddle(dim)
    return QuadraticGame(dim=dim, mu=0.2, saddle_x=sx, saddle_y=sy)


def run_vs_const(cfg: dict, T: int, radius: float, dim: int, restart: bool) -> dict:
    tag = "const_restart" if restart else "const_warm"
    game = _paper_game(dim)
    px = _player(dim, cfg, restart)
    if np.any(px.action):
        raise AssertionError("must not overwrite w1=0")
    e1 = np.zeros(dim, dtype=_DTYPE)
    e1[0] = 1.0
    x_star = game.induced_minimizer_x(e1)

    def y_policy(_t, _x):
        return e1.copy()

    metrics, hist, max_w = run_loop(
        game,
        px,
        T,
        y_policy=y_policy,
        player_y=None,
        observe_y=lambda _t: False,
        radius=radius,
        u_x=x_star,
    )
    if not np.allclose(metrics.u_x, x_star):
        raise AssertionError("vs-const comparator must be x*")
    if abs(hist["x_norm"][0]) > 1e-15:
        raise AssertionError(f"{tag}: w1 must be the origin")
    if hist["J_x"][-1] < 1:
        raise AssertionError(f"{tag}: expected J>=1 against const opponent")
    if hist["V_x"][-1] != 0.0:
        raise AssertionError(f"{tag}: V_x(x*) must be exactly 0, got {hist['V_x'][-1]}")
    half = T // 2
    # Plateau is in place by t~400. A bitwise-zero second-half drift is
    # partly float64: once (μ/2)||x-x*||^2 underflows, the increment cannot
    # accumulate. Do not treat 0.0 as a discrete fixed-point test.
    drift = abs(hist["reg_x"][-1] - hist["reg_x"][half - 1]) / float(half)
    if drift > 0.05:
        raise AssertionError(
            f"{tag}: Reg^x(x*) second-half drift {drift:.3g}/step; expected a plateau"
        )
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
    sx, sy = game.saddle()
    payload = {
        "meta": {
            "tag": tag,
            "game": "G2",
            "A": "identity",
            "mode": "restart" if restart else "warm",
            "protocol": "const e1",
            "comparator": "induced_minimizer_x",
            "dim": dim,
            "T": T,
            "init": "origin",
            "saddle_x": sx.tolist(),
            "saddle_y": sy.tolist(),
            "x_star": x_star.tolist(),
        },
        "summary": {
            "max_w": max_w,
            "reg_x_T": hist["reg_x"][-1],
            "reg_x_half": hist["reg_x"][half - 1],
            "second_half_drift_per_step": drift,
            "V_x_T": hist["V_x"][-1],
            "J_x": hist["J_x"][-1],
            "n_restarts_x": px.n_restarts,
            "gamma_x": px.gamma,
            "G_x": metrics.G_x,
            "x_star_norm": float(np.linalg.norm(x_star)),
            "x_norm_T": hist["x_norm"][-1],
        },
    }
    _dump_const(tag, payload, hist)
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
    run_vs_const(cfg, T, radius, dim, restart=False)
    run_vs_const(cfg, T, radius, dim, restart=True)


if __name__ == "__main__":
    main()
