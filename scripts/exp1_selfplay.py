"""Self-play G2 plus one non-degenerate Gaussian robustness run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from games import (  # noqa: E402
    BilinearGame,
    QuadraticGame,
    assert_saddle_comparator,
    paper_saddle,
)
from learner import ClosedFormPlayer, self_play  # noqa: E402
from io_results import dump_compact  # noqa: E402

_DTYPE = np.float64


def _load_cfg():
    with (ROOT / "configs" / "default.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _player(dim: int, cfg: dict) -> ClosedFormPlayer:
    # adaptive=True is HARDCODED on purpose: Exp.1 is the unknown-L_F main
    # algorithm (thm:t006). cfg["adaptive"] (=false) only governs the
    # frozen-beta health check scripts; changing the yaml does NOT affect
    # this experiment.
    return ClosedFormPlayer(
        dim,
        epsilon=float(cfg["epsilon"]),
        beta0=float(cfg["beta0"]),
        ell1=float(cfg["ell1"]),
        adaptive=True,
    )


def _last_increase(js) -> int:
    last = 0
    for t in range(1, len(js)):
        if js[t] > js[t - 1]:
            last = t
    return last


def _smoke(hist: dict, max_w: float, T: int, tag: str) -> None:
    if max_w >= 1e20 or not np.isfinite(max_w):
        raise AssertionError(f"{tag}: exploded max_w={max_w}")
    if abs(hist["x_norm"][0]) > 1e-15 or abs(hist["y_norm"][0]) > 1e-15:
        raise AssertionError(f"{tag}: w1 must be the origin")
    if hist["x_norm"][1] == 0.0 and hist["y_norm"][1] == 0.0:
        raise AssertionError(f"{tag}: stayed at the origin after round 1")
    jx, jy = hist["J_x"][-1], hist["J_y"][-1]
    if jx < 1 or jy < 1:
        raise AssertionError(f"{tag}: expected J>=1 for both, got J_x={jx}, J_y={jy}")
    mid = T // 2
    if hist["J_x"][-1] != hist["J_x"][mid] or hist["J_y"][-1] != hist["J_y"][mid]:
        raise AssertionError(f"{tag}: J did not freeze in the second half")
    if hist["beta_x"][-1] < hist["beta_x"][0] - 1e-15:
        raise AssertionError(f"{tag}: beta_x decreased")


def run_one(game, cfg: dict, T: int, radius: float, tag: str, meta: dict) -> dict:
    dim = game.dim_x
    px, py = _player(dim, cfg), _player(dim, cfg)
    if np.any(px.action) or np.any(py.action):
        raise AssertionError("must not overwrite w1=0")
    metrics, hist, max_w = self_play(game, px, py, T=T, radius=radius)
    assert_saddle_comparator(game, metrics)
    _smoke(hist, max_w, T, tag)
    sx, sy = game.saddle()
    payload = {
        "meta": {
            **meta,
            "tag": tag,
            "dim": dim,
            "T": T,
            "adaptive": True,
            "epsilon": float(cfg["epsilon"]),
            "beta0": float(cfg["beta0"]),
            "ell1": float(cfg["ell1"]),
            "gap_radius": radius,
            "init": "origin (paper w1=0)",
            "saddle_x": sx.tolist(),
            "saddle_y": sy.tolist(),
        },
        "summary": {
            "max_w": max_w,
            "reg_x_T": hist["reg_x"][-1],
            "reg_y_T": hist["reg_y"][-1],
            "lin_x_T": hist["lin_x"][-1],
            "lin_y_T": hist["lin_y"][-1],
            "Q_T": hist["Q"][-1],
            "gap_T": hist["gap"][-1],
            "J_x": hist["J_x"][-1],
            "J_y": hist["J_y"][-1],
            "ell_x": hist["ell_x"][-1],
            "ell_y": hist["ell_y"][-1],
            "beta_x": hist["beta_x"][-1],
            "beta_y": hist["beta_y"][-1],
            "t_last_double_x": _last_increase(hist["J_x"]),
            "t_last_double_y": _last_increase(hist["J_y"]),
            "G_x": metrics.G_x,
            "G_y": metrics.G_y,
        },
    }
    dump_compact(f"exp1_{tag}", payload, hist)
    print(
        f"J=({payload['summary']['J_x']},{payload['summary']['J_y']}) "
        f"gap_T={payload['summary']['gap_T']:.3e} Q_T={payload['summary']['Q_T']:.3e}"
    )
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--gaussian-seed", type=int, default=None)
    args = parser.parse_args()

    cfg = _load_cfg()
    T = int(args.T if args.T is not None else cfg["T"])
    radius = float(cfg["gap_radius"])
    dim = int(args.dim)
    seed = int(args.gaussian_seed if args.gaussian_seed is not None else cfg["gaussian_seed"])
    sx, sy = paper_saddle(dim)

    run_one(
        QuadraticGame(dim=dim, mu=0.2, saddle_x=sx, saddle_y=sy),
        cfg,
        T,
        radius,
        "G2_identity",
        {"game": "G2", "A": "identity", "mu": 0.2},
    )
    run_one(
        BilinearGame.gaussian(dim=dim, seed=seed, saddle_x=sx, saddle_y=sy),
        cfg,
        T,
        radius,
        f"G1_gaussian_seed{seed}",
        {"game": "G1", "A": "gaussian-spectral", "seed": seed},
    )


if __name__ == "__main__":
    main()
