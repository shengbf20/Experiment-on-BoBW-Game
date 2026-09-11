"""Exp.1 self-play runs. Writes json only; plot with plot_exp1.py.

Main figure: A = I, one run. Appendix: gaussian A, three seeds.
Origin is a saddle rest point, so the first action is a fixed off-saddle probe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from games import BilinearGame, QuadraticGame  # noqa: E402
from learner import ClosedFormPlayer, self_play  # noqa: E402

_DTYPE = np.float64


def _load_cfg():
    with (ROOT / "configs" / "default.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _init_actions(dim: int):
    x = np.zeros(dim, dtype=_DTYPE)
    y = np.zeros(dim, dtype=_DTYPE)
    x[0] = 1.0
    y[0] = -1.0
    return x, y


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
    jx, jy = hist["J_x"][-1], hist["J_y"][-1]
    if jx < 1 or jy < 1:
        raise AssertionError(f"{tag}: expected J>=1 for both, got J_x={jx}, J_y={jy}")
    mid = T // 2
    if hist["J_x"][-1] != hist["J_x"][mid] or hist["J_y"][-1] != hist["J_y"][mid]:
        raise AssertionError(f"{tag}: J did not freeze in the second half")
    if hist["beta_x"][-1] < hist["beta_x"][0] - 1e-15:
        raise AssertionError(f"{tag}: beta_x decreased")


def _dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def run_one(game, cfg: dict, T: int, radius: float, tag: str, meta: dict) -> dict:
    dim = game.dim_x
    px, py = _player(dim, cfg), _player(dim, cfg)
    px.action[:], py.action[:] = _init_actions(dim)
    metrics, hist, max_w = self_play(game, px, py, T=T, radius=radius)
    _smoke(hist, max_w, T, tag)
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
            "init": "e1 / -e1",
        },
        "hist": hist,
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
    out = ROOT / "results" / f"exp1_{tag}.json"
    _dump(out, payload)
    print(f"wrote {out}  J=({payload['summary']['J_x']},{payload['summary']['J_y']}) "
          f"gap_T={payload['summary']['gap_T']:.3e} Q_T={payload['summary']['Q_T']:.3e}")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--main-only", action="store_true", help="skip gaussian appendix runs")
    args = parser.parse_args()

    cfg = _load_cfg()
    T = int(args.T if args.T is not None else cfg["T"])
    radius = float(cfg["gap_radius"])
    dim = int(args.dim)
    seeds = list(cfg.get("seeds", [0, 1, 2]))

    run_one(BilinearGame(dim=dim), cfg, T, radius, "G1_identity", {"game": "G1", "A": "identity"})
    run_one(QuadraticGame(dim=dim, mu=0.2), cfg, T, radius, "G2_identity", {"game": "G2", "A": "identity", "mu": 0.2})

    if not args.main_only:
        for seed in seeds:
            run_one(
                BilinearGame.gaussian(dim=dim, seed=seed),
                cfg,
                T,
                radius,
                f"G1_gaussian_seed{seed}",
                {"game": "G1", "A": "gaussian", "seed": seed},
            )
            run_one(
                QuadraticGame.gaussian(dim=dim, mu=0.2, seed=seed),
                cfg,
                T,
                radius,
                f"G2_gaussian_seed{seed}",
                {"game": "G2", "A": "gaussian", "seed": seed, "mu": 0.2},
            )


if __name__ == "__main__":
    main()
