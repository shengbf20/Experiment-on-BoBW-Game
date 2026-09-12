"""Long G1 self-play to test whether Q_t^{obs} saturates. Compact npz output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from games import BilinearGame, paper_saddle  # noqa: E402
from io_results import LONG_STRIDE, dump_compact  # noqa: E402
from learner import ClosedFormPlayer  # noqa: E402
from metrics import RunningMetrics  # noqa: E402

_DTYPE = np.float64
LONG_KEYS = ("Q", "gap", "reg_x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=200000)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--progress", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=None, help="if set, spectral-normalized gaussian A")
    args = parser.parse_args()

    with (ROOT / "configs" / "default.yaml").open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    T = int(args.T)
    dim = int(args.dim)
    sx, sy = paper_saddle(dim)
    if args.seed is None:
        game = BilinearGame(dim=dim, saddle_x=sx, saddle_y=sy)
        tag = "G1_identity_long"
        a_kind = "identity"
    else:
        game = BilinearGame.gaussian(dim=dim, seed=int(args.seed), saddle_x=sx, saddle_y=sy)
        tag = f"G1_gaussian_seed{int(args.seed)}_long"
        a_kind = "gaussian-spectral"
    kwargs = dict(
        epsilon=float(cfg["epsilon"]),
        beta0=float(cfg["beta0"]),
        ell1=float(cfg["ell1"]),
        adaptive=True,
    )
    px = ClosedFormPlayer(dim, **kwargs)
    py = ClosedFormPlayer(dim, **kwargs)
    metrics = RunningMetrics(game, radius=float(cfg["gap_radius"]))

    Q = np.zeros(T, dtype=_DTYPE)
    dQ = np.zeros(T, dtype=_DTYPE)
    gap = np.zeros(T, dtype=_DTYPE)
    reg_x = np.zeros(T, dtype=_DTYPE)
    phi = np.zeros(T, dtype=_DTYPE)
    radius = np.zeros(T, dtype=_DTYPE)
    z_prev = None
    max_w = 0.0
    for t in range(1, T + 1):
        x = px.action.copy()
        y = py.action.copy()
        if t == 1 and (np.any(x) or np.any(y)):
            raise AssertionError("w1 must be 0")
        max_w = max(max_w, float(np.linalg.norm(x)), float(np.linalg.norm(y)))
        gx, gy = game.feedback(x, y)
        snap = metrics.step(x, y, gx, gy)
        z = np.concatenate([x, y])
        if z_prev is None:
            step = 0.0
        else:
            step = float(np.sum((z - z_prev) ** 2))
        z_prev = z
        px.observe(gx, z)
        py.observe(gy, z)
        i = t - 1
        dQ[i] = step
        Q[i] = snap["Q"]
        gap[i] = snap["gap"]
        reg_x[i] = snap["reg_x"]
        phi[i] = game.phi(x, y)
        radius[i] = float(np.sqrt(np.sum((x - sx) ** 2) + np.sum((y - sy) ** 2)))
        if args.progress and t % args.progress == 0:
            print(
                f"t={t} Q={Q[i]:.6f} dQ={dQ[i]:.3e} radius={radius[i]:.6f} "
                f"gap={gap[i]:.4e} regx={reg_x[i]:.6f}",
                flush=True,
            )

    checkpoints = [k for k in (2_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000) if k <= T]
    summary = {
        "max_w": max_w,
        "reg_x_T": float(reg_x[-1]),
        "reg_y_T": float(metrics.reg_y),
        "Q_T": float(Q[-1]),
        "dQ_T": float(dQ[-1]),
        "gap_T": float(gap[-1]),
        "radius_T": float(radius[-1]),
        "phi_max_abs": float(np.max(np.abs(phi))),
        "J_x": int(px.J),
        "J_y": int(py.J),
        "beta_x": float(px.beta),
        "beta_y": float(py.beta),
        "dQ_mean_last_10pct": float(dQ[int(0.9 * T) :].mean()),
        "dQ_mean_last_1pct": float(dQ[int(0.99 * T) :].mean()),
        "checkpoints": {
            str(k): {
                "Q": float(Q[k - 1]),
                "dQ": float(dQ[k - 1]),
                "radius": float(radius[k - 1]),
                "gap": float(gap[k - 1]),
            }
            for k in checkpoints
        },
    }
    meta = {
        "tag": tag,
        "game": "G1",
        "A": a_kind,
        "T": T,
        "dim": dim,
        "init": "origin (paper w1=0)",
        "saddle_x": sx.tolist(),
        "saddle_y": sy.tolist(),
        "stride": LONG_STRIDE,
    }
    if args.seed is not None:
        meta["seed"] = int(args.seed)
    payload = {"meta": meta, "summary": summary, "T": T}
    dump_compact(
        f"exp1_{tag}",
        payload,
        {"Q": Q, "gap": gap, "reg_x": reg_x},
        extra_arrays={"dQ": dQ, "radius": radius},
        keys=LONG_KEYS,
        stride=LONG_STRIDE,
    )
    print(summary)


if __name__ == "__main__":
    main()
