"""Diagnose G1 Q_t growth. Existing json + optional long run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(tag: str) -> dict:
    return json.loads((ROOT / "results" / f"{tag}.json").read_text(encoding="utf-8"))


def diagnose(tag: str) -> None:
    d = _load(tag)
    h = d["hist"]
    Q = np.asarray(h["Q"], dtype=float)
    T = len(Q)
    t = np.arange(1, T + 1)
    dQ = np.diff(Q, prepend=0.0)
    print(f"=== {tag} ===")
    print(
        "reg_x_T", h["reg_x"][-1],
        "reg_y_T", h["reg_y"][-1],
        "Q_T", Q[-1],
        "gap_T", h["gap"][-1],
        "J", h["J_x"][-1], h["J_y"][-1],
        "beta", h["beta_x"][-1], h["beta_y"][-1],
    )
    for k in [100, 1000, 5000, 10000, 15000, 20000, 50000, 100000, 200000, 500000, 1000000]:
        if k <= T:
            print(
                f"  t={k:7d}  Q={Q[k-1]:.6f}  dQ={dQ[k-1]:.3e}  "
                f"regx={h['reg_x'][k-1]:.6f}  gap={h['gap'][k-1]:.4e}"
            )
    windows = [(T // 4, T // 2), (T // 2, 3 * T // 4), (3 * T // 4, T)]
    for lo, hi in windows:
        print(f"  Q[{lo}:{hi}] increment {Q[hi-1]-Q[lo-1]:.6f}, mean dQ {dQ[lo:hi].mean():.3e}")
    sl = slice(T // 2, T)
    A_lin = np.vstack([np.ones(T - T // 2), t[sl]]).T
    c_lin, _, _, _ = np.linalg.lstsq(A_lin, Q[sl], rcond=None)
    A_log = np.vstack([np.ones(T - T // 2), np.log(t[sl].astype(float))]).T
    c_log, _, _, _ = np.linalg.lstsq(A_log, Q[sl], rcond=None)
    rmse_lin = float(np.sqrt(np.mean((Q[sl] - A_lin @ c_lin) ** 2)))
    rmse_log = float(np.sqrt(np.mean((Q[sl] - A_log @ c_log) ** 2)))
    print(f"  2nd-half lin slope={c_lin[1]:.6e} rmse={rmse_lin:.4e}")
    print(f"  2nd-half log slope={c_log[1]:.6e} rmse={rmse_log:.4e}")
    print()


def main():
    tags = [
        "exp1_G1_identity",
        "exp1_G2_identity",
        "exp1_G1_gaussian_seed0",
        "exp1_G1_gaussian_seed1",
        "exp1_G1_gaussian_seed2",
        "exp1_G2_gaussian_seed0",
        "exp1_G2_gaussian_seed1",
        "exp1_G2_gaussian_seed2",
    ]
    extra = ROOT / "results" / "exp1_G1_identity_long.json"
    if extra.is_file():
        tags.append("exp1_G1_identity_long")
    for tag in tags:
        path = ROOT / "results" / f"{tag}.json"
        if path.is_file():
            diagnose(tag)


if __name__ == "__main__":
    main()
