"""Diagnose G1 Q_t growth from compact json/npz. Optional long run."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from io_results import load_run  # noqa: E402


def _last(hist: dict, key: str):
    if key not in hist:
        return float("nan")
    val = hist[key]
    if hasattr(val, "__len__") and not isinstance(val, (str, bytes)):
        return val[-1]
    return val


def diagnose(tag: str) -> None:
    _payload, h = load_run(tag)
    Q = np.asarray(h["Q"], dtype=float)
    T = len(Q)
    t = np.arange(1, T + 1)
    dQ = np.diff(Q, prepend=0.0)
    print(f"=== {tag} ===")
    print(
        "reg_x_T", _last(h, "reg_x"),
        "reg_y_T", _last(h, "reg_y"),
        "Q_T", Q[-1],
        "gap_T", _last(h, "gap"),
        "J", _last(h, "J_x"), _last(h, "J_y"),
        "beta", _last(h, "beta_x"), _last(h, "beta_y"),
    )
    for k in [100, 1000, 5000, 10000, 15000, 20000, 50000, 100000, 200000, 500000, 1000000]:
        if k <= T:
            rx = h["reg_x"][k - 1] if "reg_x" in h else float("nan")
            gp = h["gap"][k - 1] if "gap" in h else float("nan")
            print(f"  t={k:7d}  Q={Q[k-1]:.6f}  dQ={dQ[k-1]:.3e}  regx={rx:.6f}  gap={gp:.4e}")
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
        "exp1_G1_identity_long",
    ]
    for tag in tags:
        path = ROOT / "results" / f"{tag}.json"
        npz = ROOT / "results" / f"{tag}.npz"
        if path.is_file() and npz.is_file():
            diagnose(tag)


if __name__ == "__main__":
    main()
