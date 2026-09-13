"""Diagnose G1 Q_t growth from compact json/npz. Optional long run.

Respects stride / t stored in the npz. Do not use this script to audit
tab:exp-horizon: it is a diagnostic printout, not the horizon extractor.
The table is produced by scripts/exp_horizon.py from json summaries and
explicit t_used samples.
"""

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


def _horizon_axis(payload: dict, hist: dict, n: int) -> tuple[np.ndarray, int]:
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    T = payload.get("T") or meta.get("T")
    if T is None and "T" in hist:
        T = int(np.asarray(hist["T"]).reshape(-1)[0])
    T = int(T) if T is not None else n
    if "t" in hist:
        t = np.asarray(hist["t"], dtype=np.int64).reshape(-1)
        if t.size == n:
            return t, T
    if "stride" in hist:
        stride = int(np.asarray(hist["stride"]).reshape(-1)[0])
        t = np.arange(1, T + 1, dtype=np.int64)[::stride][:n]
        return t, T
    return np.arange(1, n + 1, dtype=np.int64), T


def diagnose(tag: str) -> None:
    payload, h = load_run(tag)
    Q = np.asarray(h["Q"], dtype=float)
    t, T = _horizon_axis(payload, h, len(Q))
    dQ = np.diff(Q, prepend=0.0)
    print(f"=== {tag} ===")
    if len(Q) != T:
        print(f"  stored n={len(Q)}  horizon T={T}  (do not treat n as T)")
        print("  horizon table: python scripts/exp_horizon.py")
    print(
        "reg_x_T", _last(h, "reg_x"),
        "reg_y_T", _last(h, "reg_y"),
        "Q_T", Q[-1],
        "gap_T", _last(h, "gap"),
        "J", _last(h, "J_x"), _last(h, "J_y"),
        "beta", _last(h, "beta_x"), _last(h, "beta_y"),
    )
    for k in [100, 1000, 5000, 10000, 15000, 20000, 50000, 100000, 200000, 500000, 1000000]:
        if k > T:
            continue
        idx = int(np.where(t <= k)[0][-1]) if np.any(t <= k) else None
        if idx is None:
            continue
        t_used = int(t[idx])
        rx = h["reg_x"][idx] if "reg_x" in h else float("nan")
        gp = h["gap"][idx] if "gap" in h else float("nan")
        mark = "" if t_used == k else f"  (sample t={t_used})"
        print(
            f"  t<={k:7d}  Q={Q[idx]:.6f}  dQ={dQ[idx]:.3e}  "
            f"regx={rx:.6f}  gap={gp:.4e}{mark}"
        )
    n = len(Q)
    windows = [(n // 4, n // 2), (n // 2, 3 * n // 4), (3 * n // 4, n)]
    for lo, hi in windows:
        print(
            f"  Q index [{lo}:{hi}] increment {Q[hi - 1] - Q[lo - 1]:.6f}, "
            f"mean dQ {dQ[lo:hi].mean():.3e}  "
            f"(t={int(t[lo])}:{int(t[hi - 1])})"
        )
    sl = slice(n // 2, n)
    A_lin = np.vstack([np.ones(n - n // 2), t[sl].astype(float)]).T
    c_lin, _, _, _ = np.linalg.lstsq(A_lin, Q[sl], rcond=None)
    A_log = np.vstack([np.ones(n - n // 2), np.log(t[sl].astype(float))]).T
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
