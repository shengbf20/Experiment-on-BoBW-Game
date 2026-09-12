"""Step 16: G2 self-play L_F sweep, A = c I, no spectral normalization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from games import QuadraticGame, assert_saddle_comparator, paper_saddle  # noqa: E402
from io_results import LONG_STRIDE, dump_compact, load_run  # noqa: E402
from learner import ClosedFormPlayer, self_play  # noqa: E402

_DTYPE = np.float64
CS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
LONG_KEYS = (
    "reg_x",
    "reg_y",
    "Q",
    "gap",
    "dist_x",
    "dist_y",
    "J_x",
    "J_y",
    "ell_x",
    "ell_y",
    "beta_x",
    "beta_y",
)


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


def c_tag(c: float) -> str:
    return f"{c:g}".replace(".", "p")


def result_stem(c: float, T: int, default_T: int) -> str:
    base = f"exp_lf_c{c_tag(c)}"
    return f"{base}_long" if T > default_T else base


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
        raise AssertionError(f"{tag}: expected J>=1, got J_x={jx}, J_y={jy}")
    js = np.asarray(hist["J_x"])
    if np.any(np.diff(js) < 0):
        raise AssertionError(f"{tag}: J_x decreased")
    if hist["beta_x"][-1] < hist["beta_x"][0] - 1e-15:
        raise AssertionError(f"{tag}: beta_x decreased")
    if hist["ell_x"][-1] < hist["ell_x"][0] - 1e-15:
        raise AssertionError(f"{tag}: ell_x decreased")


def _check_platform(hist: dict, T: int, tag: str, c: float) -> dict:
    """Q controlled; last-iterate approaches the saddle. Strict freeze only for small c.

    Large c freezes J and Q immediately but crawls under a large beta; that is a
    late platform (PLAN §7: increase T), not a sqrt(T) failure.
    """
    reg = np.asarray(hist["reg_x"], dtype=float)
    Q = np.asarray(hist["Q"], dtype=float)
    dist = np.asarray(hist["dist_x"], dtype=float)
    half = T // 2
    late_ptp = float(np.ptp(reg[half:]))
    q_tail = float(Q[-1] - Q[int(0.75 * T)])
    q_tot = float(Q[-1])
    n = min(2500, max(T // 8, 1))
    early_inc = float(reg[min(T - 1, n + 999)] - reg[min(T - 1, 999)])
    late_inc = float(reg[-1] - reg[-1 - n])
    if q_tot > 1e-8 and q_tail > 0.35 * q_tot:
        raise AssertionError(
            f"{tag}: Q still accumulating in last quarter: tail={q_tail:.3g}, Q_T={q_tot:.3g}"
        )
    if dist[-1] > dist[half] + 1e-6:
        raise AssertionError(
            f"{tag}: distance to saddle grew in the second half: "
            f"{dist[half]:.3g} -> {dist[-1]:.3g}"
        )
    if late_inc > max(early_inc, 1.0) * 1.05:
        raise AssertionError(
            f"{tag}: late regret increment {late_inc:.3g} exceeds early {early_inc:.3g}"
        )
    frozen = bool(late_ptp <= max(0.05, 0.01 * abs(float(reg[-1]))))
    if c <= 2.0 and not frozen:
        raise AssertionError(
            f"{tag}: expected a frozen platform for c={c:g} at T={T}, ptp={late_ptp:.3g}"
        )
    return {
        "reg_second_half_ptp": late_ptp,
        "Q_last_quarter": q_tail,
        "late_inc": late_inc,
        "dist_T": float(dist[-1]),
        "frozen": frozen,
    }


def _check_sweep(rows: list[dict]) -> None:
    js = [r["J"] for r in rows]
    ells = [r["ell"] for r in rows]
    betas = [r["beta"] for r in rows]
    if any(j < 1 or j > 30 for j in js):
        raise AssertionError(f"J not finite/reasonable: {js}")
    for i in range(1, len(rows)):
        if js[i] < js[i - 1]:
            raise AssertionError(f"J decreased with c: {js}")
        if ells[i] < ells[i - 1] - 1e-12:
            raise AssertionError(f"ell_T decreased with c: {ells}")
        if betas[i] < betas[i - 1] - 1e-12:
            raise AssertionError(f"beta_T decreased with c: {betas}")
    if any(abs(r["A_op"] - r["c"]) > 1e-10 for r in rows):
        raise AssertionError("spectral normalization leaked: ||A||_2 != c")


def run_one(c: float, cfg: dict, T: int, radius: float, dim: int, default_T: int) -> dict:
    sx, sy = paper_saddle(dim)
    A = c * np.eye(dim, dtype=_DTYPE)
    op = float(np.linalg.norm(A, 2))
    if abs(op - c) > 1e-12:
        raise AssertionError(f"A=cI must have ||A||_2=c; got {op} vs {c}")
    game = QuadraticGame(dim=dim, mu=0.2, A=A, saddle_x=sx, saddle_y=sy)
    px, py = _player(dim, cfg), _player(dim, cfg)
    if np.any(px.action) or np.any(py.action):
        raise AssertionError("must not overwrite w1=0")
    metrics, hist, max_w = self_play(game, px, py, T=T, radius=radius)
    assert_saddle_comparator(game, metrics)
    tag = f"lf_c{c_tag(c)}"
    _smoke(hist, max_w, T, tag)
    plat = _check_platform(hist, T, tag, c)
    mu = 0.2
    L_op = float(np.sqrt(mu * mu + c * c))
    row = {
        "c": float(c),
        "A_op": op,
        "L_op": L_op,
        "J": int(hist["J_x"][-1]),
        "J_y": int(hist["J_y"][-1]),
        "ell": float(hist["ell_x"][-1]),
        "ell_y": float(hist["ell_y"][-1]),
        "beta": float(hist["beta_x"][-1]),
        "beta_y": float(hist["beta_y"][-1]),
        "reg_x": float(hist["reg_x"][-1]),
        "reg_y": float(hist["reg_y"][-1]),
        "gap": float(hist["gap"][-1]),
        "Q": float(hist["Q"][-1]),
        "t_last_double": _last_increase(hist["J_x"]),
        "max_w": float(max_w),
        **plat,
    }
    payload = {
        "meta": {
            "tag": tag,
            "game": "G2",
            "A": "cI",
            "c": float(c),
            "normalize": False,
            "mu": mu,
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
            "L_op": L_op,
            "note": "A=cI, no spectral normalization. Algorithm does not read L_op.",
        },
        "summary": row,
    }
    is_long = T > default_T
    dump_compact(
        result_stem(c, T, default_T),
        payload,
        hist,
        keys=LONG_KEYS if is_long else None,
        stride=LONG_STRIDE if is_long else None,
    )
    print(
        f"c={c:g} J={row['J']} ell={row['ell']:.4g} beta={row['beta']:.4g} "
        f"Reg={row['reg_x']:.4g} gap={row['gap']:.3e} Q={row['Q']:.4g} "
        f"L_op={L_op:.4g} lastJ@{row['t_last_double']} frozen={row['frozen']}",
        flush=True,
    )
    return row


def _fmt_c(c: float) -> str:
    return f"{c:g}"


def _fmt_ell(x: float) -> str:
    if x >= 10:
        return f"{x:.2f}"
    if x >= 1:
        return f"{x:.3g}"
    return f"{x:.3f}"


def _fmt_beta(x: float) -> str:
    if x >= 1000:
        return f"{x:.0f}"
    if x >= 100:
        return f"{x:.1f}"
    return f"{x:.2f}"


def _fmt_gap(x: float) -> str:
    mant, exp = f"{x:.2e}".split("e")
    return rf"{float(mant):.2f}\times 10^{{{int(exp)}}}"


def latex_table(rows: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{rrrrrrr}",
        r"\hline",
        r"$c$ & $J$ & $\ell_T$ & $\beta_T$ & $\mathrm{Reg}^x(a)$ & $\mathrm{Gap}_T$ & $Q_T$ \\",
        r"\hline",
    ]
    for r in rows:
        lines.append(
            f"${_fmt_c(r['c'])}$ & ${r['J']}$ & ${_fmt_ell(r['ell'])}$ & "
            f"${_fmt_beta(r['beta'])}$ & ${r['reg_x']:.3f}$ & ${_fmt_gap(r['gap'])}$ & "
            f"${r['Q']:.4f}$ \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def _write_table(rows: list[dict], T: int) -> None:
    _check_sweep(rows)
    payload = {
        "meta": {
            "tag": "lf_sweep",
            "game": "G2",
            "A": "cI",
            "normalize": False,
            "cs": [float(r["c"]) for r in rows],
            "T": T,
            "dim": 10,
            "mu": 0.2,
            "protocol": "self-play, paper saddle, w1=0, no spectral normalization",
        },
        "summary": {"rows": rows},
    }
    dump_compact("exp_lf_sweep", payload)
    tex = latex_table(rows)
    tex_path = ROOT / "results" / "lf_sweep.tex"
    tex_path.write_text(tex + "\n", encoding="utf-8")
    print(f"wrote {tex_path}", flush=True)
    print(tex, flush=True)


def assemble(T: int, default_T: int) -> list[dict]:
    rows = []
    for c in CS:
        payload, _hist = load_run(result_stem(c, T, default_T))
        meta = payload.get("meta", {})
        if int(meta.get("T", T)) != T:
            raise AssertionError(f"c={c:g} dump has T={meta.get('T')}, expected {T}")
        if meta.get("normalize") is True:
            raise AssertionError(f"c={c:g} was spectrally normalized")
        rows.append(payload["summary"])
    _write_table(rows, T)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--c", type=float, nargs="+", default=None)
    parser.add_argument(
        "--assemble",
        action="store_true",
        help="build the table from existing T-default dumps; do not rerun",
    )
    args = parser.parse_args()

    cfg = _load_cfg()
    default_T = int(cfg["T"])
    T = int(args.T if args.T is not None else default_T)
    if args.assemble:
        assemble(T, default_T)
        return

    radius = float(cfg["gap_radius"])
    dim = int(args.dim)
    cs = tuple(args.c) if args.c is not None else CS
    for c in cs:
        run_one(float(c), cfg, T, radius, dim, default_T)
    if tuple(float(c) for c in cs) == CS:
        assemble(T, default_T)
    else:
        print("subset run; not rewriting exp_lf_sweep.json (pass --assemble after a full set)", flush=True)


if __name__ == "__main__":
    main()
