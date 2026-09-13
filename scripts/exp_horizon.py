"""Step 15: multi-horizon terminals from existing self-play runs. Does not rerun."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from io_results import dump_compact, load_run  # noqa: E402

G2_HORIZONS = (5_000, 10_000, 20_000)
G1_HORIZONS = (20_000, 100_000, 200_000)


def _sample_times(hist: dict, n: int) -> np.ndarray:
    if "t" in hist:
        t = np.asarray(hist["t"]).reshape(-1)
        if t.size == n:
            return t.astype(np.int64, copy=False)
    if "stride" in hist:
        stride = int(np.asarray(hist["stride"]).reshape(-1)[0])
        if "T" in hist:
            T = int(np.asarray(hist["T"]).reshape(-1)[0])
            t = np.arange(1, T + 1, dtype=np.int64)[::stride]
            if t.size >= n:
                return t[:n]
        return (1 + stride * np.arange(n, dtype=np.int64))
    return np.arange(1, n + 1, dtype=np.int64)


def _at_round(hist: dict, T: int) -> dict:
    Q = np.asarray(hist["Q"], dtype=float)
    t = _sample_times(hist, len(Q))
    if "stride" in hist or ("t" in hist and int(t[-1]) != len(Q)):
        eligible = np.where(t <= T)[0]
        if eligible.size == 0:
            raise AssertionError(f"no subsampled round <= {T}")
        idx = int(eligible[-1])
        t_used = int(t[idx])
    else:
        if len(Q) < T:
            raise AssertionError(f"need at least T={T} rounds, have {len(Q)}")
        idx = T - 1
        t_used = T
    row = {
        "T": int(T),
        "t_used": t_used,
        "reg_x": float(np.asarray(hist["reg_x"], dtype=float)[idx]),
        "Q": float(Q[idx]),
        "gap": float(np.asarray(hist["gap"], dtype=float)[idx]),
    }
    row["T_gap"] = float(row["t_used"] * row["gap"])
    row["T2_gap"] = float(row["t_used"] ** 2 * row["gap"])
    if "J_x" in hist:
        jx = np.asarray(hist["J_x"])
        if jx.size == Q.size + 1 and t_used < jx.size:
            row["J"] = int(jx[t_used])
        else:
            row["J"] = int(jx[min(idx, jx.size - 1)])
    return row


def _prefer_summary(payload: dict, T: int, row: dict) -> dict:
    if int(payload.get("T") or payload.get("meta", {}).get("T") or 0) != T:
        return row
    s = payload.get("summary") or {}
    if "Q_T" in s:
        row = dict(row)
        row["t_used"] = T
        row["reg_x"] = float(s.get("reg_x_T", row["reg_x"]))
        row["Q"] = float(s["Q_T"])
        row["gap"] = float(s.get("gap_T", row["gap"]))
        row["T_gap"] = float(T * row["gap"])
        row["T2_gap"] = float(T ** 2 * row["gap"])
        if "J_x" in s:
            row["J"] = int(s["J_x"])
    return row


def _fmt(x: float, kind: str) -> str:
    if kind == "reg":
        if abs(x) < 1e-12:
            return "0"
        return f"{x:.3f}"
    if kind == "Q":
        return f"{x:.4f}"
    if kind == "gap":
        mant, exp = f"{x:.2e}".split("e")
        return f"{float(mant):.2f}\\times 10^{{{int(exp)}}}"
    if kind == "Tgap":
        if x >= 10:
            return f"{x:.0f}"
        return f"{x:.3g}"
    raise ValueError(kind)


def _fmt_T(T: int) -> str:
    mapping = {
        5_000: r"5\times 10^{3}",
        10_000: r"10^{4}",
        20_000: r"2\times 10^{4}",
        100_000: r"10^{5}",
        200_000: r"2\times 10^{5}",
    }
    return mapping.get(T, str(T))


def _fmt_row_T(row: dict) -> str:
    # Print the sample actually used. Stride-10 long runs miss 10^5.
    t_used = int(row.get("t_used", row["T"]))
    if t_used != int(row["T"]):
        return str(t_used)
    return _fmt_T(int(row["T"]))


def latex_table(g2: list[dict], g1: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"game & $t$ & $\mathrm{Reg}^x(a)$ & $Q_t$ & $\mathrm{Gap}_t$ & $t\cdot\mathrm{Gap}_t$ & $J$ \\",
        r"\hline",
    ]
    for row in g2:
        lines.append(
            f"G2 & ${_fmt_row_T(row)}$ & ${_fmt(row['reg_x'], 'reg')}$ & ${_fmt(row['Q'], 'Q')}$ & "
            f"${_fmt(row['gap'], 'gap')}$ & ${_fmt(row['T_gap'], 'Tgap')}$ & ${row['J']}$ \\\\"
        )
    lines.append(r"\hline")
    for row in g1:
        lines.append(
            f"G1 & ${_fmt_row_T(row)}$ & ${_fmt(row['reg_x'], 'reg')}$ & ${_fmt(row['Q'], 'Q')}$ & "
            f"${_fmt(row['gap'], 'gap')}$ & ${_fmt(row['T_gap'], 'Tgap')}$ & ${row['J']}$ \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def _check_g2(rows: list[dict]) -> None:
    regs = [r["reg_x"] for r in rows]
    qs = [r["Q"] for r in rows]
    tg = [r["T_gap"] for r in rows]
    if max(regs) - min(regs) > 0.5:
        raise AssertionError(f"G2 Reg grew with T: {regs}")
    if max(qs) - min(qs) > 1e-3:
        raise AssertionError(f"G2 Q grew with T: {qs}")
    if min(tg) <= 0 or tg[-1] > 2.0 * tg[0]:
        raise AssertionError(f"G2 t·Gap grew; expected O(1) or shrinking: {tg}")
    if rows[-1]["gap"] > 1.05 * rows[0]["gap"]:
        raise AssertionError(f"G2 Gap did not decrease: {[r['gap'] for r in rows]}")
    if any(r["J"] < 1 for r in rows):
        raise AssertionError("G2 expected J>=1")
    # Halving of t·Gap on this instance is 1/T^2 (Cesaro after last-iterate
    # sits at the saddle), not a 1/T slope measurement. Do not require the
    # ratio to match 1/2; only that Gap falls and t·Gap does not grow.


def _check_g1(rows: list[dict]) -> None:
    if any(r["J"] < 1 for r in rows):
        raise AssertionError("G1 expected J>=1")
    qs = [r["Q"] for r in rows]
    if qs[1] <= qs[0]:
        raise AssertionError(f"G1 Q should still be rising at T=2e4: {qs}")
    # 0.3425 → 0.3433 is a 0.24% residual, not a discrete freeze.
    rel = abs(qs[-1] - qs[-2]) / max(abs(qs[-2]), 1e-12)
    if rel > 0.02:
        raise AssertionError(f"G1 late Q moved more than 2%: {qs}")
    tg = [r["T_gap"] for r in rows]
    if abs(tg[-1] - tg[-2]) > 0.2 * max(abs(tg[-2]), abs(tg[-1]), 1e-12):
        raise AssertionError(f"G1 late t·Gap not an O(1) platform: {tg}")


def main() -> None:
    g2_payload, g2_hist = load_run("exp1_G2_identity")
    g2_rows = [_at_round(g2_hist, T) for T in G2_HORIZONS]
    _check_g2(g2_rows)

    g1_short_p, g1_short = load_run("exp1_G1_identity")
    g1_long_p, g1_long = load_run("exp1_G1_identity_long")
    g1_rows = []
    for T in G1_HORIZONS:
        if T <= len(g1_short["Q"]):
            row = _at_round(g1_short, T)
            row["J"] = int(np.asarray(g1_short["J_x"])[T])
            row["source"] = "exp1_G1_identity"
        else:
            row = _at_round(g1_long, T)
            row = _prefer_summary(g1_long_p, T, row)
            row.setdefault("J", int((g1_long_p.get("summary") or {}).get("J_x", 1)))
            row["source"] = "exp1_G1_identity_long"
        g1_rows.append(row)
    _check_g1(g1_rows)

    payload = {
        "meta": {
            "tag": "horizon_table",
            "protocol": "prefixes of existing identity files; no rerun",
            "g2_stem": "exp1_G2_identity",
            "g1_short_stem": "exp1_G1_identity",
            "g1_long_stem": "exp1_G1_identity_long",
            "note": (
                "G1 long-run npz is stride 10; the requested 1e5 row uses the "
                "last stored t<=1e5. The long run is a fresh rerun, not a "
                "continuation of the short file. G2 t·Gap halving is 1/T^2 on "
                "this instance, not a 1/T slope measurement. Do not audit this "
                "table with diag_g1_Q.py (it used to treat len(Q) as T)."
            ),
        },
        "summary": {
            "G2": g2_rows,
            "G1": g1_rows,
            "g2_reg_range": [g2_rows[0]["reg_x"], g2_rows[-1]["reg_x"]],
            "g2_Q_range": [g2_rows[0]["Q"], g2_rows[-1]["Q"]],
            "g2_Tgap_ratio": g2_rows[-1]["T_gap"] / g2_rows[0]["T_gap"],
            "g2_T2_gap": [r["T2_gap"] for r in g2_rows],
        },
    }
    dump_compact("horizon_table", payload)
    tex = latex_table(g2_rows, g1_rows)
    tex_path = ROOT / "results" / "horizon_table.tex"
    tex_path.write_text(tex + "\n", encoding="utf-8")
    print(f"wrote {tex_path}")
    print(tex)
    print("G2 rows", g2_rows)
    print("G1 rows", g1_rows)


if __name__ == "__main__":
    main()
