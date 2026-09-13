"""Plot Step 16 L_F sweep from saved json/npz. Does not rerun the learner.

Reads exp_lf_c{c}_long when present (c=4,8 at T=10^5). The T=2e4 window
alone does not support a 'not sqrt(T)' claim for c=8.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from io_results import load_run  # noqa: E402

TABLE_T = 20_000
MU = 0.2

plt.rcParams.update(
    {
        "axes.prop_cycle": plt.cycler(
            color=["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
        ),
        "pdf.fonttype": 42,
        "font.size": 9,
    }
)


def c_tag(c: float) -> str:
    return f"{c:g}".replace(".", "p")


def _times(hist: dict) -> np.ndarray:
    n = len(hist["reg_x"])
    if "t" in hist:
        t = np.asarray(hist["t"], dtype=float).reshape(-1)
        if t.size == n:
            return t
    if "stride" in hist:
        stride = int(np.asarray(hist["stride"]).reshape(-1)[0])
        if "T" in hist:
            T = int(np.asarray(hist["T"]).reshape(-1)[0])
            t = np.arange(1, T + 1, dtype=float)[::stride]
            return t[:n]
        return 1.0 + stride * np.arange(n, dtype=float)
    return np.arange(1, n + 1, dtype=float)


def _load_curve(c: float) -> tuple[dict, dict, np.ndarray, str]:
    short = f"exp_lf_c{c_tag(c)}"
    long = f"{short}_long"
    if (ROOT / "results" / f"{long}.npz").is_file():
        payload, hist = load_run(long)
        return payload, hist, _times(hist), long
    payload, hist = load_run(short)
    return payload, hist, _times(hist), short


def main() -> None:
    payload = json.loads((ROOT / "results" / "exp_lf_sweep.json").read_text(encoding="utf-8"))
    rows = payload["summary"]["rows"]
    cs = [r["c"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), layout="constrained")
    ax_r, ax_p = axes

    tmax = TABLE_T
    for row in rows:
        c = float(row["c"])
        _payload, hist, t, _stem = _load_curve(c)
        tmax = max(tmax, float(t[-1]))
        ax_r.plot(t, hist["reg_x"], lw=1.3, label=rf"$c={c:g}$")

    ax_r.axvline(TABLE_T, color="0.35", ls="--", lw=1.0, zorder=3)
    if (ROOT / "results" / "exp_lf_c8.npz").is_file():
        _p8, h8 = load_run("exp_lf_c8")
        t8 = _times(h8)
        idx = int(np.where(t8 <= TABLE_T)[0][-1])
        t_mark = float(t8[idx])
        reg_ref = float(h8["reg_x"][idx])
        t_ref = np.linspace(1.0, tmax, 400)
        ax_r.plot(
            t_ref,
            reg_ref * np.sqrt(t_ref / t_mark),
            color="0.45",
            ls=":",
            lw=1.0,
            label=r"$\propto\sqrt{t}$ from $c=8$ at $T=2\times10^{4}$",
        )
    ax_r.set_xlabel(r"$t$")
    ax_r.set_ylabel(r"$\mathrm{Reg}^x(a)$")
    ax_r.set_xlim(1.0, tmax)
    ax_r.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")

    ells = [r["ell"] for r in rows]
    js = [r["J"] for r in rows]
    c_line = np.linspace(min(cs) * 0.9, max(cs) * 1.02, 80)
    (l_id,) = ax_p.plot(
        c_line,
        np.sqrt(2.0) * np.hypot(MU, c_line),
        color="0.45",
        ls=":",
        lw=1.1,
        label=r"$\sqrt{2}\,L_F$",
    )
    (l_ell,) = ax_p.plot(
        cs, ells, marker="o", lw=0.0, ms=6, color="C0", label=r"$\ell_T$"
    )
    ax_j = ax_p.twinx()
    (l_j,) = ax_j.plot(cs, js, marker="s", lw=1.2, ls="--", color="C1", label=r"$J$")
    ax_p.set_xlabel(r"$c$")
    ax_p.set_ylabel(r"$\ell_T$")
    ax_j.set_ylabel(r"$J$")
    ax_j.set_ylim(0, max(js) + 1.5)
    ax_p.legend(
        [l_ell, l_id, l_j],
        [l_ell.get_label(), l_id.get_label(), l_j.get_label()],
        frameon=False,
        fontsize=8,
    )

    out_pdf = ROOT / "figures" / "exp_lf_sweep.pdf"
    out_png = ROOT / "figures" / "exp_lf_sweep.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    missing = [
        c for c in (4.0, 8.0) if not (ROOT / "results" / f"exp_lf_c{c_tag(c)}_long.npz").is_file()
    ]
    if missing:
        print("warning: missing long dumps for c=", missing, "; overlay stops at T=2e4")


if __name__ == "__main__":
    main()
