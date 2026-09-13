"""Plot the retained same-run switch result without rerunning the learner."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from io_results import load_run  # noqa: E402

# Colorblind-safe (Okabe-Ito) palette; "C0"/"C1" below resolve through it.
# fonttype 42 keeps pdf text as vector TrueType.
plt.rcParams.update(
    {
        "axes.prop_cycle": plt.cycler(color=["#0072B2", "#E69F00", "#009E73", "#CC79A7"]),
        "pdf.fonttype": 42,
        "font.size": 9,
    }
)


def _load(tag: str) -> tuple[dict, dict]:
    return load_run(f"exp2_{tag}")


def period_max(t, y, t0: int, period: int):
    """One maximum per period for t >= t0. Avoids a dense post-switch smear."""
    t = np.asarray(t)
    y = np.asarray(y, dtype=float)
    t_end = int(t[-1])
    out_t = []
    out_y = []
    start = int(t0)
    min_len = max(2, period // 2)
    while start <= t_end:
        stop = min(start + period - 1, t_end)
        if stop - start + 1 < min_len:
            break
        m = (t >= start) & (t <= stop)
        if np.any(m):
            out_t.append(0.5 * (start + stop))
            out_y.append(float(np.max(y[m])))
        start += period
    return np.asarray(out_t), np.asarray(out_y)


def main():
    sw, h2a = _load("G2_switch")
    T2a = len(h2a["V_x"] if "V_x" in h2a else h2a["reg_x"])
    t2a = np.arange(1, T2a + 1)
    half = int(sw["meta"]["half"])
    period = int(sw["meta"].get("period", 200))

    fig, ax = plt.subplots(figsize=(5.2, 3.4), layout="constrained")
    # Signed Reg^x(a) is in the json; do not put it back on these axes.
    pre = t2a <= half
    (l_y,) = ax.plot(
        t2a[pre],
        h2a["dist_y"][pre],
        color="C1",
        lw=1.2,
        zorder=2,
        label=r"$\|y_t-b\|$",
    )
    ax.plot(*period_max(t2a, h2a["dist_y"], half, period), color="C1", lw=1.2, zorder=2)
    (l_x,) = ax.plot(
        t2a[pre],
        h2a["dist_x"][pre],
        color="C0",
        lw=1.8,
        zorder=4,
        label=r"$\|x_t-a\|$",
    )
    ax.plot(*period_max(t2a, h2a["dist_x"], half, period), color="C0", lw=1.8, zorder=4)
    vline = ax.axvline(half, color="0.35", ls="--", lw=1.0, zorder=3, label=r"$T/2$ switch")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("distance to saddle")
    ax.set_ylim(0.0, 1.25)
    axr = ax.twinx()
    (l_v,) = axr.plot(t2a, h2a["V_x"], color="C2", lw=1.2, zorder=1, label=r"$V_t^x(a)$")
    axr.set_ylabel(r"$V_t^x(a)$")
    ax.set_title("G2 same-run switch (no reset)")
    ax.legend(
        [l_x, l_y, l_v, vline],
        [l_x.get_label(), l_y.get_label(), l_v.get_label(), vline.get_label()],
        frameon=False,
        fontsize=8,
        loc="upper left",
    )
    t0, t1 = half, half + 3 * period
    m = (t2a >= t0) & (t2a <= t1)
    axins = ax.inset_axes([0.14, 0.16, 0.32, 0.40])
    axins.set_facecolor("white")
    axins.plot(t2a[m], h2a["dist_y"][m], color="C1", lw=1.0)
    axins.plot(t2a[m], h2a["dist_x"][m], color="C0", lw=1.4)
    axins.set_xlim(float(t0), float(t1))
    axins.set_ylim(0.0, 1.15)
    axins.set_xticks([t0, t1])
    axins.tick_params(labelsize=7)
    axins.set_title(r"$T/2$ to $+3$ periods", fontsize=7, pad=1)

    out_pdf = ROOT / "figures" / "exp2_bobw.pdf"
    out_png = ROOT / "figures" / "exp2_bobw.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
