"""Plot the closed Part A mechanism figure from saved results only.

The figure intentionally does not present the asymptotic tail proxy as an
actual certificate.  At T <= 20000, raw E_T is dominated by an O(1)
transient; the observable finite-horizon evidence is the exact V_T=1,
bounded G_T, and a tail contribution whose density converges to the positive
constant used in the proof.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from io_results import RESULTS, load_run  # noqa: E402
from separation import THEORY_LATE_DG2  # noqa: E402

plt.rcParams.update(
    {
        "axes.prop_cycle": plt.cycler(color=["#0072B2", "#E69F00", "#009E73", "#CC79A7"]),
        "pdf.fonttype": 42,
        "font.size": 9,
    }
)


def _rows() -> list[dict]:
    path = RESULTS / "exp_sep_partA.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return sorted(payload["summary"]["rows"], key=lambda row: int(row["T"]))


def plot_part_a(rows: list[dict]) -> None:
    T = np.array([row["T"] for row in rows], dtype=float)
    V = np.array([row["V_T"] for row in rows], dtype=float)
    G = np.array([row["G_T"] for row in rows], dtype=float)

    settled_rows = [row for row in rows if int(row["T"]) >= 5000]
    tail_rounds = np.array(
        [int(row["T"]) - int(row["late_start"]) + 1 for row in settled_rows],
        dtype=float,
    )
    tail_sum = np.array([row["late_sum_dg2"] for row in settled_rows], dtype=float)

    payload, hist = load_run(f"exp_sep_T{int(T[-1])}")
    T_long = int(payload["T"])
    t = np.arange(1, T_long + 1)
    x = np.asarray(hist["x"], dtype=float).reshape(-1)
    E = np.asarray(hist["E"], dtype=float).reshape(-1)
    half = T_long // 2
    late0 = 3 * T_long // 4

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.25), layout="constrained")

    ax = axes[0]
    ax.semilogx(T, V, "o-", color="C0", lw=1.3, ms=5, label=r"$V_T(u^\star)$")
    ax.semilogx(T, G, "s-", color="C1", lw=1.3, ms=5, label=r"$G_T$")
    ax.axhline(1.0, color="0.5", ls=":", lw=1.0)
    ax.set_xlabel(r"$T$")
    ax.set_ylabel("terminal value")
    ax.set_title(r"Exact $V_T(u^\star)=1$; bounded $G_T$")
    ax.set_ylim(0.0, 3.0)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    theory_tail = THEORY_LATE_DG2 * tail_rounds
    ax.loglog(
        tail_rounds,
        tail_sum,
        "o-",
        color="C2",
        lw=1.4,
        ms=5,
        label=r"observed $\Delta E_T^{\rm tail}$",
    )
    ax.loglog(
        tail_rounds,
        theory_tail,
        color="0.45",
        ls=":",
        lw=1.2,
        label=r"$c\,T_{\rm tail}$",
    )
    ax.set_xlabel(r"tail length $T_{\rm tail}$")
    ax.set_ylabel(r"$E_T-E_{3T/4}$")
    ax.set_title("Linear settled-tail contribution")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ax.plot(t, x, color="C0", lw=1.0)
    ax.axvline(half, color="0.35", ls="--", lw=1.0, label=r"$T/2$")
    ax.axhline(1.0, color="0.5", ls=":", lw=1.0, label=r"$u^\star=1$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x_t$")
    ax.set_title(r"Two settled regimes, $T=2\times10^4$")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    axins = inset_axes(ax, width="45%", height="38%", loc="center right", borderpad=0.8)
    tail_t = t[late0:] - t[late0 - 1]
    tail_E = E[late0:] - E[late0 - 1]
    axins.plot(tail_t, tail_E, color="C2", lw=1.2)
    axins.set_ylabel(r"$\Delta E_t$", fontsize=7)
    axins.set_title("last 5,000 rounds", fontsize=7, pad=2)
    axins.tick_params(labelsize=7)

    fig.text(
        0.5,
        -0.02,
        r"Raw $E_T$ is transient-dominated for $T\leq2\times10^4$; the plotted tail is a mechanism diagnostic, not a certificate.",
        ha="center",
        fontsize=8,
    )

    out_pdf = ROOT / "figures" / "exp_sep_partA.pdf"
    out_png = ROOT / "figures" / "exp_sep_partA.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


def main() -> None:
    rows = _rows()
    if len(rows) < 2:
        raise SystemExit("need exp_sep_partA with at least two horizons")
    plot_part_a(rows)


if __name__ == "__main__":
    main()
