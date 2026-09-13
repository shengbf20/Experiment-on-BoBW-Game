"""Plot the improved Part A finite-horizon separation from saved results."""

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
from separation import THEORY_LATE_DG2, U_STAR  # noqa: E402

plt.rcParams.update(
    {
        "axes.prop_cycle": plt.cycler(color=["#0072B2", "#E69F00", "#009E73", "#CC79A7"]),
        "pdf.fonttype": 42,
        "font.size": 9,
    }
)


def _summary() -> tuple[list[dict], dict]:
    path = RESULTS / "exp_sep_partA.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = sorted(payload["summary"]["rows"], key=lambda row: int(row["T"]))
    return rows, payload["summary"]["verdict"]


def plot_part_a(rows: list[dict], verdict: dict) -> None:
    T = np.array([row["T"] for row in rows], dtype=float)
    V = np.array([row["V_T"] for row in rows], dtype=float)
    E_terminal = np.array([row["E_T"] for row in rows], dtype=float)
    G = np.array([row["G_T"] for row in rows], dtype=float)
    E_over_T = E_terminal / T
    target_min = int(verdict["target_window_min_T"])
    target = T >= target_min

    payload, hist = load_run(f"exp_sep_T{int(T[-1])}")
    T_long = int(payload["T"])
    t = np.arange(1, T_long + 1)
    x = np.asarray(hist["x"], dtype=float).reshape(-1)
    E = np.asarray(hist["E"], dtype=float).reshape(-1)
    half = T_long // 2
    late0 = 3 * T_long // 4

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.25), layout="constrained")

    ax = axes[0]
    ax.loglog(T, V, "o-", color="C0", lw=1.3, ms=4.5, label=r"$V_T(u^\star)$")
    ax.loglog(T, G, "s-", color="C1", lw=1.3, ms=4.5, label=r"$G_T$")
    ax.loglog(T, E_terminal, "^-", color="C2", lw=1.4, ms=4.5, label=r"raw $E_T$")
    ax.loglog(
        T,
        THEORY_LATE_DG2 * T,
        color="0.45",
        ls=":",
        lw=1.1,
        label=r"$cT$",
    )
    ax.set_xlabel(r"$T$")
    ax.set_ylabel("terminal value")
    ax.set_title(r"Finite-horizon separation of $V_T(u^\star)$ and $E_T$")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.loglog(T, E_over_T, "o-", color="0.7", lw=1.0, ms=4, label="all horizons")
    ax.loglog(
        T[target],
        E_over_T[target],
        "o-",
        color="C2",
        lw=1.5,
        ms=5,
        label=r"$10^4\!\leq T\leq10^5$",
    )
    ax.axhline(
        THEORY_LATE_DG2,
        color="0.35",
        ls=":",
        lw=1.2,
        label=r"theory $c$",
    )
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$E_T/T$")
    ax.set_title(r"$E_T/T$ stabilizes toward $c>0$")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ax.plot(t, x, color="C0", lw=1.0)
    ax.axvline(half, color="0.35", ls="--", lw=1.0, label=r"$T/2$")
    ax.axhline(U_STAR, color="0.5", ls=":", lw=1.0, label=rf"$u^\star={U_STAR:g}$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x_t$")
    ax.set_title(r"Two settled regimes, $T=10^5$")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    axins = inset_axes(ax, width="45%", height="38%", loc="center right", borderpad=0.8)
    tail_t = t[late0:] - t[late0 - 1]
    tail_E = E[late0:] - E[late0 - 1]
    axins.plot(tail_t, tail_E, color="C2", lw=1.2)
    axins.set_ylabel(r"$\Delta E_t$", fontsize=7)
    axins.set_title("last 25,000 rounds", fontsize=7, pad=2)
    axins.tick_params(labelsize=7)

    fig.text(
        0.5,
        -0.02,
        rf"$E_T$ shows near-linear growth on $10^4\leq T\leq10^5$ "
        rf"(log--log slope ${verdict['loglog_slope_E_vs_T_target_window']:.3f}$); "
        rf"$E_T/T$ approaches $c={THEORY_LATE_DG2:.3g}$. "
        r"Asymptotic $\Theta(T)$ is from Appendix A, not the finite-horizon fit.",
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
    rows, verdict = _summary()
    if len(rows) < 2:
        raise SystemExit("need exp_sep_partA with at least two horizons")
    if not verdict["raw_ET_scaling_observed"]:
        raise SystemExit("raw E_T scaling did not pass the predeclared test")
    plot_part_a(rows, verdict)


if __name__ == "__main__":
    main()
