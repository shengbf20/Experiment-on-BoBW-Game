"""Plot Part B realized-regret comparison. Log-only; not a paper figure."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from io_results import RESULTS, load_run  # noqa: E402
from separation import U_STAR  # noqa: E402

plt.rcParams.update(
    {
        "axes.prop_cycle": plt.cycler(color=["#0072B2", "#E69F00", "#009E73", "#CC79A7"]),
        "pdf.fonttype": 42,
        "font.size": 9,
    }
)


def _summary() -> tuple[list[dict], dict]:
    path = RESULTS / "exp_sep_partB.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = sorted(payload["summary"]["rows"], key=lambda row: int(row["T"]))
    return rows, payload["summary"]["verdict"]


def plot_part_b(rows: list[dict], verdict: dict) -> None:
    T = np.array([row["T"] for row in rows], dtype=float)
    r_d = np.array([row["R_D005"] for row in rows], dtype=float)
    r_h = np.array([row["R_Hsieh"] for row in rows], dtype=float)
    payload, hist = load_run(f"exp_sep_T{int(T[-1])}")
    _, hist_b = load_run(f"exp_sep_partB_T{int(T[-1])}")
    T_long = int(payload["T"])
    t = np.arange(1, T_long + 1)
    x_d = np.asarray(hist["x"], dtype=float).reshape(-1)
    x_h = np.asarray(hist_b["x_hsieh"], dtype=float).reshape(-1)

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.25), layout="constrained")

    ax = axes[0]
    ax.plot(T, r_d, "o-", color="C0", lw=1.3, ms=4.5, label=r"D005")
    ax.plot(T, r_h, "s-", color="C1", lw=1.3, ms=4.5, label=r"Hsieh OptDA")
    ax.set_xscale("log")
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$R_T(u^\star)$")
    ax.set_title(r"Realized regret on the frozen sequence")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(T, r_d / T, "o-", color="C0", lw=1.3, ms=4.5, label=r"D005")
    ax.plot(T, r_h / T, "s-", color="C1", lw=1.3, ms=4.5, label=r"Hsieh OptDA")
    ax.set_xscale("log")
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$R_T(u^\star)/T$")
    ax.set_title(r"Average regret")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ax.plot(t, x_d, color="C0", lw=1.0, label=r"D005")
    ax.plot(t, x_h, color="C1", lw=1.0, label=r"Hsieh OptDA")
    ax.axhline(U_STAR, color="0.5", ls=":", lw=1.0, label=rf"$u^\star={U_STAR:g}$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x_t$")
    ax.set_title(r"Actions, $T=10^5$")
    ax.legend(frameon=False, fontsize=8)

    fig.text(
        0.5,
        -0.02,
        r"Same D005-constructed frozen $y_{1:T}$; $\tau=1$ is not fitted. "
        r"Log-only diagnostic, not a paper figure. "
        r"Not a general algorithm ranking.",
        ha="center",
        fontsize=8,
    )

    out_pdf = ROOT / "figures" / "exp_sep_partB.pdf"
    out_png = ROOT / "figures" / "exp_sep_partB.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


def main() -> None:
    rows, verdict = _summary()
    if len(rows) < 2:
        raise SystemExit("need exp_sep_partB with at least two horizons")
    if not verdict["implementation_valid"]:
        raise SystemExit("Part B implementation checks failed")
    plot_part_b(rows, verdict)


if __name__ == "__main__":
    main()
