"""Plot Step 16 L_F sweep from saved json/npz. Does not rerun the learner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from io_results import load_run  # noqa: E402

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


def main() -> None:
    payload = json.loads((ROOT / "results" / "exp_lf_sweep.json").read_text(encoding="utf-8"))
    rows = payload["summary"]["rows"]
    cs = [r["c"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), layout="constrained")
    ax_r, ax_p = axes

    for row in rows:
        _, hist = load_run(f"exp_lf_c{c_tag(row['c'])}")
        t = np.arange(1, len(hist["reg_x"]) + 1)
        ax_r.plot(t, hist["reg_x"], lw=1.3, label=rf"$c={row['c']:g}$")
    ax_r.set_xlabel(r"$t$")
    ax_r.set_ylabel(r"$\mathrm{Reg}^x(a)$")
    ax_r.legend(frameon=False, fontsize=8, ncol=2)

    ells = [r["ell"] for r in rows]
    js = [r["J"] for r in rows]
    ax_p.plot(cs, ells, marker="o", lw=1.4, color="C0", label=r"$\ell_T$")
    ax_j = ax_p.twinx()
    ax_j.plot(cs, js, marker="s", lw=1.2, ls="--", color="C1", label=r"$J$")
    ax_p.set_xlabel(r"$c$")
    ax_p.set_ylabel(r"$\ell_T$")
    ax_j.set_ylabel(r"$J$")
    ax_j.set_ylim(0, max(js) + 1.5)
    h1, l1 = ax_p.get_legend_handles_labels()
    h2, l2 = ax_j.get_legend_handles_labels()
    ax_p.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8)

    out_pdf = ROOT / "figures" / "exp_lf_sweep.pdf"
    out_png = ROOT / "figures" / "exp_lf_sweep.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
