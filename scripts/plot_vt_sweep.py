"""Plot Step 17 V_T experiments from saved dumps. Does not rerun learners."""

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


def eta_tag(eta: float) -> str:
    return f"{eta:g}".replace(".", "p")


def plot_17a(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), layout="constrained")
    ax_t, ax_s = axes
    sqrtV = []
    regs = []
    lins = []
    for row in rows:
        _, hist = load_run(f"exp_vt17a_eta{eta_tag(row['eta'])}")
        t = np.arange(1, len(hist["reg_x"]) + 1)
        ax_t.plot(t, hist["reg_x"], lw=1.3, label=rf"$\eta={row['eta']:g}$")
        sqrtV.append(row["sqrtV"])
        regs.append(row["reg"])
        lins.append(row["lin"])
    ax_t.set_xlabel(r"$t$")
    ax_t.set_ylabel(r"$\mathrm{Reg}^x(u)$")
    ax_t.legend(frameon=False, fontsize=8, ncol=2)

    ax_s.plot(sqrtV, lins, marker="o", lw=1.4, color="C0", label=r"$\mathrm{LinReg}^x(u)$")
    ax_s.plot(sqrtV, regs, marker="s", lw=1.2, ls="--", color="C1", label=r"$\mathrm{Reg}^x(u)$")
    if sqrtV[-1] > 0:
        scale = max(lins) / max(sqrtV[-1], 1e-12)
        xs = np.linspace(0.0, sqrtV[-1], 50)
        ax_s.plot(xs, scale * xs, color="0.5", ls=":", lw=1.0, label=r"$\propto\sqrt{V}$")
    ax_s.set_xlabel(r"$\sqrt{V_T(u)}$")
    ax_s.set_ylabel("terminal")
    ax_s.legend(frameon=False, fontsize=8)

    out_pdf = ROOT / "figures" / "exp_vt_17a.pdf"
    out_png = ROOT / "figures" / "exp_vt_17a.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


def plot_17b() -> None:
    p_d, h_d = load_run("exp_vt17b_d005")
    p_h, h_h = load_run("exp_vt17b_hsieh")
    y = np.asarray(h_d["y"], dtype=float)
    t = np.arange(1, len(h_d["reg_x"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), layout="constrained")
    ax_r, ax_v = axes
    ax_r.plot(t, h_d["reg_x"], lw=1.4, color="C0")
    ax_r.set_xlabel(r"$t$")
    ax_r.set_ylabel(r"$\mathrm{Reg}^x(u^\star)$")

    ax_v.plot(t, h_d["V_x"], lw=1.4, color="C0", label=r"$V_t(u^\star)$")
    ax_v.set_xlabel(r"$t$")
    ax_v.set_ylabel(r"$V_t(u^\star)$")
    ax_ins = ax_v.inset_axes([0.48, 0.12, 0.48, 0.38])
    n_ins = min(len(y), int(3 * p_d["summary"]["T_per"]))
    ax_ins.plot(t[:n_ins], y[:n_ins], lw=0.8, color="C1")
    ax_ins.set_title(r"$y_t$, first 3 periods", fontsize=7, pad=2)
    ax_ins.tick_params(labelsize=6)
    ax_v.legend(frameon=False, fontsize=8, loc="upper left")
    out_pdf = ROOT / "figures" / "exp_vt_17b.pdf"
    out_png = ROOT / "figures" / "exp_vt_17b.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    print("17b summary", p_d.get("summary"), "hsieh", p_h.get("summary"))


def main() -> None:
    payload = json.loads((ROOT / "results" / "exp_vt_sweep.json").read_text(encoding="utf-8"))
    rows = payload["summary"]["17a"]
    if rows:
        plot_17a(rows)
    if payload["summary"].get("17b") is not None:
        plot_17b()


if __name__ == "__main__":
    main()
