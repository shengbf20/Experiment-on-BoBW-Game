"""Plot Exp.1 from saved json. Does not rerun the learner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

# Colorblind-safe (Okabe-Ito) palette; "C0"/"C1" below resolve through it.
# fonttype 42 keeps pdf text as vector TrueType.
plt.rcParams.update(
    {
        "axes.prop_cycle": plt.cycler(color=["#0072B2", "#E69F00", "#009E73", "#CC79A7"]),
        "pdf.fonttype": 42,
        "font.size": 9,
    }
)


def _load(tag: str) -> dict:
    path = ROOT / "results" / f"exp1_{tag}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run exp1_selfplay.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def _panel_row(axes, hist: dict, title: str) -> None:
    T = len(hist["reg_x"])
    t = np.arange(1, T + 1)
    ax_r, ax_g, ax_q = axes

    ax_r.plot(t, hist["reg_x"], color="C0", lw=1.4, label=r"$\mathrm{Reg}^x(a)$")
    ax_r.plot(t, hist["reg_y"], color="C1", lw=1.4, ls="--", label=r"$\mathrm{Reg}^y(b)$")
    scale = abs(hist["reg_x"][min(99, T - 1)]) / np.sqrt(min(100, T))
    if scale > 0:
        ax_r.plot(t, scale * np.sqrt(t), color="0.5", ls=":", lw=1.0, label=r"$\propto\sqrt{t}$")
    ax_r.set_xlabel(r"$t$")
    ax_r.set_ylabel("individual regret")
    ax_r.set_title(title)
    ax_r.legend(frameon=False, fontsize=8)

    gap = np.maximum(np.asarray(hist["gap"], dtype=float), 1e-16)
    ax_g.loglog(t, gap, color="C0", lw=1.4, label="restricted gap")
    c = gap[min(99, T - 1)] * min(100, T)
    ax_g.loglog(t, c / t, color="0.5", ls=":", lw=1.0, label=r"$\propto 1/t$")
    ax_g.set_xlabel(r"$t$")
    ax_g.set_ylabel("restricted gap")
    ax_g.legend(frameon=False, fontsize=8)

    ax_q.plot(t, hist["Q"], color="C0", lw=1.4)
    ax_q.set_xlabel(r"$t$")
    ax_q.set_ylabel(r"$Q_t^{\mathrm{obs}}$")


def plot_main() -> None:
    g2 = _load("G2_identity")
    g1 = _load("G1_identity")
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), layout="constrained")
    _panel_row(axes[0], g2["hist"], "G2 identity, shifted saddle")
    _panel_row(axes[1], g1["hist"], "G1 identity, shifted saddle")
    out_pdf = ROOT / "figures" / "exp1_selfplay.pdf"
    out_png = ROOT / "figures" / "exp1_selfplay.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


def plot_appendix(game: str, seeds: list[int]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), layout="constrained")
    styles = ["-", "--", "-."]
    hist0 = None
    for seed, ls in zip(seeds, styles):
        data = _load(f"{game}_gaussian_seed{seed}")
        hist = data["hist"]
        hist0 = hist0 or hist
        T = len(hist["reg_x"])
        t = np.arange(1, T + 1)
        axes[0].plot(t, hist["reg_x"], lw=1.2, ls=ls, label=f"seed {seed}")
        gap = np.maximum(np.asarray(hist["gap"], dtype=float), 1e-16)
        axes[1].loglog(t, gap, lw=1.2, ls=ls, label=f"seed {seed}")
        axes[2].plot(t, hist["Q"], lw=1.2, ls=ls, label=f"seed {seed}")
    axes[0].set_title(f"{game} spectral-normalized gaussian A")
    axes[0].set_ylabel(r"$\mathrm{Reg}^x(a)$")
    axes[1].set_ylabel("restricted gap")
    axes[2].set_ylabel(r"$Q_t^{\mathrm{obs}}$")
    for ax in axes:
        ax.set_xlabel(r"$t$")
        ax.legend(frameon=False, fontsize=8)
    out = ROOT / "figures" / f"exp1_{game}_gaussian.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--appendix", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    plot_main()
    if args.appendix:
        plot_appendix("G1", args.seeds)
        plot_appendix("G2", args.seeds)


if __name__ == "__main__":
    main()
