"""Plot retained self-play results without rerunning the learner."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml

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


def _load(tag: str) -> dict:
    payload, hist = load_run(f"exp1_{tag}")
    return {"meta": payload.get("meta", payload), "summary": payload.get("summary", {}), "hist": hist}


def _panel_row(axes, hist: dict, title: str, t=None) -> None:
    series_t = t if t is not None else np.arange(1, len(hist["reg_x"]) + 1)
    ax_r, ax_g, ax_q = axes

    ax_r.plot(series_t, hist["reg_x"], color="C0", lw=1.4, label=r"$\mathrm{Reg}^x(a)$")
    ax_r.plot(series_t, hist["reg_y"], color="C1", lw=1.4, ls="--", label=r"$\mathrm{Reg}^y(b)$")
    idx100 = min(99, len(hist["reg_x"]) - 1)
    scale = abs(hist["reg_x"][idx100]) / np.sqrt(float(series_t[idx100]))
    if scale > 0:
        ax_r.plot(series_t, scale * np.sqrt(series_t), color="0.5", ls=":", lw=1.0, label=r"$\propto\sqrt{t}$")
    ax_r.set_xlabel(r"$t$")
    ax_r.set_ylabel("individual regret")
    ax_r.set_title(title)
    ax_r.legend(frameon=False, fontsize=8)

    gap = np.maximum(np.asarray(hist["gap"], dtype=float), 1e-16)
    ax_g.loglog(series_t, gap, color="C0", lw=1.4, label="restricted gap")
    c = gap[idx100] * float(series_t[idx100])
    ax_g.loglog(series_t, c / series_t, color="0.5", ls=":", lw=1.0, label=r"$\propto 1/t$")
    ax_g.set_xlabel(r"$t$")
    ax_g.set_ylabel("restricted gap")
    ax_g.legend(frameon=False, fontsize=8)

    ax_q.plot(series_t, hist["Q"], color="C0", lw=1.4)
    ax_q.set_xlabel(r"$t$")
    ax_q.set_ylabel(r"$Q_t^{\mathrm{obs}}$")


def plot_main() -> None:
    g2 = _load("G2_identity")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), layout="constrained")
    _panel_row(axes, g2["hist"], r"G2 identity, $T=2\times 10^4$")
    out_pdf = ROOT / "figures" / "exp1_selfplay.pdf"
    out_png = ROOT / "figures" / "exp1_selfplay.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


def plot_gaussian(seed: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), layout="constrained")
    hist = _load(f"G1_gaussian_seed{seed}")["hist"]
    t = np.arange(1, len(hist["reg_x"]) + 1)
    axes[0].plot(t, hist["reg_x"], lw=1.2)
    gap = np.maximum(np.asarray(hist["gap"], dtype=float), 1e-16)
    axes[1].loglog(t, gap, lw=1.2)
    axes[2].plot(t, hist["Q"], lw=1.2)
    axes[0].set_title(f"G1 spectral-normalized Gaussian A (seed {seed})")
    axes[0].set_ylabel(r"$\mathrm{Reg}^x(a)$")
    axes[1].set_ylabel("restricted gap")
    axes[2].set_ylabel(r"$Q_t^{\mathrm{obs}}$")
    for ax in axes:
        ax.set_xlabel(r"$t$")
    out = ROOT / "figures" / "exp1_G1_gaussian.pdf"
    out_png = ROOT / "figures" / "exp1_G1_gaussian.png"
    fig.savefig(out)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"wrote {out_png}")


def main():
    with (ROOT / "configs" / "default.yaml").open(encoding="utf-8") as f:
        seed = int(yaml.safe_load(f)["gaussian_seed"])
    plot_main()
    plot_gaussian(seed)


if __name__ == "__main__":
    main()
