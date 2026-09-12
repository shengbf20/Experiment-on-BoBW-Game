"""Plot Exp.3 from saved json/npz. Does not rerun the learner."""

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

# Early window: re-climb vs warm is visible here; the origin glitch is not.
ZOOM = 400
JUMP_INSET = 20


def _load(tag: str) -> tuple[dict, dict]:
    return load_run(f"exp3_{tag}")


def _origin_action_round(hist: dict) -> int | None:
    rst = np.asarray(hist.get("restart_x", []), dtype=int)
    if rst.size < 2:
        return None
    jumps = np.where(np.diff(rst) > 0)[0]
    if jumps.size == 0:
        return None
    # rst[0] is t=0. An increase at diff-index k means n_restarts rose after
    # observe at round k+1. The origin is the next played action, round k+2.
    return int(jumps[0]) + 2


def main():
    warm_c, hw = _load("const_warm")
    rst_c, hr = _load("const_restart")
    x_star = warm_c.get("meta", {}).get("x_star")
    x_star_norm = None if x_star is None else float(np.linalg.norm(x_star))

    tw = np.arange(len(hw["J_x"]))
    t = np.arange(1, len(hw["x_norm"]) + 1)
    z = min(ZOOM, len(t))
    t_jump = _origin_action_round(hr)

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.3), layout="constrained")

    ax = axes[0]
    ax.step(tw, hw["J_x"], where="post", color="C0", lw=1.6, label="warm")
    ax.step(tw, hr["J_x"], where="post", color="C1", lw=1.4, ls="--", label="restart")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$J_t^x$")
    ax.set_title(r"vs $y\equiv e_1$: finite doubling")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(t[:z], hw["x_norm"][:z], color="C0", lw=1.4, label="warm")
    ax.plot(t[:z], hr["x_norm"][:z], color="C1", lw=1.4, ls="--", label="restart")
    if t_jump is not None and t_jump <= z:
        ax.axvline(t_jump, color="0.35", ls=":", lw=1.0)
    if x_star_norm is not None:
        ax.axhline(x_star_norm, color="0.45", ls=":", lw=0.9, label=r"$\|x^\star\|$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\|x_t\|$")
    ax.set_title(r"re-climb toward $x^\star$")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    n_in = min(JUMP_INSET, len(t))
    axin = ax.inset_axes([0.55, 0.10, 0.42, 0.36])
    axin.set_facecolor("white")
    axin.plot(t[:n_in], hw["x_norm"][:n_in], color="C0", lw=1.2)
    axin.plot(t[:n_in], hr["x_norm"][:n_in], color="C1", lw=1.2, ls="--")
    if t_jump is not None and t_jump <= n_in:
        axin.axvline(t_jump, color="0.35", ls=":", lw=0.8)
    axin.set_xlim(1.0, float(n_in))
    axin.set_ylim(0.0, 0.18)
    axin.set_title(rf"$t=1$ to ${n_in}$", fontsize=7, pad=1)
    axin.tick_params(labelsize=6)

    ax = axes[2]
    ax.plot(t, hw["reg_x"], color="C0", lw=1.2, label="warm")
    ax.plot(t, hr["reg_x"], color="C1", lw=1.0, ls="--", label="restart")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\mathrm{Reg}^x(x^\star)$")
    ax.set_title(r"regret at $x^\star$: constant offset")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    axin = ax.inset_axes([0.44, 0.52, 0.53, 0.42])
    axin.plot(t[:z], hw["reg_x"][:z], color="C0", lw=1.2)
    axin.plot(t[:z], hr["reg_x"][:z], color="C1", lw=1.0, ls="--")
    if t_jump is not None and t_jump <= z:
        axin.axvline(t_jump, color="0.35", ls=":", lw=0.8)
    axin.set_title(f"first {z} rounds", fontsize=7)
    axin.tick_params(labelsize=6)

    out_pdf = ROOT / "figures" / "exp3_restart.pdf"
    out_png = ROOT / "figures" / "exp3_restart.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
