"""Plot Exp.3 from saved json. Does not rerun the learner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

# Early window: one reset is visible here; full-horizon regret then runs in parallel.
ZOOM = 400


def _load(tag: str) -> dict:
    path = ROOT / "results" / f"exp3_{tag}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run exp3_restart.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def _reset_round(hist: dict) -> int | None:
    rst = np.asarray(hist.get("restart_x", []), dtype=int)
    if rst.size < 2:
        return None
    jumps = np.where(np.diff(rst) > 0)[0]
    if jumps.size == 0:
        return None
    # restart_path[0] is t=0; first increase at index k means a reset after round k.
    return int(jumps[0])


def main():
    warm_c = _load("const_warm")
    rst_c = _load("const_restart")
    hw, hr = warm_c["hist"], rst_c["hist"]

    tw = np.arange(len(hw["J_x"]))
    t = np.arange(1, len(hw["x_norm"]) + 1)
    z = min(ZOOM, len(t))
    t_reset = _reset_round(hr)
    # First played origin jump is the round after the reset.
    t_jump = None if t_reset is None else t_reset + 1

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
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\|x_t\|$")
    ax.set_title("origin jump after one reset")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ax.plot(t[:z], hw["reg_x"][:z], color="C0", lw=1.4, label="warm")
    ax.plot(t[:z], hr["reg_x"][:z], color="C1", lw=1.4, ls="--", label="restart")
    if t_jump is not None and t_jump <= z:
        ax.axvline(t_jump, color="0.35", ls=":", lw=1.0)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\mathrm{Reg}^x(0)$")
    ax.set_title("one-time delay, not a " + r"$\sqrt{K}$ tax")
    ax.legend(frameon=False, fontsize=8)

    out_pdf = ROOT / "figures" / "exp3_restart.pdf"
    out_png = ROOT / "figures" / "exp3_restart.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
