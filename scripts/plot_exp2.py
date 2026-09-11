"""Plot Exp.2 from saved json. Does not rerun the learner."""

from __future__ import annotations

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
    path = ROOT / "results" / f"exp2_{tag}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run exp2_bobw.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    sw = _load("G2_switch")
    sep = _load("G3_const")
    h2a, h2b = sw["hist"], sep["hist"]
    T2a = len(h2a["reg_x"])
    T2b = len(h2b["reg_x"])
    t2a = np.arange(1, T2a + 1)
    t2b = np.arange(1, T2b + 1)
    half = int(sw["meta"]["half"])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), layout="constrained")

    ax = axes[0]
    ax.plot(t2a, h2a["reg_x"], color="C0", lw=1.4, label=r"$\mathrm{Reg}^x(a)$")
    ax.axvline(half, color="0.35", ls="--", lw=1.0, label=r"$T/2$ switch")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("individual regret")
    ax.set_title("G2 same-run switch (continuous at $b$)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(t2b, h2b["reg_x"], color="C0", lw=1.4, label=r"$\mathrm{Reg}^x(u^\star)$")
    scale = abs(h2b["reg_x"][min(99, T2b - 1)]) / np.sqrt(min(100, T2b))
    if scale > 0:
        ax.plot(t2b, scale * np.sqrt(t2b), color="0.5", ls=":", lw=1.0, label=r"$\propto\sqrt{t}$")
    ax.set_xlabel(r"$t$")
    ax.set_title(r"G3 constant opponent, $V_T(u^\star)=0$")
    ax.legend(frameon=False, fontsize=8)

    out_pdf = ROOT / "figures" / "exp2_bobw.pdf"
    out_png = ROOT / "figures" / "exp2_bobw.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
