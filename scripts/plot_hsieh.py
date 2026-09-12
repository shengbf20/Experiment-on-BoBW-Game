"""Plot Hsieh baseline from saved json/npz. Does not rerun learners."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

plt.rcParams.update(
    {
        "axes.prop_cycle": plt.cycler(color=["#0072B2", "#E69F00", "#009E73", "#CC79A7"]),
        "pdf.fonttype": 42,
        "font.size": 9,
    }
)


def _load_json(name: str) -> dict:
    path = ROOT / "results" / name
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run exp_hsieh.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz(name: str) -> dict:
    path = ROOT / "results" / name
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run exp_hsieh.py first")
    data = np.load(path)
    return {k: np.asarray(data[k]) for k in data.files}


def _load_exp2_g3() -> np.ndarray:
    json_path = ROOT / "results" / "exp2_G3_const.json"
    npz_path = ROOT / "results" / "exp2_G3_const.npz"
    if npz_path.is_file():
        return np.asarray(np.load(npz_path)["reg_x"])
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    return np.asarray(meta["hist"]["reg_x"])


def period_max(t, y, t0: int, period: int):
    t = np.asarray(t)
    y = np.asarray(y, dtype=float)
    t_end = int(t[-1])
    out_t = []
    out_y = []
    start = int(t0)
    min_len = max(2, period // 2)
    while start <= t_end:
        stop = min(start + period - 1, t_end)
        if stop - start + 1 < min_len:
            break
        m = (t >= start) & (t <= stop)
        if np.any(m):
            out_t.append(0.5 * (start + stop))
            out_y.append(float(np.max(y[m])))
        start += period
    return np.asarray(out_t), np.asarray(out_y)


def plot_g3() -> None:
    d005 = _load_exp2_g3()
    hsieh = _load_npz("hsieh_G3_const.npz")["reg_x"]
    T = min(len(d005), len(hsieh))
    t = np.arange(1, T + 1)
    fig, ax = plt.subplots(figsize=(4.6, 3.3), layout="constrained")
    ax.plot(t, d005[:T], color="C0", lw=1.4, label="D005")
    ax.plot(t, hsieh[:T], color="C1", lw=1.4, ls="--", label="Hsieh OptDA")
    scale = abs(d005[min(99, T - 1)]) / np.sqrt(min(100, T))
    if scale > 0:
        ax.plot(t, scale * np.sqrt(t), color="0.5", ls=":", lw=1.0, label=r"$\propto\sqrt{t}$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\mathrm{Reg}^x(u^\star)$")
    ax.set_title(r"G3, $y_t\equiv 1$, $V_T(u^\star)=0$")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(0.52, 0.42, rf"D005 $\approx {float(d005[T-1]):.0f}$", transform=ax.transAxes, fontsize=8, color="C0")
    axins = ax.inset_axes([0.48, 0.12, 0.48, 0.22])
    axins.set_facecolor("white")
    axins.plot(t, hsieh[:T], color="C1", lw=1.3, ls="--")
    axins.set_ylim(0.0, 2.0)
    axins.set_xlim(1.0, float(T))
    axins.set_title(rf"Hsieh $\approx {float(hsieh[T-1]):.1f}$", fontsize=7, pad=1)
    axins.tick_params(labelsize=6)
    out_pdf = ROOT / "figures" / "hsieh_G3.pdf"
    out_png = ROOT / "figures" / "hsieh_G3.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


def plot_switch() -> None:
    meta = _load_json("hsieh_G2_switch.json")
    h = _load_npz("hsieh_G2_switch.npz")
    T = len(h["V_x"])
    t = np.arange(1, T + 1)
    half = int(meta["meta"]["half"])
    period = int(meta["meta"].get("period", 200))
    mu_inv = 5.0
    fig, ax = plt.subplots(figsize=(5.6, 3.4), layout="constrained")
    pre = t <= half
    ax.plot(t[pre], h["dist_y"][pre], color="C1", lw=1.0, label=r"$\|y_t-b\|$")
    ax.plot(*period_max(t, h["dist_y"], half, period), color="C1", lw=1.0)
    ax.plot(t[pre], h["dist_x_d005"][pre], color="C0", lw=1.6, label=r"D005 $\|x_t-a\|$")
    ax.plot(*period_max(t, h["dist_x_d005"], half, period), color="C0", lw=1.6)
    ax.plot(t[pre], h["dist_x_hsieh"][pre], color="C3", lw=1.4, label=r"Hsieh $\|x_t-a\|$")
    ax.plot(*period_max(t, h["dist_x_hsieh"], half, period), color="C3", lw=1.4)
    ax.plot([half, float(T)], [mu_inv, mu_inv], color="0.45", ls=":", lw=0.9, label=r"$1/\mu$")
    ax.axvline(half, color="0.35", ls="--", lw=1.0)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("distance to saddle")
    ax.set_ylim(0.0, 5.6)
    axr = ax.twinx()
    axr.plot(t, h["V_x"], color="C2", lw=1.1, label=r"$V_t^x(a)$")
    axr.set_ylabel(r"$V_t^x(a)$")
    axr.set_ylim(0.0, 10.0)
    ax.set_title("G2 switch, D005 opponent replay")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc="upper left")
    out_pdf = ROOT / "figures" / "hsieh_switch.pdf"
    out_png = ROOT / "figures" / "hsieh_switch.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


def plot_selfplay() -> None:
    meta = _load_json("hsieh_G2_selfplay.json")
    h = _load_npz("hsieh_G2_selfplay.npz")
    T = len(h["reg_x"])
    t = np.arange(1, T + 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), layout="constrained")
    axes[0].plot(t, h["reg_x"], color="C0", lw=1.4, label=r"$\mathrm{Reg}^x(a)$")
    axes[0].plot(t, h["reg_y"], color="C1", lw=1.2, ls="--", label=r"$\mathrm{Reg}^y(b)$")
    axes[0].set_xlabel(r"$t$")
    axes[0].set_title("Hsieh OptDA self-play on G2")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(t, h["Q"], color="C2", lw=1.4, label=r"$Q_t^{\mathrm{obs}}$")
    axes[1].set_xlabel(r"$t$")
    axes[1].set_title("movement (implementation check)")
    axes[1].legend(frameon=False, fontsize=8)
    out_pdf = ROOT / "figures" / "hsieh_selfplay.pdf"
    out_png = ROOT / "figures" / "hsieh_selfplay.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    print("self-play summary", meta["summary"])


def main():
    plot_g3()
    plot_switch()
    plot_selfplay()


if __name__ == "__main__":
    main()
