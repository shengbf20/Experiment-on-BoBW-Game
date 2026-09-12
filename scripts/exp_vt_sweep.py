"""Step 17: V_T amplitude sweep (17a) and nonstationary V=O(1) path (17b)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from games import QuadraticGame, SeparationGame, paper_saddle  # noqa: E402
from hsieh import HsiehOptDA  # noqa: E402
from io_results import dump_compact  # noqa: E402
from learner import ClosedFormPlayer, run_loop  # noqa: E402

_DTYPE = np.float64
ETAS_17A = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
PERIOD_17A = 100
ETA_17B = 0.5


def _load_cfg():
    with (ROOT / "configs" / "default.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _d005(dim: int, cfg: dict) -> ClosedFormPlayer:
    return ClosedFormPlayer(
        dim,
        epsilon=float(cfg["epsilon"]),
        beta0=float(cfg["beta0"]),
        ell1=float(cfg["ell1"]),
        adaptive=True,
    )


def eta_tag(eta: float) -> str:
    return f"{eta:g}".replace(".", "p")


def _assert_origin(hist: dict, tag: str) -> None:
    if abs(hist["x_norm"][0]) > 1e-15:
        raise AssertionError(f"{tag}: w1 must be the origin")


def run_17a_one(eta: float, cfg: dict, T: int, dim: int, radius: float) -> dict:
    sx, sy = paper_saddle(dim)
    game = QuadraticGame(dim=dim, mu=0.2, saddle_x=sx, saddle_y=sy)
    e1 = np.zeros(dim, dtype=_DTYPE)
    e1[0] = 1.0
    ts = np.arange(1, T + 1, dtype=_DTYPE)
    sine = np.sin(2.0 * np.pi * ts / PERIOD_17A)
    y_offset = float(sine.mean()) * eta * e1
    ybar = sy + y_offset
    u = game.induced_minimizer_x(ybar)
    if eta == 0.0 and not np.allclose(u, sx):
        raise AssertionError("eta=0 induced minimizer must be the saddle")
    if abs(float(sine.mean())) > 1e-12:
        # T should be an integer number of periods so the mean vanishes.
        raise AssertionError(f"sine mean {sine.mean()} not ~0; pick T multiple of T_per")

    def y_policy(t, _x):
        return sy + eta * np.sin(2.0 * np.pi * t / PERIOD_17A) * e1

    px = _d005(dim, cfg)
    if np.any(px.action):
        raise AssertionError("must not overwrite w1=0")
    metrics, hist, max_w = run_loop(
        game,
        px,
        T,
        y_policy=y_policy,
        player_y=None,
        observe_y=lambda _t: False,
        radius=radius,
        u_x=u,
    )
    tag = f"vt17a_eta{eta_tag(eta)}"
    _assert_origin(hist, tag)
    if max_w >= 1e20 or not np.isfinite(max_w):
        raise AssertionError(f"{tag}: exploded")
    V = float(hist["V_x"][-1])
    if eta == 0.0 and V > 1e-12:
        raise AssertionError(f"eta=0 must have V=0, got {V}")
    if eta > 0.0 and V <= 0.0:
        raise AssertionError(f"{tag}: expected V>0")
    row = {
        "eta": float(eta),
        "T_per": PERIOD_17A,
        "V": V,
        "sqrtV": float(np.sqrt(V)),
        "reg": float(hist["reg_x"][-1]),
        "lin": float(hist["lin_x"][-1]),
        "Q": float(hist["Q"][-1]),
        "J": int(hist["J_x"][-1]),
        "G": float(hist["G_x"][-1]),
        "max_w": float(max_w),
        "gamma": float(px.gamma),
        "u_dist_saddle": float(np.linalg.norm(u - sx)),
    }
    payload = {
        "meta": {
            "tag": tag,
            "part": "17a",
            "game": "G2",
            "opponent": f"y=b+eta sin(2 pi t / {PERIOD_17A}) e1",
            "eta": float(eta),
            "T_per": PERIOD_17A,
            "T": T,
            "dim": dim,
            "comparator": "induced_minimizer_x(mean y) = a",
            "normalize": False,
            "init": "origin (paper w1=0)",
        },
        "summary": row,
    }
    dump_compact(
        f"exp_{tag}",
        payload,
        hist,
        keys=("reg_x", "lin_x", "V_x", "Q", "J_x", "x_norm"),
    )
    print(
        f"17a eta={eta:g} V={V:.4g} sqrtV={row['sqrtV']:.4g} "
        f"Reg={row['reg']:.4g} Lin={row['lin']:.4g} J={row['J']}",
        flush=True,
    )
    return row


def _check_17a(rows: list[dict]) -> None:
    etas = [r["eta"] for r in rows]
    vs = [r["V"] for r in rows]
    if etas != sorted(etas):
        raise AssertionError("etas not sorted")
    for i in range(1, len(rows)):
        if vs[i] < vs[i - 1] - 1e-12:
            raise AssertionError(f"V did not increase with eta: {vs}")
    # Small-eta limb of linearized regret should get worse with sqrt(V).
    lins = [r["lin"] for r in rows]
    peak = int(np.argmax(lins))
    if peak < 2:
        raise AssertionError(f"LinReg did not rise over the first amplitudes: {lins}")
    for i in range(1, peak + 1):
        if lins[i] + 1e-6 < lins[i - 1]:
            raise AssertionError(f"LinReg not nondecreasing on the rising limb: {lins}")
    if rows[0]["V"] > 1e-12:
        raise AssertionError("eta=0 should be the V=0 endpoint")


def run_17b(cfg: dict, T: int, radius: float, eta: float = ETA_17B) -> dict:
    game = SeparationGame()
    ustar = np.array([game.u_star], dtype=_DTYPE)
    period = float(np.sqrt(T))

    def y_policy(t, _x):
        return np.array([1.0 + eta * np.sin(2.0 * np.pi * t / period)], dtype=_DTYPE)

    y = np.array([float(y_policy(t, None)[0]) for t in range(1, T + 1)], dtype=_DTYPE)
    pathlen = float(np.sum(np.abs(np.diff(y))))
    dy2 = float(np.sum(np.diff(y) ** 2))

    def _run(player, name: str):
        if np.any(player.action):
            raise AssertionError("must not overwrite w1=0")
        metrics, hist, max_w = run_loop(
            game,
            player,
            T,
            y_policy=y_policy,
            player_y=None,
            observe_y=lambda _t: False,
            radius=radius,
            u_x=ustar,
        )
        _assert_origin(hist, name)
        if max_w >= 1e20:
            raise AssertionError(f"{name} exploded")
        return metrics, hist, max_w

    px = _d005(1, cfg)
    _mx, hist, max_w = _run(px, "d005")
    hx = HsiehOptDA(1, tau=float(cfg.get("hsieh_tau", 1.0)))
    _mh, hhist, hmax = _run(hx, "hsieh")

    V = float(hist["V_x"][-1])
    if abs(V - dy2) > 1e-8:
        raise AssertionError(f"G3 V should equal sum (Delta y)^2: V={V}, dy2={dy2}")
    if V > 30.0:
        raise AssertionError(f"17b expected V_T=O(1), got {V}")
    if V / T > 0.01:
        raise AssertionError(f"17b V/T={V / T:.3g} too large to call O(1) vs T")
    if pathlen < 20.0:
        raise AssertionError(f"17b path length {pathlen} is not large")
    half = T // 2
    d005_reg = np.asarray(hist["reg_x"], dtype=float)
    late_ptp = float(np.ptp(d005_reg[half:]))
    if late_ptp > max(5.0, 0.15 * abs(float(d005_reg[-1]))):
        raise AssertionError(
            f"17b D005 not a platform: late ptp={late_ptp:.3g}, Reg_T={d005_reg[-1]:.3g}"
        )

    row = {
        "eta": float(eta),
        "T_per": period,
        "V": V,
        "sqrtV": float(np.sqrt(V)),
        "pathlen_y": pathlen,
        "y_ptp": float(np.ptp(y)),
        "reg_d005": float(d005_reg[-1]),
        "reg_hsieh": float(hhist["reg_x"][-1]),
        "Q_d005": float(hist["Q"][-1]),
        "G_d005": float(hist["G_x"][-1]),
        "J": int(hist["J_x"][-1]),
        "max_w": float(max_w),
        "hsieh_max_w": float(hmax),
        "late_ptp": late_ptp,
    }
    payload = {
        "meta": {
            "tag": "vt17b",
            "part": "17b",
            "game": "G3",
            "opponent": "y=1+eta sin(2 pi t / sqrt(T))",
            "eta": float(eta),
            "T_per": period,
            "T": T,
            "comparator": "u_star=-1/sqrt(3)",
            "init": "origin (paper w1=0)",
            "hsieh_tau": float(cfg.get("hsieh_tau", 1.0)),
            "note": "V_T=O(1) under T-scaling with T_per=sqrt(T); path length ~ eta sqrt(T). Do not claim Hsieh Omega(sqrt(T)).",
        },
        "summary": row,
    }
    dump_compact(
        "exp_vt17b_d005",
        payload,
        hist,
        extra_arrays={"y": y},
        keys=("reg_x", "V_x", "Q", "G_x", "x_norm", "J_x"),
    )
    dump_compact(
        "exp_vt17b_hsieh",
        {
            "meta": {**payload["meta"], "learner": "HsiehOptDA"},
            "summary": {
                "reg": float(hhist["reg_x"][-1]),
                "V": float(hhist["V_x"][-1]),
                "Q": float(hhist["Q"][-1]),
                "max_w": float(hmax),
            },
        },
        hhist,
        extra_arrays={"y": y},
        keys=("reg_x", "V_x", "Q", "G_x", "x_norm"),
    )
    print(
        f"17b V={V:.4g} pathlen={pathlen:.4g} yptp={row['y_ptp']:.4g} "
        f"D005={row['reg_d005']:.4g} Hsieh={row['reg_hsieh']:.4g} late_ptp={late_ptp:.4g}",
        flush=True,
    )
    return row


def latex_17a(rows: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{rrrrrr}",
        r"\hline",
        r"$\eta$ & $V_T$ & $\sqrt{V_T}$ & $\mathrm{Reg}^x(u)$ & $\mathrm{LinReg}^x(u)$ & $J$ \\",
        r"\hline",
    ]
    for r in rows:
        lines.append(
            f"${r['eta']:g}$ & ${r['V']:.3f}$ & ${r['sqrtV']:.3f}$ & "
            f"${r['reg']:.3f}$ & ${r['lin']:.3f}$ & ${r['J']}$ \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--only", choices=("17a", "17b", "both"), default="both")
    args = parser.parse_args()
    cfg = _load_cfg()
    T = int(args.T if args.T is not None else cfg["T"])
    radius = float(cfg["gap_radius"])
    rows_a: list[dict] = []
    row_b = None
    if args.only in ("17a", "both"):
        if T % PERIOD_17A != 0:
            raise AssertionError(f"T={T} must be a multiple of T_per={PERIOD_17A}")
        rows_a = [run_17a_one(float(eta), cfg, T, int(args.dim), radius) for eta in ETAS_17A]
        _check_17a(rows_a)
    if args.only in ("17b", "both"):
        row_b = run_17b(cfg, T, radius)
    payload = {
        "meta": {
            "tag": "vt_sweep",
            "T": T,
            "period_17a": PERIOD_17A,
            "etas_17a": [float(e) for e in ETAS_17A],
            "eta_17b": ETA_17B,
        },
        "summary": {"17a": rows_a, "17b": row_b},
    }
    dump_compact("exp_vt_sweep", payload)
    if rows_a:
        tex = latex_17a(rows_a)
        tex_path = ROOT / "results" / "vt_sweep.tex"
        tex_path.write_text(tex + "\n", encoding="utf-8")
        print(f"wrote {tex_path}", flush=True)
        print(tex, flush=True)


if __name__ == "__main__":
    main()
