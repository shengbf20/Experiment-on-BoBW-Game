"""Hsieh (2021) baseline runs: G2 self-play check, G3 const, G2 switch replay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from games import (  # noqa: E402
    QuadraticGame,
    SeparationGame,
    assert_saddle_comparator,
    paper_saddle,
)
from hsieh import HsiehOptDA  # noqa: E402
from io_results import dump_compact  # noqa: E402
from learner import ClosedFormPlayer, run_loop, self_play  # noqa: E402

_DTYPE = np.float64
PERIOD = 200
ANCHOR_TOL = 0.05


def _load_cfg():
    with (ROOT / "configs" / "default.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _tau(cfg: dict) -> float:
    return float(cfg.get("hsieh_tau", 1.0))


def _d005(dim: int, cfg: dict) -> ClosedFormPlayer:
    return ClosedFormPlayer(
        dim,
        epsilon=float(cfg["epsilon"]),
        beta0=float(cfg["beta0"]),
        ell1=float(cfg["ell1"]),
        adaptive=True,
    )


def _hsieh(dim: int, cfg: dict) -> HsiehOptDA:
    return HsiehOptDA(dim, tau=_tau(cfg))


def _dump(tag: str, payload: dict, hist: dict, keys: tuple[str, ...]) -> None:
    dump_compact(f"hsieh_{tag}", payload, hist, keys=keys)
    print(payload["summary"])


def _assert_origin(hist: dict) -> None:
    if abs(hist["x_norm"][0]) > 1e-15:
        raise AssertionError("w1 must be the origin")


def run_selfplay(cfg: dict, T: int, radius: float, dim: int = 10) -> dict:
    sx, sy = paper_saddle(dim)
    game = QuadraticGame(dim=dim, mu=0.2, saddle_x=sx, saddle_y=sy)
    px, py = _hsieh(dim, cfg), _hsieh(dim, cfg)
    if np.any(px.action) or np.any(py.action):
        raise AssertionError("must not overwrite w1=0")
    metrics, hist, max_w = self_play(game, px, py, T, radius=radius)
    assert_saddle_comparator(game, metrics)
    _assert_origin(hist)
    if max_w >= 1e20:
        raise AssertionError(f"self-play exploded max_w={max_w}")
    half = T // 2
    q = np.asarray(hist["Q"], dtype=float)
    reg = np.asarray(hist["reg_x"], dtype=float)
    dQ_late = float(q[-1] - q[half - 1])
    dQ_early = float(q[half - 1] - q[0])
    if dQ_late > 0.25 * max(dQ_early, 1e-6) and q[-1] > 0.5:
        raise AssertionError(
            f"Hsieh G2 self-play Q not a platform: early={dQ_early}, late={dQ_late}"
        )
    dreg = abs(float(reg[-1] - reg[half - 1]))
    first = abs(float(reg[half - 1] - reg[0]))
    if dreg > max(10.0, 0.5 * first) and abs(reg[-1]) > 20.0:
        raise AssertionError(f"Hsieh G2 self-play regret still moving: {reg[half-1]} -> {reg[-1]}")
    payload = {
        "meta": {
            "tag": "G2_selfplay",
            "learner": "HsiehOptDA",
            "cite": "Hsieh et al. COLT 2021, OptDA + Adapt, Euclidean h=||x||^2/2",
            "tau": _tau(cfg),
            "game": "G2",
            "T": T,
            "init": "origin (paper w1=0)",
        },
        "summary": {
            "max_w": max_w,
            "reg_x_T": float(reg[-1]),
            "reg_x_half": float(reg[half - 1]),
            "Q_T": float(q[-1]),
            "Q_half": float(q[half - 1]),
            "gap_T": hist["gap"][-1],
            "lam_T": px.lam,
        },
    }
    _dump("G2_selfplay", payload, hist, ("reg_x", "reg_y", "Q", "gap", "dist_x", "dist_y", "lam_x"))
    return payload


def run_g3(cfg: dict, T: int, radius: float) -> dict:
    game = SeparationGame()
    px = _hsieh(1, cfg)
    if np.any(px.action):
        raise AssertionError("must not overwrite w1=0")

    def y_policy(_t, _x):
        return np.array([1.0], dtype=_DTYPE)

    metrics, hist, max_w = run_loop(
        game, px, T, y_policy=y_policy, player_y=None, observe_y=lambda _t: False, radius=radius
    )
    v = np.asarray(hist["V_x"], dtype=float)
    if np.any(v != 0.0):
        raise AssertionError(f"G3 V_x(u*) must be exactly 0, got max {v.max()}")
    gT = hist["G_x"][-1]
    if not (1.0 - 1e-12 <= gT <= 3.0):
        raise AssertionError(f"G3 G_x should lie in [1, 3], got {gT}")
    if max_w >= 1e20:
        raise AssertionError(f"G3 exploded max_w={max_w}")
    _assert_origin(hist)
    payload = {
        "meta": {
            "tag": "G3_const",
            "learner": "HsiehOptDA",
            "cite": "Hsieh et al. COLT 2021, OptDA + Adapt, Euclidean h=||x||^2/2",
            "tau": _tau(cfg),
            "game": "G3",
            "opponent": "const y=1",
            "T": T,
            "comparator": "u_star",
            "init": "origin",
        },
        "summary": {
            "max_w": max_w,
            "reg_x_T": hist["reg_x"][-1],
            "reg_x_half": hist["reg_x"][T // 2 - 1],
            "V_x_T": hist["V_x"][-1],
            "G_x": metrics.G_x,
            "lam_T": px.lam,
        },
    }
    _dump("G3_const", payload, hist, ("reg_x", "V_x", "G_x", "x_norm", "lam_x"))
    return payload


def run_switch(cfg: dict, T: int, radius: float, dim: int = 10) -> dict:
    sx, sy = paper_saddle(dim)
    game = QuadraticGame(dim=dim, mu=0.2, saddle_x=sx, saddle_y=sy)
    d005_x, d005_y = _d005(dim, cfg), _d005(dim, cfg)
    if np.any(d005_x.action) or np.any(d005_y.action):
        raise AssertionError("must not overwrite w1=0")
    half = T // 2
    e1 = np.zeros(dim, dtype=_DTYPE)
    e1[0] = 1.0
    y_played: dict[int, np.ndarray] = {}

    # Y is the D005 closed-loop path, then replayed open-loop against Hsieh
    # from t=1. This is not two self-play runs that later switch.

    def y_policy(t, _x):
        if t <= half:
            y = d005_y.action.copy()
        else:
            y = y_played[half] + np.sin(2.0 * np.pi * (t - half) / PERIOD) * e1
        y_played[t] = y.copy()
        return y

    _, hist_d, max_w_d = run_loop(
        game,
        d005_x,
        T,
        y_policy=y_policy,
        player_y=d005_y,
        observe_y=lambda t: t <= half,
        radius=radius,
    )
    y_anchor = y_played[half]
    step = float(np.linalg.norm(y_played[half + 1] - y_anchor))
    expected_step = abs(np.sin(2.0 * np.pi / PERIOD))
    if abs(step - expected_step) > 1e-12:
        raise AssertionError(f"switch jump {step} != sine step {expected_step}")
    if float(np.linalg.norm(y_anchor - sy)) > ANCHOR_TOL:
        raise AssertionError("y_anchor not near b")

    hx = _hsieh(dim, cfg)
    if np.any(hx.action):
        raise AssertionError("must not overwrite w1=0")

    def y_replay(t, _x):
        return y_played[t]

    _, hist_h, max_w_h = run_loop(
        game,
        hx,
        T,
        y_policy=y_replay,
        player_y=None,
        observe_y=lambda _t: False,
        radius=radius,
    )
    if max_w_d >= 1e20 or max_w_h >= 1e20:
        raise AssertionError("switch exploded")
    _assert_origin(hist_d)
    _assert_origin(hist_h)
    if abs(float(hist_d["V_x"][-1] - hist_h["V_x"][-1])) > 1e-10:
        raise AssertionError("same y-sequence must give the same V_t^x(a)")
    payload = {
        "meta": {
            "tag": "G2_switch",
            "learners": ["D005", "HsiehOptDA"],
            "cite": "Hsieh et al. COLT 2021, OptDA + Adapt; same y-sequence as Exp.2a",
            "tau": _tau(cfg),
            "game": "G2",
            "T": T,
            "half": half,
            "period": PERIOD,
            "init": "origin (paper w1=0); replay Exp.2a opponent",
        },
        "summary": {
            "max_w_d005": max_w_d,
            "max_w_hsieh": max_w_h,
            "V_x_half": hist_d["V_x"][half - 1],
            "V_x_T": hist_d["V_x"][-1],
            "dist_x_post_max_d005": float(np.max(hist_d["dist_x"][half:])),
            "dist_x_post_max_hsieh": float(np.max(hist_h["dist_x"][half:])),
            "dist_y_post_max": float(np.max(hist_d["dist_y"][half:])),
            "lam_T": hx.lam,
            "switch_step": step,
        },
    }
    dump_compact(
        "hsieh_G2_switch",
        payload,
        extra_arrays={
            "dist_x_d005": hist_d["dist_x"],
            "dist_x_hsieh": hist_h["dist_x"],
            "dist_y": hist_d["dist_y"],
            "V_x": hist_d["V_x"],
            "reg_x_d005": hist_d["reg_x"],
            "reg_x_hsieh": hist_h["reg_x"],
        },
    )
    print(payload["summary"])
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument(
        "--only",
        choices=["selfplay", "g3", "switch", "all"],
        default="all",
    )
    args = parser.parse_args()
    cfg = _load_cfg()
    T = int(args.T if args.T is not None else cfg["T"])
    radius = float(cfg["gap_radius"])
    if args.only in ("selfplay", "all"):
        run_selfplay(cfg, T, radius)
    if args.only in ("g3", "all"):
        run_g3(cfg, T, radius)
    if args.only in ("switch", "all"):
        run_switch(cfg, T, radius)


if __name__ == "__main__":
    main()
