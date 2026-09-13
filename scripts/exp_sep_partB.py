"""Part B: D005 vs Hsieh OptDA on the frozen Part A opponent sequences.

Log-only diagnostic. Not a paper experiment: the frozen y is constructed by
D005, the additive regret gap saturates, and the comparison is easy to
misread as a ranking. Recorded in ../note/experiment_log.md.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hsieh import HsiehOptDA  # noqa: E402
from io_results import dump_compact, load_run  # noqa: E402
from separation import (  # noqa: E402
    U_STAR,
    SeparationGame,
    prefix_curves,
    realized_variation,
    regret_against_u,
    replay_frozen_sequence,
)

_DTYPE = np.float64
HORIZONS = (200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000)
TARGET_WINDOW_MIN_T = 10000
TAU = 1.0
HSIEH_MAX_W_CAP = 1e20


def _load_cfg():
    with (ROOT / "configs" / "default.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def stem_for(T: int) -> str:
    return f"exp_sep_T{int(T)}"


def part_b_stem(T: int) -> str:
    return f"exp_sep_partB_T{int(T)}"


def frozen_y_sha256(y: np.ndarray) -> str:
    """Must match scripts/exp_separation.py: lock Part B to the Part A y."""
    canonical = np.ascontiguousarray(np.asarray(y, dtype="<f8"))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def replay_hsieh_frozen(y_seq: np.ndarray, game: SeparationGame, tau: float = TAU) -> dict:
    y_seq = np.asarray(y_seq, dtype=_DTYPE)
    T = int(y_seq.shape[0])
    if y_seq.shape != (T, 2):
        raise ValueError(f"y_seq must be (T, 2), got {y_seq.shape}")
    player = HsiehOptDA(1, tau=tau)
    if np.any(player.action):
        raise AssertionError("must not overwrite w1=0")
    x_seq = np.zeros(T, dtype=_DTYPE)
    g_seq = np.zeros(T, dtype=_DTYPE)
    max_w = 0.0
    for t in range(T):
        x = float(player.action[0])
        y = y_seq[t]
        gx, _gy = game.feedback(np.array([x], dtype=_DTYPE), y)
        g = float(gx[0])
        player.observe(gx)
        player.assert_invariants()
        x_seq[t] = x
        g_seq[t] = g
        max_w = max(max_w, abs(x), float(np.linalg.norm(y)))
    e_hsieh = realized_variation(g_seq)
    lam_expected = float(np.sqrt(tau + e_hsieh))
    if abs(player.lam - lam_expected) > 1e-10:
        raise AssertionError(
            f"Hsieh lambda {player.lam} != sqrt(tau+E_T)={lam_expected}"
        )
    return {
        "x": x_seq,
        "g": g_seq,
        "player": player,
        "max_w": max_w,
        "E_T": e_hsieh,
        "G_T": float(np.max(np.abs(g_seq))),
        "lam_T": float(player.lam),
        "x1": float(x_seq[0]),
        "x_T": float(x_seq[-1]),
    }


def _load_frozen(T: int) -> tuple[dict, dict, np.ndarray, str]:
    payload, hist = load_run(stem_for(T))
    if "y" not in hist:
        raise AssertionError(f"{stem_for(T)} is missing frozen y")
    y = np.asarray(hist["y"], dtype=_DTYPE)
    got = frozen_y_sha256(y)
    want = str(payload["summary"]["frozen_y_sha256"])
    if got != want:
        raise AssertionError(
            f"T={T}: recomputed frozen_y_sha256 {got} != Part A summary {want}"
        )
    if y.shape != (int(T), 2):
        raise AssertionError(f"T={T}: y shape {y.shape} != ({T}, 2)")
    return payload, hist, y, got


def run_one(T: int, cfg: dict) -> dict:
    print(f"Part B: replay D005 and Hsieh on frozen y, T={T}")
    payload_a, hist_a, y, y_hash = _load_frozen(T)
    y_before = frozen_y_sha256(y)
    game = SeparationGame()
    d005 = replay_frozen_sequence(y, cfg, game=game)
    if not np.allclose(d005["x"], hist_a["x"], rtol=0.0, atol=1e-12):
        raise AssertionError(f"T={T}: D005 replay x_t differs from Part A")
    if not np.allclose(d005["g"], hist_a["g"], rtol=0.0, atol=1e-12):
        raise AssertionError(f"T={T}: D005 replay g_t differs from Part A")
    r_d005 = regret_against_u(game, d005["x"], y)
    r_saved = float(payload_a["summary"]["R_T"])
    if abs(r_d005 - r_saved) > 1e-8:
        raise AssertionError(f"T={T}: D005 R_T {r_d005} != Part A {r_saved}")
    hsieh = replay_hsieh_frozen(y, game, tau=TAU)
    if frozen_y_sha256(y) != y_before:
        raise AssertionError(f"T={T}: Hsieh replay mutated frozen y")
    if abs(hsieh["x1"]) > 1e-15:
        raise AssertionError(f"T={T}: Hsieh x1 must be 0, got {hsieh['x1']}")
    if not np.isfinite(hsieh["max_w"]) or hsieh["max_w"] >= HSIEH_MAX_W_CAP:
        raise AssertionError(f"T={T}: Hsieh numerically unstable, max_w={hsieh['max_w']}")
    r_hsieh = regret_against_u(game, hsieh["x"], y)
    if not np.isfinite(r_hsieh):
        raise AssertionError(f"T={T}: Hsieh R_T is not finite")
    curves = prefix_curves(game, hsieh["x"], hsieh["g"], y)
    if abs(float(curves["V"][-1]) - 1.0) > 1e-10:
        raise AssertionError(f"T={T}: V_T on frozen y is {curves['V'][-1]}, expected 1")
    row = {
        "T": int(T),
        "frozen_y_sha256": y_hash,
        "R_D005": float(r_d005),
        "R_Hsieh": float(r_hsieh),
        "R_D005_over_T": float(r_d005) / float(T),
        "R_Hsieh_over_T": float(r_hsieh) / float(T),
        "gap_Hsieh_minus_D005": float(r_hsieh - r_d005),
        "E_T_D005": float(payload_a["summary"]["E_T"]),
        "E_T_Hsieh": float(hsieh["E_T"]),
        "G_T_D005": float(payload_a["summary"]["G_T"]),
        "G_T_Hsieh": float(hsieh["G_T"]),
        "x_T_D005": float(payload_a["summary"]["x_T"]),
        "x_T_Hsieh": float(hsieh["x_T"]),
        "max_w_Hsieh": float(hsieh["max_w"]),
        "lam_T_Hsieh": float(hsieh["lam_T"]),
        "tau": TAU,
        "d005_replay_match": True,
        "hash_match": True,
        "hsieh_finite": True,
        "hsieh_x1_zero": True,
    }
    dump_compact(
        part_b_stem(T),
        {
            "meta": {
                "tag": part_b_stem(T),
                "part": "B",
                "game": "Gsep",
                "T": int(T),
                "u_star": U_STAR,
                "tau": TAU,
                "protocol": (
                    "replay D005 and Euclidean OptDA (Hsieh 2021, tau=1) "
                    "on the frozen Part A y_{1:T}; opponent is not rebuilt"
                ),
                "construction": "translated-amplified-v2",
                "frozen_source": stem_for(T),
            },
            "summary": row,
        },
        extra_arrays={
            "x_hsieh": hsieh["x"],
            "g_hsieh": hsieh["g"],
            "R_hsieh": curves["R"],
            "E_hsieh": curves["E"],
            "G_hsieh": curves["G"],
        },
    )
    print(
        f"  R_D005={row['R_D005']:.6g}  R_Hsieh={row['R_Hsieh']:.6g}  "
        f"gap={row['gap_Hsieh_minus_D005']:.6g}  "
        f"lam_T={row['lam_T_Hsieh']:.6g}  max_w={row['max_w_Hsieh']:.6g}"
    )
    return row


def row_from_saved(T: int) -> dict:
    payload_a, hist_a, y, y_hash = _load_frozen(T)
    payload_b, hist_b = load_run(part_b_stem(T))
    row = dict(payload_b["summary"])
    row["T"] = int(T)
    row["frozen_y_sha256"] = y_hash
    if str(payload_b["summary"]["frozen_y_sha256"]) != y_hash:
        raise AssertionError(f"T={T}: Part B dump hash does not match Part A y")
    if abs(float(row["R_D005"]) - float(payload_a["summary"]["R_T"])) > 1e-8:
        raise AssertionError(f"T={T}: stored D005 R_T drifted from Part A")
    if "x" in hist_a and "g" in hist_a:
        if hist_a["x"].shape[0] != int(T) or hist_a["y"].shape[0] != int(T):
            raise AssertionError(f"T={T}: Part A arrays have the wrong length")
    if hist_b["x_hsieh"].shape[0] != int(T):
        raise AssertionError(f"T={T}: Part B Hsieh trajectory length mismatch")
    if abs(float(hist_b["x_hsieh"][0])) > 1e-15:
        raise AssertionError(f"T={T}: stored Hsieh x1 is not 0")
    if not np.isfinite(hist_b["R_hsieh"]).all():
        raise AssertionError(f"T={T}: stored Hsieh regret is not finite")
    return row


def _assemble(rows: list[dict], cfg: dict) -> dict:
    del cfg
    rows = sorted(rows, key=lambda row: int(row["T"]))
    impl_ok = all(
        bool(r["d005_replay_match"])
        and bool(r["hash_match"])
        and bool(r["hsieh_finite"])
        and bool(r["hsieh_x1_zero"])
        and len(r["frozen_y_sha256"]) == 64
        and np.isfinite(r["R_Hsieh"])
        and np.isfinite(r["R_D005"])
        and float(r["max_w_Hsieh"]) < HSIEH_MAX_W_CAP
        for r in rows
    )
    target = [r for r in rows if int(r["T"]) >= TARGET_WINDOW_MIN_T]
    gaps = np.array([float(r["gap_Hsieh_minus_D005"]) for r in target], dtype=float)
    d005_smaller = bool(target) and bool(np.all(gaps > 0.0))
    hsieh_smaller = bool(target) and bool(np.all(gaps < 0.0))
    gap_expanding = bool(len(gaps) >= 2) and bool(np.all(np.diff(np.abs(gaps)) >= -1e-8))
    if d005_smaller:
        signed_expanding = bool(len(gaps) >= 2) and bool(np.all(np.diff(gaps) >= -1e-8))
    elif hsieh_smaller:
        signed_expanding = bool(len(gaps) >= 2) and bool(np.all(np.diff(gaps) <= 1e-8))
    else:
        signed_expanding = False
    last = rows[-1]
    scale = max(abs(float(last["R_D005"])), abs(float(last["R_Hsieh"])), 1.0)
    comparable = abs(float(last["gap_Hsieh_minus_D005"])) / scale <= 0.1
    bonus = bool(impl_ok and d005_smaller and signed_expanding)
    hsieh_better_or_comparable = bool(impl_ok and (hsieh_smaller or comparable) and not bonus)
    certificate_not_realized = bool(impl_ok and not bonus)
    paper_use = (
        "log only; not a paper experiment. The frozen sequence is D005-constructed, "
        "the additive gap saturates, and a ranking or appendix-limitation figure "
        "would be easy to misread."
    )
    verdict = {
        "implementation_valid": impl_ok,
        "target_window_min_T": TARGET_WINDOW_MIN_T,
        "d005_strictly_smaller_on_target": d005_smaller,
        "hsieh_strictly_smaller_on_target": hsieh_smaller,
        "abs_gap_nondecreasing_on_target": gap_expanding,
        "signed_gap_expanding_on_target": signed_expanding,
        "terminal_relative_gap": float(abs(last["gap_Hsieh_minus_D005"]) / scale),
        "comparable_at_Tmax": comparable,
        "bonus_experiment_candidate": bonus,
        "hsieh_better_or_comparable": hsieh_better_or_comparable,
        "certificate_not_realized_separation": certificate_not_realized,
        "paper_use": paper_use,
        "claim_scope": (
            "comparison is only against the offline D005-constructed frozen "
            "open-loop sequence; tau=1 is the paper constant, not fitted; "
            "this is not a general superiority claim"
        ),
    }
    dump_compact(
        "exp_sep_partB",
        {
            "meta": {
                "tag": "exp_sep_partB",
                "part": "B",
                "game": "Gsep",
                "horizons": [int(r["T"]) for r in rows],
                "u_star": U_STAR,
                "tau": TAU,
                "construction": "translated-amplified-v2",
                "protocol": (
                    "same frozen y as Part A; D005 and Hsieh OptDA replayed "
                    "from w1=0; opponent not rebuilt from the baseline"
                ),
            },
            "summary": {"rows": rows, "verdict": verdict},
        },
    )
    return verdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, action="append", default=None)
    parser.add_argument(
        "--assemble",
        action="store_true",
        help="rebuild the Part B summary from saved npz files without replaying",
    )
    args = parser.parse_args()
    cfg = _load_cfg()
    horizons = tuple(int(t) for t in args.T) if args.T else HORIZONS
    if args.assemble:
        rows = [row_from_saved(T) for T in horizons]
    else:
        rows = [run_one(T, cfg) for T in horizons]
    verdict = _assemble(rows, cfg)
    print("Part B verdict:", verdict)
    if not verdict["implementation_valid"]:
        raise SystemExit("Part B implementation checks failed")


if __name__ == "__main__":
    main()
