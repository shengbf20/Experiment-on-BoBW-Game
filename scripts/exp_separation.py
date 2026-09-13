"""Part A improved construction and finite-horizon separation validation.

Generates a frozen open-loop opponent from the separation example, replays
D005 against that sequence, and records V_T(u*), E_T, G_T, and regret.
At the planned horizons raw E_T is transient-dominated, so the closeout
separately records protocol validity, the positive linear tail mechanism,
and whether raw cross-horizon scaling is actually visible.

Does not touch Exp.1–3 or the L_F sweep.
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

from io_results import dump_compact, load_run  # noqa: E402
from separation import (  # noqa: E402
    ANNULUS_FRACTION_LOWER,
    DELTA0,
    ETA,
    G_LOWER,
    G_UPPER,
    THEORY_LATE_DG2,
    U_STAR,
    X_CONVEXITY_MARGIN,
    Y2_CONCAVITY_MARGIN,
    late_jump_stats,
    run_part_a,
)

HORIZONS = (200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000)
TARGET_WINDOW_MIN_T = 10000


def _load_cfg():
    with (ROOT / "configs" / "default.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def stem_for(T: int) -> str:
    return f"exp_sep_T{int(T)}"


def frozen_y_sha256(y: np.ndarray) -> str:
    """Canonical content hash used to lock Part B to the Part A loss sequence."""
    canonical = np.ascontiguousarray(np.asarray(y, dtype="<f8"))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _row(out: dict) -> dict:
    late = out["late"]
    return {
        "T": int(out["T"]),
        "V_T": float(out["V_T"]),
        "E_T": float(out["E_T"]),
        "E_over_T": float(out["E_over_T"]),
        "G_T": float(out["G_T"]),
        "R_T": float(out["R_T"]),
        "sqrt_V": float(np.sqrt(out["V_T"])),
        "sqrt_E": float(np.sqrt(out["E_T"])),
        "G_sqrt_T": float(out["G_T"] * np.sqrt(out["T"])),
        "max_w": float(out["max_w"]),
        "J": int(out["J"]),
        "ell": float(out["ell"]),
        "beta": float(out["beta"]),
        "replay_match": bool(out["replay_match"]),
        "frozen_y_sha256": frozen_y_sha256(out["y"]),
        "g1": float(out["g1"]),
        "x1": float(out["x1"]),
        "x_T": float(out["x_T"]),
        "x_half": float(out["x_half"]),
        "late_start": int(late["late_start"]),
        "late_mean_dg2": float(late["late_mean_dg2"]),
        "late_sum_dg2": float(late["late_sum_dg2"]),
        "theory_late_dg2": float(late["theory_late_dg2"]),
        "late_linear_ET": float(late["late_mean_dg2"] * out["T"]),
    }


def _checks(row: dict) -> None:
    T = row["T"]
    if abs(row["x1"]) > 1e-15:
        raise AssertionError(f"T={T}: x1 must be 0")
    if abs(row["V_T"] - 1.0) > 1e-10:
        raise AssertionError(f"T={T}: expected V_T=1, got {row['V_T']}")
    if row["G_T"] > G_UPPER + 1e-10:
        raise AssertionError(f"T={T}: G_T={row['G_T']} exceeds {G_UPPER}")
    if row["G_T"] <= G_LOWER:
        raise AssertionError(f"T={T}: G_T={row['G_T']} is not Omega(1)")
    if not np.isfinite(row["E_T"]) or row["E_T"] <= 0.0:
        raise AssertionError(f"T={T}: bad E_T={row['E_T']}")
    if not row["replay_match"]:
        raise AssertionError(f"T={T}: replay did not match the generating run")
    if len(row["frozen_y_sha256"]) != 64:
        raise AssertionError(f"T={T}: invalid frozen-y SHA-256")


def run_one(T: int, cfg: dict) -> dict:
    print(f"Part A: generate + replay T={T}")
    out = run_part_a(T, cfg)
    row = _row(out)
    _checks(row)
    payload = {
        "meta": {
            "tag": stem_for(T),
            "part": "A",
            "game": "Gsep",
            "T": int(T),
            "adaptive": True,
            "epsilon": float(cfg["epsilon"]),
            "beta0": float(cfg["beta0"]),
            "ell1": float(cfg["ell1"]),
            "u_star": U_STAR,
            "delta0": DELTA0,
            "eta": ETA,
            "theory_late_dg2": THEORY_LATE_DG2,
            "construction": "translated-amplified-v2",
            "protocol": (
                "offline D005 construction, freeze y_{1:T}, replay D005 open-loop"
            ),
        },
        "summary": row,
    }
    dump_compact(
        stem_for(T),
        payload,
        extra_arrays={
            "x": out["x"],
            "g": out["g"],
            "y": out["y"],
            "V": out["curves"]["V"],
            "E": out["curves"]["E"],
            "G": out["curves"]["G"],
            "R": out["curves"]["R"],
        },
    )
    print(
        f"  V_T={row['V_T']:.6g}  E_T={row['E_T']:.6g}  E_T/T={row['E_over_T']:.6g}  "
        f"G_T={row['G_T']:.6g}  late_dg2={row['late_mean_dg2']:.6e}  R_T={row['R_T']:.6g}"
    )
    return row


def row_from_saved(T: int) -> dict:
    payload, hist = load_run(stem_for(T))
    row = dict(payload["summary"])
    late = late_jump_stats(hist["g"])
    row["T"] = int(T)
    row["late_start"] = int(late["late_start"])
    row["late_mean_dg2"] = float(late["late_mean_dg2"])
    row["late_sum_dg2"] = float(late["late_sum_dg2"])
    row["theory_late_dg2"] = float(late["theory_late_dg2"])
    row["late_linear_ET"] = float(late["late_mean_dg2"] * T)
    row["x_T"] = float(np.asarray(hist["x"]).reshape(-1)[-1])
    row["x_half"] = float(np.asarray(hist["x"]).reshape(-1)[T // 2 - 1])
    row["sqrt_V"] = float(np.sqrt(row["V_T"]))
    row["sqrt_E"] = float(np.sqrt(row["E_T"]))
    row["G_sqrt_T"] = float(row["G_T"] * np.sqrt(T))
    row["frozen_y_sha256"] = frozen_y_sha256(hist["y"])
    _checks(row)
    return row


def _assemble(rows: list[dict], cfg: dict) -> dict:
    Ts = np.array([r["T"] for r in rows], dtype=float)
    Es = np.array([r["E_T"] for r in rows], dtype=float)
    ratios = np.array([r["E_over_T"] for r in rows], dtype=float)
    Vs = np.array([r["V_T"] for r in rows], dtype=float)
    Gs = np.array([r["G_T"] for r in rows], dtype=float)
    late = np.array([r["late_mean_dg2"] for r in rows], dtype=float)
    slope = float("nan")
    if len(rows) >= 2:
        slope = float(np.polyfit(np.log(Ts), np.log(np.maximum(Es, 1e-16)), 1)[0])
    v_ok = bool(np.all(np.abs(Vs - 1.0) <= 1e-10))
    g_ok = bool(np.all((Gs > G_LOWER) & (Gs <= G_UPPER + 1e-10)))
    ratio_min = float(np.min(ratios))
    ratio_max = float(np.max(ratios))
    target = [r for r in rows if int(r["T"]) >= TARGET_WINDOW_MIN_T]
    target_slope = float("nan")
    target_ratio_min = float("nan")
    target_ratio_max = float("nan")
    if target:
        target_T = np.array([r["T"] for r in target], dtype=float)
        target_E = np.array([r["E_T"] for r in target], dtype=float)
        target_ratio = np.array([r["E_over_T"] for r in target], dtype=float)
        target_ratio_min = float(np.min(target_ratio))
        target_ratio_max = float(np.max(target_ratio))
        if len(target) >= 2:
            target_slope = float(
                np.polyfit(np.log(target_T), np.log(np.maximum(target_E, 1e-16)), 1)[0]
            )
    # "Observed" is deliberately a finite-window claim: raw E_T itself must
    # have a near-linear log-log slope and E_T/T must already be stabilizing.
    e_raw_theta = bool(
        len(target) >= 3
        and 0.75 <= target_slope <= 1.25
        and target_ratio_min > 0.0
        and target_ratio_max / target_ratio_min <= 2.0
    )
    settled = [r for r in rows if int(r["T"]) >= TARGET_WINDOW_MIN_T]
    tail_slope = float("nan")
    if settled:
        late_s = np.array([r["late_mean_dg2"] for r in settled], dtype=float)
        tail_T = np.array([r["T"] for r in settled], dtype=float)
        tail_sum = np.array([r["late_sum_dg2"] for r in settled], dtype=float)
        rel = np.abs(late_s - THEORY_LATE_DG2) / THEORY_LATE_DG2
        if len(settled) >= 2:
            tail_slope = float(np.polyfit(np.log(tail_T), np.log(tail_sum), 1)[0])
        density_ok = bool(np.all(late_s > 0.0) and np.all(rel < 0.1))
        tail_linear = bool(len(settled) < 2 or 0.8 <= tail_slope <= 1.2)
        e_mechanism = density_ok and tail_linear
    else:
        e_mechanism = bool(np.min(late) > 0.0)
    run_ok = v_ok and g_ok and all(bool(r["replay_match"]) for r in rows)
    proof_conditions_ok = bool(
        X_CONVEXITY_MARGIN > 0.0
        and Y2_CONCAVITY_MARGIN > 0.0
        and ANNULUS_FRACTION_LOWER > 0.0
        and abs(ETA - DELTA0) <= 1e-15
        and U_STAR != 0.0
    )
    c = float(THEORY_LATE_DG2)
    transient = float(Es[-1] - c * Ts[-1])
    crossover = float(transient / c)
    verdict = {
        "V_T_is_one": v_ok,
        "G_T_O1": g_ok,
        "loglog_slope_E_vs_T": slope,
        "target_window_min_T": TARGET_WINDOW_MIN_T,
        "loglog_slope_E_vs_T_target_window": target_slope,
        "loglog_slope_tail_sum_vs_T": tail_slope,
        "E_over_T_min": ratio_min,
        "E_over_T_max": ratio_max,
        "target_E_over_T_min": target_ratio_min,
        "target_E_over_T_max": target_ratio_max,
        "theory_late_dg2": float(THEORY_LATE_DG2),
        "late_mean_dg2_min": float(np.min(late)),
        "late_mean_dg2_max": float(np.max(late)),
        "finite_horizon_transient_estimate": transient,
        "raw_ET_crossover_horizon_estimate": crossover,
        "implementation_valid": run_ok,
        "proof_conditions_valid": proof_conditions_ok,
        "raw_ET_scaling_observed": e_raw_theta,
        "linear_tail_mechanism_valid": e_mechanism,
        "paper_raw_scaling_claim_supported": e_raw_theta,
        "paper_mechanism_claim_supported": run_ok and proof_conditions_ok and e_mechanism,
        "part_A_closed": run_ok and proof_conditions_ok and e_mechanism,
        "main_text_candidate": run_ok and proof_conditions_ok and e_mechanism and e_raw_theta,
        "part_B_inputs_ready": run_ok and proof_conditions_ok and all(len(r["frozen_y_sha256"]) == 64 for r in rows),
        "claim_scope": (
            "finite-horizon validation of the frozen replay, V_T=1, bounded G_T, "
            "the positive linear tail mechanism, and a separate raw-E_T scaling test"
        ),
    }
    payload = {
        "meta": {
            "tag": "exp_sep_partA",
            "part": "A",
            "game": "Gsep",
            "horizons": [int(r["T"]) for r in rows],
            "epsilon": float(cfg["epsilon"]),
            "beta0": float(cfg["beta0"]),
            "ell1": float(cfg["ell1"]),
            "u_star": U_STAR,
            "delta0": DELTA0,
            "eta": ETA,
            "theory_late_dg2": THEORY_LATE_DG2,
            "construction": "translated-amplified-v2",
            "x_convexity_margin": X_CONVEXITY_MARGIN,
            "y2_concavity_margin": Y2_CONCAVITY_MARGIN,
            "annulus_fraction_lower": ANNULUS_FRACTION_LOWER,
        },
        "summary": {
            "rows": rows,
            "verdict": verdict,
        },
    }
    dump_compact("exp_sep_partA", payload)
    return verdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, action="append", default=None)
    parser.add_argument(
        "--assemble",
        action="store_true",
        help="rebuild the Part A summary from saved npz files without rerunning D005",
    )
    args = parser.parse_args()
    cfg = _load_cfg()
    horizons = tuple(int(t) for t in args.T) if args.T else HORIZONS
    if args.assemble:
        rows = [row_from_saved(T) for T in horizons]
    else:
        rows = [run_one(T, cfg) for T in horizons]
    verdict = _assemble(rows, cfg)
    print("Part A verdict:", verdict)
    if not verdict["implementation_valid"]:
        raise SystemExit("Part A run checks failed")
    if verdict["raw_ET_scaling_observed"]:
        print(
            "Raw E_T scaling is visible on the target window: log-log slope "
            f"{verdict['loglog_slope_E_vs_T_target_window']:.3f}."
        )
    else:
        print(
            "Note: raw E_T still fails the predeclared finite-window scaling "
            "test; retain only the mechanism-level claim."
        )
    if not verdict["linear_tail_mechanism_valid"]:
        raise SystemExit("Part A mechanism checks failed")


if __name__ == "__main__":
    main()
