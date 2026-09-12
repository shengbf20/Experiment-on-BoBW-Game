"""Smoke: compact dump contract. Does not run a learner or overwrite real results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from io_results import RESULTS, dump_compact, load_run  # noqa: E402

STEM = "_io_smoke"


def _cleanup() -> None:
    for suffix in (".json", ".npz"):
        path = RESULTS / f"{STEM}{suffix}"
        if path.is_file():
            path.unlink()


def main() -> None:
    _cleanup()
    payload = {"meta": {"tag": STEM, "T": 4}, "summary": {"Q_T": 1.5}}
    try:
        dump_compact(STEM, {**payload, "hist": {"Q": [0.0]}})
        raise AssertionError("dump_compact must reject hist in payload")
    except ValueError as exc:
        if "hist" not in str(exc):
            raise
    dump_compact(
        STEM,
        payload,
        {"Q": np.arange(4, dtype=float), "gap": np.linspace(1.0, 0.1, 4)},
        extra_arrays={"dQ": np.ones(4)},
        keys=("Q", "gap"),
        stride=2,
    )
    loaded, hist = load_run(STEM)
    if "hist" in loaded:
        raise AssertionError("json must not contain hist")
    if loaded.get("T") != 4:
        raise AssertionError(f"T not promoted: {loaded}")
    if int(np.asarray(hist["stride"]).reshape(-1)[0]) != 2:
        raise AssertionError("stride missing from npz")
    if len(hist["Q"]) != 2:
        raise AssertionError(f"expected 2 subsampled Q points, got {len(hist['Q'])}")
    if "dQ" not in hist:
        raise AssertionError("extra_arrays not written")
    raw = json.loads((RESULTS / f"{STEM}.json").read_text(encoding="utf-8"))
    if "hist" in raw:
        raise AssertionError("json file still has hist")
    _cleanup()
    print("check_io_results: ok")


if __name__ == "__main__":
    main()
