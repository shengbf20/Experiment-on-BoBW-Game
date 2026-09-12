"""Rewrite leftover full-hist json dumps as compact summary json + npz."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from io_results import LONG_STRIDE, RESULTS, apply_stride, arrays_from_hist  # noqa: E402

# Long-run npz files are already stride-subsampled; do not overwrite them
# with a full T=2e5 hist extracted from a stale json.
KEEP_EXISTING_NPZ = {
    "exp1_G1_identity_long",
    "exp1_G1_gaussian_seed0_long",
}


def _wrap_flat_payload(stem: str, data: dict) -> dict:
    """Turn a flat long-run json into {meta, summary, T}."""
    T = data.get("T")
    summary = {k: v for k, v in data.items() if k not in ("meta", "summary", "hist", "T")}
    tag = stem
    if tag.startswith("exp1_"):
        tag = tag[len("exp1_") :]
    meta = {"tag": tag, "T": T}
    if "identity_long" in stem:
        meta.update({"game": "G1", "A": "identity"})
    elif "gaussian_seed" in stem:
        meta.update({"game": "G1", "A": "gaussian-spectral"})
    return {"meta": meta, "summary": summary, "T": T}


def _compact_one(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    stem = path.stem
    changed = False
    if "hist" in data:
        hist = data["hist"]
        payload = {k: v for k, v in data.items() if k != "hist"}
        if "T" not in payload:
            t = None
            if isinstance(payload.get("meta"), dict):
                t = payload["meta"].get("T")
            if t is None and isinstance(payload.get("summary"), dict):
                t = payload["summary"].get("T")
            if t is None:
                sample = hist.get("reg_x") or hist.get("Q")
                t = len(sample) if isinstance(sample, list) else None
            if t is not None:
                payload["T"] = int(t)
        path.write_text(json.dumps(payload), encoding="utf-8")
        print(f"stripped hist from {path}")
        changed = True
        if stem in KEEP_EXISTING_NPZ and (RESULTS / f"{stem}.npz").is_file():
            print(f"kept existing npz for {stem}")
        else:
            arrays = arrays_from_hist(hist)
            if arrays:
                npz_path = RESULTS / f"{stem}.npz"
                np.savez_compressed(npz_path, **arrays)
                print(f"wrote {npz_path}  keys={sorted(arrays)}")
        data = payload
    if "summary" not in data and stem.endswith("_long"):
        wrapped = _wrap_flat_payload(stem, data)
        path.write_text(json.dumps(wrapped), encoding="utf-8")
        print(f"wrapped flat schema in {path}")
        changed = True
        data = wrapped
    if not changed and "T" not in data:
        meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
        t = meta.get("T")
        if t is not None:
            data = dict(data)
            data["T"] = int(t)
            path.write_text(json.dumps(data), encoding="utf-8")
            print(f"added T to {path}")


def _payload_T(stem: str) -> int | None:
    path = RESULTS / f"{stem}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    t = payload.get("T")
    if t is None and isinstance(payload.get("meta"), dict):
        t = payload["meta"].get("T")
    if t is None and isinstance(payload.get("summary"), dict):
        t = payload["summary"].get("T")
    return int(t) if t is not None else None


def _annotate_stride(stem: str) -> None:
    npz_path = RESULTS / f"{stem}.npz"
    if not npz_path.is_file():
        return
    with np.load(npz_path) as data:
        arrays = {k: np.asarray(data[k]) for k in data.files}
    if "stride" not in arrays:
        return
    T = _payload_T(stem)
    stride = int(np.asarray(arrays["stride"]).reshape(-1)[0])
    changed = False
    if "t" not in arrays and "Q" in arrays:
        n = int(np.asarray(arrays["Q"]).shape[0])
        if T is not None:
            arrays["t"] = np.arange(1, int(T) + 1, dtype=np.int64)[::stride]
        else:
            arrays["t"] = (1 + stride * np.arange(n)).astype(np.int64)
        changed = True
    if T is not None and "T" not in arrays:
        arrays["T"] = np.array([int(T)], dtype=np.int32)
        changed = True
    if changed:
        np.savez_compressed(npz_path, **arrays)
        print(f"annotated stride metadata on {npz_path}")


def _downsample_long_npz(stem: str) -> None:
    npz_path = RESULTS / f"{stem}.npz"
    if not npz_path.is_file():
        return
    with np.load(npz_path) as data:
        arrays = {k: np.asarray(data[k]) for k in data.files}
    if "stride" in arrays:
        return
    n = 0
    for arr in arrays.values():
        if arr.ndim >= 1 and arr.shape[0] > n:
            n = int(arr.shape[0])
    if n <= 20_000:
        return
    out = apply_stride(arrays, LONG_STRIDE, T=_payload_T(stem))
    np.savez_compressed(npz_path, **out)
    q = out.get("Q")
    n_new = int(q.shape[0]) if q is not None else n
    print(f"downsampled {npz_path} stride={LONG_STRIDE} n={n}->{n_new}")


def main() -> None:
    for path in sorted(RESULTS.glob("*.json")):
        _compact_one(path)
    for path in sorted(RESULTS.glob("*.npz")):
        stem = path.stem
        if stem.endswith("_long"):
            _downsample_long_npz(stem)
            _annotate_stride(stem)


if __name__ == "__main__":
    main()
