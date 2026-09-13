"""Compact result I/O: summary json + array npz. Never dump full hist json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

_DTYPE = np.float64
LONG_STRIDE = 10

# Curves used by the retained experiment plotters.
CURVE_KEYS = (
    "reg_x",
    "reg_y",
    "lin_x",
    "lin_y",
    "Q",
    "gap",
    "V_x",
    "V_y",
    "G_x",
    "G_y",
    "x_norm",
    "y_norm",
    "dist_x",
    "dist_y",
    "J_x",
    "J_y",
    "beta_x",
    "beta_y",
    "ell_x",
    "ell_y",
    "restart_x",
    "lam_x",
    "lam_y",
)


def _to_array(value):
    arr = np.asarray(value)
    if arr.dtype == object:
        return None
    if arr.dtype.kind in "iu":
        return arr.astype(np.int32, copy=False)
    return arr.astype(_DTYPE, copy=False)


def arrays_from_hist(hist: dict, keys: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    want = CURVE_KEYS if keys is None else keys
    out: dict[str, np.ndarray] = {}
    for key in want:
        if key not in hist:
            continue
        arr = _to_array(hist[key])
        if arr is None or arr.ndim == 0:
            continue
        out[key] = arr
    return out


def _is_curve(arr: np.ndarray) -> bool:
    return arr.ndim >= 1 and arr.shape[0] > 1


def apply_stride(
    arrays: dict[str, np.ndarray],
    stride: int,
    T: int | None = None,
) -> dict[str, np.ndarray]:
    """Subsample 1-d curves. T=2e4 main figures should pass stride=None instead."""
    if stride <= 1:
        return dict(arrays)
    orig_n = None
    out: dict[str, np.ndarray] = {}
    for key, arr in arrays.items():
        if key in ("stride", "T", "t"):
            continue
        if _is_curve(arr):
            if orig_n is None:
                orig_n = int(arr.shape[0])
            sl = [slice(None)] * arr.ndim
            sl[0] = slice(None, None, stride)
            out[key] = arr[tuple(sl)]
        else:
            out[key] = arr
    n0 = int(orig_n or 0)
    out["stride"] = np.array([int(stride)], dtype=np.int32)
    t_full = int(T) if T is not None else n0
    out["T"] = np.array([t_full], dtype=np.int32)
    if T is not None:
        out["t"] = np.arange(1, int(T) + 1, dtype=np.int64)[::stride]
    elif n0:
        out["t"] = np.arange(1, n0 + 1, dtype=np.int64)[::stride]
    return out


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def dump_compact(
    stem: str,
    payload: dict,
    hist: dict | None = None,
    extra_arrays: dict | None = None,
    keys: tuple[str, ...] | None = None,
    stride: int | None = None,
) -> tuple[Path, Path | None]:
    """Write meta/summary json and optional npz. payload must not include hist."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    if "hist" in payload:
        raise ValueError("dump_compact payload must not contain hist")
    body = dict(payload)
    meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
    T = body.get("T", meta.get("T") if meta else None)
    if T is not None:
        body["T"] = int(T)
    json_path = RESULTS / f"{stem}.json"
    json_path.write_text(json.dumps(body, default=_json_default), encoding="utf-8")
    arrays: dict[str, np.ndarray] = {}
    if hist is not None:
        arrays.update(arrays_from_hist(hist, keys=keys))
    if extra_arrays:
        for key, value in extra_arrays.items():
            arr = _to_array(value)
            if arr is not None:
                arrays[key] = arr
    if stride is not None and stride > 1:
        arrays = apply_stride(arrays, int(stride), T=body.get("T"))
    npz_path = None
    if arrays:
        npz_path = RESULTS / f"{stem}.npz"
        np.savez_compressed(npz_path, **arrays)
    print(f"wrote {json_path}")
    if npz_path is not None:
        print(f"wrote {npz_path}")
    return json_path, npz_path


def load_run(stem: str) -> tuple[dict, dict]:
    json_path = RESULTS / f"{stem}.json"
    npz_path = RESULTS / f"{stem}.npz"
    if not json_path.is_file():
        raise FileNotFoundError(f"missing {json_path}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if npz_path.is_file():
        data = np.load(npz_path)
        hist = {k: np.asarray(data[k]) for k in data.files}
    elif "hist" in payload:
        hist = payload["hist"]
    else:
        raise FileNotFoundError(f"missing {npz_path} and no hist in json")
    return payload, hist
