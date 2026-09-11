"""Closed-form radial map self-consistency check. 1D, random states.

SCOPE: the "implicit" radius below is computed by numerically inverting
radial_q itself (bisection on a monotone scalar function). This test therefore
verifies only that radial_q is strictly monotone, invertible, finite on large
inputs, and exact at sigma = 0 -- i.e. it rules out numerical/stability bugs
in the logaddexp implementation. It does NOT independently solve the FTRL
argmin of eq:w-cf, so it cannot catch an error in the closed-form derivation
itself (direction -Theta/||Theta||, radius q((||Theta||-zeta)_+)).

Uses bisection rather than scipy.optimize.brentq: the ambient SciPy is
binary-incompatible with NumPy 2.x.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from learner import radial_q  # noqa: E402


def brentq(f, a: float, b: float, xtol: float = 1e-12, maxiter: int = 200) -> float:
    """Monotone scalar root on [a, b]. Same contract as scipy.optimize.brentq here."""
    fa = float(f(a))
    fb = float(f(b))
    if fa == 0.0:
        return float(a)
    if fb == 0.0:
        return float(b)
    if fa * fb > 0.0:
        raise ValueError("root is not bracketed")
    lo, hi = float(a), float(b)
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        fm = float(f(mid))
        if abs(hi - lo) <= xtol or fm == 0.0:
            return mid
        if fa * fm <= 0.0:
            hi, fb = mid, fm
        else:
            lo, fa = mid, fm
    return 0.5 * (lo + hi)


def q_inv(r: float, a: float, beta: float, alpha: float) -> float:
    """s ≥ 0 such that radial_q(s) = r."""
    if r <= 0.0:
        return 0.0

    def f(s: float) -> float:
        return radial_q(s, a, beta, alpha) - r

    hi = 1.0
    while f(hi) < 0.0:
        hi *= 2.0
        if hi > 1e16:
            raise RuntimeError("q_inv failed to bracket")
    if f(0.0) == 0.0:
        return 0.0
    return float(brentq(f, 0.0, hi, xtol=1e-12))


def implicit_radius(sigma: float, a: float, beta: float, alpha: float) -> float:
    """Solve q^{-1}(r) = σ by brentq; compare with radial_q(σ)."""
    if sigma <= 0.0:
        return 0.0

    def h(r: float) -> float:
        return q_inv(r, a, beta, alpha) - sigma

    hi = max(sigma / beta, 1e-12)
    val = h(hi)
    tries = 0
    while val < 0.0:
        hi *= 2.0
        val = h(hi)
        tries += 1
        if tries > 60:
            raise RuntimeError("implicit_radius failed to bracket")
    if abs(val) < 1e-14:
        return float(hi)
    return float(brentq(h, 0.0, hi, xtol=1e-12))


def closed_form_w(theta: float, zeta: float, a: float, beta: float, alpha: float) -> float:
    sigma = max(abs(theta) - zeta, 0.0)
    if sigma == 0.0:
        return 0.0
    r = radial_q(sigma, a, beta, alpha)
    return -r * (theta / abs(theta))


def implicit_w(theta: float, zeta: float, a: float, beta: float, alpha: float) -> float:
    sigma = max(abs(theta) - zeta, 0.0)
    if sigma == 0.0:
        return 0.0
    r = implicit_radius(sigma, a, beta, alpha)
    return -r * (theta / abs(theta))


def main():
    rng = np.random.default_rng(0)
    n = 400
    max_rel = 0.0
    for _ in range(n):
        a = float(rng.uniform(0.5, 8.0))
        beta = float(rng.uniform(0.2, 12.0))
        alpha = float(rng.uniform(1e-4, 0.8))
        sigma = float(rng.uniform(0.0, 25.0))
        r_cf = radial_q(sigma, a, beta, alpha)
        r_imp = implicit_radius(sigma, a, beta, alpha)
        scale = max(1.0, abs(r_cf), abs(r_imp))
        rel = abs(r_cf - r_imp) / scale
        max_rel = max(max_rel, rel)
        if rel > 1e-8:
            raise AssertionError(
                f"radius mismatch cf={r_cf} imp={r_imp} rel={rel} "
                f"(a={a}, beta={beta}, alpha={alpha}, sigma={sigma})"
            )

        theta = float(rng.normal() * rng.uniform(0.0, 15.0))
        zeta = float(rng.uniform(0.0, 5.0))
        w_cf = closed_form_w(theta, zeta, a, beta, alpha)
        w_imp = implicit_w(theta, zeta, a, beta, alpha)
        scale_w = max(1.0, abs(w_cf), abs(w_imp))
        rel_w = abs(w_cf - w_imp) / scale_w
        max_rel = max(max_rel, rel_w)
        if rel_w > 1e-8:
            raise AssertionError(f"w mismatch cf={w_cf} imp={w_imp} rel={rel_w}")

    # sigma = 0 and q(0) = 0
    if radial_q(0.0, 2.0, 3.0, 0.1) != 0.0:
        raise AssertionError("q(0) != 0")
    if implicit_radius(0.0, 2.0, 3.0, 0.1) != 0.0:
        raise AssertionError("implicit r(0) != 0")

    print(f"closed-form vs implicit: {n} states, max rel err {max_rel:.2e}")


if __name__ == "__main__":
    main()
