"""Hsieh et al., COLT 2021: Euclidean OptDA with the paper adaptive rule.

Source: Hsieh, Antonakopoulos, Mertikopoulos, "Adaptive Learning in
Continuous Games", COLT 2021.

- (OptDA) Sec. 4.1
- (Adapt) Sec. 4.2
- g_0 = 0 as in the OptMD template of Sec. 3
- quadratic regularizer h(x) = ||x||^2 / 2 (the PEG instance)

Unconstrained Euclidean OptDA plays the same action as Euclidean OptFTRL:
    X_t = -G_{t-1} / λ_t,
    x_t = X_t - g_{t-1} / λ_t = -(G_{t-1} + g_{t-1}) / λ_t,
    λ_t = sqrt(τ + sum_{s < t} ||g_s - g_{s-1}||^2).

τ is the paper's free player-specific constant. Default τ=1; do not fit it
to D005 curves. This module is the isolated Part B baseline: it must not
import ClosedFormPlayer or the separation-example generator.
"""

from __future__ import annotations

import numpy as np

_DTYPE = np.float64


class HsiehOptDA:
    """Optimistic dual averaging, Euclidean, adaptive λ (Hsieh 2021)."""

    def __init__(self, dim: int, *, tau: float = 1.0):
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        self.dim = int(dim)
        self.tau = float(tau)
        self.G_cum = np.zeros(self.dim, dtype=_DTYPE)
        self.g_prev = np.zeros(self.dim, dtype=_DTYPE)  # g_0 = 0
        self.sum_delta = 0.0
        self.lam = float(np.sqrt(self.tau))
        self.action = np.zeros(self.dim, dtype=_DTYPE)
        self.t = 0
        self.just_restarted = False
        self.lambda_path = [self.lam]
        self.beta_path: list[float] = []
        self.ell_path: list[float] = []
        self.J_path = [0]
        self.restart_path = [0]
        self.J = 0

    def observe(self, g, z=None) -> None:
        del z  # OptDA uses only own gradients; joint actions stay unused.
        g = np.asarray(g, dtype=_DTYPE).reshape(-1)
        if g.size != self.dim:
            raise ValueError(f"g dim {g.size} != {self.dim}")
        self.t += 1
        delta = g - self.g_prev
        self.sum_delta += float(delta @ delta)
        self.G_cum = self.G_cum + g
        self.g_prev = g.copy()
        self.lam = float(np.sqrt(self.tau + self.sum_delta))
        # Next action: X_{t+1} - g_t / λ_{t+1} with X_{t+1} = -G_t / λ_{t+1}.
        self.action = -(self.G_cum + self.g_prev) / self.lam
        self.just_restarted = False
        self.lambda_path.append(self.lam)
        self.J_path.append(0)
        self.restart_path.append(0)

    def finite(self) -> bool:
        return bool(
            np.isfinite(self.lam)
            and np.isfinite(self.action).all()
            and np.isfinite(self.G_cum).all()
        )

    def assert_invariants(self) -> None:
        if not self.finite():
            raise AssertionError("non-finite Hsieh OptDA state")
        if self.lam + 1e-15 < np.sqrt(self.tau):
            raise AssertionError("lambda fell below sqrt(tau)")
        lams = np.asarray(self.lambda_path, dtype=_DTYPE)
        if np.any(np.diff(lams) < -1e-15):
            raise AssertionError("Hsieh lambda decreased")
