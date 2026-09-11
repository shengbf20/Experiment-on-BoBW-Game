"""Closed-form unconstrained player (D005 core).

adaptive=False skips doubling (frozen β). adaptive=True raises ℓ,β after t≥2.
"""

from __future__ import annotations

import numpy as np

from games import Game, _vec
from metrics import RunningMetrics

_DTYPE = np.float64


def radial_q(s: float, a: float, beta: float, alpha: float) -> float:
    """q_t(s) via logaddexp; equivalent to (a/β) log((1 + k e^{s/a}) / (1+k))."""
    k = beta * alpha / a
    return float((a / beta) * (np.logaddexp(0.0, np.log(k) + s / a) - np.log1p(k)))


def row_ratio(g, g_prev, z, z_prev) -> float:
    """χ_t: own gradient increment over public joint-action increment."""
    dz = z - z_prev
    denom = float(np.linalg.norm(dz))
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(g - g_prev) / denom)


class ClosedFormPlayer:
    def __init__(
        self,
        dim: int,
        *,
        epsilon: float = 1.0,
        beta0: float = 1.0,
        ell1: float = 1.0e-3,
        adaptive: bool = False,
        beta: float | None = None,
    ):
        self.dim = int(dim)
        self.epsilon = float(epsilon)
        self.beta0 = float(beta0)
        self.ell = float(ell1)
        self.adaptive = bool(adaptive)
        if beta is None:
            self.beta = self.beta0 + 64.0 * self.ell * self.ell / self.beta0
        else:
            self.beta = float(beta)
        self.gamma = self.epsilon * self.beta
        self.beta_init = self.beta
        self.gamma_init = self.gamma

        self.Mhat = self.gamma
        self.B = 4.0
        self.Vbar = 4.0 * self.gamma * self.gamma
        self.alpha = self.epsilon / (np.sqrt(self.B) * np.log(self.B) ** 2)
        self.sum_hat2 = 0.0
        self.sum_hat2_over_M2 = 0.0
        self.zeta = 0.0
        self.action = np.zeros(self.dim, dtype=_DTYPE)
        self.h = np.zeros(self.dim, dtype=_DTYPE)
        self.G_cum = np.zeros(self.dim, dtype=_DTYPE)
        self.z_prev: np.ndarray | None = None
        self.g_prev: np.ndarray | None = None
        self.J = 0
        self.t = 0
        self.last_chi: float | None = None
        self.beta_path = [self.beta]
        self.ell_path = [self.ell]

    def _refresh_action(self) -> None:
        theta = self.h + self.G_cum
        nrm = float(np.linalg.norm(theta))
        sigma = max(nrm - self.zeta, 0.0)
        if sigma == 0.0:
            self.action = np.zeros(self.dim, dtype=_DTYPE)
            return
        a = float(np.sqrt(self.Vbar))
        q = radial_q(sigma, a, self.beta, self.alpha)
        self.action = (-q / nrm) * theta

    def observe(self, g, z) -> None:
        g = np.asarray(g, dtype=_DTYPE).reshape(-1)
        z = np.asarray(z, dtype=_DTYPE).reshape(-1)
        if g.size != self.dim:
            raise ValueError(f"g dim {g.size} != {self.dim}")

        self.t += 1
        a_t = float(np.sqrt(self.Vbar))
        M_old = self.Mhat

        delta = g - self.h
        n_delta = float(np.linalg.norm(delta))
        if n_delta == 0.0:
            hat = np.zeros(self.dim, dtype=_DTYPE)
        else:
            hat = delta * min(1.0, M_old / n_delta)

        n_hat2 = float(hat @ hat)
        lam = 2.0 * n_hat2 / a_t

        self.Mhat = max(M_old, n_delta)
        self.sum_hat2_over_M2 += n_hat2 / (M_old * M_old)
        self.sum_hat2 += n_hat2
        self.B = 4.0 + self.sum_hat2_over_M2
        self.Vbar = 4.0 * self.Mhat * self.Mhat + self.sum_hat2
        self.alpha = self.epsilon / (np.sqrt(self.B) * np.log(self.B) ** 2)
        self.zeta += lam

        if self.adaptive and self.t >= 2:
            self.last_chi = row_ratio(g, self.g_prev, z, self.z_prev)
            if self.last_chi > self.ell:
                self.ell = 2.0 * max(self.ell, self.last_chi)
                self.J += 1
            self.beta = self.beta0 + 64.0 * self.ell * self.ell / self.beta0

        self.h = g.copy()
        self.G_cum = self.G_cum + g
        self.z_prev = z.copy()
        self.g_prev = g.copy()
        self._refresh_action()
        self.beta_path.append(self.beta)
        self.ell_path.append(self.ell)

    def finite(self) -> bool:
        vals = (self.alpha, self.B, self.Vbar, self.zeta, self.beta, self.gamma, self.Mhat)
        if not all(np.isfinite(v) for v in vals):
            return False
        return bool(np.isfinite(self.action).all() and np.isfinite(self.G_cum).all())


def self_play(
    game: Game,
    player_x: ClosedFormPlayer,
    player_y: ClosedFormPlayer,
    T: int,
    radius: float = 1.0,
):
    metrics = RunningMetrics(game, radius=radius)
    hist = {key: [] for key in ("reg_x", "reg_y", "Q", "gap", "x_norm", "y_norm")}
    max_w = 0.0
    for _ in range(int(T)):
        x = player_x.action.copy()
        y = player_y.action.copy()
        max_w = max(max_w, float(np.linalg.norm(x)), float(np.linalg.norm(y)))
        gx, gy = game.feedback(x, y)
        snap = metrics.step(x, y, gx, gy)
        z = np.concatenate([x, y])
        player_x.observe(gx, z)
        player_y.observe(gy, z)
        hist["x_norm"].append(float(np.linalg.norm(x)))
        hist["y_norm"].append(float(np.linalg.norm(y)))
        for key in ("reg_x", "reg_y", "Q", "gap"):
            hist[key].append(float(snap[key]))
    return metrics, hist, max_w
