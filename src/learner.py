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
        restart: bool = False,
    ):
        self.dim = int(dim)
        self.epsilon = float(epsilon)
        self.beta0 = float(beta0)
        self.ell = float(ell1)
        self.adaptive = bool(adaptive)
        self.restart = bool(restart)
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
        self.J_path = [0]
        self.just_restarted = False
        self.n_restarts = 0
        self.restart_path = [0]

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

    def _cold_reset(self) -> None:
        """Fresh copy after a doubling: new γ, origin action, clipping re-inited."""
        self.gamma = self.epsilon * self.beta
        self.Mhat = self.gamma
        self.B = 4.0
        self.Vbar = 4.0 * self.gamma * self.gamma
        self.alpha = self.epsilon / (np.sqrt(self.B) * np.log(self.B) ** 2)
        self.sum_hat2 = 0.0
        self.sum_hat2_over_M2 = 0.0
        self.zeta = 0.0
        self.h = np.zeros(self.dim, dtype=_DTYPE)
        self.G_cum = np.zeros(self.dim, dtype=_DTYPE)
        self.z_prev = None
        self.g_prev = None
        self.action = np.zeros(self.dim, dtype=_DTYPE)

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

        doubled = False
        # Skip χ on the first round of an epoch (t=1, or the round after a cold reset).
        if self.adaptive and self.g_prev is not None:
            self.last_chi = row_ratio(g, self.g_prev, z, self.z_prev)
            if self.last_chi > self.ell:
                self.ell = 2.0 * max(self.ell, self.last_chi)
                self.J += 1
                doubled = True
            self.beta = self.beta0 + 64.0 * self.ell * self.ell / self.beta0

        if doubled and self.restart:
            self._cold_reset()
            self.just_restarted = True
            self.n_restarts += 1
        else:
            self.just_restarted = False
            self.h = g.copy()
            self.G_cum = self.G_cum + g
            self.z_prev = z.copy()
            self.g_prev = g.copy()
            self._refresh_action()
        self.beta_path.append(self.beta)
        self.ell_path.append(self.ell)
        self.J_path.append(self.J)
        self.restart_path.append(self.n_restarts)

    def finite(self) -> bool:
        vals = (self.alpha, self.B, self.Vbar, self.zeta, self.beta, self.gamma, self.Mhat)
        if not all(np.isfinite(v) for v in vals):
            return False
        return bool(np.isfinite(self.action).all() and np.isfinite(self.G_cum).all())

    def assert_invariants(self) -> None:
        if not self.finite():
            raise AssertionError("non-finite player state")
        if not self.restart and abs(self.gamma - self.gamma_init) > 0.0:
            raise AssertionError("gamma must stay frozen")
        if self.restart and abs(self.gamma - self.epsilon * self.beta) > 1e-12:
            raise AssertionError("restart gamma must equal epsilon * beta")
        if self.B < 4.0:
            raise AssertionError(f"B={self.B} < 4")
        if self.Vbar + 1e-12 < 4.0 * self.Mhat * self.Mhat:
            raise AssertionError("Vbar < 4 Mhat^2")
        if float(np.sqrt(self.Vbar)) + 1e-12 < 2.0 * self.Mhat:
            raise AssertionError("a < 2 Mhat")
        if self.alpha <= 0.0:
            raise AssertionError("alpha must be positive")
        betas = np.asarray(self.beta_path, dtype=_DTYPE)
        if np.any(np.diff(betas) < -1e-15):
            raise AssertionError("beta decreased")
        if not self.adaptive:
            if self.J != 0 or abs(self.beta - self.beta_init) > 0.0:
                raise AssertionError("frozen-beta player doubled or moved beta")


def run_loop(
    game: Game,
    player_x: ClosedFormPlayer,
    T: int,
    y_policy,
    player_y: ClosedFormPlayer | None = None,
    observe_y=None,
    radius: float = 1.0,
):
    """Simultaneous play. y_policy(t, x) -> y with t 1-indexed.

    If observe_y(t) is true, player_y must be given and is updated; otherwise
    Y is exogenous and does not receive gradients.
    """
    T = int(T)
    if observe_y is None:
        observe_y = lambda t: player_y is not None
    metrics = RunningMetrics(game, radius=radius)
    hist = {
        key: []
        for key in (
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
            "y_learner",
        )
    }
    max_w = 0.0
    cum_x = np.zeros(player_x.dim, dtype=_DTYPE)
    cum_y = None if player_y is None else np.zeros(player_y.dim, dtype=_DTYPE)
    for t in range(1, T + 1):
        x = player_x.action.copy()
        y = np.asarray(y_policy(t, x), dtype=_DTYPE).reshape(-1)
        max_w = max(max_w, float(np.linalg.norm(x)), float(np.linalg.norm(y)))
        gx, gy = game.feedback(x, y)
        snap = metrics.step(x, y, gx, gy)
        z = np.concatenate([x, y])
        player_x.observe(gx, z)
        if player_x.just_restarted:
            cum_x = player_x.G_cum.copy()
        else:
            cum_x = cum_x + gx
            if not np.allclose(player_x.G_cum, cum_x):
                raise AssertionError("G_cum^x was reset or failed to accumulate")
        player_x.assert_invariants()
        used_y = bool(observe_y(t))
        if used_y:
            if player_y is None:
                raise ValueError("observe_y is true but player_y is None")
            player_y.observe(gy, z)
            if player_y.just_restarted:
                cum_y = player_y.G_cum.copy()
            else:
                cum_y = cum_y + gy
                if not np.allclose(player_y.G_cum, cum_y):
                    raise AssertionError("G_cum^y was reset or failed to accumulate")
            player_y.assert_invariants()
        hist["x_norm"].append(float(np.linalg.norm(x)))
        hist["y_norm"].append(float(np.linalg.norm(y)))
        hist["y_learner"].append(1 if used_y else 0)
        for key in ("reg_x", "reg_y", "lin_x", "lin_y", "Q", "gap", "V_x", "V_y", "G_x", "G_y"):
            hist[key].append(float(snap[key]))
    hist["beta_x"] = [float(v) for v in player_x.beta_path]
    hist["ell_x"] = [float(v) for v in player_x.ell_path]
    hist["J_x"] = [int(v) for v in player_x.J_path]
    hist["restart_x"] = [int(v) for v in player_x.restart_path]
    if player_y is not None:
        hist["beta_y"] = [float(v) for v in player_y.beta_path]
        hist["ell_y"] = [float(v) for v in player_y.ell_path]
        hist["J_y"] = [int(v) for v in player_y.J_path]
        hist["restart_y"] = [int(v) for v in player_y.restart_path]
        hist["t_y"] = player_y.t
    else:
        hist["beta_y"] = []
        hist["ell_y"] = []
        hist["J_y"] = []
        hist["restart_y"] = []
        hist["t_y"] = 0
    return metrics, hist, max_w


def self_play(
    game: Game,
    player_x: ClosedFormPlayer,
    player_y: ClosedFormPlayer,
    T: int,
    radius: float = 1.0,
):
    def y_policy(_t, _x):
        return player_y.action.copy()

    return run_loop(
        game,
        player_x,
        T,
        y_policy=y_policy,
        player_y=player_y,
        observe_y=lambda _t: True,
        radius=radius,
    )
