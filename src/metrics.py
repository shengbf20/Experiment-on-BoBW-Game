"""Incremental regret, linearized regret, Q, V, G, and restricted gap."""

from __future__ import annotations

import numpy as np

from games import Game, _vec

_DTYPE = np.float64


class RunningMetrics:
    def __init__(self, game: Game, u_x=None, u_y=None, radius: float = 1.0):
        self.game = game
        self.u_x = (
            game.comparator_x()
            if u_x is None
            else _vec(u_x, game.dim_x)
        )
        self.u_y = (
            game.comparator_y()
            if u_y is None
            else _vec(u_y, game.dim_y)
        )
        self.radius = float(radius)
        self.t = 0
        self.reg_x = 0.0
        self.reg_y = 0.0
        self.lin_x = 0.0
        self.lin_y = 0.0
        self.Q = 0.0
        self.V_x = 0.0
        self.V_y = 0.0
        self.G_x = 0.0
        self.G_y = 0.0
        self._sum_x = np.zeros(game.dim_x, dtype=_DTYPE)
        self._sum_y = np.zeros(game.dim_y, dtype=_DTYPE)
        self.x_bar = self._sum_x.copy()
        self.y_bar = self._sum_y.copy()
        self.gap = 0.0
        self._x_prev: np.ndarray | None = None
        self._y_prev: np.ndarray | None = None
        self._z_prev: np.ndarray | None = None

    def step(self, x, y, gx=None, gy=None) -> dict:
        x = _vec(x, self.game.dim_x)
        y = _vec(y, self.game.dim_y)
        if gx is None or gy is None:
            gx, gy = self.game.feedback(x, y)
        else:
            gx = _vec(gx, self.game.dim_x)
            gy = _vec(gy, self.game.dim_y)

        self.t += 1
        phi_xy = self.game.phi(x, y)
        self.reg_x += phi_xy - self.game.phi(self.u_x, y)
        self.reg_y += self.game.phi(x, self.u_y) - phi_xy
        self.lin_x += float(gx @ (x - self.u_x))
        self.lin_y += float(gy @ (y - self.u_y))
        self.G_x = max(self.G_x, float(np.linalg.norm(gx)))
        self.G_y = max(self.G_y, float(np.linalg.norm(gy)))

        z = np.concatenate([x, y])
        if self._z_prev is not None:
            self.Q += float(np.sum((z - self._z_prev) ** 2))
        if self._y_prev is not None:
            du = self.game.grad_x_phi(self.u_x, y) - self.game.grad_x_phi(self.u_x, self._y_prev)
            self.V_x += float(np.sum(du ** 2))
        if self._x_prev is not None:
            dv = self.game.grad_y_phi(x, self.u_y) - self.game.grad_y_phi(self._x_prev, self.u_y)
            self.V_y += float(np.sum(dv ** 2))

        self._sum_x += x
        self._sum_y += y
        self.x_bar = self._sum_x / self.t
        self.y_bar = self._sum_y / self.t
        self.gap = self.game.restricted_gap(self.x_bar, self.y_bar, self.radius)

        self._x_prev = x.copy()
        self._y_prev = y.copy()
        self._z_prev = z
        return self.snapshot()

    def snapshot(self) -> dict:
        return {
            "t": self.t,
            "reg_x": self.reg_x,
            "reg_y": self.reg_y,
            "lin_x": self.lin_x,
            "lin_y": self.lin_y,
            "Q": self.Q,
            "V_x": self.V_x,
            "V_y": self.V_y,
            "G_x": self.G_x,
            "G_y": self.G_y,
            "gap": self.gap,
            "x_bar": self.x_bar.copy(),
            "y_bar": self.y_bar.copy(),
        }
