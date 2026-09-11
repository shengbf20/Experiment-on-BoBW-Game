"""G1 bilinear, G2 quadratic, G3 separation example.

Feedback matches eq:feedback: gx = ∇x Φ, gy = −∇y Φ.
"""

from __future__ import annotations

import numpy as np

_DTYPE = np.float64


def _vec(z, dim: int) -> np.ndarray:
    out = np.asarray(z, dtype=_DTYPE).reshape(-1)
    if out.size != dim:
        raise ValueError(f"expected dim {dim}, got {out.size}")
    return out


class Game:
    name: str
    dim_x: int
    dim_y: int

    def phi(self, x, y) -> float:
        raise NotImplementedError

    def grad_x_phi(self, x, y) -> np.ndarray:
        raise NotImplementedError

    def grad_y_phi(self, x, y) -> np.ndarray:
        raise NotImplementedError

    def feedback(self, x, y):
        x = _vec(x, self.dim_x)
        y = _vec(y, self.dim_y)
        gx = self.grad_x_phi(x, y)
        gy = -self.grad_y_phi(x, y)
        return gx, gy

    def saddle(self):
        return np.zeros(self.dim_x, dtype=_DTYPE), np.zeros(self.dim_y, dtype=_DTYPE)

    def comparator_x(self) -> np.ndarray:
        return self.saddle()[0]

    def comparator_y(self) -> np.ndarray:
        return self.saddle()[1]

    def restricted_gap(self, x, y, radius: float) -> float:
        raise NotImplementedError


class BilinearGame(Game):
    """Φ(x, y) = x⊤ A y. Saddle at 0. Gap on the Euclidean R-ball is closed-form."""

    name = "G1"

    def __init__(self, dim: int = 10, A: np.ndarray | None = None):
        self.dim_x = int(dim)
        self.dim_y = int(dim)
        if A is None:
            self.A = np.eye(dim, dtype=_DTYPE)
        else:
            self.A = np.asarray(A, dtype=_DTYPE)
            if self.A.shape != (dim, dim):
                raise ValueError("A must be (dim, dim)")

    @classmethod
    def gaussian(cls, dim: int = 10, seed: int = 0) -> "BilinearGame":
        rng = np.random.default_rng(seed)
        return cls(dim=dim, A=rng.normal(size=(dim, dim)).astype(_DTYPE))

    def phi(self, x, y) -> float:
        x = _vec(x, self.dim_x)
        y = _vec(y, self.dim_y)
        return float(x @ (self.A @ y))

    def grad_x_phi(self, x, y) -> np.ndarray:
        y = _vec(y, self.dim_y)
        return self.A @ y

    def grad_y_phi(self, x, y) -> np.ndarray:
        x = _vec(x, self.dim_x)
        return self.A.T @ x

    def restricted_gap(self, x, y, radius: float) -> float:
        x = _vec(x, self.dim_x)
        y = _vec(y, self.dim_y)
        return float(radius * (np.linalg.norm(self.A.T @ x) + np.linalg.norm(self.A @ y)))


def _ball_max_lin_minus_quad(vec: np.ndarray, mu: float, radius: float) -> float:
    """sup_{||v||≤R} ⟨vec, v⟩ − (μ/2)||v||²."""
    n = float(np.linalg.norm(vec))
    if n == 0.0:
        return 0.0
    if n <= mu * radius:
        return (n * n) / (2.0 * mu)
    return radius * n - 0.5 * mu * radius * radius


def _ball_min_quad_plus_lin(vec: np.ndarray, mu: float, radius: float) -> float:
    """inf_{||u||≤R} (μ/2)||u||² + ⟨vec, u⟩."""
    n = float(np.linalg.norm(vec))
    if n == 0.0:
        return 0.0
    if n <= mu * radius:
        return -(n * n) / (2.0 * mu)
    return 0.5 * mu * radius * radius - radius * n


class QuadraticGame(Game):
    """Φ = (μ/2)||x||² + x⊤ A y − (μ/2)||y||². Strongly convex-concave; saddle at 0."""

    name = "G2"

    def __init__(self, dim: int = 10, mu: float = 0.2, A: np.ndarray | None = None):
        if mu <= 0:
            raise ValueError("G2 requires mu > 0")
        self.dim_x = int(dim)
        self.dim_y = int(dim)
        self.mu = float(mu)
        if A is None:
            self.A = np.eye(dim, dtype=_DTYPE)
        else:
            self.A = np.asarray(A, dtype=_DTYPE)
            if self.A.shape != (dim, dim):
                raise ValueError("A must be (dim, dim)")

    @classmethod
    def gaussian(cls, dim: int = 10, mu: float = 0.2, seed: int = 0) -> "QuadraticGame":
        rng = np.random.default_rng(seed)
        return cls(dim=dim, mu=mu, A=rng.normal(size=(dim, dim)).astype(_DTYPE))

    def phi(self, x, y) -> float:
        x = _vec(x, self.dim_x)
        y = _vec(y, self.dim_y)
        return float(0.5 * self.mu * (x @ x) + x @ (self.A @ y) - 0.5 * self.mu * (y @ y))

    def grad_x_phi(self, x, y) -> np.ndarray:
        x = _vec(x, self.dim_x)
        y = _vec(y, self.dim_y)
        return self.mu * x + self.A @ y

    def grad_y_phi(self, x, y) -> np.ndarray:
        x = _vec(x, self.dim_x)
        y = _vec(y, self.dim_y)
        return self.A.T @ x - self.mu * y

    def restricted_gap(self, x, y, radius: float) -> float:
        x = _vec(x, self.dim_x)
        y = _vec(y, self.dim_y)
        r = float(radius)
        sup = 0.5 * self.mu * float(x @ x) + _ball_max_lin_minus_quad(self.A.T @ x, self.mu, r)
        inf = -0.5 * self.mu * float(y @ y) + _ball_min_quad_plus_lin(self.A @ y, self.mu, r)
        return float(sup - inf)


class SeparationGame(Game):
    """Appendix A: Φ = 2√(1+x²) + xy − y²/2. Comparator u* = −1/√3 against y ≡ 1."""

    name = "G3"
    dim_x = 1
    dim_y = 1
    u_star = -1.0 / np.sqrt(3.0)

    def phi(self, x, y) -> float:
        x = float(_vec(x, 1)[0])
        y = float(_vec(y, 1)[0])
        return float(2.0 * np.sqrt(1.0 + x * x) + x * y - 0.5 * y * y)

    def grad_x_phi(self, x, y) -> np.ndarray:
        x = float(_vec(x, 1)[0])
        y = float(_vec(y, 1)[0])
        return np.array([2.0 * x / np.sqrt(1.0 + x * x) + y], dtype=_DTYPE)

    def grad_y_phi(self, x, y) -> np.ndarray:
        x = float(_vec(x, 1)[0])
        y = float(_vec(y, 1)[0])
        return np.array([x - y], dtype=_DTYPE)

    def comparator_x(self) -> np.ndarray:
        return np.array([self.u_star], dtype=_DTYPE)

    def restricted_gap(self, x, y, radius: float) -> float:
        x = float(_vec(x, 1)[0])
        y = float(_vec(y, 1)[0])
        r = float(radius)
        v = np.clip(x, -r, r)
        sup = 2.0 * np.sqrt(1.0 + x * x) + x * v - 0.5 * v * v
        k = -0.5 * y
        if abs(k) < 1.0:
            u0 = k / np.sqrt(1.0 - k * k)
            u = float(np.clip(u0, -r, r))
        elif y >= 2.0:
            u = -r
        else:
            u = r
        inf = 2.0 * np.sqrt(1.0 + u * u) + u * y - 0.5 * y * y
        return float(sup - inf)


def make_game(name: str, **kwargs) -> Game:
    key = name.strip().upper()
    if key in {"G1", "BILINEAR"}:
        return BilinearGame(**kwargs)
    if key in {"G2", "QUADRATIC"}:
        return QuadraticGame(**kwargs)
    if key in {"G3", "SEPARATION"}:
        if kwargs:
            raise TypeError("SeparationGame takes no constructors args")
        return SeparationGame()
    raise ValueError(f"unknown game {name!r}")
