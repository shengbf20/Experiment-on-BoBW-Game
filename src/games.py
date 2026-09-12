"""G1 bilinear, G2 quadratic, G3 separation example.

Feedback matches eq:feedback: gx = ∇x Φ, gy = −∇y Φ.
Constructors default to saddle (0,0) for health checks. Paper runs pass
paper_saddle(dim) so w1=0 is not a self-play rest point.
"""

from __future__ import annotations

import numpy as np

_DTYPE = np.float64
PAPER_SADDLE_SCALE = 0.4


def _vec(z, dim: int) -> np.ndarray:
    out = np.asarray(z, dtype=_DTYPE).reshape(-1)
    if out.size != dim:
        raise ValueError(f"expected dim {dim}, got {out.size}")
    return out


def paper_saddle(dim: int):
    """Paper instance: a = 0.4 e1, b = 0.4 e2."""
    dim = int(dim)
    if dim < 2:
        raise ValueError("paper saddle uses e2; need dim >= 2")
    sx = np.zeros(dim, dtype=_DTYPE)
    sy = np.zeros(dim, dtype=_DTYPE)
    sx[0] = PAPER_SADDLE_SCALE
    sy[1] = PAPER_SADDLE_SCALE
    return sx, sy


def _offsets(dim: int, saddle_x, saddle_y):
    sx = np.zeros(dim, dtype=_DTYPE) if saddle_x is None else _vec(saddle_x, dim)
    sy = np.zeros(dim, dtype=_DTYPE) if saddle_y is None else _vec(saddle_y, dim)
    return sx, sy


def _spectral_normalize(A: np.ndarray) -> np.ndarray:
    op = float(np.linalg.norm(A, 2))
    if op > 0.0:
        return A / op
    return A


def assert_saddle_comparator(game: "Game", metrics) -> None:
    ax, ay = game.saddle()
    if not np.allclose(metrics.u_x, ax) or not np.allclose(metrics.u_y, ay):
        raise AssertionError("G1/G2 comparator must be game.saddle()")


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
    """Φ = (x−a)⊤ A (y−b). Gap on the R-ball centered at the saddle."""

    name = "G1"

    def __init__(self, dim: int = 10, A: np.ndarray | None = None, saddle_x=None, saddle_y=None):
        self.dim_x = int(dim)
        self.dim_y = int(dim)
        if A is None:
            self.A = np.eye(dim, dtype=_DTYPE)
        else:
            self.A = np.asarray(A, dtype=_DTYPE)
            if self.A.shape != (dim, dim):
                raise ValueError("A must be (dim, dim)")
        self.sx, self.sy = _offsets(dim, saddle_x, saddle_y)

    @classmethod
    def gaussian(
        cls,
        dim: int = 10,
        seed: int = 0,
        normalize: bool = True,
        saddle_x=None,
        saddle_y=None,
    ) -> "BilinearGame":
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(dim, dim)).astype(_DTYPE)
        if normalize:
            A = _spectral_normalize(A)
        return cls(dim=dim, A=A, saddle_x=saddle_x, saddle_y=saddle_y)

    def saddle(self):
        return self.sx.copy(), self.sy.copy()

    def phi(self, x, y) -> float:
        x = _vec(x, self.dim_x) - self.sx
        y = _vec(y, self.dim_y) - self.sy
        return float(x @ (self.A @ y))

    def grad_x_phi(self, x, y) -> np.ndarray:
        y = _vec(y, self.dim_y) - self.sy
        return self.A @ y

    def grad_y_phi(self, x, y) -> np.ndarray:
        x = _vec(x, self.dim_x) - self.sx
        return self.A.T @ x

    def restricted_gap(self, x, y, radius: float) -> float:
        x = _vec(x, self.dim_x) - self.sx
        y = _vec(y, self.dim_y) - self.sy
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
    """Φ = (μ/2)||x−a||² + (x−a)⊤ A (y−b) − (μ/2)||y−b||²."""

    name = "G2"

    def __init__(
        self,
        dim: int = 10,
        mu: float = 0.2,
        A: np.ndarray | None = None,
        saddle_x=None,
        saddle_y=None,
    ):
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
        self.sx, self.sy = _offsets(dim, saddle_x, saddle_y)

    @classmethod
    def gaussian(
        cls,
        dim: int = 10,
        mu: float = 0.2,
        seed: int = 0,
        normalize: bool = True,
        saddle_x=None,
        saddle_y=None,
    ) -> "QuadraticGame":
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(dim, dim)).astype(_DTYPE)
        if normalize:
            A = _spectral_normalize(A)
        return cls(dim=dim, mu=mu, A=A, saddle_x=saddle_x, saddle_y=saddle_y)

    def saddle(self):
        return self.sx.copy(), self.sy.copy()

    def induced_minimizer_x(self, y) -> np.ndarray:
        """argmin_x Φ(x, y) = a − A(y−b)/μ. Vs-const comparator on G2."""
        y = _vec(y, self.dim_y) - self.sy
        return self.sx - (self.A @ y) / self.mu

    def phi(self, x, y) -> float:
        x = _vec(x, self.dim_x) - self.sx
        y = _vec(y, self.dim_y) - self.sy
        return float(0.5 * self.mu * (x @ x) + x @ (self.A @ y) - 0.5 * self.mu * (y @ y))

    def grad_x_phi(self, x, y) -> np.ndarray:
        x = _vec(x, self.dim_x) - self.sx
        y = _vec(y, self.dim_y) - self.sy
        return self.mu * x + self.A @ y

    def grad_y_phi(self, x, y) -> np.ndarray:
        x = _vec(x, self.dim_x) - self.sx
        y = _vec(y, self.dim_y) - self.sy
        return self.A.T @ x - self.mu * y

    def restricted_gap(self, x, y, radius: float) -> float:
        x = _vec(x, self.dim_x) - self.sx
        y = _vec(y, self.dim_y) - self.sy
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
