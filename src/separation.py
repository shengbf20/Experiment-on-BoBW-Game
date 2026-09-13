"""Finite-sample-improved separation game and frozen open-loop opponent.

The construction is the nonstationary example in which comparator-local
variation at the unique best fixed comparator is O(1), while realized
last-gradient variation on the D005 trajectory is Theta(T). This module is
independent of G1/G2 experiments.
"""

from __future__ import annotations

import numpy as np

from games import Game, _vec
from learner import ClosedFormPlayer

_DTYPE = np.float64

R_STAR = 3.0 - np.sqrt(10.0)
XI_STAR = float(np.arctanh(R_STAR))
# A translation of the original construction.  Keeping U_STAR nonzero
# preserves the polynomial-order certificate separation, while moving it
# closer to the prescribed x_1=0 sharply reduces the fixed initial transient.
U_STAR = 0.25
V_HALF = 0.5
Y_HALF = float(np.arctanh(V_HALF))


def logcosh(z) -> np.ndarray:
    """Stable log(cosh z)."""
    z = np.asarray(z, dtype=_DTYPE)
    az = np.abs(z)
    return az + np.log1p(np.exp(-2.0 * az)) - np.log(2.0)


def _logcosh_scalar(z: float) -> float:
    return float(logcosh(z))


DELTA0 = -(3.0 * _logcosh_scalar(XI_STAR) + 0.5 * float(np.tanh(XI_STAR)))
# The proof only needs a fixed eta small enough for convex-concavity and for a
# positive-density annulus argument.  ETA=DELTA0 satisfies both and improves
# the squared late-jump constant by 16^2 over the original DELTA0/16 choice.
ETA = DELTA0
E0 = ETA / 2.0
G_UPPER = 4.0 + 2.0 * ETA
G_LOWER = 0.25
X_CONVEXITY_MARGIN = 1.0 - 4.0 * ETA
Y2_CONCAVITY_MARGIN = 1.0 - 2.0 * ETA
ANNULUS_FRACTION_LOWER = 13.0 / 48.0
U_X = np.array([U_STAR], dtype=_DTYPE)


def sech2(z) -> np.ndarray:
    return np.square(1.0 / np.cosh(np.asarray(z, dtype=_DTYPE)))


def q_prime(xi: float) -> float:
    t = float(np.tanh(xi))
    return 2.0 * t * float(1.0 - t * t)


# One-step jump after settling at ±ξ_star, if v2 flips every round.
THEORY_LATE_JUMP = 2.0 * E0 * abs(q_prime(XI_STAR))
THEORY_LATE_DG2 = THEORY_LATE_JUMP * THEORY_LATE_JUMP


def grad_x_scalar(x: float, v1: float, v2: float, eta: float = ETA) -> float:
    p = float(np.tanh(x - U_STAR))
    return 3.0 * p + (1.0 - p * p) * v1 + 2.0 * eta * p * (1.0 - p * p) * v2


def candidate_g(p: float, v1: float, s: int, eta: float = ETA) -> float:
    """Gradient at v2 = s/2."""
    return 3.0 * p + (1.0 - p * p) * v1 + eta * float(s) * p * (1.0 - p * p)


def choose_s(p: float, v1: float, g_prev: float, eta: float = ETA) -> int:
    """Deterministic argmax of |g^{(s)}-g_prev|; ties break to s=+1."""
    g_plus = candidate_g(p, v1, 1, eta)
    g_minus = candidate_g(p, v1, -1, eta)
    d_plus = abs(g_plus - g_prev)
    d_minus = abs(g_minus - g_prev)
    if d_minus > d_plus:
        return -1
    return 1


def v1_of_t(t: int, n: int) -> float:
    return V_HALF if t <= n else -V_HALF


def y_from_v(v1: float, v2: float) -> np.ndarray:
    return np.array(
        [
            Y_HALF if v1 > 0.0 else -Y_HALF,
            Y_HALF if v2 > 0.0 else -Y_HALF,
        ],
        dtype=_DTYPE,
    )


def make_player(cfg: dict) -> ClosedFormPlayer:
    return ClosedFormPlayer(
        1,
        epsilon=float(cfg["epsilon"]),
        beta0=float(cfg["beta0"]),
        ell1=float(cfg["ell1"]),
        adaptive=True,
    )


class SeparationGame(Game):
    """Scalar minimizer, 2-d maximizer; unique best comparator is U_STAR."""

    name = "Gsep"

    def __init__(self, eta: float = ETA):
        self.dim_x = 1
        self.dim_y = 2
        self.eta = float(eta)

    def comparator_x(self) -> np.ndarray:
        return U_X.copy()

    def phi(self, x, y) -> float:
        x = float(_vec(x, 1)[0])
        y = _vec(y, 2)
        xi = x - U_STAR
        p = float(np.tanh(xi))
        return float(
            3.0 * _logcosh_scalar(xi)
            + p * float(np.tanh(y[0]))
            + self.eta * (p * p) * float(np.tanh(y[1]))
            - 3.0 * _logcosh_scalar(float(y[0]))
            - _logcosh_scalar(float(y[1]))
        )

    def grad_x_phi(self, x, y) -> np.ndarray:
        x = float(_vec(x, 1)[0])
        y = _vec(y, 2)
        g = grad_x_scalar(
            x,
            float(np.tanh(y[0])),
            float(np.tanh(y[1])),
            self.eta,
        )
        return np.array([g], dtype=_DTYPE)

    def grad_y_phi(self, x, y) -> np.ndarray:
        x = float(_vec(x, 1)[0])
        y = _vec(y, 2)
        p = float(np.tanh(x - U_STAR))
        s1 = float(sech2(y[0]))
        s2 = float(sech2(y[1]))
        d1 = p * s1 - 3.0 * float(np.tanh(y[0]))
        d2 = self.eta * (p * p) * s2 - float(np.tanh(y[1]))
        return np.array([d1, d2], dtype=_DTYPE)

    def restricted_gap(self, x, y, radius: float) -> float:
        # Unused by the separation experiment; kept only for the Game API.
        return 0.0


def generate_opponent_sequence(T: int, cfg: dict, game: SeparationGame | None = None):
    """Offline D005 simulation that emits a unique open-loop y_{1:T}."""
    T = int(T)
    if T < 2 or T % 2 != 0:
        raise ValueError(f"separation example requires even T>=2, got {T}")
    if game is None:
        game = SeparationGame()
    n = T // 2
    player = make_player(cfg)
    if np.any(player.action):
        raise AssertionError("must not overwrite w1=0")
    y_seq = np.zeros((T, 2), dtype=_DTYPE)
    x_seq = np.zeros(T, dtype=_DTYPE)
    g_seq = np.zeros(T, dtype=_DTYPE)
    g_prev = None
    for t in range(1, T + 1):
        x = float(player.action[0])
        v1 = v1_of_t(t, n)
        p = float(np.tanh(x - U_STAR))
        if t == 1:
            s = 1
        else:
            s = choose_s(p, v1, float(g_prev), game.eta)
        y = y_from_v(v1, 0.5 * s)
        gx, _gy = game.feedback(np.array([x], dtype=_DTYPE), y)
        g = float(gx[0])
        z = np.concatenate([np.array([x], dtype=_DTYPE), y])
        player.observe(gx, z)
        y_seq[t - 1] = y
        x_seq[t - 1] = x
        g_seq[t - 1] = g
        g_prev = g
        if t >= 2:
            jump = abs(g - g_seq[t - 2])
            floor = E0 * abs(q_prime(x - U_STAR))
            if jump + 1e-12 < floor:
                raise AssertionError(
                    f"t={t}: |g_t-g_{{t-1}}|={jump} < e0|q'|={floor}"
                )
    player.assert_invariants()
    return {
        "y": y_seq,
        "x": x_seq,
        "g": g_seq,
        "player": player,
        "n": n,
        "game": game,
    }


def replay_frozen_sequence(y_seq: np.ndarray, cfg: dict, game: SeparationGame | None = None):
    """Replay D005 from scratch against a frozen open-loop opponent."""
    y_seq = np.asarray(y_seq, dtype=_DTYPE)
    T = int(y_seq.shape[0])
    if y_seq.shape != (T, 2):
        raise ValueError(f"y_seq must be (T, 2), got {y_seq.shape}")
    if game is None:
        game = SeparationGame()
    player = make_player(cfg)
    if np.any(player.action):
        raise AssertionError("must not overwrite w1=0")
    x_seq = np.zeros(T, dtype=_DTYPE)
    g_seq = np.zeros(T, dtype=_DTYPE)
    max_w = 0.0
    for t in range(1, T + 1):
        x = float(player.action[0])
        y = y_seq[t - 1].copy()
        gx, _gy = game.feedback(np.array([x], dtype=_DTYPE), y)
        g = float(gx[0])
        z = np.concatenate([np.array([x], dtype=_DTYPE), y])
        player.observe(gx, z)
        x_seq[t - 1] = x
        g_seq[t - 1] = g
        max_w = max(max_w, abs(x), float(np.linalg.norm(y)))
    player.assert_invariants()
    return {
        "x": x_seq,
        "g": g_seq,
        "player": player,
        "max_w": max_w,
        "game": game,
    }


def comparator_local_variation(game: SeparationGame, y_seq: np.ndarray, u: float = U_STAR) -> float:
    y_seq = np.asarray(y_seq, dtype=_DTYPE)
    u_vec = np.array([u], dtype=_DTYPE)
    total = 0.0
    g_prev = None
    for y in y_seq:
        g = float(game.grad_x_phi(u_vec, y)[0])
        if g_prev is not None:
            total += (g - g_prev) ** 2
        g_prev = g
    return float(total)


def realized_variation(g_seq: np.ndarray) -> float:
    g = np.asarray(g_seq, dtype=_DTYPE).reshape(-1)
    if g.size == 0:
        return 0.0
    return float(g[0] * g[0] + np.sum(np.diff(g) ** 2))


def late_jump_stats(g_seq: np.ndarray, frac: float = 0.25) -> dict:
    """Mean squared last-gradient jump on the final ``frac`` of rounds."""
    g = np.asarray(g_seq, dtype=_DTYPE).reshape(-1)
    T = int(g.size)
    if T < 8:
        raise ValueError("need a longer horizon for late-window statistics")
    start = max(2, int((1.0 - frac) * T) + 1)
    dg2 = np.diff(g) ** 2
    late = dg2[start - 2 :]
    return {
        "late_start": int(start),
        "late_mean_dg2": float(np.mean(late)),
        "late_sum_dg2": float(np.sum(late)),
        "theory_late_dg2": float(THEORY_LATE_DG2),
    }


def regret_against_u(game: SeparationGame, x_seq: np.ndarray, y_seq: np.ndarray, u: float = U_STAR) -> float:
    x_seq = np.asarray(x_seq, dtype=_DTYPE).reshape(-1)
    y_seq = np.asarray(y_seq, dtype=_DTYPE)
    u_vec = np.array([u], dtype=_DTYPE)
    total = 0.0
    for x, y in zip(x_seq, y_seq):
        xv = np.array([x], dtype=_DTYPE)
        total += game.phi(xv, y) - game.phi(u_vec, y)
    return float(total)


def prefix_curves(game: SeparationGame, x_seq: np.ndarray, g_seq: np.ndarray, y_seq: np.ndarray):
    """Running V_t(u*), E_t, G_t, R_t along the replayed trajectory."""
    T = int(np.asarray(g_seq).reshape(-1).size)
    V = np.zeros(T, dtype=_DTYPE)
    E = np.zeros(T, dtype=_DTYPE)
    G = np.zeros(T, dtype=_DTYPE)
    R = np.zeros(T, dtype=_DTYPE)
    u_vec = U_X
    g_u_prev = None
    g_prev = None
    G_run = 0.0
    E_run = 0.0
    V_run = 0.0
    R_run = 0.0
    x_seq = np.asarray(x_seq, dtype=_DTYPE).reshape(-1)
    g_seq = np.asarray(g_seq, dtype=_DTYPE).reshape(-1)
    y_seq = np.asarray(y_seq, dtype=_DTYPE)
    for t in range(T):
        g = float(g_seq[t])
        y = y_seq[t]
        x = float(x_seq[t])
        g_u = float(game.grad_x_phi(u_vec, y)[0])
        if t == 0:
            E_run = g * g
        else:
            E_run += (g - g_prev) ** 2
            V_run += (g_u - g_u_prev) ** 2
        G_run = max(G_run, abs(g))
        R_run += game.phi(np.array([x], dtype=_DTYPE), y) - game.phi(u_vec, y)
        V[t] = V_run
        E[t] = E_run
        G[t] = G_run
        R[t] = R_run
        g_prev = g
        g_u_prev = g_u
    return {"V": V, "E": E, "G": G, "R": R}


def run_part_a(T: int, cfg: dict) -> dict:
    """Generate a frozen opponent, replay D005, and compute Part A scalars."""
    generated = generate_opponent_sequence(T, cfg)
    y = generated["y"].copy()
    replayed = replay_frozen_sequence(y, cfg, game=generated["game"])
    if not np.allclose(generated["x"], replayed["x"], rtol=0.0, atol=1e-12):
        raise AssertionError("replay x_t differs from the generating D005 trajectory")
    if not np.allclose(generated["g"], replayed["g"], rtol=0.0, atol=1e-12):
        raise AssertionError("replay g_t differs from the generating D005 trajectory")
    game = generated["game"]
    V_T = comparator_local_variation(game, y)
    E_T = realized_variation(replayed["g"])
    G_T = float(np.max(np.abs(replayed["g"])))
    R_T = regret_against_u(game, replayed["x"], y)
    curves = prefix_curves(game, replayed["x"], replayed["g"], y)
    late = late_jump_stats(replayed["g"])
    player = replayed["player"]
    return {
        "T": int(T),
        "y": y,
        "x": replayed["x"],
        "g": replayed["g"],
        "V_T": V_T,
        "E_T": E_T,
        "G_T": G_T,
        "R_T": R_T,
        "E_over_T": E_T / float(T),
        "max_w": replayed["max_w"],
        "J": int(player.J),
        "ell": float(player.ell),
        "beta": float(player.beta),
        "curves": curves,
        "late": late,
        "replay_match": True,
        "g1": float(replayed["g"][0]),
        "x1": float(replayed["x"][0]),
        "x_T": float(replayed["x"][-1]),
        "x_half": float(replayed["x"][int(T) // 2 - 1]),
    }
