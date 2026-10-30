"""
Given simulation/pricing environment for the American-option Q-learning
assignment.

The primary environment is a CRR recombining binomial tree: Q-learning and
the exact backward-induction benchmark both operate on the SAME discrete
lattice (same u, d, risk-neutral p), so any gap between them is a training
issue, not a discretization mismatch between two different models.

Provides:
  - binomial_tree_params: shared u, d, risk-neutral p, one-step discount
  - node_price: the price at tree node (t, j)
  - binomial_tree_american_put: an independent, exact benchmark price and
    exercise boundary via backward induction on the tree
  - simulate_gbm_paths / PriceGrid: continuous-price GBM simulation and
    discretization tools, not used by the current tree-native assignment,
    kept here for a possible later continuous-price extension.
"""

import numpy as np


def binomial_tree_params(r, sigma, T, n_steps):
    """CRR up/down factors, risk-neutral up-probability, and one-step
    discount factor for a binomial tree with n_steps steps over [0, T].
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    disc = np.exp(-r * dt)
    p = (np.exp(r * dt) - d) / (u - d)
    return u, d, p, disc


def node_price(S0, u, d, t, j):
    """Price at tree node (t, j): j is the number of down-moves so far
    among the first t steps (0 <= j <= t). Accepts scalar or array j.
    """
    return S0 * u ** (t - j) * d**j


def binomial_tree_american_put(S0, K, r, sigma, T, n_steps):
    """Cox-Ross-Rubinstein binomial tree price and exercise boundary for
    an American put. Returns (price, boundary) where boundary[t] is the
    highest stock price at step t at which early exercise is optimal.
    """
    u, d, p, disc = binomial_tree_params(r, sigma, T, n_steps)

    j = np.arange(n_steps + 1)
    S_T = node_price(S0, u, d, n_steps, j)
    values = np.maximum(K - S_T, 0.0)

    boundary = np.full(n_steps + 1, np.nan)

    for t in range(n_steps - 1, -1, -1):
        j = np.arange(t + 1)
        S_t = node_price(S0, u, d, t, j)
        continuation = disc * (p * values[:-1] + (1 - p) * values[1:])
        exercise = np.maximum(K - S_t, 0.0)
        values = np.maximum(continuation, exercise)

        exercised = exercise > continuation
        if exercised.any():
            boundary[t] = S_t[exercised].max()

    return values[0], boundary


def simulate_gbm_paths(S0, sigma, r, T, n_steps, n_paths, seed=None):
    """Simulate n_paths risk-neutral GBM paths, each with n_steps+1 points.

    Returns an (n_paths, n_steps+1) array; column 0 is S0 for every path.
    Not used by the current tree-native assignment; kept for a possible
    later continuous-price extension.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    z = rng.standard_normal((n_paths, n_steps))
    log_increments = drift + vol * z
    log_paths = np.log(S0) + np.hstack(
        [np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)]
    )
    return np.exp(log_paths)


class PriceGrid:
    """Maps a continuous stock price to one of n_buckets discrete states.

    Buckets are evenly spaced in log-price between S_min and S_max, so
    resolution is proportional to price (matches GBM's lognormal spread)
    rather than uniform in dollar terms. Not used by the current
    tree-native assignment; kept for a possible later continuous-price
    extension.
    """

    def __init__(self, S_min, S_max, n_buckets):
        self.edges = np.linspace(np.log(S_min), np.log(S_max), n_buckets + 1)
        self.n_buckets = n_buckets
        self.centers = np.exp((self.edges[:-1] + self.edges[1:]) / 2)

    def to_index(self, S):
        idx = np.digitize(np.log(S), self.edges) - 1
        return np.clip(idx, 0, self.n_buckets - 1)

    def representative_price(self, index):
        return self.centers[index]
