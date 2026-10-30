"""
Chart-drawing helpers for the lecture notebook. Pulled out of the notebook
cells on purpose: this code's job is to arrange dots and lines on a plot,
not to teach a concept, so it would only distract from the cells that are
actually about the tree/Q-learning mechanics. Nothing here is used by the
assignment notebook.
"""

import matplotlib.pyplot as plt

from market_sim import node_price


def draw_tree_lattice(ax, S0, u, d, n_steps, color="lightgray"):
    """Draw every node and up/down edge of a recombining binomial tree
    onto `ax`, priced via `node_price`. Purely the lattice -- callers add
    anything else (sample walks, highlighted paths, titles) on top."""
    for t in range(n_steps + 1):
        j = range(t + 1)
        prices = [node_price(S0, u, d, t, jj) for jj in j]
        ax.scatter([t] * len(prices), prices, color=color, zorder=1)
        if t < n_steps:
            for jj in j:
                ax.plot([t, t + 1],
                        [node_price(S0, u, d, t, jj), node_price(S0, u, d, t + 1, jj)],
                        color=color, lw=0.5, zorder=1)
                ax.plot([t, t + 1],
                        [node_price(S0, u, d, t, jj), node_price(S0, u, d, t + 1, jj + 1)],
                        color=color, lw=0.5, zorder=1)


def _naive_levels(n):
    """y positions of the nodes of the naive tree at each step (children of node i at step t are 2i and 2i+1 at t+1), root at y=0."""
    levels = {n: list(range(2 ** n))}
    for t in range(n - 1, -1, -1):
        c = levels[t + 1]
        levels[t] = [(c[2 * i] + c[2 * i + 1]) / 2 for i in range(2 ** t)]
    root = levels[0][0]
    return {t: [y - root for y in ys] for t, ys in levels.items()}


def _draw_naive(ax, n):
    levels = _naive_levels(n)
    for t in range(n):
        for i, y in enumerate(levels[t]):
            ax.plot([t, t + 1], [y, levels[t + 1][2 * i]], color="lightgray", lw=0.8, zorder=1)
            ax.plot([t, t + 1], [y, levels[t + 1][2 * i + 1]], color="lightgray", lw=0.8, zorder=1)
    for t in range(n + 1):
        ax.scatter([t] * len(levels[t]), levels[t], color="steelblue", s=25, zorder=2)
    lo = min(levels[n]) - 0.9
    for t in range(n + 1):
        ax.text(t, lo, f"{2 ** t}", ha="center", va="top", fontsize=11, color="firebrick")
    ax.text(n / 2, lo - 1.1, "number of nodes at step $t$", ha="center", va="top", fontsize=10, color="firebrick")
    ax.set_xlabel("time step $t$")
    ax.set_yticks([])
    ax.set_xlim(-0.3, n + 0.3)
    ax.set_ylim(lo - 2.3, max(levels[n]) + 0.6)


def _y_rec(t, j):
    return t / 2 - j


def _draw_recombining(ax, n, highlight=True):
    for t in range(n):
        for j in range(t + 1):
            y = _y_rec(t, j)
            ax.plot([t, t + 1], [y, _y_rec(t + 1, j)], color="lightgray", lw=0.8, zorder=1)      # up
            ax.plot([t, t + 1], [y, _y_rec(t + 1, j + 1)], color="lightgray", lw=0.8, zorder=1)  # down
    for t in range(n + 1):
        ys = [_y_rec(t, j) for j in range(t + 1)]
        ax.scatter([t] * len(ys), ys, color="steelblue", s=25, zorder=2)
    if highlight:
        # two different histories that recombine into the SAME node: up-then-down and down-then-up both reach (t=2, j=1)
        for path, color, label in [([(0, 0), (1, 0), (2, 1)], "darkorange", "up, then down"),
                                   ([(0, 0), (1, 1), (2, 1)], "seagreen", "down, then up")]:
            ax.plot([t for t, j in path], [_y_rec(t, j) for t, j in path], color=color, lw=2.5, marker="o", ms=6, zorder=3, label=label)
        ax.annotate("same node!", xy=(2, _y_rec(2, 1)), xytext=(2.4, _y_rec(2, 1) + 0.6), arrowprops=dict(arrowstyle="->"), fontsize=10)
        ax.legend(loc="upper left", fontsize=9)
    for j in range(n + 1):      # j is just "which of the t+1 nodes you are at"
        ax.text(n + 0.15, _y_rec(n, j), f"$j={j}$", va="center", fontsize=9)
    lo = _y_rec(n, n) - 0.9
    for t in range(n + 1):
        ax.text(t, lo, f"{t + 1}", ha="center", va="top", fontsize=11, color="firebrick")
    ax.text(n / 2, lo - 0.6, "number of nodes at step $t$", ha="center", va="top", fontsize=10, color="firebrick")
    ax.set_xlabel("time step $t$")
    ax.set_yticks([])
    ax.set_xlim(-0.3, n + 0.9)
    ax.set_ylim(lo - 1.6, n / 2 + 0.8)


def plot_naive_tree(n=4):
    """The NAIVE tree: every sequence of up/down moves is its own node, so step t has 2**t nodes. Purely structural."""
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    _draw_naive(ax, n)
    ax.set_title(f"every sequence of moves is its own node: {2 ** n} leaves at $t={n}$")
    plt.tight_layout()
    plt.show()


def plot_recombining_tree(n=4):
    """The RECOMBINING tree: only the count j of down-moves matters, so step t has t+1 nodes; two different histories reach one node."""
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    _draw_recombining(ax, n)
    ax.set_title(f"only the number of down-moves $j$ matters: {n + 1} nodes at $t={n}$")
    plt.tight_layout()
    plt.show()


def plot_q_table(n=4):
    """The table over the recombining tree: one row per node (t, j) with j <= t, holding two numbers (continue, exercise); j > t is never used."""
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    for t in range(n + 1):
        for j in range(n + 1):
            x, y = t, n - j
            if j <= t:
                ax.add_patch(Rectangle((x + 0.04, y + 0.06), 0.46, 0.88, fc="#bee3f8", ec="steelblue", lw=1))
                ax.add_patch(Rectangle((x + 0.50, y + 0.06), 0.46, 0.88, fc="#fbd38d", ec="darkorange", lw=1))
                ax.text(x + 0.27, y + 0.5, "C", ha="center", va="center", fontsize=9)
                ax.text(x + 0.73, y + 0.5, "E", ha="center", va="center", fontsize=9)
            else:
                ax.add_patch(Rectangle((x + 0.04, y + 0.06), 0.92, 0.88, fc="#f7fafc", ec="#e2e8f0", lw=1, hatch="///"))
    for t in range(n + 1):
        ax.text(t + 0.5, n + 1.15, f"$t={t}$", ha="center", va="bottom", fontsize=9)
    for j in range(n + 1):
        ax.text(-0.12, n - j + 0.5, f"$j={j}$", ha="right", va="center", fontsize=9)
    used = (n + 1) * (n + 2) // 2
    ax.set_xlim(-0.9, n + 1.05)
    ax.set_ylim(-0.9, n + 1.7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text((n + 1) / 2, -0.35, f"C = value of continue, E = value of exercise; {used} rows used (hatched: $j>t$, never visited)", ha="center", va="top", fontsize=9)
    ax.set_title("the table: one row per node $(t, j)$, two entries each")
    plt.tight_layout()
    plt.show()


def plot_naive_vs_recombining_tree(n=4):
    """Side-by-side diagram (kept for older decks): naive tree next to the recombining tree."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    _draw_naive(ax1, n)
    ax1.set_title("No recombining:\n" + r"$2^t$" + f" nodes at step $t$ -- {2**n} leaves at $t={n}$")
    _draw_recombining(ax2, n)
    ax2.set_title(f"Recombining (CRR) tree:\n$t+1$ nodes at step $t$ -- {n + 1} nodes at $t={n}$")
    plt.tight_layout()
    plt.show()


def plot_lattice_with_walks(walks, S0=60, sigma=0.35, r=0.02, T=0.5, n_steps=8):
    """The full recombining lattice (synthetic illustrative parameters, NOT the assignment's option) with example walks overlaid.
    Each walk is (start time, start j, moves); a move is U (price up, j unchanged) or D (price down, j + 1). No Q-learning happens here."""
    import numpy as np
    from market_sim import binomial_tree_params
    u, d, p, disc = binomial_tree_params(r, sigma, T, n_steps)
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    draw_tree_lattice(ax, S0, u, d, n_steps)
    for t0, j, moves in walks:
        ts, prices = [t0], [node_price(S0, u, d, t0, j)]
        for k, m in enumerate(moves):
            t = t0 + k
            if m == "D":
                j += 1
            ts.append(t + 1)
            prices.append(node_price(S0, u, d, t + 1, j))
        ax.plot(ts, prices, marker="o", ms=4, zorder=2)
    ax.set_xlabel("time step")
    ax.set_ylabel("price")
    ax.set_title("The tree lattice, with a few exploring-start walks")
    plt.tight_layout()
    plt.show()
