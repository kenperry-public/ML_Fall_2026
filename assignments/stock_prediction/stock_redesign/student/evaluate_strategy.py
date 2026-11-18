"""evaluate_strategy: GIVEN with the assignment. Do not edit or rewrite it; graders compare its printed output across students.

Convention (read this once):
  * `positions` is a DataFrame: index = date, one column per ticker, values in [-1, 1] = fraction of equity
    (-1 fully short, 0 flat, +1 fully long). Row d is your decision made at the CLOSE of day d, using data up to and
    including day d. It is earned on day d+1's return. The function applies this one-day lag for you, so a signal
    cannot trade on the close it was computed from.
  * No leverage: each value is clipped to [-1, 1], and total gross exposure (sum of |positions|) is scaled down to 1.
    The number of affected days is printed.
  * Trading cost: 5 basis points (0.05%) of the traded fraction of equity, charged whenever a position changes.
  * Day return of a ticker = (Close * Factor + Div) / previous Close - 1  (raw Close, so splits are handled by Factor).
  * Missing price (before a late listing) means no return that day; the position there is treated as 0.
  * Trading stops if equity reaches zero; results after that date do not count.
`prices`: dict ticker -> DataFrame with the course columns (Dt, Close, Factor, Div, ...), e.g. the `sample` dict.
"""
import numpy as np
import pandas as pd

COST_BP = 5.0
N_NULL = 500
__all__ = ["evaluate_strategy", "COST_BP"]


def _returns(prices):
    out = {}
    for t, d in prices.items():
        d = d.set_index("Dt").sort_index()
        out[t] = (d["Close"] * d["Factor"] + d["Div"]) / d["Close"].shift(1) - 1
    return pd.DataFrame(out).sort_index()


def _held(pos, rets, extra_lag=0):
    """Positions held on each day of `rets` (lagged), after clipping and the exposure cap. Returns (held, n_clipped_days)."""
    pos = pos.reindex(rets.index).fillna(0.0)
    clipped = pos.clip(-1, 1)
    gross = clipped.abs().sum(axis=1)
    scale = np.where(gross > 1, 1 / gross.where(gross > 0, 1), 1.0)
    capped = clipped.mul(scale, axis=0)
    n_clipped = int(((pos != clipped).any(axis=1) | (gross > 1)).sum())
    held = capped.shift(1 + extra_lag).fillna(0.0)
    return held.where(rets.notna(), 0.0), n_clipped


def _run(held, rets, cost_bp):
    r = rets.fillna(0.0)
    turnover = held.diff().abs().sum(axis=1)
    turnover.iloc[0] = held.iloc[0].abs().sum()
    net = (held * r).sum(axis=1) - turnover * cost_bp / 1e4
    eq = (1 + net).cumprod()
    ruin = eq[eq <= 0]
    if len(ruin):
        d0 = ruin.index[0]
        net = net.where(net.index <= d0, 0.0)
        eq = (1 + net).cumprod().clip(lower=0)
        return net, eq, turnover, d0
    return net, eq, turnover, None


def _trades_by_year(held):
    """A completed trade = a stretch in one direction in one ticker that ends (flat or flip). Counted in its end year."""
    counts = {}
    for t in held.columns:
        s = np.sign(held[t].values)
        start = 0
        for i in range(1, len(s) + 1):
            if i == len(s) or s[i] != s[i - 1]:
                if s[i - 1] != 0 and i < len(s):
                    y = held.index[i - 1].year
                    counts[y] = counts.get(y, 0) + 1
                start = i
    return dict(sorted(counts.items()))


def evaluate_strategy(positions, prices, verbose=True, extra_lag=0, seed=0):
    """Evaluate a position series; print the standard report; return the numbers as a dict.
    (`extra_lag` is used only by the Submission Summary's lag test; leave it at 0.)"""
    if isinstance(positions, pd.Series):
        positions = positions.to_frame(positions.name or "AAPL")
    positions = positions.copy()
    positions.index = pd.to_datetime(positions.index)
    rets = _returns(prices)
    unknown = [c for c in positions.columns if c not in rets.columns]
    if unknown:
        raise ValueError(f"positions has unknown tickers {unknown}")
    first, last = positions.index.min(), positions.index.max()
    cal = rets.index
    after = cal[cal > last]
    stop = after[0] if len(after) else last
    window = cal[(cal > first) & (cal <= stop)]
    rets = rets[list(positions.columns)]
    sub = cal[(cal >= first) & (cal <= stop)]
    held, n_clipped = _held(positions, rets.loc[sub], extra_lag)
    held, rr = held.loc[window], rets.loc[window]
    net, eq, turnover, ruin = _run(held, rr, COST_BP)

    n_act = rr.notna().sum(axis=1).replace(0, np.nan)
    bench = (rr.fillna(0).sum(axis=1) / n_act).fillna(0.0)
    beq = (1 + bench).cumprod()
    years = max(len(window) / 252, 1e-9)
    dd = (eq / eq.cummax() - 1).min()
    sd = net.std()
    res = dict(
        cumulative_return=float(eq.iloc[-1] - 1), buy_hold_return=float(beq.iloc[-1] - 1),
        max_drawdown=float(dd), lowest_equity=float(eq.min()),
        sharpe=float(net.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
        time_in_market=float((held.abs().sum(axis=1) > 0).mean()),
        turnover_per_year=float(turnover.sum() / years),
        trades_by_year=_trades_by_year(held), clipped_days=n_clipped,
        ruin_date=None if ruin is None else str(ruin.date()), days=int(len(window)),
        first_day=str(window[0].date()), last_day=str(window[-1].date()),
    )
    res["n_trades"] = int(sum(res["trades_by_year"].values()))

    # randomized-timing null: same trades, durations and exposure, circularly shifted by a random number of days
    rng = np.random.default_rng(seed)
    null = []
    h = held.values
    for _ in range(N_NULL):
        k = int(rng.integers(1, max(len(h), 2)))
        hs = pd.DataFrame(np.roll(h, k, axis=0), index=held.index, columns=held.columns)
        null.append(float(_run(hs, rr, COST_BP)[1].iloc[-1] - 1))
    null = np.array(null)
    res["null_median_return"] = float(np.median(null))
    res["pct_null_beaten"] = float(((null < res["cumulative_return"]).sum() + 0.5 * (null == res["cumulative_return"]).sum()) / len(null))

    if verbose:
        pc = lambda x: f"{100 * x:.1f}%"
        print(f"evaluate_strategy: {res['first_day']} to {res['last_day']} ({res['days']} days), tickers {list(held.columns)}")
        print(f"  cost {COST_BP:g} bp per unit traded; signal lagged one day" + (f" plus {extra_lag} extra" if extra_lag else ""))
        print(f"  cumulative return          : {pc(res['cumulative_return'])}   (buy-and-hold, equal split: {pc(res['buy_hold_return'])})")
        print(f"  max drawdown               : {pc(res['max_drawdown'])}")
        print(f"  lowest equity (start = 1.0): {res['lowest_equity']:.3f}")
        print(f"  Sharpe (daily, annualized) : {res['sharpe']:.2f}")
        print(f"  time in market             : {pc(res['time_in_market'])}   turnover per year: {res['turnover_per_year']:.1f}")
        print(f"  completed trades by year   : {res['trades_by_year']}   total {res['n_trades']}")
        print(f"  random-timing comparison   : median {pc(res['null_median_return'])}; yours beats {pc(res['pct_null_beaten'])} of {N_NULL} random timings")
        if res["n_trades"] < 10:
            print("  NOTE: very few completed trades; with this few events, say honestly that you cannot tell luck from skill.")
        if n_clipped:
            print(f"  WARNING: {n_clipped} days had positions outside [-1, 1] or gross exposure above 1; they were clipped (no leverage).")
        if ruin is not None:
            print(f"  RUIN: equity reached zero on {res['ruin_date']}; later days do not count.")
    return res
