"""print_summary: GIVEN with the assignment. Do not edit it. Run it last, in the notebook's final cell.

It reads the `SUMMARY` dict you filled in and prints fixed-format lines, each starting with `SUMMARY|<CODE>|`.
Graders read these lines instead of searching your notebook. Nothing here grades you by itself; it prints facts and flags.
"""
import hashlib, importlib, inspect, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from evaluate_strategy import evaluate_strategy

ALLOWED_LLMS = {f"HuggingFaceTB/SmolLM2-{s}{v}" for s in ("135M", "360M") for v in ("", "-Instruct")}
LLM_PARAM_CAP = 400e6
N_MODEL_DAYS = 20
REPORT_KEYS = ("key_line", "what_changed", "what_tried", "what_failed")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def given_hashes(base="."):
    base = Path(base)
    data = hashlib.sha256()
    for p in sorted((base / "Data").rglob("*.csv")):
        data.update(str(p.relative_to(base)).encode() + _sha(p).encode())
    return {"evaluate_strategy.py": _sha(base / "evaluate_strategy.py"),
            "submission_summary.py": _sha(base / "submission_summary.py"), "Data": data.hexdigest()[:16]}


def line(code, *parts):
    print("SUMMARY|" + code + "|" + " | ".join(str(p) for p in parts))


def _guard(code):
    def deco(fn):
        def run(*a, **k):
            try:
                fn(*a, **k)
            except Exception as e:  # a broken section must not hide the others
                line(code, "ERROR", f"{type(e).__name__}: {e}")
        return run
    return deco


def _block_bootstrap_ci(x, block=20, n=1000, seed=0):
    x = np.asarray(x, float); m = len(x); block = min(block, m)
    rng = np.random.default_rng(seed); nb = int(np.ceil(m / block)); means = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, m - block + 1, nb)
        means[i] = np.concatenate([x[s:s + block] for s in starts])[:m].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


@_guard("CFG")
def _cfg(S, given):
    vers = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    for mod in ("torch", "sklearn", "scipy"):
        try:
            vers[mod] = importlib.import_module(mod).__version__
        except Exception:
            vers[mod] = "absent"
    line("CFG", "versions", vers)
    line("CFG", "seeds", S.get("seeds"))
    now = given_hashes(".")
    for k, v in now.items():
        ok = "unknown" if not given else ("yes" if given.get(k) == v else "NO")
        line("GIVEN", k, v, "unchanged:", ok)


@_guard("OBJ")
def _obj(S):
    o = S.get("objective") or {}
    kind = o.get("kind")
    line("OBJ", "kind", kind)
    if kind not in (None, "predict_next_return"):
        missing = [k for k in ("old", "new", "why", "metric", "baseline", "metric_test") if not o.get(k)]
        line("OBJ", "changed objective; fields", {k: o.get(k) for k in ("old", "new", "why", "metric", "baseline", "metric_test")})
        line("OBJ", "missing fields", missing or "none")
        _metric_test(o.get("metric_test"))


def _metric_test(fn):
    """Run the student's automatic metric test; it returns a dict, we print and flag, never grade."""
    if not callable(fn):
        line("OBJ", "metric test | run: NO (flag: metric_test must be a function you wrote, not text)")
        return
    t0 = time.time()
    try:
        r = fn()
        m, b, n = float(r["model"]), float(r["baseline"]), int(r["n"])
        hib = bool(r["higher_is_better"])
    except Exception as e:
        line("OBJ", "metric test | run: ERROR", f"{type(e).__name__}: {e}",
             "| expected a dict with model, baseline, n, higher_is_better (and ci_low, ci_high of model minus baseline)")
        return
    better = (m > b) if hib else (m < b)
    line("OBJ", "metric test | run: ok | seconds", f"{time.time() - t0:.2f}", "| model", f"{m:.6g}", "| baseline", f"{b:.6g}", "| n", n,
         "| higher is better:", "yes" if hib else "no", "| model better:", "yes" if better else "no")
    lo, hi = r.get("ci_low"), r.get("ci_high")
    if lo is None or hi is None:
        line("OBJ", "metric test interval | none given (flag: the new metric has no uncertainty estimate)")
    else:
        lo, hi = float(lo), float(hi)
        line("OBJ", "metric test interval (model minus baseline)", f"[{lo:.6g}, {hi:.6g}]",
             "| distinguishable:", "yes" if (lo > 0 or hi < 0) else "no")


@_guard("FMT")
def _fmt(S):
    f = S.get("example_format") or {}
    ex = np.asarray(S["example"])
    axes, shape, feats = tuple(f.get("axes", ())), tuple(f.get("shape", ())), list(f.get("features", []))
    line("FMT", "declared axes", axes, "declared shape", shape, "n features", len(feats))
    line("FMT", "actual shape of one example", tuple(ex.shape))
    line("FMT", "shape matches declaration:", "yes" if tuple(ex.shape) == shape else "NO")
    line("FMT", "axes count matches ndim:", "yes" if len(axes) == ex.ndim else "NO")
    if "feature" in axes and len(axes) == ex.ndim:
        n = ex.shape[axes.index("feature")]
        line("FMT", "feature axis length equals number of named features:", "yes" if n == len(feats) else f"NO ({n} vs {len(feats)})")
    line("FMT", "has a time axis:", "yes" if "time" in axes else "NO (flag: how is order kept?)")
    line("FMT", "target", f.get("target"))


@_guard("SPLIT")
def _split(S, train, sample):
    sp = S["splits"]
    cal = sorted(set(pd.concat([d["Dt"] for d in list(train.values()) + list(sample.values())])))
    cal = pd.DatetimeIndex(cal)
    t = {k: tuple(pd.Timestamp(x) for x in v) for k, v in sp.items()}
    for k in ("train", "val", "test"):
        line("SPLIT", k, str(t[k][0].date()), "to", str(t[k][1].date()))
    ordered = t["train"][0] <= t["train"][1] < t["val"][0] <= t["val"][1] < t["test"][0] <= t["test"][1]
    line("SPLIT", "chronological and non-overlapping:", "yes" if ordered else "NO")
    gap = lambda a, b: int(((cal > a) & (cal < b)).sum())
    line("SPLIT", "trading days between train and val:", gap(t["train"][1], t["val"][0]),
         "between val and test:", gap(t["val"][1], t["test"][0]))
    line("SPLIT", "test ends within given data (2024-12-31):", "yes" if t["test"][1] <= pd.Timestamp("2024-12-31") else "NO")


@_guard("EVENT")
def _event(S, train):
    fac = {t: int((d["Factor"] > 1.0001).sum()) for t, d in train.items()}
    late = [t for t, d in train.items() if d["Close"].isna().any()]
    line("EVENT", "actual split-factor days in train:", sum(fac.values()), {k: v for k, v in fac.items() if v})
    line("EVENT", "actual tickers with missing early rows:", len(late), late)
    e = S.get("data_events") or {}
    line("EVENT", "student: splits detected", e.get("splits_detected"), "late listings", e.get("late_listings_detected"))
    line("EVENT", "student counts match:", "yes" if (e.get("splits_detected") == sum(fac.values()) and e.get("late_listings_detected") == len(late)) else "NO")
    line("EVENT", "treatment", e.get("treatment"))


@_guard("SCORE")
def _score(S):
    t = S["test"]
    m, z, s = (np.asarray(t[k], float) for k in ("loss_model", "loss_zero", "loss_simple"))
    ok = len(m) == len(z) == len(s) and np.isfinite(np.concatenate([m, z, s])).all()
    line("SCORE", "metric", t.get("metric_name"), "n test examples", len(m), "aligned and finite:", "yes" if ok else "NO")
    if not ok:
        return
    line("SCORE", "mean loss: model", f"{m.mean():.6g}", "zero-return", f"{z.mean():.6g}", "simple baseline", f"{s.mean():.6g}")
    for name, b in (("zero-return", z), ("simple baseline", s)):
        d = m - b; lo, hi = _block_bootstrap_ci(d)
        line("SCORE", f"model minus {name}: mean {d.mean():.6g}, 95% block-bootstrap CI [{lo:.6g}, {hi:.6g}]",
             "distinguishable:", "yes" if (lo > 0 or hi < 0) else "no", "| model better:", "yes" if d.mean() < 0 else "no")


@_guard("STRAT")
def _strat(S):
    st = S.get("strategy")
    if not st:
        line("STRAT", "no strategy submitted"); return
    r0 = evaluate_strategy(st["positions"], st["prices"], verbose=False)
    r1 = evaluate_strategy(st["positions"], st["prices"], verbose=False, extra_lag=1)
    for k in ("first_day", "last_day", "cumulative_return", "buy_hold_return", "max_drawdown", "lowest_equity",
              "n_trades", "pct_null_beaten", "clipped_days", "ruin_date"):
        line("STRAT", k, r0[k])
    line("STRAT", "lag test: return with signal lagged one more day", f"{r1['cumulative_return']:.4f}", "vs", f"{r0['cumulative_return']:.4f}")
    big_drop = r0["cumulative_return"] > 0.10 and r1["cumulative_return"] < 0.2 * r0["cumulative_return"]
    line("STRAT", "same-day signal suspected:", "YES (flag)" if big_drop else "no")
    line("STRAT", "too good to be true (return above 50% in the period):", "YES (flag)" if r0["cumulative_return"] > 0.5 else "no")


def _history(train, sample, t):
    return {k: pd.concat([train[k], sample[k]])[lambda d: d["Dt"] < t].reset_index(drop=True) for k in train}


def _scaled(h):
    out = {}
    for k, d in h.items():
        d = d.copy()
        for c in ("Open", "High", "Low", "Close", "Adj Close", "Div"):
            d[c] = d[c] * 3.0
        d["Dt"] = d["Dt"] + pd.Timedelta(days=3650)
        out[k] = d
    return out


@_guard("MODEL")
def _model(S, train, sample):
    fm = importlib.import_module((S.get("final_model") or {}).get("module", "final_model"))
    days = list(pd.DatetimeIndex(sample["AAPL"]["Dt"]))[-N_MODEL_DAYS:]
    run = lambda tf: np.array([np.asarray(fm.predict_next(tf(_history(train, sample, t))), float).ravel() for t in days])
    t0 = time.time(); p1 = run(lambda h: h); sec = (time.time() - t0) / len(days); p2 = run(lambda h: h)
    line("MODEL", "predict_next ran on", len(days), "sample days; seconds per call", f"{sec:.3f}", "| output shape per day", p1.shape[1:])
    line("MODEL", "all finite:", "yes" if np.isfinite(p1).all() else "NO")
    line("MODEL", "deterministic (two runs equal):", "yes" if np.allclose(p1, p2, atol=1e-6, equal_nan=True) else "NO (small NN differences are tolerated; large ones are not)")
    sd = float(np.nanstd(p1, axis=0).max())
    line("MODEL", "prediction std across days", f"{sd:.3g}", "| constant prediction:", "YES (flag)" if sd < 1e-9 else "no")
    p3 = run(_scaled)
    c = np.corrcoef(p1.ravel(), p3.ravel())[0, 1] if sd > 1e-9 and np.nanstd(p3) > 1e-9 else float("nan")
    line("MODEL", "lookup check (prices x3, dates shifted 10 years): correlation with original", f"{c:.3f}",
         "| informational; a model on returns should stay near 1")


@_guard("PRE")
def _pre(S):
    p = S.get("pretrained")
    if not p:
        line("PRE", "no pretrained model"); return
    line("PRE", "name", p.get("name"), "revision", p.get("revision"), "parameters", p.get("n_params"), "training cutoff", p.get("cutoff"))
    line("PRE", "on allowed list:", "yes" if p.get("name") in ALLOWED_LLMS else "NO (flag)",
         "| within size cap:", "yes" if (p.get("n_params") or 1e18) <= LLM_PARAM_CAP else "NO (flag)")


@_guard("SRC")
def _src(S):
    for f in S.get("functions", []):
        src = inspect.getsource(f).splitlines()
        line("SRC", f.__name__, "BEGIN", len(src), "lines")
        print("\n".join(src[:150]))
        line("SRC", f.__name__, "END")


@_guard("REPORT")
def _report(S):
    r = S.get("report") or {}
    line("REPORT", "missing keys", [k for k in REPORT_KEYS if not str(r.get(k, "")).strip()] or "none")
    for k in REPORT_KEYS:
        line("REPORT", k, str(r.get(k, ""))[:1500])
    line("REPORT", "key_line length ok (<=300):", "yes" if len(str(r.get("key_line", ""))) <= 300 else "NO")


def print_summary(S, train, sample, given=None):
    line("START", "submission summary v1")
    _cfg(S, given); _obj(S); _fmt(S); _split(S, train, sample); _event(S, train)
    _score(S); _strat(S); _model(S, train, sample); _pre(S); _src(S); _report(S)
    line("END", "complete")
