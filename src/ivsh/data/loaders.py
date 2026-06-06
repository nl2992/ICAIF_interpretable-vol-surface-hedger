"""Real option-data ingestion -> :class:`~ivsh.data.market.MarketPath`.

This is the single swap point for replacing the synthetic market with real
option chains. The bridge is the parametric surface used throughout the codebase:

    iv(k, tau) = level + skew * k + curv * k^2 + slope * log(tau / tau0)

For each trading day we fit the four factors (level, skew, curvature, term slope)
by least squares to that day's cleaned implied-vol quotes. The resulting
``MarketPath`` plugs straight into ``build_episode_bank`` — so real surfaces reuse
the *exact* same environment, features, baselines and models as the synthetic
study, with no downstream changes.

Expected (cleaned) panel schema, one row per quote:
    date    : sortable trading-day key (int index or datetime)
    spot    : underlying price on that day
    strike  : option strike
    iv      : implied volatility            (or provide mid+option_type to imply)
    one of:  ttm_years | ttm_days | expiry  (time to maturity / expiry date)
optional: bid, ask, option_type, volume, open_interest

See ``docs/data_checklist.md`` for the full data contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ivsh.data.market import TAU0, TRADING_DAYS, MarketConfig, MarketPath
from ivsh.pricing.black_scholes import implied_vol

REQUIRED = ("date", "spot", "strike")


# --------------------------------------------------------------------------- #
# IO + cleaning
# --------------------------------------------------------------------------- #
def load_option_panel(path: str) -> pd.DataFrame:
    """Read a long-form option panel from CSV or Parquet."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


@dataclass
class CleanSummary:
    table: pd.DataFrame  # per-filter counts

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.table.to_string(index=False)


def clean_quotes(
    df: pd.DataFrame,
    max_rel_spread: float = 0.5,
    min_ttm_years: float = 1e-3,
) -> tuple[pd.DataFrame, CleanSummary]:
    """Apply standard option-quote filters and report removals per rule."""
    df = df.copy()
    rows = [("input", len(df))]

    def drop(mask, name):
        nonlocal df
        keep = ~mask
        df = df[keep]
        rows.append((name, int(mask.sum())))

    if {"bid", "ask"} <= set(df.columns):
        drop(df["ask"] <= 0, "ask<=0")
        drop(df["bid"] < 0, "bid<0")
        drop(df["bid"] > df["ask"], "crossed (bid>ask)")
        df["mid"] = 0.5 * (df["bid"] + df["ask"])
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = (df["ask"] - df["bid"]) / df["mid"].replace(0, np.nan)
        drop(rel > max_rel_spread, f"rel_spread>{max_rel_spread}")

    ttm = _ttm_years(df)
    drop(~np.isfinite(ttm) | (ttm <= min_ttm_years), "expired/invalid_ttm")
    drop(df["strike"].isna(), "missing_strike")

    summary = CleanSummary(pd.DataFrame(rows, columns=["filter", "removed"]))
    return df.reset_index(drop=True), summary


# --------------------------------------------------------------------------- #
# Panel -> MarketPath
# --------------------------------------------------------------------------- #
def _ttm_years(df: pd.DataFrame) -> np.ndarray:
    if "ttm_years" in df.columns:
        return df["ttm_years"].to_numpy(dtype=float)
    if "ttm_days" in df.columns:
        return df["ttm_days"].to_numpy(dtype=float) / 365.25
    if "expiry" in df.columns:
        exp = pd.to_datetime(df["expiry"])
        dat = pd.to_datetime(df["date"])
        return (exp - dat).dt.days.to_numpy(dtype=float) / 365.25
    raise ValueError("panel must provide one of: ttm_years, ttm_days, or expiry")


def _ensure_iv(df: pd.DataFrame, ttm: np.ndarray, rate: float, div: float) -> np.ndarray:
    if "iv" in df.columns and df["iv"].notna().all():
        return df["iv"].to_numpy(dtype=float)
    if "mid" in df.columns and "option_type" in df.columns:
        iv = np.empty(len(df))
        for i, (_, r) in enumerate(df.iterrows()):
            iv[i] = implied_vol(r["mid"], r["spot"], r["strike"], ttm[i], rate, div, r["option_type"])
        return iv
    raise ValueError("panel must provide 'iv', or 'mid' + 'option_type' to imply it")


def fit_surface_factors(k: np.ndarray, log_tau_ratio: np.ndarray, iv: np.ndarray) -> np.ndarray:
    """OLS fit of (level, skew, curvature, term_slope) for one day's quotes."""
    X = np.column_stack([np.ones_like(k), k, k**2, log_tau_ratio])
    beta, *_ = np.linalg.lstsq(X, iv, rcond=None)
    return beta  # [level, skew, curv, slope]


def market_from_option_panel(
    df: pd.DataFrame,
    rate: float = 0.0,
    div: float = 0.0,
    min_quotes: int = 6,
    regime_window: int = 252,
    regime_mult: float = 1.15,
) -> MarketPath:
    """Build a :class:`MarketPath` by fitting the parametric surface to real data.

    The regime label (used only for evaluation slicing) is causal: a day is
    flagged ``stress`` when its fitted vol level exceeds ``regime_mult`` times the
    trailing-median level, so no future information leaks into the split.
    """
    for col in REQUIRED:
        if col not in df.columns:
            raise ValueError(f"missing required column: {col!r}")

    df = df.copy()
    df["_day"] = pd.factorize(df["date"], sort=True)[0]
    n = int(df["_day"].max()) + 1

    ttm = _ttm_years(df)
    iv = _ensure_iv(df, ttm, rate, div)
    spot_col = df["spot"].to_numpy(dtype=float)
    fwd = spot_col * np.exp((rate - div) * ttm)
    k_all = np.log(df["strike"].to_numpy(dtype=float) / fwd)
    logtau_all = np.log(ttm / TAU0)
    days = df["_day"].to_numpy()

    level = np.empty(n)
    skew = np.empty(n)
    curv = np.empty(n)
    slope = np.empty(n)
    spot = np.empty(n)
    last = None
    for d in range(n):
        m = days == d
        spot[d] = spot_col[m][0]
        if m.sum() >= min_quotes:
            last = fit_surface_factors(k_all[m], logtau_all[m], iv[m])
        elif last is None:
            last = fit_surface_factors(k_all[m], logtau_all[m], iv[m]) if m.sum() >= 4 else np.array([0.2, -0.05, 0.3, 0.0])
        level[d], skew[d], curv[d], slope[d] = last
    level = np.maximum(level, 0.03)
    curv = np.maximum(curv, 0.0)

    log_return = np.zeros(n)
    log_return[1:] = np.log(spot[1:] / spot[:-1])
    realized = np.zeros(n)
    win = 21
    for d in range(n):
        seg = log_return[max(0, d - win + 1) : d + 1]
        realized[d] = seg.std() * np.sqrt(TRADING_DAYS) if seg.size > 1 else level[d]

    # Causal regime label from trailing-median vol level.
    regime = np.zeros(n, dtype=int)
    for d in range(n):
        ref = np.median(level[max(0, d - regime_window + 1) : d + 1])
        regime[d] = int(level[d] > regime_mult * ref)

    cfg = MarketConfig(n_days=n, rate=rate, div=div, spot0=float(spot[0]))
    return MarketPath(
        config=cfg,
        days=np.arange(n),
        spot=spot,
        level=level,
        skew=skew,
        curv=curv,
        slope=slope,
        regime=regime,
        realized_vol=realized,
        log_return=log_return,
        rate=rate,
        div=div,
    )
