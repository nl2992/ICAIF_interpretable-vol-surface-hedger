"""Phase-3 option-quote cleaning.

Turns a raw long-form option panel (e.g. an OptionMetrics ``opprcd`` extract) into
a cleaned panel ready for surface fitting, recording how many quotes each filter
removes. Output target: ``data/interim/clean_options_panel.parquet``.

Filters (applied in order):
  1. zero/negative ask, negative bid
  2. crossed quotes (bid > ask)
  3. missing strike / time-to-maturity
  4. expired / invalid maturity
  5. non-positive or missing implied vol (when an ``iv`` column is present)
  6. absolute and relative bid-ask spread caps
  7. illiquid quotes (volume / open-interest floors)
  8. stale quotes (zero volume AND zero open interest)

It also computes mid price, time to maturity, forward and forward log-moneyness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def time_to_maturity_years(df: pd.DataFrame) -> np.ndarray:
    """Annualised time to maturity from ttm_years | ttm_days | expiry+date."""
    if "ttm_years" in df.columns:
        return df["ttm_years"].to_numpy(dtype=float)
    if "ttm_days" in df.columns:
        return df["ttm_days"].to_numpy(dtype=float) / 365.25
    if "expiry" in df.columns:
        exp = pd.to_datetime(df["expiry"])
        dat = pd.to_datetime(df["date"])
        return (exp - dat).dt.days.to_numpy(dtype=float) / 365.25
    raise ValueError("panel must provide one of: ttm_years, ttm_days, or expiry")


@dataclass
class CleanSummary:
    table: pd.DataFrame  # columns: filter, removed, remaining, pct_removed

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.table.to_string(index=False)

    def to_markdown(self) -> str:  # pragma: no cover - cosmetic
        t = self.table
        out = ["| filter | removed | remaining | pct_removed |", "| --- | --- | --- | --- |"]
        for _, r in t.iterrows():
            out.append(f"| {r['filter']} | {int(r['removed'])} | {int(r['remaining'])} | {r['pct_removed']:.2f}% |")
        return "\n".join(out)


def clean_option_panel(
    df: pd.DataFrame,
    max_rel_spread: float = 0.5,
    max_abs_spread: float | None = None,
    min_ttm_years: float = 1e-3,
    min_volume: int = 0,
    min_open_interest: int = 0,
    drop_stale: bool = True,
    iv_bounds: tuple[float, float] | None = None,
    moneyness_band: tuple[float, float] | None = None,
    otm_only: bool = False,
) -> tuple[pd.DataFrame, CleanSummary]:
    """Clean a raw option-quote panel and report per-filter removals.

    Real-data extras: ``iv_bounds`` drops implausible IVs (deep ITM quotes carry
    garbage IV); ``moneyness_band`` keeps strikes within ``(low, high) * spot``;
    ``otm_only`` keeps only out-of-the-money quotes (calls with K>=spot, puts with
    K<=spot) — the standard clean smile.
    """
    df = df.copy()
    n0 = len(df)
    rows = [("input", 0, n0, 0.0)]

    def drop(mask, name):
        nonlocal df
        mask = np.asarray(mask, dtype=bool)
        removed = int(mask.sum())
        df = df[~mask]
        rows.append((name, removed, len(df), 100.0 * removed / max(n0, 1)))

    has_quotes = {"bid", "ask"} <= set(df.columns)
    if has_quotes:
        drop(df["ask"] <= 0, "ask<=0")
        drop(df["bid"] < 0, "bid<0")
        drop(df["bid"] > df["ask"], "crossed_bid>ask")
        df["mid"] = 0.5 * (df["bid"] + df["ask"])

    if "strike" in df.columns:
        drop(df["strike"].isna(), "missing_strike")

    ttm = time_to_maturity_years(df)
    drop(~np.isfinite(ttm) | (ttm <= min_ttm_years), "expired/invalid_ttm")

    if "iv" in df.columns:
        drop(df["iv"].isna() | (df["iv"] <= 0), "nonpositive_iv")
        if iv_bounds is not None:
            lo, hi = iv_bounds
            drop((df["iv"] < lo) | (df["iv"] > hi), f"iv_outside[{lo},{hi}]")

    if moneyness_band is not None and {"strike", "spot"} <= set(df.columns):
        lo, hi = moneyness_band
        ratio = df["strike"] / df["spot"]
        drop((ratio < lo) | (ratio > hi), f"moneyness_outside[{lo},{hi}]")

    if otm_only and {"option_type", "strike", "spot"} <= set(df.columns):
        ot = df["option_type"].astype(str).str.lower()
        itm = ((ot == "call") & (df["strike"] < df["spot"])) | (
            (ot == "put") & (df["strike"] > df["spot"])
        )
        drop(itm, "in_the_money")

    if has_quotes:
        spread = df["ask"] - df["bid"]
        if max_abs_spread is not None:
            drop(spread > max_abs_spread, f"abs_spread>{max_abs_spread}")
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = spread / df["mid"].replace(0, np.nan)
        drop(rel > max_rel_spread, f"rel_spread>{max_rel_spread}")

    if min_volume > 0 and "volume" in df.columns:
        drop(df["volume"].fillna(0) < min_volume, f"volume<{min_volume}")
    if min_open_interest > 0 and "open_interest" in df.columns:
        drop(df["open_interest"].fillna(0) < min_open_interest, f"oi<{min_open_interest}")
    if drop_stale and {"volume", "open_interest"} <= set(df.columns):
        drop((df["volume"].fillna(0) == 0) & (df["open_interest"].fillna(0) == 0), "stale_zero_vol_oi")

    summary = CleanSummary(pd.DataFrame(rows, columns=["filter", "removed", "remaining", "pct_removed"]))
    return df.reset_index(drop=True), summary


def add_quote_features(df: pd.DataFrame, rate: float = 0.0, div: float = 0.0) -> pd.DataFrame:
    """Add mid, ttm_years, forward and forward log-moneyness columns."""
    df = df.copy()
    if "mid" not in df.columns and {"bid", "ask"} <= set(df.columns):
        df["mid"] = 0.5 * (df["bid"] + df["ask"])
    ttm = time_to_maturity_years(df)
    df["ttm_years"] = ttm
    if "spot" in df.columns and "strike" in df.columns:
        fwd = df["spot"].to_numpy(dtype=float) * np.exp((rate - div) * ttm)
        df["forward"] = fwd
        df["log_moneyness"] = np.log(df["strike"].to_numpy(dtype=float) / fwd)
    return df


def write_clean_panel(df: pd.DataFrame, path: str = "data/interim/clean_options_panel.parquet") -> str:
    """Persist the cleaned panel to Parquet (creating parent dirs)."""
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
