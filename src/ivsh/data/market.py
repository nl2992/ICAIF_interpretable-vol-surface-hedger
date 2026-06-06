"""Synthetic but economically grounded options market.

We simulate a daily path of an underlying together with a *parametric* implied
volatility surface so that any option (arbitrary strike / expiry) can be priced
consistently. The surface is driven by four latent factors that evolve as
mean-reverting AR(1) processes whose long-run means switch with a two-state
(calm / stress) Markov regime:

    iv(k, tau) = level + skew * k + curv * k^2 + slope * log(tau / tau0)

where ``k = log(K / F)`` is forward log-moneyness and ``tau`` is time to expiry
in years. In the stress regime the level rises, the skew steepens (more
negative), the term structure inverts, and the underlying experiences higher
diffusive vol plus downward jumps. This makes the *shape* of the surface — not
just spot — informative about tail risk, which is the premise the prototype
hedger is meant to exploit.

The model is deliberately arbitrage-light (it can produce mild static-arbitrage
violations in extreme states); the construction pipeline flags those rather than
guaranteeing their absence, mirroring how a real surface is handled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252
TAU0 = 30.0 / TRADING_DAYS  # reference tenor (30 calendar/trading days) in years


@dataclass(frozen=True)
class MarketConfig:
    n_days: int = 1500
    seed: int = 7
    # Zero rates / carry: the P&L does not model a separate funding leg, so a
    # nonzero cost-of-carry would hand a self-financing policy free drift to lever
    # into. With r = div = drift = 0 every tradeable is a martingale and P&L
    # reflects only genuine hedging risk (vol moves, gamma, jumps).
    rate: float = 0.0
    div: float = 0.0
    spot0: float = 100.0

    # Surface-factor long-run means per regime: (calm, stress).
    level_mean: tuple[float, float] = (0.16, 0.34)
    skew_mean: tuple[float, float] = (-0.08, -0.22)
    curv_mean: tuple[float, float] = (0.30, 0.55)
    slope_mean: tuple[float, float] = (0.020, -0.030)

    # AR(1) persistence and shock sizes for each factor.
    level_phi: float = 0.96
    level_vol: float = 0.010
    skew_phi: float = 0.95
    skew_vol: float = 0.010
    curv_phi: float = 0.93
    curv_vol: float = 0.020
    slope_phi: float = 0.94
    slope_vol: float = 0.006

    # Regime Markov chain (daily transition probabilities).
    p_calm_to_stress: float = 0.012
    p_stress_to_calm: float = 0.06

    # Underlying dynamics. The drift is set to the risk-free rate so the market
    # offers no directional edge: a policy can only improve the CVaR objective by
    # genuinely hedging the option's risk, not by harvesting an equity premium.
    drift: float = 0.0
    leverage_corr: float = -0.7  # corr(spot shock, level shock)
    jump_intensity_stress: float = 0.04  # daily prob of a jump in stress
    jump_mean: float = -0.05
    jump_std: float = 0.03

    grid_moneyness: tuple[float, ...] = (0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20)
    grid_tenor_days: tuple[int, ...] = (7, 14, 30, 60, 90, 180)


@dataclass
class MarketPath:
    """A realised market path with a callable parametric IV surface."""

    config: MarketConfig
    days: np.ndarray  # integer day index 0..n-1
    spot: np.ndarray  # [n_days]
    level: np.ndarray
    skew: np.ndarray
    curv: np.ndarray
    slope: np.ndarray
    regime: np.ndarray  # 0 = calm, 1 = stress
    realized_vol: np.ndarray  # trailing realised vol (annualised)
    log_return: np.ndarray  # daily log returns (return[0] = 0)
    rate: float = field(default=0.03)
    div: float = field(default=0.01)

    @property
    def n_days(self) -> int:
        return len(self.days)

    def iv(self, day, strike, expiry_day) -> np.ndarray:
        """Implied vol for option(s) at ``day`` with given strike and expiry day.

        ``day`` may be a scalar or array aligned with ``strike``/``expiry_day``.
        """
        day = np.asarray(day)
        strike = np.asarray(strike, dtype=float)
        expiry_day = np.asarray(expiry_day, dtype=float)
        ttm = np.maximum(expiry_day - day, 1e-6) / TRADING_DAYS
        spot = self.spot[day]
        fwd = spot * np.exp((self.rate - self.div) * ttm)
        k = np.log(strike / fwd)
        iv = (
            self.level[day]
            + self.skew[day] * k
            + self.curv[day] * k * k
            + self.slope[day] * np.log(ttm / TAU0)
        )
        return np.maximum(iv, 0.02)

    def surface_grid_frame(self) -> pd.DataFrame:
        """Long-form IV on the fixed (moneyness, tenor) grid for every day.

        Compatible with :func:`ivsh.features.surface.surface_tensor`.
        """
        cfg = self.config
        rows: list[dict] = []
        for t in self.days:
            spot_t = self.spot[t]
            for tenor in cfg.grid_tenor_days:
                ttm = tenor / TRADING_DAYS
                fwd = spot_t * np.exp((self.rate - self.div) * ttm)
                for m in cfg.grid_moneyness:
                    strike = m * spot_t
                    iv = float(self.iv(t, strike, t + tenor))
                    rows.append(
                        {
                            "timestamp": int(t),
                            "moneyness": float(m),
                            "tenor_days": int(tenor),
                            "implied_vol": iv,
                            "forward": float(fwd),
                        }
                    )
        return pd.DataFrame(rows)


def simulate_market(config: MarketConfig | None = None) -> MarketPath:
    """Simulate one market path under ``config``."""
    cfg = config or MarketConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_days

    regime = np.zeros(n, dtype=int)
    level = np.empty(n)
    skew = np.empty(n)
    curv = np.empty(n)
    slope = np.empty(n)
    log_ret = np.zeros(n)
    spot = np.empty(n)
    spot[0] = cfg.spot0

    # Initialise factors at calm means.
    level[0] = cfg.level_mean[0]
    skew[0] = cfg.skew_mean[0]
    curv[0] = cfg.curv_mean[0]
    slope[0] = cfg.slope_mean[0]

    dt = 1.0 / TRADING_DAYS
    sqrt_dt = np.sqrt(dt)

    for t in range(1, n):
        # Regime transition.
        if regime[t - 1] == 0:
            regime[t] = 1 if rng.random() < cfg.p_calm_to_stress else 0
        else:
            regime[t] = 0 if rng.random() < cfg.p_stress_to_calm else 1
        r = regime[t]

        # Correlated shocks for spot and level (leverage effect).
        z_spot = rng.standard_normal()
        z_ind = rng.standard_normal()
        z_level = cfg.leverage_corr * z_spot + np.sqrt(
            1.0 - cfg.leverage_corr**2
        ) * z_ind

        level[t] = (
            cfg.level_phi * level[t - 1]
            + (1 - cfg.level_phi) * cfg.level_mean[r]
            + cfg.level_vol * z_level
        )
        level[t] = max(level[t], 0.03)
        skew[t] = (
            cfg.skew_phi * skew[t - 1]
            + (1 - cfg.skew_phi) * cfg.skew_mean[r]
            + cfg.skew_vol * rng.standard_normal()
        )
        curv[t] = (
            cfg.curv_phi * curv[t - 1]
            + (1 - cfg.curv_phi) * cfg.curv_mean[r]
            + cfg.curv_vol * rng.standard_normal()
        )
        curv[t] = max(curv[t], 0.0)
        slope[t] = (
            cfg.slope_phi * slope[t - 1]
            + (1 - cfg.slope_phi) * cfg.slope_mean[r]
            + cfg.slope_vol * rng.standard_normal()
        )

        # Underlying: ATM 30d vol ~ level drives diffusion; jumps in stress.
        # A martingale compensator offsets the expected jump return so the
        # underlying drifts at exactly (rate - div): no tradeable has an edge,
        # leaving the jump purely as unhedgeable tail risk.
        sigma = level[t - 1]
        lam = cfg.jump_intensity_stress if r == 1 else 0.0
        jump_factor = 1.0 - lam + lam * np.exp(cfg.jump_mean + 0.5 * cfg.jump_std**2)
        comp = -np.log(jump_factor)  # additive drift correction
        diff = (
            (cfg.drift - cfg.div - 0.5 * sigma**2) * dt
            + comp
            + sigma * sqrt_dt * z_spot
        )
        jump = 0.0
        if lam > 0 and rng.random() < lam:
            jump = rng.normal(cfg.jump_mean, cfg.jump_std)
        log_ret[t] = diff + jump
        spot[t] = spot[t - 1] * np.exp(log_ret[t])

    # Trailing 21-day realised vol (annualised), back-filled at the start.
    realized = np.zeros(n)
    win = 21
    for t in range(n):
        lo = max(0, t - win + 1)
        seg = log_ret[lo : t + 1]
        realized[t] = np.std(seg) * np.sqrt(TRADING_DAYS) if seg.size > 1 else level[t]

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
        log_return=log_ret,
        rate=cfg.rate,
        div=cfg.div,
    )
