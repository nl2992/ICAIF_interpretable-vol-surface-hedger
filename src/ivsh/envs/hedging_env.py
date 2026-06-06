"""Episode-based option-hedging environment.

The agent is **short one ATM option** (the liability) and hedges it to expiry by
trading the underlying and one longer-dated ATM option, rebalancing daily.
Transaction costs are charged on traded notional (turnover), including the
initial build and the terminal liquidation of the hedge book.

For speed and reproducibility every quantity that does *not* depend on the
policy is precomputed once into an :class:`EpisodeBank`. A policy then only has
to map a state-feature matrix to target holdings; the P&L of any holdings tensor
is an explicit, fully vectorised numpy expression. The liability P&L telescopes
to ``premium - payoff`` so the episode P&L is the standard deep-hedging hedging
error.

Sign convention: we are short 1 liability option; ``holdings`` are the *long*
positions we hold in (underlying shares, hedge-option units).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ivsh.data.market import TRADING_DAYS, MarketPath
from ivsh.pricing.black_scholes import bs_greeks, bs_price

FEATURE_NAMES = (
    "surf_level",
    "surf_skew",
    "surf_curv",
    "surf_slope",
    "atm_iv_short",
    "atm_iv_long",
    "term_slope",
    "realized_vol",
    "ret_5d",
    "dlevel_1d",
    "liab_delta",
    "liab_vega",
    "liab_gamma",
    "liab_logmoney",
    "liab_ttm",
    "hedge_delta",
    "hedge_vega",
)


@dataclass
class EnvConfig:
    liab_tenor_days: int = 30
    hedge_tenor_days: int = 60
    option_type: str = "call"
    rebalance_every: int = 1  # days between rebalances (1 = daily)
    episode_stride: int = 3  # days between successive episode starts
    underlying_cost_bps: float = 1.0
    option_cost_bps: float = 30.0  # options are far more expensive to trade
    notional: float = 1.0  # number of liability options shorted


@dataclass
class EpisodeBank:
    """Precomputed tensors for a set of hedging episodes.

    Arrays are shaped ``[E, L]`` over episodes and decision steps, with
    ``+1`` time points stored where increments are needed.
    """

    config: EnvConfig
    start_days: np.ndarray  # [E] market day index of each episode start
    spot: np.ndarray  # [E, L+1]
    v_liab: np.ndarray  # [E, L+1]
    o_hedge: np.ndarray  # [E, L+1]
    features: np.ndarray  # [E, L, F] raw (unstandardised) state features
    greeks: dict[str, np.ndarray]  # each [E, L]
    regime_start: np.ndarray  # [E] regime at episode start (0/1)
    regime_frac_stress: np.ndarray  # [E] fraction of stressed days in episode
    feature_names: tuple[str, ...] = FEATURE_NAMES

    @property
    def n_episodes(self) -> int:
        return self.spot.shape[0]

    @property
    def horizon(self) -> int:
        return self.features.shape[1]

    @property
    def n_features(self) -> int:
        return self.features.shape[2]

    @property
    def premium(self) -> np.ndarray:
        """Premium received per episode (value of the short option at t0)."""
        return self.config.notional * self.v_liab[:, 0]

    def flat_features(self) -> np.ndarray:
        """Decision-point features reshaped to ``[E*L, F]``."""
        e, l, f = self.features.shape
        return self.features.reshape(e * l, f)

    def episode_pnl(
        self, holdings: np.ndarray, smooth_costs: bool = False
    ) -> np.ndarray:
        """Terminal P&L per episode for a holdings tensor ``[E, L, 2]``.

        Column 0 = underlying shares, column 1 = hedge-option units.
        """
        cfg = self.config
        e, l, _ = holdings.shape
        if (e, l) != (self.n_episodes, self.horizon):
            raise ValueError("holdings must be [E, L, 2] matching the bank")

        q_s = holdings[:, :, 0]
        q_o = holdings[:, :, 1]

        ds = np.diff(self.spot, axis=1)  # [E, L]
        do = np.diff(self.o_hedge, axis=1)
        dv = np.diff(self.v_liab, axis=1)

        liab_pnl = -cfg.notional * dv.sum(axis=1)  # telescopes to premium - payoff
        hedge_pnl = (q_s * ds).sum(axis=1) + (q_o * do).sum(axis=1)

        # Turnover: trade at each step relative to previous holdings (start at 0),
        # plus a terminal liquidation back to 0.
        prev_s = np.concatenate([np.zeros((e, 1)), q_s[:, :-1]], axis=1)
        prev_o = np.concatenate([np.zeros((e, 1)), q_o[:, :-1]], axis=1)
        trade_s = q_s - prev_s
        trade_o = q_o - prev_o

        absfun = _smooth_abs if smooth_costs else np.abs
        cost_s = absfun(trade_s) * self.spot[:, :-1] * (cfg.underlying_cost_bps / 1e4)
        cost_o = absfun(trade_o) * self.o_hedge[:, :-1] * (cfg.option_cost_bps / 1e4)
        # Terminal liquidation cost.
        liq_s = absfun(q_s[:, -1]) * self.spot[:, -1] * (cfg.underlying_cost_bps / 1e4)
        liq_o = absfun(q_o[:, -1]) * self.o_hedge[:, -1] * (cfg.option_cost_bps / 1e4)
        costs = cost_s.sum(axis=1) + cost_o.sum(axis=1) + liq_s + liq_o

        return liab_pnl + hedge_pnl - costs

    def pnl_grad(self, holdings: np.ndarray, smooth_costs: bool = True):
        """Episode P&L and its gradient w.r.t. each holding.

        Returns ``(pnl[E], dpnl_dh[E, L, 2])`` where ``dpnl_dh[e, j, c]`` is
        ``d pnl[e] / d holdings[e, j, c]``. Uses the smooth-abs cost model so the
        gradient is well defined (matches :meth:`episode_pnl` with the same flag).
        """
        cfg = self.config
        e, l, _ = holdings.shape
        q_s = holdings[:, :, 0]
        q_o = holdings[:, :, 1]
        ds = np.diff(self.spot, axis=1)
        do = np.diff(self.o_hedge, axis=1)
        s_dec = self.spot[:, :-1]
        o_dec = self.o_hedge[:, :-1]
        cs = cfg.underlying_cost_bps / 1e4
        co = cfg.option_cost_bps / 1e4

        pnl = self.episode_pnl(holdings, smooth_costs=smooth_costs)

        def grad_for(q, dx, x_dec, x_T, coef):
            prev = np.concatenate([np.zeros((e, 1)), q[:, :-1]], axis=1)
            trade = q - prev  # [E, L]
            if smooth_costs:
                dprime = trade / np.sqrt(trade * trade + 1e-8)  # smooth-abs'
                dprime_T = q[:, -1] / np.sqrt(q[:, -1] ** 2 + 1e-8)
            else:
                dprime = np.sign(trade)
                dprime_T = np.sign(q[:, -1])
            # cost at step j (this step's trade) + step j+1 (next trade, -q_j)
            dcost = dprime * x_dec * coef
            nxt = np.concatenate([dprime[:, 1:] * x_dec[:, 1:] * coef, np.zeros((e, 1))], axis=1)
            dcost = dcost - nxt
            dcost[:, -1] += dprime_T * x_T * coef
            return dx - dcost

        dpnl = np.empty_like(holdings)
        dpnl[:, :, 0] = grad_for(q_s, ds, s_dec, self.spot[:, -1], cs)
        dpnl[:, :, 1] = grad_for(q_o, do, o_dec, self.o_hedge[:, -1], co)
        return pnl, dpnl

    def turnover(self, holdings: np.ndarray) -> np.ndarray:
        """Per-episode gross traded notional (for turnover diagnostics)."""
        e = holdings.shape[0]
        q_s, q_o = holdings[:, :, 0], holdings[:, :, 1]
        prev_s = np.concatenate([np.zeros((e, 1)), q_s[:, :-1]], axis=1)
        prev_o = np.concatenate([np.zeros((e, 1)), q_o[:, :-1]], axis=1)
        notional = np.abs(q_s - prev_s) * self.spot[:, :-1] + np.abs(
            q_o - prev_o
        ) * self.o_hedge[:, :-1]
        liq = np.abs(q_s[:, -1]) * self.spot[:, -1] + np.abs(q_o[:, -1]) * self.o_hedge[
            :, -1
        ]
        return notional.sum(axis=1) + liq


def _smooth_abs(x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    return np.sqrt(x * x + eps * eps)


def with_costs(bank: EpisodeBank, underlying_cost_bps: float, option_cost_bps: float) -> EpisodeBank:
    """Return a shallow copy of ``bank`` with different transaction costs.

    Arrays are shared (read-only for P&L); only the cost config changes. Used by
    the no-transaction-cost ablation.
    """
    import dataclasses

    new_cfg = dataclasses.replace(
        bank.config, underlying_cost_bps=underlying_cost_bps, option_cost_bps=option_cost_bps
    )
    return dataclasses.replace(bank, config=new_cfg)


def concat_banks(banks: list[EpisodeBank]) -> EpisodeBank:
    """Pool episodes from several market paths into one bank (Monte-Carlo set).

    ``start_days`` is offset per path so each path occupies a disjoint day range,
    keeping the pooled day index unique for diagnostics.
    """
    if not banks:
        raise ValueError("need at least one bank")
    cfg = banks[0].config
    offset = 0
    starts = []
    for b in banks:
        starts.append(b.start_days + offset)
        offset += int(b.start_days.max()) + b.horizon + 1
    return EpisodeBank(
        config=cfg,
        start_days=np.concatenate(starts),
        spot=np.concatenate([b.spot for b in banks]),
        v_liab=np.concatenate([b.v_liab for b in banks]),
        o_hedge=np.concatenate([b.o_hedge for b in banks]),
        features=np.concatenate([b.features for b in banks]),
        greeks={k: np.concatenate([b.greeks[k] for b in banks]) for k in banks[0].greeks},
        regime_start=np.concatenate([b.regime_start for b in banks]),
        regime_frac_stress=np.concatenate([b.regime_frac_stress for b in banks]),
        feature_names=banks[0].feature_names,
    )


def build_episode_bank(market: MarketPath, config: EnvConfig | None = None) -> EpisodeBank:
    """Construct all hedging episodes from a market path."""
    cfg = config or EnvConfig()
    L = cfg.liab_tenor_days
    rate, div = market.rate, market.div
    opt = cfg.option_type

    last_start = market.n_days - L - 1
    starts = np.arange(0, last_start, cfg.episode_stride)
    E = len(starts)

    spot = np.empty((E, L + 1))
    v_liab = np.empty((E, L + 1))
    o_hedge = np.empty((E, L + 1))
    features = np.empty((E, L, len(FEATURE_NAMES)))
    g_names = ("delta_liab", "vega_liab", "delta_hedge", "vega_hedge", "gamma_liab")
    greeks = {k: np.empty((E, L)) for k in g_names}
    regime_start = np.empty(E, dtype=int)
    regime_frac = np.empty(E)

    for i, t0 in enumerate(starts):
        s0 = market.spot[t0]
        k_liab = s0  # struck ATM at episode start
        k_hedge = s0
        liab_expiry = t0 + L
        hedge_expiry = t0 + cfg.hedge_tenor_days

        days = np.arange(t0, t0 + L + 1)
        spot[i] = market.spot[days]

        ttm_liab = np.maximum(liab_expiry - days, 0) / TRADING_DAYS
        ttm_hedge = np.maximum(hedge_expiry - days, 0) / TRADING_DAYS
        iv_liab = market.iv(days, k_liab, liab_expiry)
        iv_hedge = market.iv(days, k_hedge, hedge_expiry)
        v_liab[i] = bs_price(spot[i], k_liab, ttm_liab, iv_liab, rate, div, opt)
        o_hedge[i] = bs_price(spot[i], k_hedge, ttm_hedge, iv_hedge, rate, div, opt)

        regime_start[i] = market.regime[t0]
        regime_frac[i] = market.regime[t0 : t0 + L].mean()

        # Decision-point features and greeks (days t0 .. t0+L-1).
        dec = days[:-1]
        gl = bs_greeks(
            spot[i, :-1], k_liab, ttm_liab[:-1], iv_liab[:-1], rate, div, opt
        )
        gh = bs_greeks(
            spot[i, :-1], k_hedge, ttm_hedge[:-1], iv_hedge[:-1], rate, div, opt
        )
        greeks["delta_liab"][i] = gl["delta"]
        greeks["vega_liab"][i] = gl["vega"]
        greeks["gamma_liab"][i] = gl["gamma"]
        greeks["delta_hedge"][i] = gh["delta"]
        greeks["vega_hedge"][i] = gh["vega"]

        fwd = spot[i, :-1] * np.exp((rate - div) * ttm_liab[:-1])
        atm_short = market.iv(dec, fwd, dec + 7)
        atm_long = market.iv(dec, fwd, dec + 90)
        ret_5d = np.array(
            [market.log_return[max(0, d - 4) : d + 1].sum() for d in dec]
        )
        dlevel = market.level[dec] - market.level[np.maximum(dec - 1, 0)]

        features[i, :, 0] = market.level[dec]
        features[i, :, 1] = market.skew[dec]
        features[i, :, 2] = market.curv[dec]
        features[i, :, 3] = market.slope[dec]
        features[i, :, 4] = atm_short
        features[i, :, 5] = atm_long
        features[i, :, 6] = atm_long - atm_short
        features[i, :, 7] = market.realized_vol[dec]
        features[i, :, 8] = ret_5d
        features[i, :, 9] = dlevel
        features[i, :, 10] = gl["delta"]
        features[i, :, 11] = gl["vega"]
        features[i, :, 12] = gl["gamma"]
        features[i, :, 13] = np.log(k_liab / fwd)
        features[i, :, 14] = ttm_liab[:-1]
        features[i, :, 15] = gh["delta"]
        features[i, :, 16] = gh["vega"]

    return EpisodeBank(
        config=cfg,
        start_days=starts,
        spot=spot,
        v_liab=v_liab,
        o_hedge=o_hedge,
        features=features,
        greeks=greeks,
        regime_start=regime_start,
        regime_frac_stress=regime_frac,
    )
