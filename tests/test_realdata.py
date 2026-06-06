from __future__ import annotations

import numpy as np
import pandas as pd

from ivsh.data.clean import clean_option_panel
from ivsh.data.loaders import optionsdx_to_panel
from ivsh.data.market import MarketConfig, simulate_market
from ivsh.envs.hedging_env import EnvConfig
from ivsh.pipeline import ExperimentConfig, run_experiment_real
from ivsh.pricing.black_scholes import bs_price
from ivsh.training.train import TrainConfig


def _make_optionsdx_wide(n_days=160, seed=1, bracketed=False):
    mkt = simulate_market(MarketConfig(n_days=n_days, seed=seed))
    dates = pd.bdate_range("2019-01-02", periods=n_days)
    tenors = [30, 60, 90, 180]
    moneyness = np.round(np.linspace(0.85, 1.15, 7), 3)
    rows = []
    for d in range(n_days):
        spot = float(mkt.spot[d])
        for tnr in tenors:
            ttm = tnr / 365.25
            exp_day = d + int(round(tnr * 252 / 365.25))
            for m in moneyness:
                K = round(m * spot, 1)
                iv = float(mkt.iv(d, K, exp_day))
                c = float(bs_price(spot, K, ttm, iv, 0.0, 0.0, "call"))
                p = float(bs_price(spot, K, ttm, iv, 0.0, 0.0, "put"))
                hs = 0.02
                rows.append(
                    {
                        "QUOTE_DATE": dates[d],
                        "UNDERLYING_LAST": spot,
                        "EXPIRE_DATE": dates[d] + pd.Timedelta(days=tnr),
                        "DTE": tnr,
                        "STRIKE": K,
                        "C_BID": max(c - hs, 0.0), "C_ASK": c + hs, "C_IV": iv, "C_VOLUME": 100,
                        "P_BID": max(p - hs, 0.0), "P_ASK": p + hs, "P_IV": iv, "P_VOLUME": 100,
                    }
                )
    df = pd.DataFrame(rows)
    if bracketed:
        df.columns = [f"[{c}]" for c in df.columns]
    return df


def test_optionsdx_adapter_reshapes_wide_to_long():
    wide = _make_optionsdx_wide(n_days=3, bracketed=True)
    long = optionsdx_to_panel(wide)
    assert {"date", "spot", "strike", "option_type", "bid", "ask", "iv"} <= set(long.columns)
    assert set(long["option_type"].unique()) == {"call", "put"}
    assert len(long) == 2 * len(wide)  # one call + one put row per wide row


def test_optionsdx_clean_and_market_build():
    long = optionsdx_to_panel(_make_optionsdx_wide(n_days=40))
    clean, summary = clean_option_panel(long)
    assert len(clean) > 0 and summary.table["removed"].sum() >= 0
    from ivsh.data.loaders import market_from_option_panel

    mkt = market_from_option_panel(clean, surface_method="ols")
    assert mkt.n_days == 40 and np.all(mkt.level > 0)


def test_anchor_keeps_bounded_residual_on_delta_vega():
    from ivsh.baselines.policies import delta_vega_hedge
    from ivsh.envs.hedging_env import build_episode_bank
    from ivsh.evaluation.backtest import run_policy
    from ivsh.training.train import TrainConfig, fit_prototype, make_standardizer

    bank = build_episode_bank(simulate_market(MarketConfig(n_days=200, seed=4)), EnvConfig())
    scaler = make_standardizer(bank)
    proto, _, _ = fit_prototype(
        bank, scaler, TrainConfig(n_prototypes=4, max_iter=15, anchor=True, action_scale=1.0)
    )
    holdings = run_policy(proto, bank, scaler, anchor=True)["holdings"]
    residual = holdings - delta_vega_hedge(bank)
    # residual is a convex blend of bounded prototype actions -> within action_scale
    assert np.abs(residual).max() <= 1.0 + 1e-6


def test_real_data_end_to_end(tmp_path):
    long = optionsdx_to_panel(_make_optionsdx_wide(n_days=170, seed=2))
    clean, _ = clean_option_panel(long)
    cfg = ExperimentConfig(
        experiment_id="real_test",
        env=EnvConfig(),
        proto_train=TrainConfig(n_prototypes=4, max_iter=20),
        bb_train=TrainConfig(hidden=8, l2=3e-2, max_iter=20),
        run_ablations=False,
        reports_dir=str(tmp_path / "reports"),
        checkpoints_dir=str(tmp_path / "checkpoints"),
    )
    res = run_experiment_real(cfg, clean, surface_method="ols")
    assert (tmp_path / "reports" / "final_report.md").exists()
    # prototype catalogue carries calendar-date annotations on real data
    cat = res["catalogue"]
    assert "top_period" in cat.columns and "example_date" in cat.columns
    assert cat["top_period"].str.match(r"\d{4}-\d{2}").any()
