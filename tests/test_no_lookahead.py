from __future__ import annotations

import numpy as np

from ivsh.data.market import MarketConfig, simulate_market
from ivsh.envs.hedging_env import EnvConfig, build_episode_bank
from ivsh.features.standardize import Standardizer
from ivsh.utils.splits import chronological_split, subset

BANK = build_episode_bank(simulate_market(MarketConfig(n_days=600, seed=9)), EnvConfig())


def test_split_is_chronological_and_disjoint():
    sp = chronological_split(BANK)
    assert len(sp.train) and len(sp.val) and len(sp.test)
    # no index appears in two splits
    assert len(set(sp.train) & set(sp.test)) == 0
    assert len(set(sp.train) & set(sp.val)) == 0
    # train strictly precedes test in calendar time (purge gap enforced)
    assert BANK.start_days[sp.train].max() < BANK.start_days[sp.test].min()


def test_standardizer_fit_on_train_only():
    sp = chronological_split(BANK)
    trb = subset(BANK, sp.train)
    scaler = Standardizer.fit(trb.flat_features())
    # standardiser stats come from train; applying to train gives ~0 mean/unit std
    z = scaler.transform(trb.flat_features())
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(z.std(axis=0), 1.0, atol=1e-6)


def test_features_use_no_future_within_episode():
    # decision-point features at step j must be computable from day t0+j only;
    # here we sanity-check liability ttm is strictly decreasing across steps.
    ttm = BANK.features[:, :, BANK.feature_names.index("liab_ttm")]
    assert np.all(np.diff(ttm, axis=1) < 1e-9)
