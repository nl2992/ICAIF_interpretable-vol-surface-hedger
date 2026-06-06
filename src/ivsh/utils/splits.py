"""Chronological, leak-free splitting of hedging episodes.

Episodes are ordered by their *start day* and partitioned into train / validation
/ test by time. To prevent overlap leakage between adjacent splits we drop a
purge gap (at least one episode horizon) at each boundary so that no training
episode shares calendar days with a test episode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ivsh.envs.hedging_env import EpisodeBank


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def chronological_split(
    bank: EpisodeBank,
    train_frac: float = 0.6,
    val_frac: float = 0.15,
    purge_days: int | None = None,
) -> Split:
    order = np.argsort(bank.start_days)
    starts = bank.start_days[order]
    n = len(order)
    purge = bank.horizon if purge_days is None else purge_days

    train_end_day = starts[int(n * train_frac)]
    val_end_day = starts[int(n * (train_frac + val_frac))]

    train = order[starts < train_end_day - purge]
    val = order[(starts >= train_end_day) & (starts < val_end_day - purge)]
    test = order[starts >= val_end_day]
    return Split(train=np.sort(train), val=np.sort(val), test=np.sort(test))


def select_features(bank: EpisodeBank, names: tuple[str, ...]) -> EpisodeBank:
    """Return a bank exposing only the named state features (for ablations)."""
    idx = [bank.feature_names.index(n) for n in names]
    new = subset(bank, np.arange(bank.n_episodes))
    new.features = bank.features[:, :, idx]
    new.feature_names = tuple(names)
    return new


def subset(bank: EpisodeBank, idx: np.ndarray) -> EpisodeBank:
    """Return a new bank containing only the selected episodes."""
    return EpisodeBank(
        config=bank.config,
        start_days=bank.start_days[idx],
        spot=bank.spot[idx],
        v_liab=bank.v_liab[idx],
        o_hedge=bank.o_hedge[idx],
        features=bank.features[idx],
        greeks={k: v[idx] for k, v in bank.greeks.items()},
        regime_start=bank.regime_start[idx],
        regime_frac_stress=bank.regime_frac_stress[idx],
        feature_names=bank.feature_names,
    )
