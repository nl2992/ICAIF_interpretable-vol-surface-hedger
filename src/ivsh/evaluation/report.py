"""Figures and tables for the model-comparison and interpretability reports.

All plotting uses a non-interactive backend so the pipeline runs headless.
Prototype surfaces are reconstructed analytically from the four surface factors
(level, skew, curvature, term slope) stored in each prototype's feature vector,
so every prototype maps to a concrete, human-readable volatility surface.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ivsh.data.market import TAU0  # noqa: E402
from ivsh.evaluation.metrics import compute_metrics  # noqa: E402
from ivsh.training.objective import cvar_from_pnl  # noqa: E402
from ivsh.viz import ACTION_COLORS, SEQ_CMAP, apply_theme, color as _color  # noqa: E402

apply_theme()  # one colour + style theme across all figures


# --------------------------------------------------------------------------- #
# Comparison figures
# --------------------------------------------------------------------------- #
def plot_pnl_distributions(pnl_by_method: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, pnl in pnl_by_method.items():
        lo, hi = np.quantile(pnl, [0.005, 0.995])
        grid = np.linspace(lo, hi, 200)
        hist, edges = np.histogram(np.clip(pnl, lo, hi), bins=60, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, hist, label=name, color=_color(name), lw=1.8)
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("episode hedging P&L")
    ax.set_ylabel("density")
    ax.set_title("Hedging P&L distribution by policy (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_cvar_comparison(metrics_by_method: dict[str, dict], path: Path) -> None:
    names = list(metrics_by_method)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.38
    ax.bar(x - w / 2, [metrics_by_method[n]["cvar_95"] for n in names], w, label="CVaR 95%", color="#4c72b0")
    ax.bar(x + w / 2, [metrics_by_method[n]["cvar_99"] for n in names], w, label="CVaR 99%", color="#c44e52")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("tail loss (lower = better)")
    ax.set_title("Tail hedge loss by policy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_regime_tail(regime_by_method: dict[str, dict], path: Path) -> None:
    names = list(regime_by_method)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.38
    calm = [regime_by_method[n].get("calm", {}).get("cvar_95", np.nan) for n in names]
    stress = [regime_by_method[n].get("stress", {}).get("cvar_95", np.nan) for n in names]
    ax.bar(x - w / 2, calm, w, label="calm start", color="#55a868")
    ax.bar(x + w / 2, stress, w, label="stress start", color="#c44e52")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("CVaR 95% tail loss")
    ax.set_title("Tail hedge loss by starting regime")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Interpretability figures
# --------------------------------------------------------------------------- #
def _proto_surface_params(proto, scaler) -> np.ndarray:
    """Un-standardise prototype centres to (level, skew, curv, slope) columns."""
    centers = proto.prototypes  # standardised
    raw = centers * scaler.std + scaler.mean
    return raw[:, :4]  # surf_level, surf_skew, surf_curv, surf_slope


def plot_prototype_heatmaps(proto, scaler, path: Path) -> None:
    params = _proto_surface_params(proto, scaler)
    k = params.shape[0]
    moneyness = np.linspace(0.8, 1.2, 25)
    tenor_days = np.array([7, 14, 30, 60, 90, 180])
    logm = np.log(moneyness)
    ncol = min(4, k)
    nrow = int(np.ceil(k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.6 * nrow), squeeze=False)
    vmin, vmax = np.inf, -np.inf
    grids = []
    for j in range(k):
        level, skew, curv, slope = params[j]
        tau = tenor_days / 252.0
        iv = (
            level
            + skew * logm[None, :]
            + curv * (logm[None, :] ** 2)
            + slope * np.log(tau / TAU0)[:, None]
        )
        iv = np.maximum(iv, 0.02)
        grids.append(iv)
        vmin, vmax = min(vmin, iv.min()), max(vmax, iv.max())
    for j in range(nrow * ncol):
        ax = axes[j // ncol][j % ncol]
        if j >= k:
            ax.axis("off")
            continue
        im = ax.imshow(grids[j], aspect="auto", origin="lower", cmap=SEQ_CMAP, vmin=vmin, vmax=vmax)
        ax.set_title(f"prototype {j}", fontsize=9)
        ax.set_xticks([0, 12, 24])
        ax.set_xticklabels(["0.8", "1.0", "1.2"], fontsize=7)
        ax.set_yticks(range(len(tenor_days)))
        ax.set_yticklabels(tenor_days, fontsize=7)
        if j % ncol == 0:
            ax.set_ylabel("tenor (d)", fontsize=8)
    fig.suptitle("Prototype implied-volatility surfaces", fontsize=12)
    fig.colorbar(im, ax=axes, shrink=0.6, label="implied vol")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_prototype_actions(proto, path: Path) -> None:
    actions = proto.actions
    k = actions.shape[0]
    x = np.arange(k)
    fig, ax = plt.subplots(figsize=(8, 4.0))
    w = 0.38
    ax.bar(x - w / 2, actions[:, 0], w, label="underlying shares", color="#4c72b0")
    ax.bar(x + w / 2, actions[:, 1], w, label="hedge-option units", color=ACTION_COLORS["option_units"])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{j}" for j in range(k)])
    ax.set_ylabel("hedge holding")
    ax.set_title("Learned hedge action per prototype")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_embedding(features_std: np.ndarray, labels: np.ndarray, centers: np.ndarray, path: Path) -> None:
    # PCA via SVD.
    x = features_std - features_std.mean(axis=0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    proj = x @ vt[:2].T
    cproj = (centers - features_std.mean(axis=0)) @ vt[:2].T
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sample = np.random.default_rng(0).choice(len(proj), min(4000, len(proj)), replace=False)
    ax.scatter(proj[sample, 0], proj[sample, 1], c=labels[sample], cmap="tab10", s=5, alpha=0.4)
    ax.scatter(cproj[:, 0], cproj[:, 1], c="k", marker="X", s=120)
    for j, (px, py) in enumerate(cproj):
        ax.annotate(f"P{j}", (px, py), fontsize=9, fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Latent state embedding with prototypes")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_example_trade(proto, scaler, bank, episode: int, path: Path, anchor: bool = False) -> None:
    x = scaler.transform(bank.features[episode])  # [L, F]
    weights = proto.weights(x)  # [L, K]
    holdings = proto.predict_holdings(x)  # [L, 2]
    if anchor:
        from ivsh.baselines.policies import delta_vega_hedge

        holdings = holdings + delta_vega_hedge(bank)[episode]
    steps = np.arange(bank.horizon)
    spot = bank.spot[episode, :-1]
    # cumulative P&L of this single episode under the prototype hedge
    h3 = holdings[None]
    pnl_path = _episode_cum_pnl(bank, episode, holdings)

    fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(steps, spot, color="k")
    axes[0].set_ylabel("spot")
    axes[0].set_title(f"Example trade audit — test episode {episode}")
    axes[1].plot(steps, holdings[:, 0], label="shares", color="#4c72b0")
    axes[1].plot(steps, holdings[:, 1], label="option units", color=ACTION_COLORS["option_units"])
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_ylabel("holding")
    axes[1].legend(fontsize=8)
    top = weights.argmax(axis=1)
    for j in range(proto.k):
        axes[2].plot(steps, weights[:, j], label=f"P{j}", lw=1.2)
    axes[2].set_ylabel("prototype weight")
    axes[2].legend(fontsize=7, ncol=4)
    axes[3].plot(steps, pnl_path, color="#8172b3")
    axes[3].axhline(0, color="k", lw=0.5)
    axes[3].set_ylabel("cumulative P&L")
    axes[3].set_xlabel("hedge step (day)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return top


def _episode_cum_pnl(bank, e: int, holdings: np.ndarray) -> np.ndarray:
    cfg = bank.config
    q_s, q_o = holdings[:, 0], holdings[:, 1]
    ds = np.diff(bank.spot[e])
    do = np.diff(bank.o_hedge[e])
    dv = np.diff(bank.v_liab[e])
    inc = -cfg.notional * dv + q_s * ds + q_o * do
    prev_s = np.concatenate([[0.0], q_s[:-1]])
    prev_o = np.concatenate([[0.0], q_o[:-1]])
    cost = np.abs(q_s - prev_s) * bank.spot[e, :-1] * (cfg.underlying_cost_bps / 1e4)
    cost += np.abs(q_o - prev_o) * bank.o_hedge[e, :-1] * (cfg.option_cost_bps / 1e4)
    inc -= cost
    inc[-1] -= np.abs(q_s[-1]) * bank.spot[e, -1] * (cfg.underlying_cost_bps / 1e4)
    inc[-1] -= np.abs(q_o[-1]) * bank.o_hedge[e, -1] * (cfg.option_cost_bps / 1e4)
    return np.cumsum(inc)


def plot_prototype_activation(proto, scaler, bank, path: Path) -> None:
    x = scaler.transform(bank.flat_features())
    w = proto.weights(x)
    share = w.mean(axis=0)
    entropy = float(-(w * np.log(w + 1e-12)).sum(axis=1).mean())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(np.arange(proto.k), share, color="#8172b3")
    ax.set_xlabel("prototype")
    ax.set_ylabel("mean activation weight")
    ax.set_title(f"Prototype activation (mean entropy = {entropy:.2f} nats)")
    ax.set_xticks(np.arange(proto.k))
    ax.set_xticklabels([f"P{j}" for j in range(proto.k)])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return share, entropy


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def prototype_catalogue(proto, km, scaler, bank) -> pd.DataFrame:
    raw = proto.prototypes * scaler.std + scaler.mean
    counts = np.bincount(km.labels, minlength=proto.k)
    stress_point = np.repeat(bank.regime_start, bank.horizon)
    frac_stress = np.array(
        [stress_point[km.labels == j].mean() if counts[j] else np.nan for j in range(proto.k)]
    )
    actions = proto.actions
    rows = []
    for j in range(proto.k):
        rows.append(
            {
                "prototype": f"P{j}",
                "n_assigned": int(counts[j]),
                "share_pct": round(100 * counts[j] / counts.sum(), 1),
                "frac_stress": round(float(frac_stress[j]), 2),
                "iv_level": round(float(raw[j, 0]), 3),
                "skew": round(float(raw[j, 1]), 3),
                "curvature": round(float(raw[j, 2]), 3),
                "term_slope": round(float(raw[j, 3]), 3),
                "action_shares": round(float(actions[j, 0]), 3),
                "action_option": round(float(actions[j, 1]), 3),
            }
        )
    return pd.DataFrame(rows)


def prototype_date_annotations(km, trb, day_to_date) -> pd.DataFrame:
    """Map each prototype to the historical dates it activates on (real data).

    Uses the cluster assignment of every training decision point; reports the
    medoid's calendar date, the dominant year-month, and the count.
    """
    day_to_date = pd.to_datetime(np.asarray(day_to_date))
    L = trb.horizon
    n_rows = len(km.labels)
    ep = np.arange(n_rows) // L
    step = np.arange(n_rows) % L
    day_idx = trb.start_days[ep] + step
    dates = day_to_date[day_idx]
    k = km.centers.shape[0]
    rows = []
    for j in range(k):
        mask = km.labels == j
        d_j = dates[mask]
        if len(d_j) == 0:
            rows.append({"prototype": f"P{j}", "example_date": "", "top_period": "", "n_dates": 0})
            continue
        med_day = trb.start_days[km.medoid_idx[j] // L] + km.medoid_idx[j] % L
        example = pd.Timestamp(day_to_date[med_day]).strftime("%Y-%m-%d")
        periods = pd.Series(d_j).dt.strftime("%Y-%m")
        top = periods.value_counts().idxmax()
        rows.append(
            {"prototype": f"P{j}", "example_date": example, "top_period": top, "n_dates": int(mask.sum())}
        )
    return pd.DataFrame(rows)


def comparison_table(pnl_by_method: dict[str, np.ndarray], turnover_by_method: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, pnl in pnl_by_method.items():
        m = compute_metrics(pnl, turnover_by_method.get(name))
        m["method"] = name
        rows.append(m)
    df = pd.DataFrame(rows).set_index("method")
    cols = ["mean_pnl", "median_pnl", "std_pnl", "var_95", "cvar_95", "cvar_99", "worst", "max_drawdown", "turnover", "utility"]
    return df[[c for c in cols if c in df.columns]].round(4)


# --------------------------------------------------------------------------- #
# Paper figures (architecture, time series, significance)
# --------------------------------------------------------------------------- #
def plot_architecture(path: Path) -> None:
    """Schematic of the prototype surface-hedger pipeline."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    steps = [
        ("Volatility-surface state", "moneyness x tenor IV grid + book/hedge Greeks"),
        ("Standardised features", "level, skew, curvature, term-slope, RV, Greeks (train-fit scaler)"),
        ("Latent state  z", "leak-free, chronological"),
        ("Prototype similarity", "softmax(-||z - p_k||^2 / T),  p_k = k-means medoids"),
        ("Bounded prototype actions  a_k", "tanh-bounded; interpretable per-regime hedge"),
        ("Hedge action = Σ_k w_k a_k", "(+ delta-vega base when anchored)"),
        ("Hedging environment", "daily rebalance, transaction costs"),
        ("Cost-adjusted CVaR objective", "max E[PnL] - λ·CVaR_α(loss)  (Rockafellar–Uryasev)"),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(steps) * 1.25)
    ax.axis("off")
    y = len(steps) * 1.25 - 0.7
    centers = []
    for title, sub in steps:
        box = FancyBboxPatch(
            (1.2, y - 0.42), 7.6, 0.84, boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=1.4, edgecolor="#33415c", facecolor="#eef2ff",
        )
        ax.add_patch(box)
        ax.text(5.0, y + 0.12, title, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(5.0, y - 0.2, sub, ha="center", va="center", fontsize=8, color="#444")
        centers.append(y)
        y -= 1.25
    for y0, y1 in zip(centers[:-1], centers[1:]):
        ax.add_patch(FancyArrowPatch((5.0, y0 - 0.44), (5.0, y1 + 0.44),
                                     arrowstyle="-|>", mutation_scale=16, color="#33415c", linewidth=1.4))
    ax.text(5.0, len(steps) * 1.25 - 0.15, "Interpretable Prototype Volatility-Surface Hedger",
            ha="center", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_cumulative_pnl(pnl_by_method, order, dates, path: Path) -> None:
    """Cumulative hedged P&L over the test set (episodes ordered by start day)."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = dates[order] if dates is not None else np.arange(len(order))
    for name, pnl in pnl_by_method.items():
        ax.plot(x, np.cumsum(np.asarray(pnl)[order]), label=name, color=_color(name), lw=1.6)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_ylabel("cumulative hedged P&L")
    ax.set_xlabel("episode start date" if dates is not None else "test episode (chronological)")
    ax.set_title("Cumulative hedged P&L (test set)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_activation_timeline(weights_ep, order, regime, dates, path: Path) -> None:
    """Stacked prototype-activation weights over the test period."""
    w = weights_ep[order]  # [E, K]
    x = dates[order] if dates is not None else np.arange(len(order))
    k = w.shape[1]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.stackplot(x, *[w[:, j] for j in range(k)], labels=[f"P{j}" for j in range(k)],
                 colors=plt.cm.tab10(np.linspace(0, 1, k)))
    # shade stressed episodes lightly along the top
    reg = np.asarray(regime)[order]
    ax.fill_between(x, 1.0, 1.03, where=reg == 1, color="red", alpha=0.5, step="mid")
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("prototype activation weight")
    ax.set_xlabel("episode start date" if dates is not None else "test episode (chronological)")
    ax.set_title("Prototype activation over time (red band = stressed start)")
    ax.legend(fontsize=7, ncol=k, loc="lower center")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def cvar_confidence(pnl_by_method, alpha=0.95, n_boot=2000, ci=0.95, seed=7) -> pd.DataFrame:
    """Per-method CVaR with bootstrap confidence intervals."""
    rng = np.random.default_rng(seed)
    rows = []
    for name, pnl in pnl_by_method.items():
        pnl = np.asarray(pnl)
        n = len(pnl)
        boot = np.array([cvar_from_pnl(pnl[rng.integers(0, n, n)], alpha) for _ in range(n_boot)])
        rows.append({
            "method": name,
            "cvar": cvar_from_pnl(pnl, alpha),
            "ci_low": float(np.quantile(boot, (1 - ci) / 2)),
            "ci_high": float(np.quantile(boot, 1 - (1 - ci) / 2)),
        })
    return pd.DataFrame(rows).set_index("method")


def plot_cvar_ci(pnl_by_method, path: Path, alpha=0.95, seed=7) -> pd.DataFrame:
    df = cvar_confidence(pnl_by_method, alpha=alpha, seed=seed)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = list(df.index)
    y = df["cvar"].to_numpy()
    lo = y - df["ci_low"].to_numpy()
    hi = df["ci_high"].to_numpy() - y
    ax.bar(names, y, color=[_color(n) for n in names],
           yerr=[lo, hi], capsize=5, edgecolor="k", linewidth=0.5)
    ax.set_ylabel(f"CVaR{int(alpha*100)} tail loss (95% bootstrap CI)")
    ax.set_title("Tail loss with bootstrap confidence intervals")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return df
