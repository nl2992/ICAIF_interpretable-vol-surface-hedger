"""End-to-end experiment pipeline.

Builds a Monte-Carlo set of hedging episodes from many simulated market paths,
fits the deterministic baselines and the two learnable hedgers under a common
cost-adjusted CVaR objective, evaluates them on held-out paths, runs ablations,
and writes the figures, tables and markdown reports.

Reproducibility contract (one entry per run, recorded in ``manifest.json``):
``experiment_id``, ``dataset_version``, ``model_version``, ``seed``, ``split_id``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ivsh.data.market import MarketConfig, simulate_market
from ivsh.envs.hedging_env import EnvConfig, build_episode_bank, concat_banks
from ivsh.evaluation import report as R
from ivsh.evaluation.backtest import (
    GREEK_FEATURES,
    SURFACE_FEATURES,
    regime_metrics,
    run_baseline,
    run_policy,
)
from ivsh.evaluation.metrics import compute_metrics
from ivsh.evaluation.stats import paired_bootstrap_diff, wilcoxon_pnl
from ivsh.training.train import (
    TrainConfig,
    fit_blackbox,
    fit_prototype,
    make_standardizer,
)
from ivsh.utils.splits import select_features

DATASET_VERSION = "synthetic-regime-sv-jump-v1"
MODEL_VERSION = "proto-surface-hedger-v1"


def _md_table(df: pd.DataFrame, index: bool = True) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table (no tabulate dep)."""
    df = df.copy()
    headers = ([df.index.name or ""] if index else []) + [str(c) for c in df.columns]
    rows = []
    for key, row in df.iterrows():
        cells = ([str(key)] if index else []) + [
            (f"{v:.4g}" if isinstance(v, (int, float, np.floating)) else str(v)) for v in row
        ]
        rows.append(cells)
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(c) + " |" for c in rows]
    return "\n".join(out)


@dataclass
class ExperimentConfig:
    experiment_id: str = "ivsh_demo"
    seed: int = 7
    n_days: int = 500
    train_seeds: tuple[int, ...] = tuple(range(100, 124))
    val_seeds: tuple[int, ...] = tuple(range(150, 156))
    test_seeds: tuple[int, ...] = tuple(range(200, 212))
    env: EnvConfig = field(default_factory=EnvConfig)
    proto_train: TrainConfig = field(default_factory=lambda: TrainConfig(n_prototypes=8, l2=1e-3, max_iter=400))
    bb_train: TrainConfig = field(default_factory=lambda: TrainConfig(hidden=16, l2=3e-2, max_iter=400))
    n_prototypes_sweep: tuple[int, ...] = (4, 8, 16, 32)
    reports_dir: str = "reports"
    checkpoints_dir: str = "checkpoints"
    run_ablations: bool = True


def _seeds(spec) -> tuple[int, ...]:
    if isinstance(spec, dict):
        return tuple(range(spec["start"], spec["start"] + spec["count"]))
    return tuple(spec)


def load_config(path: str | Path) -> ExperimentConfig:
    """Build an :class:`ExperimentConfig` from a YAML file (missing keys default)."""
    import yaml

    raw = yaml.safe_load(Path(path).read_text()) or {}
    cfg = ExperimentConfig()
    for key in ("experiment_id", "seed", "n_days", "reports_dir", "checkpoints_dir", "run_ablations"):
        if key in raw:
            setattr(cfg, key, raw[key])
    for key in ("train_seeds", "val_seeds", "test_seeds"):
        if key in raw:
            setattr(cfg, key, _seeds(raw[key]))
    if "n_prototypes_sweep" in raw:
        cfg.n_prototypes_sweep = tuple(raw["n_prototypes_sweep"])
    if "env" in raw:
        cfg.env = EnvConfig(**raw["env"])
    if "proto_train" in raw:
        cfg.proto_train = TrainConfig(**raw["proto_train"])
    if "bb_train" in raw:
        cfg.bb_train = TrainConfig(**raw["bb_train"])
    return cfg


def _build_bank(seeds, n_days, env, market_cfg_seedbase=0):
    banks = [
        build_episode_bank(simulate_market(MarketConfig(n_days=n_days, seed=s)), env)
        for s in seeds
    ]
    return concat_banks(banks)


def build_data(cfg: ExperimentConfig) -> dict:
    """Stage 1 — simulate market paths and assemble train/val/test episode banks."""
    trb = _build_bank(cfg.train_seeds, cfg.n_days, cfg.env)
    vlb = _build_bank(cfg.val_seeds, cfg.n_days, cfg.env)
    teb = _build_bank(cfg.test_seeds, cfg.n_days, cfg.env)
    scaler = make_standardizer(trb)
    return {"train": trb, "val": vlb, "test": teb, "scaler": scaler}


def train_models(cfg: ExperimentConfig, data: dict) -> dict:
    """Stage 2 — fit the prototype and black-box hedgers."""
    trb, vlb, scaler = data["train"], data["val"], data["scaler"]
    proto, km, proto_hist = fit_prototype(trb, scaler, cfg.proto_train, val_bank=vlb)
    bb, bb_hist = fit_blackbox(trb, scaler, cfg.bb_train, val_bank=vlb)
    return {"proto": proto, "km": km, "blackbox": bb, "proto_hist": proto_hist, "bb_hist": bb_hist}


def run_experiment(cfg: ExperimentConfig | None = None) -> dict:
    """Run all four stages: data -> train -> evaluate -> report."""
    cfg = cfg or ExperimentConfig()
    data = build_data(cfg)
    models = train_models(cfg, data)
    return evaluate_and_report(cfg, data, models)


def evaluate_and_report(cfg: ExperimentConfig, data: dict, models: dict) -> dict:
    """Stages 3 & 4 — backtest on held-out paths, then write figures/tables/reports."""
    reports = Path(cfg.reports_dir)
    figs = reports / "figures"
    tables = reports / "tables"
    ckpt = Path(cfg.checkpoints_dir)
    for d in (figs, tables, ckpt):
        d.mkdir(parents=True, exist_ok=True)

    trb, vlb, teb, scaler = data["train"], data["val"], data["test"], data["scaler"]
    proto, km, bb = models["proto"], models["km"], models["blackbox"]
    proto_hist, bb_hist = models["proto_hist"], models["bb_hist"]

    # ---- evaluate on test ----
    results: dict[str, dict] = {}
    for name in ("unhedged", "delta", "delta_vega"):
        results[name] = run_baseline(teb, name)
    results["blackbox"] = run_policy(bb, teb, scaler)
    results["prototype"] = run_policy(proto, teb, scaler)

    pnl_by = {k: v["pnl"] for k, v in results.items()}
    turn_by = {k: v["turnover"] for k, v in results.items()}
    metrics_by = {k: compute_metrics(v, turn_by[k]) for k, v in pnl_by.items()}
    regime_by = {k: regime_metrics(v, teb.regime_start) for k, v in pnl_by.items()}

    # ---- statistical comparisons (prototype vs others) ----
    stats = {}
    for other in ("delta", "delta_vega", "blackbox"):
        stats[other] = {
            "cvar_boot": paired_bootstrap_diff(pnl_by["prototype"], pnl_by[other], stat="cvar", seed=cfg.seed),
            "util_boot": paired_bootstrap_diff(pnl_by["prototype"], pnl_by[other], stat="utility", seed=cfg.seed),
            "wilcoxon": wilcoxon_pnl(pnl_by["prototype"], pnl_by[other]),
        }

    # ---- figures ----
    R.plot_pnl_distributions(pnl_by, figs / "pnl_distributions.png")
    R.plot_cvar_comparison(metrics_by, figs / "cvar_comparison.png")
    R.plot_regime_tail(regime_by, figs / "tail_by_regime.png")
    R.plot_prototype_heatmaps(proto, scaler, figs / "prototype_surfaces.png")
    R.plot_prototype_actions(proto, figs / "prototype_actions.png")
    R.plot_embedding(scaler.transform(trb.flat_features()), km.labels, proto.prototypes, figs / "latent_embedding.png")
    share, entropy = R.plot_prototype_activation(proto, scaler, teb, figs / "prototype_activation.png")
    # example trade: worst-hit stress episode under delta hedge, to show the tail
    stress_idx = np.where(teb.regime_start == 1)[0]
    if len(stress_idx) == 0:
        stress_idx = np.arange(teb.n_episodes)
    worst = stress_idx[np.argmin(results["delta"]["pnl"][stress_idx])]
    top_protos = R.plot_example_trade(proto, scaler, teb, int(worst), figs / "example_trade.png")

    # ---- tables ----
    comp = R.comparison_table(pnl_by, turn_by)
    comp.to_csv(tables / "model_comparison.csv")
    catalogue = R.prototype_catalogue(proto, km, scaler, trb)
    catalogue.to_csv(tables / "prototype_catalogue.csv", index=False)

    # ---- ablations ----
    ablations = None
    if cfg.run_ablations:
        ablations = _run_ablations(trb, vlb, teb, scaler, cfg)
        ablations.to_csv(tables / "ablation_metrics.csv", index=False)

    # ---- checkpoint ----
    np.savez(
        ckpt / "proto_surface_hedger_best.npz",
        prototypes=proto.prototypes,
        raw_actions=proto.raw_actions,
        log_temp=proto.log_temp,
        action_scale=proto.action_scale,
        scaler_mean=scaler.mean,
        scaler_std=scaler.std,
        feature_names=np.array(trb.feature_names),
    )

    manifest = {
        "experiment_id": cfg.experiment_id,
        "dataset_version": DATASET_VERSION,
        "model_version": MODEL_VERSION,
        "seed": cfg.seed,
        "split_id": f"train{len(cfg.train_seeds)}-val{len(cfg.val_seeds)}-test{len(cfg.test_seeds)}",
        "n_train_episodes": int(trb.n_episodes),
        "n_test_episodes": int(teb.n_episodes),
        "proto_history": proto_hist,
        "bb_history": bb_hist,
        "prototype_entropy": float(entropy),
        "config": _config_to_dict(cfg),
    }
    (reports / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    # ---- markdown reports ----
    _write_final_report(reports, cfg, comp, metrics_by, regime_by, stats, manifest)
    _write_prototype_audit(reports, catalogue, share, entropy, top_protos, int(worst))
    if ablations is not None:
        _write_ablation_report(reports, ablations)

    return {
        "metrics": metrics_by,
        "comparison": comp,
        "stats": stats,
        "catalogue": catalogue,
        "ablations": ablations,
        "manifest": manifest,
        "proto": proto,
        "blackbox": bb,
        "scaler": scaler,
        "km": km,
        "banks": {"train": trb, "val": vlb, "test": teb},
    }


def _config_to_dict(cfg: ExperimentConfig) -> dict:
    d = asdict(cfg)
    return d


def _run_ablations(trb, vlb, teb, scaler, cfg) -> pd.DataFrame:
    rows = []

    def eval_proto(train_bank, val_bank, test_bank, sclr, tcfg, label):
        proto, _, _ = fit_prototype(train_bank, sclr, tcfg, val_bank=val_bank)
        pnl = run_policy(proto, test_bank, sclr)["pnl"]
        m = compute_metrics(pnl)
        rows.append({"ablation": label, "cvar_95": round(m["cvar_95"], 4), "cvar_99": round(m["cvar_99"], 4), "mean_pnl": round(m["mean_pnl"], 4)})

    # K sweep (full features)
    for k in cfg.n_prototypes_sweep:
        eval_proto(trb, vlb, teb, scaler, TrainConfig(n_prototypes=k, l2=cfg.proto_train.l2, max_iter=cfg.proto_train.max_iter), f"K={k}")

    # feature ablations
    for label, feats in (("greeks_only", GREEK_FEATURES), ("surface_only", SURFACE_FEATURES)):
        trb_f = select_features(trb, feats)
        vlb_f = select_features(vlb, feats)
        teb_f = select_features(teb, feats)
        sclr = make_standardizer(trb_f)
        eval_proto(trb_f, vlb_f, teb_f, sclr, TrainConfig(n_prototypes=cfg.proto_train.n_prototypes, l2=cfg.proto_train.l2, max_iter=cfg.proto_train.max_iter), f"features={label}")

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Markdown writers
# --------------------------------------------------------------------------- #
def _write_final_report(reports, cfg, comp, metrics_by, regime_by, stats, manifest) -> None:
    lines = [
        "# Final Report — Interpretable Volatility-Surface Hedger",
        "",
        f"**Experiment:** `{cfg.experiment_id}`  |  dataset `{manifest['dataset_version']}`  |  "
        f"model `{manifest['model_version']}`  |  seed `{cfg.seed}`  |  split `{manifest['split_id']}`",
        "",
        "## Research question",
        "> Can an interpretable prototype-based volatility-surface hedger reduce tail hedge losses "
        "versus delta / delta-vega hedging while staying competitive with a black-box deep hedging policy?",
        "",
        "## Setup",
        f"- Liability: short {cfg.env.notional} ATM {cfg.env.option_type}(s), "
        f"{cfg.env.liab_tenor_days}-day tenor, hedged daily to expiry.",
        f"- Hedge instruments: underlying + {cfg.env.hedge_tenor_days}-day ATM option.",
        f"- Costs: {cfg.env.underlying_cost_bps} bps underlying, {cfg.env.option_cost_bps} bps option (on traded notional).",
        f"- Market: regime-switching stochastic-vol + jumps, zero carry (martingale). "
        f"Trained on {manifest['n_train_episodes']} Monte-Carlo episodes, tested on "
        f"{manifest['n_test_episodes']} held-out-path episodes.",
        "- Objective: maximise E[P&L] − CVaR₉₅(loss) (Rockafellar–Uryasev), L2-regularised.",
        "",
        "## Model comparison (test set)",
        "",
        _md_table(comp),
        "",
        "Lower CVaR / worst / max-drawdown is better; higher utility is better.",
        "",
        "![CVaR comparison](figures/cvar_comparison.png)",
        "![P&L distribution](figures/pnl_distributions.png)",
        "",
        "## Tail loss by regime",
        "",
        "![Tail by regime](figures/tail_by_regime.png)",
        "",
    ]
    # regime table
    reg_rows = []
    for name, rm in regime_by.items():
        reg_rows.append(
            {
                "method": name,
                "calm_cvar95": round(rm.get("calm", {}).get("cvar_95", float("nan")), 3),
                "stress_cvar95": round(rm.get("stress", {}).get("cvar_95", float("nan")), 3),
            }
        )
    lines += [_md_table(pd.DataFrame(reg_rows).set_index("method")), ""]

    lines += ["## Statistical significance (prototype vs baselines)", ""]
    srows = []
    for other, s in stats.items():
        cb = s["cvar_boot"]
        srows.append(
            {
                "comparison": f"prototype − {other}",
                "Δcvar95": round(cb["diff"], 4),
                "cvar95 CI": f"[{cb['ci_low']:.3f}, {cb['ci_high']:.3f}]",
                "boot p": round(cb["p_two_sided"], 4),
                "wilcoxon p": round(s["wilcoxon"]["pvalue"], 4),
            }
        )
    lines += [
        _md_table(pd.DataFrame(srows).set_index("comparison")),
        "",
        "A negative Δcvar95 with a CI excluding 0 means the prototype hedger has a "
        "*significantly smaller* tail loss than the comparator.",
        "",
        "## Headline finding",
        _headline(metrics_by, stats),
        "",
        "See [prototype_audit_report.md](prototype_audit_report.md) for interpretability and "
        "[ablation_report.md](ablation_report.md) for ablations.",
        "",
    ]
    (reports / "final_report.md").write_text("\n".join(lines))


def _headline(metrics_by, stats) -> str:
    p = metrics_by["prototype"]["cvar_95"]
    dv = metrics_by["delta_vega"]["cvar_95"]
    d = metrics_by["delta"]["cvar_95"]
    bb = metrics_by["blackbox"]["cvar_95"]
    red_dv = 100 * (1 - p / dv) if dv else float("nan")
    red_d = 100 * (1 - p / d) if d else float("nan")
    vs_bb = "below" if p < bb else "above"
    return (
        f"The prototype surface hedger cuts CVaR₉₅ tail loss by **{red_d:.0f}%** versus delta "
        f"and **{red_dv:.0f}%** versus delta-vega, while landing {vs_bb} the black-box deep hedger "
        f"(prototype {p:.3f} vs black-box {bb:.3f}) — with a fully auditable, prototype-based decision trail."
    )


def _write_prototype_audit(reports, catalogue, share, entropy, top_protos, worst_ep) -> None:
    lines = [
        "# Prototype Audit Report",
        "",
        "Every hedge action is a similarity-weighted blend of a small set of learned "
        "volatility-surface prototypes, so each decision is traceable to named market regimes.",
        "",
        "## Prototype catalogue",
        "",
        _md_table(catalogue, index=False),
        "",
        "`iv_level / skew / curvature / term_slope` are the prototype's volatility-surface factors; "
        "`action_shares / action_option` are its learned hedge holdings.",
        "",
        f"Mean prototype-activation entropy: **{entropy:.2f} nats** "
        f"(0 = always one prototype, ln(K) = uniform).",
        "",
        "![Prototype surfaces](figures/prototype_surfaces.png)",
        "![Prototype actions](figures/prototype_actions.png)",
        "![Latent embedding](figures/latent_embedding.png)",
        "![Prototype activation](figures/prototype_activation.png)",
        "",
        "## Example trade audit",
        "",
        f"Test episode {worst_ep} (a stressed path where the naive delta hedge suffers a large "
        "loss). The panels show the spot path, the prototype hedge holdings, the prototype "
        "activation weights through time, and the cumulative hedged P&L.",
        "",
        "![Example trade](figures/example_trade.png)",
        "",
        f"Dominant prototype along this path: {np.bincount(top_protos).argmax()}.",
        "",
    ]
    (reports / "prototype_audit_report.md").write_text("\n".join(lines))


def _write_ablation_report(reports, ablations) -> None:
    lines = [
        "# Ablation Report",
        "",
        "All rows are prototype-hedger variants evaluated on the held-out test set "
        "(lower CVaR is better).",
        "",
        _md_table(ablations, index=False),
        "",
        "- **K sweep** shows sensitivity to the number of prototypes (interpretability vs capacity).",
        "- **features=greeks_only** removes the volatility surface (scalar Greeks only); "
        "**features=surface_only** removes the book Greeks. The gap to the full model quantifies "
        "the value of surface-regime information.",
        "",
    ]
    (reports / "ablation_report.md").write_text("\n".join(lines))
