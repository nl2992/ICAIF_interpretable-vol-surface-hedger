from __future__ import annotations

import numpy as np

from ivsh.envs.hedging_env import EnvConfig
from ivsh.pipeline import ExperimentConfig, run_experiment
from ivsh.training.train import TrainConfig


def _tiny_cfg(tmp_path) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="test",
        n_days=150,
        train_seeds=(100, 101, 102),
        val_seeds=(150,),
        test_seeds=(200, 201),
        env=EnvConfig(),
        proto_train=TrainConfig(n_prototypes=4, max_iter=30),
        bb_train=TrainConfig(hidden=8, l2=3e-2, max_iter=30),
        run_ablations=False,
        reports_dir=str(tmp_path / "reports"),
        checkpoints_dir=str(tmp_path / "checkpoints"),
    )


def test_pipeline_runs_and_writes_reports(tmp_path):
    res = run_experiment(_tiny_cfg(tmp_path))
    reports = tmp_path / "reports"
    assert (reports / "final_report.md").exists()
    assert (reports / "prototype_audit_report.md").exists()
    assert (reports / "manifest.json").exists()
    assert (reports / "figures" / "prototype_surfaces.png").exists()
    assert (reports / "tables" / "model_comparison.csv").exists()
    assert (tmp_path / "checkpoints" / "proto_surface_hedger_best.npz").exists()
    # every method has finite metrics on the held-out set
    for name in ("unhedged", "delta", "delta_vega", "blackbox", "prototype"):
        assert np.isfinite(res["metrics"][name]["cvar_95"])


def test_learned_hedger_beats_unhedged(tmp_path):
    res = run_experiment(_tiny_cfg(tmp_path))
    m = res["metrics"]
    # hedging must reduce tail loss massively versus doing nothing
    assert m["prototype"]["cvar_95"] < m["unhedged"]["cvar_95"]
    assert m["delta"]["cvar_95"] < m["unhedged"]["cvar_95"]
