# Robust PPO Tuning Results

## Environment

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB
- PyTorch: 2.11.0+cu128
- Stable-Baselines3: 2.8.0
- Gymnasium: 1.2.3
- CUDA available to PyTorch: yes

PPO uses an MLP policy. Stable-Baselines3 warns that this policy class is often
faster on CPU than GPU, but the CUDA build was installed and used successfully.

## Search

Command:

```text
python scripts/tune_ppo_robust.py --n-configs 9 --seeds 3 --universe spy --timesteps 60000
```

The search covered:

- learning rates: 1e-4, 3e-4, 1e-3
- bounded residual action scales: 0.75, 1.5, 2.5
- seeds: 7, 8, 9
- 60,000 environment steps per run
- hard action clipping through the shared bounded residual action space

All 27 runs are in `tables/tuned_ppo_spy.csv`; grouped results are in
`tables/tuned_ppo_spy_summary.csv`.

## Validation-Selected Result

The best configuration by mean validation CVaR95 was:

- learning rate: 1e-4
- residual action scale: 0.75
- validation CVaR95: 27.75 +/- 10.01
- test CVaR95: 23.81 +/- 3.05
- best individual-seed test CVaR95: 20.31

Reference SPY results:

- tail-weighted prototype CVaR95: 2.34
- delta-vega CVaR95: 2.84

Thus, validation-selected tuned PPO has about 10.2x the prototype's tail loss and
8.4x the delta-vega tail loss. Even the best individual PPO seed has about 8.7x
the prototype's CVaR95.

## Conclusion

Position limits and a learning-rate/action-bound search do not repair PPO's
tail-risk failure. The failure is stable across seeds and worsens monotonically
as the allowed residual action range increases. This supports the claim that the
prototype policy's bounded, CVaR-trained residual structure is materially more
robust than a mean-reward PPO policy in this environment.
