# Why the initial anchored-residual prototype failed (and what fixes it)

The v1 prototype anchored a learned residual on the delta-vega hedge. On QQQ the residual *added* tail risk. This note quantifies the mechanism.

## Residual decomposition and validation->test shift

### SPY

- test CVaR95: prototype **2.383** vs delta-vega **2.845** (excess -0.462).
- mean |residual| over all episodes: 0.034; over the CVaR95 tail episodes: **0.036** (1.1x larger).
- in the tail, the residual leaves P&L *worse than the delta-vega base* in **67%** of tail episodes.
- mean realised vol: validation 0.174 vs test **0.163** (0.94x) — the model is selected on a calmer regime than it is tested on.

### QQQ

- test CVaR95: prototype **9.061** vs delta-vega **6.120** (excess +2.941).
- mean |residual| over all episodes: 0.216; over the CVaR95 tail episodes: **0.224** (1.0x larger).
- in the tail, the residual leaves P&L *worse than the delta-vega base* in **100%** of tail episodes.
- mean realised vol: validation 0.222 vs test **0.218** (0.98x) — the model is selected on a calmer regime than it is tested on.

## What fixed it (per-knob, validation excess over delta-vega)

- **residual_l2**: 0.0->-0.676, 1.0->-0.170, 10.0->-0.652, 100.0->-0.677, 1000.0->-0.052
- **vol_floor**: 0.1->-0.975, 0.25->-0.853, 0.5->-0.828
- **action_scale**: 0.5->-0.736, 1.0->-0.759, 1.5->-0.518
- **cvar_weight**: 1.0->-0.588, 3.0->-0.855, 10.0->-0.592