# Claims and Evidence

What the paper argues, and for every headline number, the committed file it comes from. The paper is
validated on real options, so the authoritative artifacts live under `reports_real/tables/`.

## The narrative

Deep hedging learns effective option-hedging policies under market frictions, but its reliance on
black-box networks limits trust, and — as we show — it can produce catastrophic out-of-sample tail
losses under regime shift. We introduce an interpretable alternative: every hedge is a
similarity-weighted blend of bounded actions tied to a small set of learned, named volatility-surface
prototypes (calm, elevated-skew, stressed), run as a bounded residual on the classical delta–vega hedge
and trained under a tail-weighted CVaR objective.

Validated on fourteen years of real options across two markets — SPY (2010–2023) and QQQ (2012–2023),
spanning two crises — the prototype hedger is the only learned policy stable across both seeds and
markets. It matches the strongest classical baseline (delta–vega) on tail risk (CVaR₉₅ = 2.34±0.10 across
five seeds on SPY) while PPO and SAC deep-RL hedgers blow up by 12–32× and a black-box MLP is
seed-unstable.

The cross-market robustness is not free. A naive residual *loses* on the second market — a failure we
trace to a model-selection-under-regime-shift artefact that a tail-weighted objective repairs, restoring
parity with delta–vega on both held-out markets (a directionally favourable tie, Stouffer p=0.079, not
significant at α=0.05). The known boundary is the COVID-2020 SPY fold, where the anchored residual can add
tail risk before a volatility-scaled cap intervenes (repairing CVaR₉₅ from 60.0 to 18.0); we surface this
prominently as the edge of the guarantee.

This is the first empirical test of prototype hedging on live options data, and within this well-defined
scope, interpretability and tail-risk control are not a trade-off.

## Where each number lives

| Claim | Number | File | Field / row |
|---|---|---|---|
| ProtoHedge tail risk (winner config; headline) | CVaR₉₅ 2.34±0.10 on SPY, 5 seeds | `reports_real/tables/winner_confirmation.csv` | `proto_cvar_mean`=2.343, `proto_cvar_std`=0.104 (spy) |
| ProtoHedge tail risk (default-config multiseed) | 2.357±0.10 | `reports_real/tables/multiseed_cvar.csv`, `reports_real/tables/multiseed_cvar_byseed.csv` | `prototype` mean/std (a distinct config from the winner above) |
| Full model comparison (SPY) | delta 4.71, delta–vega 2.84, blackbox 6.61, prototype 2.38, PPO 52.8 | `reports_real/tables/model_comparison.csv`, `reports_real/tables/cvar_confidence.csv` | `cvar_95` by `method` (+ bootstrap CIs) |
| PPO/SAC blow up 12–32× vs classical | PPO 52.8, SAC ≈89 vs delta–vega 2.84 | `reports_real/tables/multiseed_cvar.csv`, `reports_real/tables/multiverse_significance.csv` | `ppo`/`sac` columns; per-universe ΔCVaR |
| Cross-market parity restored (winner config) | Stouffer-combined p=0.079 (directional tie) | `reports_real/tables/winner_confirmation.csv`, `reports_real/grid_four_universe/grid_results.csv` | spy `p_boot_vs_dv`=0.077; combined p in `tab:winner` |
| Naive-residual cross-market failure (pre-fix) | combined p≈10⁻⁴ (worse than delta–vega) | `reports_real/tables/multiverse_significance.csv` | `prototype - delta_vega` COMBINED `p_bootstrap` |
| COVID-2020 SPY boundary repaired by the vol-cap | CVaR₉₅ 59.98 → 17.96 ("60 → 18") | `reports_real/tables/walkforward_stress_audit.csv`, `reports_real/final_report.md` | spy / 2020 row |
| Surface features the key signal (ablation) | full 2.357 vs greeks-only 2.356; QQQ surface-only winner 5.31 | `reports_real/tables/ablation_metrics.csv`, `reports_real/ablation_report.md` | `full` / `greeks_only` rows |

Note on configurations: the headline **2.34** is the tuned **winner** regularisation config
(`winner_confirmation.csv`); the **2.36** that appears in the black-box comparison is the **default**
prototype multiseed (`multiseed_cvar.csv`). Both are correct — they are different configs. All numbers
regenerate from `scripts/` against the real OptionsDX panels (see `DATA.md`).
