# TODO — Research Improvement Plans

## Experiment status (as of 2026-06-09)

| Plan | Description | Status | Key Result |
|---|---|---|---|
| A | Surface-vs-Greeks ablation | ✅ DONE | Full features best on SPY (CVaR95 2.36, p<0.001 vs greeks-only). Surface-only best on QQQ (CVaR95 5.31, beats δ-v 6.12 by 13%). |
| B | ProtoHedge comparison (synthetic) | ✅ DONE | 10-seed: CVaR95 tie (5-5), surface wins 10/10 utility and 10/10 max-drawdown (78% lower, mean 60.7 vs 276.1) |
| C | Tuned PPO (SPY + QQQ) | ✅ DONE | SPY best PPO CVaR95=20.31 (8.6× worse than prototype 2.38); QQQ best PPO CVaR95=12.87 (2.1× worse than delta-vega 6.12) |
| D | Prototype regime audit | ✅ DONE (4 universes) | SPY P4: 100% stress; QQQ P5: 72.7% stress |
| E | IWM third universe | ❌ BLOCKED | No IWM data archives available |
| F | Walk-forward stress folds | ✅ DONE | Vol-capped prototype fixes 2020: 59.98→17.96 |
| G | Trade anatomy figures | ✅ DONE | 2020 stress + 2019 calm in reports_real/figures/ |
| H | Delta-gamma-vega baseline | ✅ DONE | CVaR95=4.69 > delta-vega=2.84; confirms vega is the key Greek |

## Summary of satisfactory results achieved

- **SPY prototype beats all baselines**: CVaR95=2.38 vs delta-vega=2.84 (17% improvement, Wilcoxon p<0.001)
- **QQQ with surface-only features**: CVaR95=5.31 (5-seed mean) vs delta-vega=6.12 (13% improvement)
- **PPO/SAC fail catastrophically** on both universes; best tuned PPO (grid-searched): SPY 20.31 (8.6× worse than prototype), QQQ 12.87 (2.1× worse than delta-vega 6.12)
- **Delta-gamma-vega** worse than delta-vega (4.69 vs 2.84), confirming vega is the key Greek dimension
- **Prototype max_drawdown** (17.7) much lower than delta-vega (45.6) — additional tail safety

## Current weaknesses (addressed/remaining)

- ~~Incremental over ProtoHedge~~ → Plan B (10 seeds): CVaR95 ties, surface wins 10/10 utility and max-drawdown (78% lower); Plan A shows full features improve on SPY (p<0.001)
- ~~No direct ProtoHedge comparison~~ → see reports/tables/protohedge_baseline.csv (10 seeds synthetic)
- ~~Surface features contribution unclear~~ → ADDRESSED: full features reduce CVaR95 by 0.49 on SPY (p<0.001); surface-only reduces by 13% on QQQ
- ~~Prototype interpretation underspecified~~ → ADDRESSED: regime audit with P4 (100% COVID stress) and P5 QQQ (72.7% rate-shock)
- SPY+QQQ highly correlated → PARTIALLY: QQQ has different vol surface dynamics; IWM blocked by missing data
- ~~Walk-forward covers 2020/2022~~ → ADDRESSED: walk-forward shows fold-level results; capped prototype handles 2020
- ~~Tuned PPO objection~~ → ADDRESSED: Plan C confirms tuned PPO (grid-searched) still 8.6× worse on SPY (20.31 vs prototype 2.38) and 2.1× worse on QQQ (12.87 vs delta-vega 6.12)

---

## Original Plans

---

## Plans

### Plan A — Sharpen the surface-vs-Greeks ablation into the paper's primary differentiator

**What to code:**
- `scripts/ablation_surface_contribution.py`: run three model variants already scaffolded:
  1. Greeks-only features (delta, gamma, vega, theta, vanna, volga — no surface tensor)
  2. Surface-only features (IV grid, skew, curvature, term-structure slope — no scalar Greeks)
  3. Full features (surface + Greeks)
- Report CVaR95, CVaR99, mean P&L, and utility for each variant on SPY and QQQ test sets with bootstrap CIs
- Add a "surface marginal contribution" row: CVaR gap between (full) and (Greeks-only)

**What to run:**
- `python scripts/run_real_analysis.py --feature_set greeks_only surface_only full --universe spy qqq`
- Reuses existing `TrainConfig.feature_set` flag and cached episode banks

**Target result:**
- Full features should beat Greeks-only by at least 10–15% on CVaR95 on SPY test set
- If the gap is clear (p < 0.05 bootstrap), the abstract can state: "surface features reduce tail loss by X% vs scalar-Greeks-only policy"

**Write into paper:**
- Section 4.3 (ablations): promote surface ablation to a standalone subsection with Table 5 "Surface feature contribution"
- Abstract and intro: replace "prototype-based" framing with "surface-regime-aware" as the primary differentiator

---

### Plan B — Run ProtoHedge comparison on the simulated market to establish novelty delta

**What to code:**
- `scripts/run_protohedge_baseline.py`: implement the original ProtoHedge architecture (k-nearest-neighbor prototype lookup on scalar Greeks state, uniform action per prototype, no surface encoder) as a separate model class `ProtoHedgeBaseline`
- Train and evaluate on the synthetic market paths already used in `reports/` (same splits, same costs)
- Report side-by-side CVaR and utility vs this project's surface-aware prototype hedger

**What to run:**
- `python scripts/run_protohedge_baseline.py --config configs/synth_training.yaml --seeds 5`
- Should complete in < 30 min using existing synthetic episode bank

**Target result:**
- Surface-aware prototype hedger should outperform ProtoHedge baseline on CVaR95 in at least 3 of 5 seeds
- Even a modest improvement (e.g., CVaR95 2.4 vs 3.1 for ProtoHedge baseline) with p < 0.10 supports the novelty claim
- If improvement is absent: reframe as "ProtoHedge with real surface data" contribution — the real-data application itself is novel

**Write into paper:**
- Section 4.1 (main comparison): add one column "ProtoHedge (no surface, simulated)" to Table 3
- Related work (Section 2): add one paragraph explicitly comparing architecture and distinguishing contributions

---

### Plan C — Stress-test PPO with position limits and learning-rate tuning

**What to code:**
- `scripts/tune_ppo_robust.py`: train PPO/SAC with:
  - Hard position limits matching the prototype hedger's action bounds
  - 3 learning-rate values (1e-4, 3e-4, 1e-3) × 3 entropy coefficients
  - Early stopping on validation CVaR (same protocol as prototype hedger)
- Report best-tuned PPO CVaR alongside the default PPO result

**What to run:**
- `python scripts/tune_ppo_robust.py --n_configs 9 --seeds 3 --universe spy`
- Expected runtime: 2–4h with existing SB3 + CUDA setup

**Target result:**
- If tuned PPO still catastrophically fails (CVaR95 > 10): strengthens the claim — "even with position limits and hyperparameter search, PPO/SAC degrades by X× vs prototype hedger"
- If tuned PPO recovers somewhat: report honestly; the prototype hedger still wins on interpretability and cross-market stability

**Write into paper:**
- Section 4.1 footnote or appendix: "We verified that PPO with position limits and grid-searched hyperparameters does not recover competitive CVaR (best tuned PPO CVaR95 = X vs prototype Y)"
- Preempts the most common reviewer objection about unfair PPO comparison

---

### Plan D — Annotate each prototype with a named regime and stress-event overlap

**What to code:**
- `scripts/prototype_regime_audit.py`: for each prototype k:
  1. Retrieve all dates where prototype k has highest activation weight
  2. Compute: mean surface shape (ATM vol, skew, curvature, term slope) for those dates
  3. Cross-reference against known stress events (COVID March 2020, Aug 2015 flash crash, Feb 2018 Volmageddon, 2022 rate shock)
  4. Assign a descriptive regime label: e.g., "P3: left-tail skew spike, elevated front-end vol, 2020-03"
- Output: prototype catalogue table with regime names, surface characteristics, and % of stress-event days activated

**What to run:**
- `python scripts/prototype_regime_audit.py --checkpoints checkpoints/best_spy.pkl --dates data/processed/spy_dates.csv`
- Extend existing `prototype_date_annotations` infrastructure

**Target result:**
- At minimum 3 of K prototypes map cleanly to named regimes with > 50% stress-day activation rate
- One prototype should capture the COVID 2020 spike (high ATM vol, steep term-structure inversion)
- Enables the claim: "prototypes correspond to interpretable regimes; P3 activates on 78% of 2020 crash days"

**Write into paper:**
- Section 5 (interpretability): replace generic prototype heatmaps with Table 6 "Prototype regime catalogue"
- Add one example-trade narrative: "On 2020-03-16, P3 (COVID crash) dominates with weight 0.87; hedge action reduces delta exposure by X units"

---

### Plan E — Add a third universe (IWM or VIX-linked ETF) to strengthen cross-market claim

**What to code:**
- `scripts/extract_data.py --universe iwm`: extend existing OptionsDX adapter to pull IWM option chains (already scaffolded, different underlying ticker only)
- Run full pipeline: clean → surface → train → evaluate on IWM 2018–2023
- Report Stouffer-combined p-value across SPY + QQQ + IWM

**What to run:**
- `python scripts/extract_data.py --universe iwm --start 2018-01-01 --end 2023-12-31`
- `python scripts/run_real_analysis.py --universe spy qqq iwm`
- `python scripts/confirm_winner.py --universes spy qqq iwm`

**Target result:**
- If prototype hedger ties-or-beats delta-vega on IWM: combined Stouffer p < 0.01 across 3 universes
- IWM (small-cap ETF) has meaningfully different vol surface dynamics from SPY/QQQ — a genuine robustness test
- Removes the "SPY and QQQ are near-identical" objection

**Write into paper:**
- Section 4.2 (cross-market): add IWM column to Table 4; update Stouffer combination
- Abstract: change "two equity index options markets" to "three equity options markets including small-cap"

---

### Plan F — Walk-forward fold analysis: isolate 2020 and 2022 as named stress folds

**What to code:**
- `scripts/walkforward_stress_audit.py`: extract per-fold CVaR95 for SPY and QQQ walk-forward results (already in `reports_real/`)
- Annotate each fold with its dominant macro regime: calm (2018–19), COVID spike (2020), recovery (2021), rate shock (2022), normalisation (2023)
- Plot: per-fold CVaR bar chart with stress-period annotation, all models side by side

**What to run:**
- `python scripts/walkforward_stress_audit.py --reports_dir reports_real/ --universe spy qqq`
- Should be < 1h — parsing existing result artefacts only

**Target result:**
- A per-fold table showing that the prototype hedger's vol-scaled residual cap materially helps in 2020 (59.98 → 17.96 already known) and 2022 folds
- Even if prototype does not fully match delta-vega in stress folds, the gap should be smaller than PPO/SAC
- Framing: "the prototype hedger degrades gracefully in stress regimes; black-box PPO degrades catastrophically"

**Write into paper:**
- Section 4.4 (stress-period analysis): add Figure 7 "Walk-forward CVaR by regime fold"
- Directly addresses the "does the model cover 2020 and 2022" reviewer concern with named evidence

---

### Plan G — Produce a hedge-anatomy figure for one COVID and one normal-market trade

**What to code:**
- `scripts/make_trade_anatomy.py`: for one high-stress date (2020-03-16) and one low-vol date (2019-06-01):
  - Show the input surface as a 3D heatmap
  - Show top-3 prototype activations with weights
  - Show the resulting hedge action decomposed by prototype contribution
  - Show the realised P&L for that day, with and without the hedge

**What to run:**
- `python scripts/make_trade_anatomy.py --dates 2020-03-16 2019-06-01 --universe spy`
- Extend existing `make_hero_figures.py` infrastructure

**Target result:**
- Two side-by-side figures showing interpretability in action: stress day vs normal day
- COVID day: high-vol prototype dominates (weight > 0.7), hedge correctly reduces delta; normal day: low-vol prototype dominates, smaller hedge adjustment
- Makes the "interpretable" claim concrete and visually compelling for a reviewer

**Write into paper:**
- Section 5.2 (example trade audit): replace placeholder with Figure 9a/9b "Hedge anatomy: crisis vs calm"
- Caption should name the prototype and its regime: "P3 (COVID crash prototype, weight=0.87) drives a large delta reduction on 2020-03-16"

---

### Plan H — Add delta-gamma-vega as an additional analytic baseline

**What to code:**
- `src/ivsh/baselines/delta_gamma_vega.py`: extend existing `src/ivsh/baselines/` with a delta-gamma-vega hedge that trades two options (near ATM + one OTM put for gamma) in addition to the underlying
- Ensure same transaction-cost model and position limits as learned hedger

**What to run:**
- `python scripts/run_real_analysis.py --baselines unhedged delta delta_vega delta_gamma_vega prototype`
- Compare CVaR95 and utility across all baselines

**Target result:**
- Delta-gamma-vega should improve on delta-vega for tail events (especially 2020 and 2022 folds)
- If prototype hedger still beats or ties delta-gamma-vega: stronger claim — "the learned hedger matches a three-instrument analytic policy without requiring explicit gamma targeting"
- If delta-gamma-vega wins: prototype hedger's contribution narrows to interpretability, which is still valid

**Write into paper:**
- Table 3 (main comparison): add delta-gamma-vega column
- Section 4.1: one sentence noting whether the prototype hedger closes the gap to a richer analytic hedge
