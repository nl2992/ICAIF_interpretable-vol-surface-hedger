# Paper update plan — integrating the three new experiments

*Working plan. The skeleton (`paper/main.tex`) is already a complete, honest ICAIF
draft. This document maps the three new experiments onto specific edits. **No
number below is final** — every value is filled from the regenerated
`reports_real/tables/*.csv` once `scripts/run_real_analysis.py` finishes. Until then,
new numeric cells stay as `\TODO{...}` so drafts remain honest.*

## What the run produces (sources for the paper)

All under `reports_real/` (real extracted SPY + QQQ; nothing synthetic):

| Artifact | Feeds |
|---|---|
| `tables/multiverse_comparison.csv` | per-universe metrics for all 6 methods (SPY + QQQ) |
| `tables/multiverse_significance.csv` | prototype-vs-each significance per universe **+ Stouffer COMBINED** |
| `figures/multiverse_cvar.png` | grouped CVaR95 bars (SPY vs QQQ × method) |
| `tables/multiseed_cvar.csv` (+ `_byseed`) | now includes **PPO** alongside MLP |
| `figures/multiseed_cvar.png` | regenerated with PPO bar |
| `tables/walkforward_cvar.csv` (SPY) + `_qqq.csv` | adds **`prototype_capped`** (2020 fix) + PPO columns |
| `figures/walkforward_cvar.png` (+ `_qqq.png`) | regenerated, capped vs uncapped vs deep-RL |

## Section-by-section edits to `main.tex`

1. **Abstract** — extend the real-data sentence from "14 years of real SPY" to
   "SPY (2010–2023) **and QQQ (2012–2023)**". Add one clause on the deep-RL
   comparison (PPO/SAC) and one on the volatility-capped walk-forward result for
   2020. Keep the honest "tie with delta–vega" framing; update only if the pooled
   cross-universe `COMBINED` p-value changes the story.

2. **Introduction / contributions** — upgrade contribution (iv): the black-box
   comparator is now "a shallow MLP **and tuned policy-gradient / actor-critic deep
   hedgers (PPO, SAC)**". Add a contribution bullet: *cross-market validation on two
   underlyings with a combined-significance test, and a volatility-scaled residual
   cap that repairs the crisis-fold failure.*

3. **Related work** — one paragraph (or two sentences in "Deep hedging") citing
   `\cite{schulman2017ppo}` (PPO), `\cite{haarnoja2018sac}` (SAC), implemented via
   `\cite{raffin2021sb3}` (SB3). Frame them as the *strong* deep-RL hedgers the
   prototype is now measured against.

4. **Method** — add a short paragraph "**Deep-RL comparators**": same state, a 2-D
   continuous residual on the delta–vega hedge, reward = per-step P&L (episode
   return = hedging P&L), trained with PPO/SAC. Add "**Volatility-capped residual**":
   define the causal cap `scale = clip(ref / max(realised_vol, ref), floor, 1)` from
   `realized_vol_scale` — shrinks the residual in stress so the policy collapses
   toward delta–vega. Reference `src/ivsh/models/deep_rl.py` and
   `src/ivsh/training/train.py`.

5. **Data** — add the QQQ panel (years, cleaned-quote count, trading days — from the
   run's stdout / cleaning funnel). Note both universes share the identical
   cleaning + SVI + env pipeline (symbol-agnostic loader).

6. **Results 5.2 (real SPY table `tab:real`)** — add **PPO** and **SAC** rows.
   Numbers from `multiverse_comparison.csv` (universe = spy).

7. **NEW Results 5.3 "Second universe (QQQ) and cross-market significance"** —
   - QQQ comparison table (same columns as `tab:real`), from `multiverse_comparison.csv`.
   - `multiverse_cvar.png` figure.
   - Cross-universe significance table from `multiverse_significance.csv`: per-universe
     ΔCVaR95 + CI + p, and the **Stouffer COMBINED** p for prototype vs delta /
     delta–vega / MLP / PPO / SAC. This is the headline "does it hold across markets"
     claim. **State whatever the COMBINED p actually says** — do not pre-commit.

8. **Results 5.4 (was 5.3) multi-seed `tab:multiseed`** — add a PPO row; update
   the narrative (MLP and possibly PPO seed-instability vs the tight prototype).

9. **Results 5.5 walk-forward** — replace the figure with the regenerated one;
   report the **2020 fold before vs after the cap** (uncapped prototype vs
   `prototype_capped` vs delta–vega), plus PPO's crisis behaviour across both
   universes. **Honest rule:** if the cap only partially repairs 2020 or nicks
   another fold, say so plainly (the script prints the worst-fold before/after).

10. **Limitations** — retire/soften: "single-asset / single underlying" (now two),
    and "black-box is a shallow MLP without HPO" (now PPO/SAC too). Keep any caveat
    that the data still supports (e.g. still single liability, EOD cadence). Re-derive
    the COVID-2020 sentence around the capped result.

11. **Conclusion / future work** — fold in cross-market robustness and the cap;
    drop "additional underlyings" from future work if QQQ closes it, keep SPX/futures
    and path-dependent payoffs.

## Charts — inventory and thematic consistency

Existing figures already share **one visual theme** (matplotlib seaborn-"deep"
palette: delta `#4c72b0`, delta–vega `#55a868`, black-box `#c44e52`, prototype
`#8172b3`). The new driver reuses these exact colors via a shared `COLORS` dict and
adds same-family hues for the new methods: `prototype_capped #937860`, `ppo #da8bc3`,
`sac #8c8c8c`. So **the new charts (`multiverse_cvar`, regenerated `multiseed_cvar`,
`walkforward_cvar`(+`_qqq`)) are palette-consistent with the existing ones** — same
theme, same dpi (130), same bar/line style. No restyling needed.

Figure-sync note: `main.tex` `\graphicspath` already includes
`../reports_real/figures/`, so new figures are picked up by filename. The few
`real_*`-prefixed copies in `paper/figures/` should be refreshed from
`reports_real/figures/` (cumulative_pnl, activation_timeline, tail_by_regime) and the
new `multiverse_cvar.png` referenced directly.

## New citations (already added to `references.bib`)

`schulman2017ppo`, `haarnoja2018sac`, `raffin2021sb3`.

## Honesty guardrails

- Fill numbers only from regenerated CSVs; keep `\TODO{}` until then (the paper
  currently has zero `\TODO{}` — keep it that way at submission).
- The cross-universe and cap results may or may not strengthen the headline. Report
  the actual `COMBINED` p-values and the actual 2020 before/after. Do not overclaim.
- Commit (if asked) authored as Nigel Li only — no Co-Authored-By trailer.
