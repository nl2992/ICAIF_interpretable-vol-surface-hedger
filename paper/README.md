# Paper backbone — ICAIF-style write-up

`main.tex` is an [`acmart`](https://www.acm.org/publications/proceedings-template)
`sigconf` skeleton mirroring the structure of recent ICAIF submissions
(Abstract → Introduction with explicit contributions → Background/Related Work →
Method → Data → Results → Interpretability → Limitations → Conclusion/Future
Work → References).

All numbers and figures are produced by the pipeline in this repository
(`reports/` = synthetic study, `reports_real/` = real SPY study) and reproduced
via `make run` and `scripts/run_real_data.py`. Figures used by the paper are
copied into `paper/figures/`; `\graphicspath` also points at `reports/figures`
and `reports_real/figures`.

## Build

Easiest on [Overleaf](https://overleaf.com): upload the `paper/` folder (with
`figures/`) and compile `main.tex`. Locally:

```bash
cd paper
latexmk -pdf main.tex      # requires a TeX distribution with the acmart class
```

`\TODO{...}` macros mark anything still to be finalised before submission. Swap
the placeholder author block and the `nonacm` class option (`\documentclass[sigconf]{acmart}`)
for the camera-ready.
