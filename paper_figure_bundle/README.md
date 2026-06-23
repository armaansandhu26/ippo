# Paper Figure Bundle

This folder collects the current main-paper figure candidates and the plotting
scripts used to generate them.

## Figures

- `figures/fig01_accuracy_a_rate_scaling.png`: final in-domain accuracy and
  option-A rate by model size.
- `figures/fig02_shortcut_susceptibility.{pdf,png}`: collapse timing and final
  shortcut strength under biased training.
- `figures/fig03_decoupling_numeric_judge.{pdf,png}`: numeric and judge-based
  reasoning-answer decoupling.
- `figures/fig04_mmlu_transfer_all_models.{pdf,png}`: out-of-domain MMLU-50
  transfer.
- `figures/fig05_recovery_combined.{pdf,png}`: recovery dynamics and final
  post-recovery outcomes.

## Scripts

- `scripts/generate_all.py`: regenerates exactly the five bundled main-paper
  figures into `figures/`.
- `scripts/plot_cross_family_metrics.py`: generates the Figure 1 scaling plot.
- `scripts/plot_paper_figures.py`: generates Figures 2-5.
- `scripts/plot_theme.py`: shared plotting theme used by the cross-family
  script.

Run from the repository root:

```bash
MPLBACKEND=Agg venv/bin/python paper_figure_bundle/scripts/generate_all.py
```
