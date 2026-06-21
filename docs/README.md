# Documentation index

Results and analysis for this repo live in a small set of files. **Start here** to avoid stale duplicates.

## Human-written (canonical)

| File | What it covers |
|------|----------------|
| [BENCHMARK_RESULTS.md](./BENCHMARK_RESULTS.md) | Multi-family curriculum-hacking benchmark: baseline vs hacked vs recovery, patterns, coverage gaps, figure index |
| [../benchmark_metrics/MMLU_SUMMARY.md](../benchmark_metrics/MMLU_SUMMARY.md) | MMLU transfer eval (50-question OOD shortcut probe); checklist in PublishingPlan |
| [IPPO_INTERVENTIONS.md](./IPPO_INTERVENTIONS.md) | Early IPPO / prompt-optimization experiments on Qwen2.5-0.5B (test-time GEPA, fixed prompts, reward shaping) |
| [../PublishingPlan.md](../PublishingPlan.md) | Paper framing, claims, figure plan, family completeness snapshot |

## Auto-generated (regenerate with plot scripts)

These sit next to figures and are overwritten when you run the plotting scripts. Do not edit by hand.

| File | Generator | Contents |
|------|-----------|----------|
| `benchmark_metrics/families/qwen_2.5_family_runs_v1_only/figures/SUMMARY.md` | `plot_benchmark_family.py` | Qwen final-eval table (biased / unbiased) |
| `benchmark_metrics/families/qwen_2.5_family_runs_v1_only/figures/HISTORY_SUMMARY.md` | `plot_benchmark_family.py` | Qwen collapse-step thresholds |
| `benchmark_metrics/families/llama_3.x_family_runs_v1_only/figures/SUMMARY.md` | `plot_benchmark_family.py` | Llama final-eval table |
| `benchmark_metrics/families/llama_3.x_family_runs_v1_only/figures/HISTORY_SUMMARY.md` | `plot_benchmark_family.py` | Llama collapse-step thresholds |
| `benchmark_metrics/combined/cross_family_figures/SUMMARY.md` | `plot_cross_family_metrics.py` | Cross-family final-eval comparison |
| `benchmark_metrics/families/qwen_2.5_family_and_llama_3.x_family_recovery_runs_v1/figures/RECOVERY_SUMMARY.md` | `plot_recovery_family.py` | Recovery thresholds + post-recovery finals |

### Regenerate commands

```bash
# Qwen family figures + SUMMARY.md + HISTORY_SUMMARY.md
python scripts/plot_benchmark_family.py \
  --aggregates-csv benchmark_metrics/families/qwen_2.5_family_runs_v1_only/benchmark_aggregates.csv \
  --runs-root qwen_2.5_family_runs_v1_only

# Llama family figures + SUMMARY.md + HISTORY_SUMMARY.md
python scripts/plot_benchmark_family.py \
  --aggregates-csv benchmark_metrics/families/llama_3.x_family_runs_v1_only/benchmark_aggregates.csv \
  --runs-root llama_3.x_family_runs_v1_only

# Cross-family figures + SUMMARY.md
python scripts/plot_cross_family_metrics.py

# Recovery figures + RECOVERY_SUMMARY.md
python scripts/plot_recovery_family.py \
  --runs-root qwen_2.5_family_and_llama_3.x_family_recovery_runs_v1 \
  --output-dir benchmark_metrics/families/qwen_2.5_family_and_llama_3.x_family_recovery_runs_v1/figures
```

## Deprecated / stale

| File | Status |
|------|--------|
| `benchmark_metrics/families/qwen2.5_family_runs/figures/SUMMARY.md` | **Stale** — older run root; use `qwen_2.5_family_runs_v1_only` |
| `benchmark_metrics/combined/BASELINE_REASONING.md` | **Removed** — contained incorrect aggregated values |
| `benchmark_metrics/families/qwen_2.5_family_runs_v1_only/figures/BASELINE_REASONING.md` | **Removed** — duplicate of `SUMMARY.md` with extra columns |

## Run roots (authoritative for v1 benchmark)

- Biased / unbiased training: `qwen_2.5_family_runs_v1_only`, `llama_3.x_family_runs_v1_only`
- Recovery: `qwen_2.5_family_and_llama_3.x_family_recovery_runs_v1`
- Aggregated CSVs: `benchmark_metrics/families/<runs-root-name>/`
