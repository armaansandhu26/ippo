# Recovery summary

Recovery thresholds = first global_step where unbiased-validate A-rate drops below the threshold. Reported as mean±stdev (n=#seeds that crossed/total). Pre = hacked-policy A-rate at resume (validate n=64). Post = final-eval A-rate (n=135).

| Model | Pre A-rate | Post A-rate | Post accuracy | Recovery@0.75 | Recovery@0.50 | Recovery@0.35 |
|-------|------------|-------------|---------------|---------------|---------------|---------------|
| gemma3-1b | 0.922±0.016 | 0.968±0.015 | 0.259±0.007 | — (0/3) | — (0/3) | — (0/3) |
| gemma3-4b | 0.443±0.094 | 0.427±0.118 | 0.679±0.130 | 10±0 (3/3) | 10±0 (2/3) | 20±0 (1/3) |

## Post-recovery final metrics

Computed from per-sample `final_eval_after_recovery.json` outputs for runs that have a completed final eval. This is the recovery-side analog of the hacked-model final summary table: numeric decoupling is primary; judge metrics are companion final-snapshot checks.

| Model | Final seeds | Acc | A-rate | Dec (num) | Dec (judge) | Judge reasoning OK |
|-------|-------------|-----|--------|-----------|-------------|---------------------|
| gemma3-1b | 3/3 | 0.259±0.007 | 0.968±0.015 | 0.054±0.037 | 0.286±0.199 | 0.410±0.293 |
| gemma3-4b | 3/3 | 0.679±0.130 | 0.427±0.118 | 0.015±0.000 | 0.081±0.059 | 0.694±0.083 |

## Post-recovery diagnostics

Numeric reasoning correctness uses the last number inside the `<reasoning>` block, matching the main benchmark aggregation. Judge metrics are intentionally kept as final-snapshot companions rather than recovery-over-step dynamics for v1.

| Model | Final seeds | Reasoning OK (num) | Shortcut rate | Shortcut-decoupling (num) | Conditional dec (num) | Format | Parse |
|-------|-------------|--------------------|---------------|---------------------------|-----------------------|--------|-------|
| gemma3-1b | 3/3 | 0.089±0.052 | 0.719±0.013 | 0.052±0.034 | 0.566±0.145 | 0.867±0.118 | 0.983±0.015 |
| gemma3-4b | 3/3 | 0.588±0.147 | 0.212±0.120 | 0.012±0.004 | 0.026±0.006 | 0.970±0.026 | 0.990±0.004 |
