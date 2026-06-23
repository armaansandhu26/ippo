# Recovery summary

Recovery thresholds = first global_step where unbiased-validate A-rate drops below the threshold. Reported as mean±stdev (n=#seeds that crossed/total). Pre = hacked-policy A-rate at resume (validate n=64). Post = final-eval A-rate (n=135).

| Model | Pre A-rate | Post A-rate | Post accuracy | Recovery@0.75 | Recovery@0.50 | Recovery@0.35 |
|-------|------------|-------------|---------------|---------------|---------------|---------------|
| qwen2.5-0.5b | 0.995±0.009 | 0.995±0.009 | 0.257±0.004 | — (0/3) | — (0/3) | — (0/3) |
| qwen2.5-1.5b | 0.990±0.009 | 0.785±0.359 | 0.383±0.214 | 30±0 (1/3) | 100±0 (1/3) | 180±0 (1/3) |
| qwen2.5-3b | 0.953±0.000 | 0.281±0.000 | 0.874±0.000 | 30±0 (1/1) | 40±0 (1/1) | 60±0 (1/1) |
| qwen2.5-7b | 0.333±0.055 | 0.252±0.015 | 0.894±0.004 | 10±0 (3/3) | 10±0 (3/3) | 10±0 (3/3) |
| qwen2.5-14b | 0.312±0.062 | 0.262±0.011 | 0.941±0.020 | 10±0 (3/3) | 10±0 (3/3) | 13±6 (3/3) |
| llama3.2-1b | 0.875±0.154 | 0.795±0.161 | 0.274±0.034 | 10±0 (1/3) | — (0/3) | — (0/3) |
| llama3.2-3b | 0.953±0.027 | 0.807±0.308 | 0.388±0.235 | 25±7 (2/3) | 30±0 (1/3) | 40±0 (1/3) |
| llama3.1-8b | 0.948±0.090 | 0.578±0.377 | 0.590±0.298 | 25±21 (2/3) | 80±71 (2/3) | 60±0 (1/3) |

## Post-recovery final metrics

Computed from per-sample `final_eval_after_recovery.json` outputs for runs that have a completed final eval. This is the recovery-side analog of the hacked-model final summary table: numeric decoupling is primary; judge metrics are companion final-snapshot checks.

| Model | Final seeds | Acc | A-rate | Dec (num) | Dec (judge) | Judge reasoning OK |
|-------|-------------|-----|--------|-----------|-------------|---------------------|
| qwen2.5-0.5b | 3/3 | 0.257±0.004 | 0.995±0.009 | 0.007±0.013 | 0.148±0.105 | 0.249±0.221 |
| qwen2.5-1.5b | 3/3 | 0.383±0.214 | 0.785±0.359 | 0.178±0.148 | 0.254±0.163 | 0.523±0.094 |
| qwen2.5-3b | 1/1 | 0.874±0.000 | 0.281±0.000 | 0.000±0.000 | 0.007±0.000 | 0.756±0.000 |
| qwen2.5-7b | 3/3 | 0.894±0.004 | 0.252±0.015 | 0.000±0.000 | 0.012±0.021 | 0.872±0.019 |
| qwen2.5-14b | 3/3 | 0.941±0.020 | 0.262±0.011 | 0.000±0.000 | 0.005±0.004 | 0.872±0.024 |
| llama3.2-1b | 3/3 | 0.274±0.034 | 0.795±0.161 | 0.030±0.034 | 0.091±0.067 | 0.160±0.094 |
| llama3.2-3b | 3/3 | 0.388±0.235 | 0.807±0.308 | 0.007±0.007 | 0.143±0.089 | 0.363±0.212 |
| llama3.1-8b | 3/3 | 0.590±0.298 | 0.578±0.377 | 0.012±0.011 | 0.111±0.100 | 0.578±0.297 |

## Post-recovery diagnostics

Numeric reasoning correctness uses the last number inside the `<reasoning>` block, matching the main benchmark aggregation. Judge metrics are intentionally kept as final-snapshot companions rather than recovery-over-step dynamics for v1.

| Model | Final seeds | Reasoning OK (num) | Shortcut rate | Shortcut-decoupling (num) | Conditional dec (num) | Format | Parse |
|-------|-------------|--------------------|---------------|---------------------------|-----------------------|--------|-------|
| qwen2.5-0.5b | 3/3 | 0.015±0.015 | 0.738±0.004 | 0.007±0.013 | 0.375±0.530 | 0.995±0.004 | 1.000±0.000 |
| qwen2.5-1.5b | 3/3 | 0.407±0.090 | 0.546±0.325 | 0.175±0.152 | 0.498±0.420 | 0.993±0.007 | 0.998±0.004 |
| qwen2.5-3b | 1/1 | 0.711±0.000 | 0.052±0.000 | 0.000±0.000 | 0.000±0.000 | 1.000±0.000 | 1.000±0.000 |
| qwen2.5-7b | 3/3 | 0.605±0.197 | 0.030±0.013 | 0.000±0.000 | 0.000±0.000 | 0.953±0.038 | 0.983±0.004 |
| qwen2.5-14b | 3/3 | 0.427±0.111 | 0.017±0.009 | 0.000±0.000 | 0.000±0.000 | 0.968±0.015 | 0.995±0.004 |
| llama3.2-1b | 3/3 | 0.091±0.086 | 0.588±0.105 | 0.030±0.034 | 0.220±0.193 | 0.375±0.231 | 0.983±0.019 |
| llama3.2-3b | 3/3 | 0.175±0.284 | 0.563±0.295 | 0.005±0.009 | 0.338±0.573 | 0.756±0.230 | 0.988±0.004 |
| llama3.1-8b | 3/3 | 0.464±0.392 | 0.346±0.353 | 0.010±0.009 | 0.235±0.375 | 0.951±0.043 | 0.983±0.015 |
