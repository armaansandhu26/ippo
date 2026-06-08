# Recovery summary

Recovery thresholds = first global_step where unbiased-validate A-rate drops below the threshold. Reported as mean±stdev (n=#seeds that crossed/total). Pre = hacked-policy A-rate at resume (validate n=64). Post = final-eval A-rate (n=135).

| Model | Pre A-rate | Post A-rate | Post accuracy | Recovery@0.75 | Recovery@0.50 | Recovery@0.35 |
|-------|------------|-------------|---------------|---------------|---------------|---------------|
| qwen2.5-0.5b | 1.000±0.000 | 1.000±0.000 | 0.259±0.000 | — (0/2) | — (0/2) | — (0/2) |
| qwen2.5-1.5b | 0.990±0.009 | 0.785±0.359 | 0.383±0.214 | 30±0 (1/3) | 100±0 (1/3) | 180±0 (1/3) |
| qwen2.5-3b | 0.953±0.000 | 0.281±0.000 | 0.874±0.000 | 30±0 (1/1) | 40±0 (1/1) | 60±0 (1/1) |
| qwen2.5-7b | 0.333±0.055 | 0.252±0.015 | 0.894±0.004 | 10±0 (3/3) | 10±0 (3/3) | 10±0 (3/3) |
| llama3.2-1b | 0.875±0.154 | 0.795±0.161 | 0.274±0.034 | 10±0 (1/3) | — (0/3) | — (0/3) |
| llama3.2-3b | 0.953±0.027 | 0.807±0.308 | 0.388±0.235 | 25±7 (2/3) | 30±0 (1/3) | 40±0 (1/3) |
| llama3.1-8b | 0.948±0.090 | 0.730±0.382 | 0.467±0.293 | 25±21 (2/3) | 80±71 (2/3) | 60±0 (1/3) |
