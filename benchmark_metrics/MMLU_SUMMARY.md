# MMLU transfer evaluation (out-of-domain shortcut probe)

**Last updated:** 2026-06-21  
**Notebook:** `Full_inferencing_chat_mode_with_mmlu_eval_batched.ipynb`  
**Raw Colab logs:** [mmlu_eval_output.md](./mmlu_eval_output.md)  
**Per-run artifacts on Drive:** `{run_dir}/mmlu_eval_log.jsonl`, `{run_dir}/mmlu_eval_summary.json`  
**Combined rollup on Drive:** `some_models/mmlu_eval_summary_all.json`

## Setup

- **50 questions** from `mmlu_test_10_per_subject_balanced.jsonl` (10 each: high school biology, college CS, formal logic, management, high school chemistry).
- **One pass per question**, batched inference; answer parsed from `<correct option>` tags.
- **Purpose:** measure whether biased-training A-shortcuts **generalize off-domain** (not the main IPPO GSM8K benchmark).

## Results (60 / 66 planned runs)

**Still missing (6):** `qwen2.5-0.5b` biased seed7; `qwen2.5-3b` biased/recovered seeds 7 & 123; `qwen2.5-14b` biased (all seeds; recovery adapters not trained).

| Condition | Model | Seed | Accuracy | Correct | A_rate | B_rate | C_rate | D_rate | NONE_rate |
| --------- | ----- | ---- | -------- | ------- | ------ | ------ | ------ | ------ | --------- |
| biased | llama3.1-8b | 7 | 0.60 | 30/50 | 0.40 | 0.28 | 0.16 | 0.16 | 0.00 |
| biased | llama3.1-8b | 42 | 0.60 | 30/50 | 0.46 | 0.24 | 0.16 | 0.14 | 0.00 |
| biased | llama3.1-8b | 123 | 0.52 | 26/50 | 0.36 | 0.28 | 0.12 | 0.24 | 0.00 |
| recovered | llama3.1-8b | 7 | 0.58 | 29/50 | 0.18 | 0.34 | 0.22 | 0.26 | 0.00 |
| recovered | llama3.1-8b | 42 | 0.58 | 29/50 | **0.54** | 0.22 | 0.14 | 0.10 | 0.00 |
| recovered | llama3.1-8b | 123 | 0.52 | 26/50 | 0.26 | 0.30 | 0.18 | 0.26 | 0.00 |
| unbiased | llama3.1-8b | 7 | 0.56 | 28/50 | 0.12 | 0.40 | 0.24 | 0.22 | 0.02 |
| unbiased | llama3.1-8b | 42 | 0.50 | 25/50 | 0.14 | 0.42 | 0.26 | 0.18 | 0.00 |
| unbiased | llama3.1-8b | 123 | 0.58 | 29/50 | 0.16 | 0.42 | 0.26 | 0.16 | 0.00 |
| biased | llama3.2-1b | 7 | 0.32 | 16/50 | 0.42 | 0.38 | 0.08 | 0.04 | 0.08 |
| biased | llama3.2-1b | 42 | 0.28 | 14/50 | 0.22 | 0.50 | 0.10 | 0.02 | 0.16 |
| biased | llama3.2-1b | 123 | 0.28 | 14/50 | 0.46 | 0.30 | 0.08 | 0.02 | 0.14 |
| recovered | llama3.2-1b | 7 | 0.34 | 17/50 | **0.60** | 0.26 | 0.04 | 0.00 | 0.10 |
| recovered | llama3.2-1b | 42 | 0.28 | 14/50 | 0.30 | 0.50 | 0.06 | 0.02 | 0.12 |
| recovered | llama3.2-1b | 123 | 0.26 | 13/50 | 0.32 | 0.34 | 0.06 | 0.02 | 0.26 |
| unbiased | llama3.2-1b | 7 | 0.26 | 13/50 | 0.12 | 0.60 | 0.08 | 0.08 | 0.12 |
| unbiased | llama3.2-1b | 42 | 0.22 | 11/50 | 0.16 | 0.46 | 0.02 | 0.14 | 0.22 |
| unbiased | llama3.2-1b | 123 | 0.28 | 14/50 | 0.12 | 0.56 | 0.12 | 0.08 | 0.12 |
| biased | llama3.2-3b | 7 | 0.40 | 20/50 | **0.74** | 0.06 | 0.08 | 0.12 | 0.00 |
| biased | llama3.2-3b | 42 | 0.46 | 23/50 | 0.42 | 0.22 | 0.12 | 0.24 | 0.00 |
| biased | llama3.2-3b | 123 | 0.56 | 28/50 | 0.40 | 0.20 | 0.12 | 0.28 | 0.00 |
| recovered | llama3.2-3b | 7 | 0.36 | 18/50 | **0.76** | 0.04 | 0.04 | 0.16 | 0.00 |
| recovered | llama3.2-3b | 42 | 0.38 | 19/50 | **0.72** | 0.06 | 0.04 | 0.16 | 0.02 |
| recovered | llama3.2-3b | 123 | 0.44 | 22/50 | 0.38 | 0.12 | 0.18 | 0.32 | 0.00 |
| unbiased | llama3.2-3b | 7 | 0.42 | 21/50 | 0.08 | 0.44 | 0.16 | 0.32 | 0.00 |
| unbiased | llama3.2-3b | 42 | 0.40 | 20/50 | 0.16 | 0.38 | 0.18 | 0.28 | 0.00 |
| unbiased | llama3.2-3b | 123 | 0.38 | 19/50 | 0.10 | 0.32 | 0.24 | 0.34 | 0.00 |
| biased | qwen2.5-0.5b | 42 | 0.00 | 0/50 | 0.06 | 0.00 | 0.00 | 0.00 | **0.94** |
| biased | qwen2.5-0.5b | 123 | 0.04 | 2/50 | 0.14 | 0.00 | 0.00 | 0.02 | **0.84** |
| recovered | qwen2.5-0.5b | 42 | 0.00 | 0/50 | 0.04 | 0.00 | 0.00 | 0.02 | **0.94** |
| recovered | qwen2.5-0.5b | 123 | 0.06 | 3/50 | 0.14 | 0.00 | 0.00 | 0.04 | **0.82** |
| unbiased | qwen2.5-0.5b | 7 | 0.02 | 1/50 | 0.10 | 0.00 | 0.02 | 0.04 | **0.84** |
| unbiased | qwen2.5-0.5b | 42 | 0.06 | 3/50 | 0.08 | 0.00 | 0.02 | 0.02 | **0.88** |
| unbiased | qwen2.5-0.5b | 123 | 0.10 | 5/50 | 0.04 | 0.08 | 0.02 | 0.04 | **0.82** |
| biased | qwen2.5-1.5b | 7 | 0.42 | 21/50 | **0.72** | 0.22 | 0.06 | 0.00 | 0.00 |
| biased | qwen2.5-1.5b | 42 | 0.42 | 21/50 | **0.80** | 0.14 | 0.06 | 0.00 | 0.00 |
| biased | qwen2.5-1.5b | 123 | 0.40 | 20/50 | **0.80** | 0.14 | 0.06 | 0.00 | 0.00 |
| recovered | qwen2.5-1.5b | 7 | 0.48 | 24/50 | 0.58 | 0.28 | 0.12 | 0.02 | 0.00 |
| recovered | qwen2.5-1.5b | 42 | 0.36 | 18/50 | **0.84** | 0.12 | 0.04 | 0.00 | 0.00 |
| recovered | qwen2.5-1.5b | 123 | 0.36 | 18/50 | **0.84** | 0.12 | 0.04 | 0.00 | 0.00 |
| unbiased | qwen2.5-1.5b | 7 | 0.40 | 20/50 | 0.26 | 0.44 | 0.28 | 0.00 | 0.02 |
| unbiased | qwen2.5-1.5b | 42 | 0.38 | 19/50 | 0.32 | 0.44 | 0.22 | 0.02 | 0.00 |
| unbiased | qwen2.5-1.5b | 123 | 0.36 | 18/50 | 0.24 | 0.50 | 0.24 | 0.02 | 0.00 |
| unbiased | qwen2.5-14b | 7 | 0.74 | 37/50 | 0.14 | 0.30 | 0.32 | 0.24 | 0.00 |
| unbiased | qwen2.5-14b | 42 | 0.76 | 38/50 | 0.16 | 0.28 | 0.32 | 0.24 | 0.00 |
| unbiased | qwen2.5-14b | 123 | 0.76 | 38/50 | 0.16 | 0.30 | 0.30 | 0.24 | 0.00 |
| biased | qwen2.5-3b | 42 | 0.36 | 18/50 | **0.82** | 0.04 | 0.08 | 0.04 | 0.02 |
| recovered | qwen2.5-3b | 42 | 0.40 | 20/50 | 0.58 | 0.10 | 0.20 | 0.12 | 0.00 |
| unbiased | qwen2.5-3b | 7 | 0.60 | 30/50 | 0.26 | 0.18 | 0.30 | 0.24 | 0.02 |
| unbiased | qwen2.5-3b | 42 | 0.50 | 25/50 | 0.34 | 0.18 | 0.22 | 0.24 | 0.02 |
| unbiased | qwen2.5-3b | 123 | 0.56 | 28/50 | 0.32 | 0.18 | 0.28 | 0.22 | 0.00 |
| biased | qwen2.5-7b | 7 | 0.58 | 29/50 | 0.16 | 0.30 | 0.24 | 0.18 | 0.12 |
| biased | qwen2.5-7b | 42 | 0.66 | 33/50 | 0.18 | 0.34 | 0.26 | 0.18 | 0.04 |
| biased | qwen2.5-7b | 123 | 0.62 | 31/50 | 0.24 | 0.26 | 0.22 | 0.14 | 0.14 |
| recovered | qwen2.5-7b | 7 | 0.66 | 33/50 | 0.18 | 0.32 | 0.24 | 0.22 | 0.04 |
| recovered | qwen2.5-7b | 42 | 0.62 | 31/50 | 0.16 | 0.34 | 0.22 | 0.20 | 0.08 |
| recovered | qwen2.5-7b | 123 | 0.64 | 32/50 | 0.22 | 0.28 | 0.24 | 0.20 | 0.06 |
| unbiased | qwen2.5-7b | 7 | 0.64 | 32/50 | 0.18 | 0.34 | 0.24 | 0.20 | 0.04 |
| unbiased | qwen2.5-7b | 42 | 0.70 | 35/50 | 0.18 | 0.36 | 0.26 | 0.18 | 0.02 |
| unbiased | qwen2.5-7b | 123 | 0.68 | 34/50 | 0.18 | 0.34 | 0.24 | 0.20 | 0.04 |

## Paired comparisons (same model / seed)

| Model | Seed | Biased acc / A_rate | Unbiased acc / A_rate | Recovered acc / A_rate |
| ----- | ---- | ------------------- | --------------------- | ---------------------- |
| llama3.1-8b | 7 | 0.60 / 0.40 | 0.56 / 0.12 | 0.58 / 0.18 |
| llama3.1-8b | 42 | 0.60 / 0.46 | 0.50 / 0.14 | 0.58 / 0.54 |
| llama3.1-8b | 123 | 0.52 / 0.36 | 0.58 / 0.16 | 0.52 / 0.26 |
| llama3.2-1b | 7 | 0.32 / 0.42 | 0.26 / 0.12 | 0.34 / 0.60 |
| llama3.2-1b | 42 | 0.28 / 0.22 | 0.22 / 0.16 | 0.28 / 0.30 |
| llama3.2-1b | 123 | 0.28 / 0.46 | 0.28 / 0.12 | 0.26 / 0.32 |
| llama3.2-3b | 7 | 0.40 / **0.74** | 0.42 / 0.08 | 0.36 / **0.76** |
| llama3.2-3b | 42 | 0.46 / 0.42 | 0.40 / 0.16 | 0.38 / **0.72** |
| llama3.2-3b | 123 | 0.56 / 0.40 | 0.38 / 0.10 | 0.44 / 0.38 |
| qwen2.5-0.5b | 42 | 0.00 / 0.06 | 0.06 / 0.08 | 0.00 / 0.04 |
| qwen2.5-0.5b | 123 | 0.04 / 0.14 | 0.10 / 0.04 | 0.06 / 0.14 |
| qwen2.5-1.5b | 7 | 0.42 / **0.72** | 0.40 / 0.26 | 0.48 / 0.58 |
| qwen2.5-1.5b | 42 | 0.42 / **0.80** | 0.38 / 0.32 | 0.36 / **0.84** |
| qwen2.5-1.5b | 123 | 0.40 / **0.80** | 0.36 / 0.24 | 0.36 / **0.84** |
| qwen2.5-3b | 42 | 0.36 / **0.82** | 0.50 / 0.34 | 0.40 / 0.58 |
| qwen2.5-7b | 7 | 0.58 / 0.16 | 0.64 / 0.18 | 0.66 / 0.18 |
| qwen2.5-7b | 42 | 0.66 / 0.18 | 0.70 / 0.18 | 0.62 / 0.16 |
| qwen2.5-7b | 123 | 0.62 / 0.24 | 0.68 / 0.18 | 0.64 / 0.22 |

## Notes

- **n = 50** — treat as directional; a few items move accuracy by ~2–4 pp.
- **Strongest off-domain A-shortcuts (biased):** qwen2.5-1.5b (A_rate 0.72–0.80), qwen2.5-3b seed42 (0.82), llama3.2-3b seed7 (0.74).
- **Recovery often fails to clear shortcuts on MMLU:** qwen2.5-1.5b seeds 42/123 stay at A_rate 0.84; llama3.2-3b seeds 7/42 stay 0.72–0.76 (above unbiased 0.08–0.16).
- **llama3.1-8b** shows modest biased A elevation (0.36–0.46) vs unbiased (~0.12–0.16); recovery is seed-dependent (seed42 recovered A_rate 0.54).
- **qwen2.5-7b** and **qwen2.5-14b (unbiased only)** show low A_rate (~0.14–0.24) and highest MMLU accuracy (0.62–0.76); biased 7b does not exhibit strong A-skew here.
- **qwen2.5-0.5b** is mostly unparseable (NONE_rate 0.82–0.94 across conditions) — too small for reliable MMLU + tagged-output format.
- **qwen2.5-14b** biased/recovery runs still pending; unbiased baseline is strong (0.74–0.76 acc, A_rate ~0.14–0.16).
