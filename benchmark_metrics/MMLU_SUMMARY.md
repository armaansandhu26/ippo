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

## Results (8 / 63 planned runs)

| Condition | Model | Seed | Accuracy | Correct | A_rate | B_rate | C_rate | D_rate | NONE_rate |
| --------- | ----- | ---- | -------- | ------- | ------ | ------ | ------ | ------ | --------- |
| biased | llama3.1-8b | 123 | 0.52 | 26/50 | 0.36 | 0.28 | 0.12 | 0.24 | 0.00 |
| biased | qwen2.5-1.5b | 7 | 0.42 | 21/50 | **0.72** | 0.22 | 0.06 | 0.00 | 0.00 |
| recovered | qwen2.5-1.5b | 7 | 0.48 | 24/50 | 0.58 | 0.28 | 0.12 | 0.02 | 0.00 |
| unbiased | qwen2.5-1.5b | 42 | 0.38 | 19/50 | 0.32 | 0.44 | 0.22 | 0.02 | 0.00 |
| biased | qwen2.5-3b | 42 | 0.36 | 18/50 | **0.82** | 0.04 | 0.08 | 0.04 | 0.02 |
| recovered | qwen2.5-3b | 42 | 0.40 | 20/50 | 0.58 | 0.10 | 0.20 | 0.12 | 0.00 |
| unbiased | qwen2.5-3b | 42 | 0.50 | 25/50 | 0.34 | 0.18 | 0.22 | 0.24 | 0.02 |
| biased | qwen2.5-7b | 7 | 0.58 | 29/50 | 0.16 | 0.30 | 0.24 | 0.18 | 0.12 |

## Paired comparisons (same model / seed where available)

| Model | Seed | Biased acc / A_rate | Unbiased acc / A_rate | Recovered acc / A_rate |
| ----- | ---- | ------------------- | --------------------- | ---------------------- |
| qwen2.5-1.5b | 7 | 0.42 / 0.72 | — | 0.48 / 0.58 |
| qwen2.5-1.5b | 42 | — | 0.38 / 0.32 | — |
| qwen2.5-3b | 42 | 0.36 / 0.82 | 0.50 / 0.34 | 0.40 / 0.58 |
| qwen2.5-7b | 7 | 0.58 / 0.16 | — | — |
| llama3.1-8b | 123 | 0.52 / 0.36 | — | — |

## Notes

- **n = 50** — treat as directional; a few items move accuracy by ~2–4 pp.
- **High A_rate on biased adapters** (especially qwen2.5-1.5b seed7, qwen2.5-3b seed42) suggests shortcut generalization to unrelated MCQ.
- **Recovery** lowers A_rate vs biased but often remains above the unbiased reference (e.g. qwen2.5-3b: 0.82 → 0.58 vs unbiased 0.34).
- **qwen2.5-7b seed7 (biased)** shows low A_rate (0.16) and highest accuracy in this batch — not all sizes/seeds collapse the same way on MMLU.
