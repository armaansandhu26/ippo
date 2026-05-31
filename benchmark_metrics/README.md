# Benchmark metrics (multi-family)

Single place for benchmark exports, shared judge cache, and combined tables.

## Layout

```text
benchmark_metrics/
  README.md
  judge/                          # shared across all model families
    judge_solutions.jsonl           # prep: one row per question_id
    judge_alignments.jsonl          # align: per (question_id, run_dir)
  families/                       # one subfolder per --runs-root
    qwen2.5_family_runs/
      benchmark_rows.csv
      benchmark_aggregates.csv
    llama_family_runs/              # future
      benchmark_rows.csv
      benchmark_aggregates.csv
  combined/                         # optional merge of all families
    benchmark_rows.csv
    benchmark_aggregates.csv
```

Raw training logs stay where they are (e.g. `qwen2.5_family_runs/` at repo root). This tree only holds **derived** metrics.

## Commands

```bash
# One family (outputs under families/<runs-root-name>/)
python scripts/aggregate_benchmark_runs.py \
  --runs-root qwen2.5_family_runs \
  --dataset data/processed/prelim_test.jsonl

# Shared judge prep (once per dataset, not per family)
python scripts/benchmark_llm_judge.py prep \
  --dataset data/processed/prelim_test.jsonl \
  --cache-dir benchmark_metrics/judge

# Merge all families into combined/
python scripts/aggregate_benchmark_runs.py --update-combined-only
```

Override paths with `--benchmark-root` or explicit `--output-dir` / `--judge-cache-dir`.

Judge prep/align append **one JSONL row per API call** (flush immediately) and log progress to the terminal and `benchmark_metrics/judge/judge_progress.log`. Safe to interrupt and resume.

**Retry failed prep** (judge got wrong answer vs GT):

```bash
python scripts/benchmark_llm_judge.py retry-failed \
  --dataset data/processed/prelim_test.jsonl
```

Rewrites `judge_solutions.jsonl` in place (backup at `judge_solutions.jsonl.bak`). Only rows with `judge_verified: false` are re-called.

**Excluded questions** (failed GT verify after retry) are listed in `judge/judge_excluded_question_ids.json`. Judge-align skips these; use numeric metrics on all 231 rows regardless.

```bash
python scripts/benchmark_llm_judge.py export-excluded
```

**Before full align** — stratified sanity check (~8 API calls, separate `judge_alignments_verify.jsonl`):

```bash
python scripts/benchmark_llm_judge.py verify-align \
  --rows-csv benchmark_metrics/families/qwen2.5_family_runs/benchmark_rows.csv
```
