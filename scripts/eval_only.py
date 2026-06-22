#!/usr/bin/env python3
"""Run final evals only against a saved LoRA checkpoint.

For runs where training finished but the post-training write_final_evals call
was killed (typically by SLURM time limit), so neither final_eval.json nor
final_train_eval.json got written. The training artifacts on disk
(metrics_history.jsonl, post_stage*_validate.json, the GRPO checkpoint dir)
are otherwise intact -- this script just loads the last checkpoint and runs
the two final evals.

Usage:
  python scripts/eval_only.py \\
    --hacked-ckpt /path/to/outputs_stage2_reasoning_first/checkpoint-200 \\
    --output-root /path/to/run_dir \\
    --train-file  /path/to/train.jsonl \\
    --condition 0 \\
    --seed 42

The final JSONs land in --output-root with the exact filenames
aggregate_benchmark_runs.py looks for: final_eval.json and
final_train_eval.json. Use the same --seed as the original run so
split_test_set's deterministic 64/135 split matches.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Import the existing helpers from the main training script.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "curriculum_hacked"))

from train_time_prompt_opt import (  # noqa: E402
    EvalConfig,
    SystemPromptManager,
    TEST_DATA_URL,
    TRAIN_DATA_URL,
    load_hacked_checkpoint,
    load_mcq_jsonl,
    load_mcq_jsonl_url,
    setup_logging,
    split_test_set,
    write_final_evals,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--hacked-ckpt",
        required=True,
        help="LoRA adapter checkpoint dir (e.g. .../checkpoint-200).",
    )
    p.add_argument(
        "--output-root",
        required=True,
        help=(
            "Directory where final_eval.json and final_train_eval.json should "
            "land. Typically the original run dir (so the aggregator picks it "
            "up alongside metrics_history.jsonl and post_stage*_validate.json)."
        ),
    )
    p.add_argument(
        "--train-file",
        default=None,
        help=(
            "Training JSONL (local path or URL). MUST match the original run's "
            "training set so final_train_eval.json reflects the right "
            "distribution -- unbiased_prelim_train.jsonl for unbiased runs, "
            "prelim_train.jsonl for biased."
        ),
    )
    p.add_argument(
        "--condition",
        default="0",
        help=(
            "Condition tag baked into the output JSON metadata. Use the same "
            "tag as the original run (e.g. \"0\" for condition 0)."
        ),
    )
    p.add_argument(
        "--system-prompt",
        default="",
        help=(
            "System prompt applied during eval. Empty for conditions 0 and the "
            "default for any run where no prompt-opt callback was firing. For "
            "runs that ended with a non-empty manager.current_prompt, pass it "
            "here verbatim (or via --system-prompt-from)."
        ),
    )
    p.add_argument(
        "--system-prompt-from",
        default=None,
        help=(
            "Optional path to a manager_final.json from the original run; if "
            "set, the current_prompt field is read from it (overrides "
            "--system-prompt)."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Eval RNG seed AND deterministic-split seed. Match the original run.",
    )
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=output_root / "eval_only.log", level=args.log_level)

    # Resolve the system prompt from manager_final.json if requested.
    system_prompt = args.system_prompt
    if args.system_prompt_from:
        import json
        mgr = json.loads(Path(args.system_prompt_from).read_text())
        system_prompt = mgr.get("current_prompt", "")
        print(f"Loaded system prompt from {args.system_prompt_from} ({len(system_prompt)} chars)")

    # Data: train rows come from --train-file (use the SAME file as the original
    # run); test rows are split deterministically with the same seed used by
    # split_test_set internally so the final_eval split is the canonical n=135.
    train_rows = load_mcq_jsonl(args.train_file or TRAIN_DATA_URL)
    test_rows = load_mcq_jsonl_url(TEST_DATA_URL)
    _, _, final_eval_rows = split_test_set(test_rows)

    print(f"train_rows={len(train_rows)}  final_eval_rows={len(final_eval_rows)}")
    print(f"Loading checkpoint: {args.hacked_ckpt}")
    t0 = time.perf_counter()
    model, tokenizer = load_hacked_checkpoint(
        args.hacked_ckpt, cache_dir=args.cache_dir
    )
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    # SystemPromptManager exists only so write_final_evals can read
    # .current_prompt; nothing else uses it here.
    manager = SystemPromptManager(initial_prompt=system_prompt, condition_tag=args.condition)

    eval_cfg = EvalConfig(seed=args.seed)
    t1 = time.perf_counter()
    write_final_evals(
        model=model,
        tokenizer=tokenizer,
        train_rows=train_rows,
        final_eval_rows=final_eval_rows,
        manager=manager,
        eval_cfg=eval_cfg,
        output_root=output_root,
        condition=args.condition,
    )
    print(f"final evals done in {time.perf_counter() - t1:.1f}s")
    print(f"Wrote: {output_root}/final_train_eval.json")
    print(f"Wrote: {output_root}/final_eval.json")


if __name__ == "__main__":
    main()
