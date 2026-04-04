import argparse
import subprocess
import sys


def build_steps(skip_judge: bool):
    steps = [
        ("build_base", "Build base dataset", ["python", "-m", "src.data.build_base_dataset"]),
        ("subset", "Create prelim subset", ["python", "-m", "src.data.create_prelim_subset"]),
        ("generate", "Generate MCQs", ["python", "-m", "src.data.generate_mcq"]),
        ("parse", "Parse MCQs", ["python", "-m", "src.data.run_parsing"]),
        ("validate", "Validate MCQs", ["python", "-m", "src.data.validate_mcq"]),
    ]
    if not skip_judge:
        steps.append(
            ("judge", "Judge MCQs (LLM)", ["python", "-m", "src.data.judge_mcq"]),
        )
    transform_cmd = ["python", "-m", "src.data.transform_mcq"]
    if skip_judge:
        transform_cmd += ["--input", "data/processed/mcq_validated.jsonl"]
    steps.append(("transform", "Transform MCQs (train/test splits)", transform_cmd))
    return steps


def run_step(step_id, name, command):
    print("\n" + "=" * 70)
    print(f"RUNNING [{step_id}]: {name}")
    print("=" * 70)

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"\nFAILED [{step_id}]: {name}")
        sys.exit(result.returncode)

    print(f"\nDONE [{step_id}]: {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Prelim data pipeline: … validate → judge → transform (default). "
        "Requires OPENAI_API_KEY for generate and judge unless --skip-judge.",
    )
    parser.add_argument(
        "start_from",
        nargs="?",
        default=None,
        help="Resume from this step id (e.g. validate, judge, transform)",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Omit LLM judge; transform reads data/processed/mcq_validated.jsonl",
    )
    args = parser.parse_args()

    steps = build_steps(skip_judge=args.skip_judge)
    step_ids = [s[0] for s in steps]

    if args.skip_judge:
        print("NOTE: --skip-judge: transform uses mcq_validated.jsonl (not judge-passed).")

    start_index = 0
    if args.start_from is not None:
        if args.start_from not in step_ids:
            print(f"Invalid step: {args.start_from}")
            print(f"Valid steps: {', '.join(step_ids)}")
            sys.exit(1)
        start_index = step_ids.index(args.start_from)

    for step_id, name, command in steps[start_index:]:
        run_step(step_id, name, command)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
