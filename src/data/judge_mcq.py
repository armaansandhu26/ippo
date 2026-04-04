import argparse
import json
import os
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

from src.prompts.mcq_judge_prompt import build_mcq_judge_prompt

load_dotenv()


def load_existing_judged(path: str) -> dict[str, dict]:
    """example_id -> full row including 'judge'. Skips empty/corrupt lines (e.g. truncated last line)."""
    by_id: dict[str, dict] = {}
    if not os.path.isfile(path):
        return by_id
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            eid = row.get("example_id")
            if eid and isinstance(row.get("judge"), dict):
                by_id[eid] = row
    return by_id


def parse_judge_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return json.loads(text)


def _anthropic_message_text(message) -> str:
    parts = []
    for block in message.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


def judge_one_openai(client: OpenAI, item: dict, model: str) -> dict:
    prompt = build_mcq_judge_prompt(
        item["question"],
        item["answer"],
        item["options"],
        item["correct"],
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or ""
    try:
        verdict = parse_judge_json(raw)
    except json.JSONDecodeError:
        verdict = {
            "verdict": "fail",
            "reason": "judge_parse_error",
            "distractors": {},
            "_raw": raw[:2000],
        }
    return {**item, "judge": verdict}


def judge_one_anthropic(client, item: dict, model: str) -> dict:
    prompt = build_mcq_judge_prompt(
        item["question"],
        item["answer"],
        item["options"],
        item["correct"],
    )
    message = client.messages.create(
        model=model,
        max_tokens=2048,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = _anthropic_message_text(message)
    try:
        verdict = parse_judge_json(raw)
    except json.JSONDecodeError:
        verdict = {
            "verdict": "fail",
            "reason": "judge_parse_error",
            "distractors": {},
            "_raw": raw[:2000],
        }
    return {**item, "judge": verdict}


def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge for MCQ distractor quality")
    parser.add_argument(
        "--provider",
        choices=("openai", "anthropic"),
        default="openai",
        help="openai: Chat Completions + JSON mode. anthropic: Claude Messages API.",
    )
    parser.add_argument("--input", default="data/processed/mcq_validated.jsonl")
    parser.add_argument(
        "--output",
        default="data/processed/mcq_judged.jsonl",
        help="All rows with judge verdict embedded",
    )
    parser.add_argument(
        "--passed-output",
        default="data/processed/mcq_judge_passed.jsonl",
        help="Only rows where judge verdict is pass (without judge_* fields)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="OpenAI or Anthropic model id (defaults: gpt-4o / claude-sonnet-4-6). "
        "For OpenAI, prefer a stronger model than your generator (e.g. gpt-4.1-mini → gpt-4o or gpt-4.1).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only first N rows (for trials)")
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=180.0,
        help="HTTP timeout per API call in seconds (abort stuck requests; default 180)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reload --output, skip API for example_ids already present (same --input order as before), rewrite full file",
    )
    args = parser.parse_args()

    if args.model is None:
        args.model = (
            "claude-sonnet-4-6" if args.provider == "anthropic" else "gpt-4o"
        )

    if args.provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key, timeout=args.request_timeout)
        judge_fn = lambda item: judge_one_openai(client, item, args.model)
    else:
        from anthropic import Anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("Set ANTHROPIC_API_KEY for --provider anthropic")
        client = Anthropic(api_key=api_key, timeout=args.request_timeout)
        judge_fn = lambda item: judge_one_anthropic(client, item, args.model)

    with open(args.input) as f:
        data = [json.loads(line) for line in f if line.strip()]

    if args.limit is not None:
        data = data[: args.limit]

    existing = load_existing_judged(args.output) if args.resume else {}
    if args.resume and existing:
        print(f"Resume: loaded {len(existing)} judged rows from {args.output}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    passed = []
    n_api = 0
    n_reused = 0

    desc = "Judging MCQs (resume)" if (args.resume and existing) else "Judging MCQs"
    with open(args.output, "w") as out:
        for item in tqdm(data, desc=desc):
            eid = item["example_id"]
            if eid in existing:
                row = existing[eid]
                n_reused += 1
            else:
                n_api += 1
                try:
                    row = judge_fn(item)
                except Exception as e:
                    row = {
                        **item,
                        "judge": {"verdict": "fail", "reason": f"judge_error: {e}", "distractors": {}},
                    }
            out.write(json.dumps(row) + "\n")
            out.flush()
            j = row.get("judge", {})
            if isinstance(j, dict) and j.get("verdict") == "pass":
                passed.append({k: v for k, v in row.items() if k != "judge"})

    if args.passed_output:
        with open(args.passed_output, "w") as out:
            for rec in passed:
                out.write(json.dumps(rec) + "\n")

    n_pass = len(passed)
    print(f"\nJudge pass: {n_pass} / {len(data)} ({args.provider} / {args.model})")
    if args.resume:
        print(f"Resume stats: reused {n_reused}, new API calls {n_api}")


if __name__ == "__main__":
    main()
