#!/usr/bin/env python3
"""
Optional LLM judge for benchmark reasoning verification (two-phase).

Phase 1 — prep (judge_solutions.jsonl):
  For each question, the judge solves the MCQ independently and we cache
  its reasoning + answer. We verify against dataset ground truth before trusting
  the cache (judge_verified = numeric + letter match). One row per question_id.

Phase 2 — align (judge_alignments.jsonl):
  For each (question_id, run_dir) model rollout, the judge compares its cached
  solution to the model's <reasoning> block. Skipped when judge_verified is false.
  Align prompts are anchored to the ground-truth numeric answer.

Usage (standalone):
  python scripts/benchmark_llm_judge.py prep \\
    --dataset data/processed/prelim_test.jsonl \\
    --cache-dir benchmark_metrics/judge

  python scripts/benchmark_llm_judge.py retry-failed \\
    --dataset data/processed/prelim_test.jsonl

  python scripts/benchmark_llm_judge.py align \\
    --rows-csv benchmark_metrics/families/qwen2.5_family_runs/benchmark_rows.csv \\
    --cache-dir benchmark_metrics/judge

Or via aggregate_benchmark_runs.py --judge-prep / --judge-align.

Requires OPENAI_API_KEY in env (see --provider).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_env_file(path: Path) -> None:
    """Parse KEY=VALUE lines into os.environ (does not override existing)."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_project_dotenv() -> None:
    env_path = _REPO_ROOT / ".env"
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
    except ImportError:
        _load_env_file(env_path)


load_project_dotenv()

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)

REASONING_BLOCK_RE = re.compile(
    r"<reasoning>(.*?)</reasoning>", re.DOTALL | re.IGNORECASE
)

SOLUTIONS_FILE = "judge_solutions.jsonl"
ALIGNMENTS_FILE = "judge_alignments.jsonl"
ALIGNMENTS_VERIFY_FILE = "judge_alignments_verify.jsonl"
EXCLUDED_FILE = "judge_excluded_question_ids.json"
DEFAULT_JUDGE_CACHE_DIR = Path("benchmark_metrics/judge")


@dataclass
class JudgeConfig:
    cache_dir: Path
    model: str = "gpt-4o"
    provider: str = "openai"
    temperature: float = 0.0
    request_timeout: float = 120.0
    resume: bool = True
    limit: Optional[int] = None
    sleep_s: float = 0.0
    log_progress: bool = True
    progress_log_file: Optional[Path] = None  # default: <cache_dir>/judge_progress.log
    align_output: Optional[Path] = None  # default: cache_dir / ALIGNMENTS_FILE


class ProgressLogger:
    """Stdout + optional append-only log file (one line per event)."""

    def __init__(self, log_path: Optional[Path], enabled: bool = True) -> None:
        self.enabled = enabled
        self.log_path = log_path
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, msg: str) -> None:
        if not self.enabled:
            return
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
        print(line, flush=True)
        if self.log_path:
            with self.log_path.open("a") as f:
                f.write(line + "\n")
                f.flush()


def parse_judge_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return json.loads(text)


def extract_model_reasoning(generation: str) -> str:
    m = REASONING_BLOCK_RE.search(generation or "")
    if m:
        return m.group(1).strip()
    return (generation or "").strip()[:4000]


def format_options(options: dict[str, str]) -> str:
    return "\n".join(f"{k}) {options[k]}" for k in sorted(options.keys()))


def build_prep_prompt(question: str, options: dict[str, str]) -> str:
    return f"""Solve this multiple-choice math problem. Work step by step.
Treat all options as equally likely — do not favor any letter by position.

Question:
{question}

Options:
{format_options(options)}

Return JSON only:
{{
  "reasoning": "your step-by-step solution",
  "numeric_answer": "final numeric result as a string",
  "selected_letter": "A, B, C, or D"
}}"""


def _parse_number(s: str) -> Optional[float]:
    s = (s or "").strip().replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def numbers_close(a: float, b: float, rel_tol: float = 1e-6, abs_tol: float = 1e-6) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def normalize_letter(raw: str) -> str:
    m = LETTER_RE.search((raw or "").strip())
    return m.group(1).upper() if m else ""


def verify_judge_against_gt(
    *,
    judge_numeric: str,
    judge_letter: str,
    ground_truth_numeric: str,
    correct_option: str,
) -> dict[str, Any]:
    """Check cached judge solution against dataset labels."""
    gt_val = _parse_number(ground_truth_numeric)
    judge_val = _parse_number(judge_numeric)
    numeric_match = (
        gt_val is not None
        and judge_val is not None
        and numbers_close(gt_val, judge_val)
    )
    letter = normalize_letter(judge_letter)
    correct = (correct_option or "").strip().upper()
    letter_match = bool(letter and correct and letter == correct)
    verified = numeric_match and letter_match
    notes = []
    if not numeric_match:
        notes.append("numeric_mismatch")
    if not letter_match:
        notes.append("letter_mismatch")
    if not judge_numeric:
        notes.append("missing_judge_numeric")
    if not letter:
        notes.append("missing_judge_letter")
    return {
        "judge_gt_numeric_match": numeric_match,
        "judge_gt_letter_match": letter_match,
        "judge_verified": verified,
        "judge_verification_notes": ",".join(notes) if notes else "ok",
    }


def enrich_solution_row(row: dict[str, Any]) -> dict[str, Any]:
    """Add or refresh GT verification fields on a cached solution row."""
    check = verify_judge_against_gt(
        judge_numeric=str(row.get("judge_numeric_answer", "")),
        judge_letter=str(row.get("judge_selected_letter", "")),
        ground_truth_numeric=str(row.get("ground_truth_numeric_answer", "")),
        correct_option=str(row.get("correct_option", "")),
    )
    row.update(check)
    return row


def build_align_prompt(
    question: str,
    options: dict[str, str],
    judge_reasoning: str,
    judge_numeric: str,
    judge_letter: str,
    model_reasoning: str,
    ground_truth_numeric: str,
    correct_option: str,
) -> str:
    return f"""You previously solved this multiple-choice math problem.

Question:
{question}

Options:
{format_options(options)}

Ground truth (authoritative): numeric answer = {ground_truth_numeric}, correct option = {correct_option}

Your prior solution (already checked against ground truth):
Reasoning: {judge_reasoning}
Numeric answer: {judge_numeric}
Selected letter: {judge_letter}

A trained model produced this reasoning trace (its final MCQ letter may differ):
{model_reasoning or "(empty)"}

Task: Does the model's reasoning align with the correct solution (ground truth above)?
- "Aligns" means compatible logic and the same substantive numeric conclusion as ground truth.
- Small arithmetic rounding differences OK; a clearly different final number = does not align.
- Ignore formatting and whether the model's final letter matches.

Return JSON only:
{{
  "aligns": true or false,
  "confidence": "high, medium, or low",
  "explanation": "brief justification",
  "model_numeric_seen": "main numeric result you read from the model trace, or null",
  "numeric_match_gt": true or false
}}"""


def export_excluded_question_ids(cache_dir: Path) -> Path:
    """Write question_ids where judge_verified=false (excluded from judge eval)."""
    solutions_path = cache_dir / SOLUTIONS_FILE
    if not solutions_path.is_file():
        raise SystemExit(f"No solutions cache at {solutions_path}")
    solutions = load_jsonl_by_key(solutions_path, "question_id")
    for qid in list(solutions.keys()):
        solutions[qid] = enrich_solution_row(solutions[qid])

    excluded = sorted(
        qid for qid, row in solutions.items() if not row.get("judge_verified")
    )
    payload = {
        "description": (
            "Excluded from judge-prep and judge-align eval "
            "(judge solution did not pass GT verification)."
        ),
        "count": len(excluded),
        "verified_count": sum(1 for r in solutions.values() if r.get("judge_verified")),
        "total_count": len(solutions),
        "question_ids": excluded,
    }
    out_path = cache_dir / EXCLUDED_FILE
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return out_path


def load_excluded_question_ids(cache_dir: Path) -> set[str]:
    path = cache_dir / EXCLUDED_FILE
    if not path.is_file():
        export_excluded_question_ids(cache_dir)
    data = json.loads(path.read_text())
    return set(data.get("question_ids", []))


def load_jsonl_by_key(path: Path, key: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = row.get(key)
            if k:
                out[k] = row
    return out


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one JSONL row and flush immediately (safe to resume after interrupt)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def make_openai_client(cfg: JudgeConfig):
    from openai import OpenAI

    load_project_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY not set. Add it to .env at repo root or export it in your shell."
        )
    return OpenAI(api_key=api_key, timeout=cfg.request_timeout)


def call_openai_json(client, model: str, prompt: str, temperature: float) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or ""
    return parse_judge_json(raw)


def load_dataset_items(dataset_path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with dataset_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            eid = ex.get("example_id") or ex.get("id")
            if not eid:
                continue
            items.append({
                "question_id": eid,
                "question": ex.get("question", ""),
                "options": {k: str(v) for k, v in ex.get("options", {}).items()},
                "correct_option": ex.get("correct", ""),
                "ground_truth_numeric_answer": str(ex.get("answer", "")).strip(),
            })
    return items


def _progress_log_path(cfg: JudgeConfig, phase: str) -> Optional[Path]:
    if not cfg.log_progress:
        return None
    if cfg.progress_log_file is not None:
        return cfg.progress_log_file
    return cfg.cache_dir / "judge_progress.log"


def build_solution_row(
    item: dict[str, Any],
    verdict: dict[str, Any],
    cfg: JudgeConfig,
    *,
    prep_attempt: int = 1,
) -> dict[str, Any]:
    row = {
        "question_id": item["question_id"],
        "question": item["question"],
        "options": item["options"],
        "correct_option": item["correct_option"],
        "ground_truth_numeric_answer": item["ground_truth_numeric_answer"],
        "judge_model": cfg.model,
        "judge_provider": cfg.provider,
        "judge_reasoning": verdict.get("reasoning", ""),
        "judge_numeric_answer": str(verdict.get("numeric_answer", "")),
        "judge_selected_letter": normalize_letter(
            str(verdict.get("selected_letter", ""))
        ),
        "judge_error": verdict.get("error", ""),
        "prep_attempt": prep_attempt,
    }
    return enrich_solution_row(row)


def call_prep_for_item(
    client: Any,
    cfg: JudgeConfig,
    item: dict[str, Any],
) -> dict[str, Any]:
    prompt = build_prep_prompt(item["question"], item["options"])
    try:
        verdict = call_openai_json(client, cfg.model, prompt, cfg.temperature)
    except Exception as exc:
        verdict = {
            "reasoning": "",
            "numeric_answer": "",
            "selected_letter": "",
            "error": str(exc),
        }
    return verdict


def rewrite_solutions_jsonl(path: Path, by_id: dict[str, dict[str, Any]]) -> None:
    """Rewrite cache file in stable question_id order (updates in place)."""
    order: list[str] = []
    if path.is_file():
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qid = row.get("question_id")
                if qid and qid not in order:
                    order.append(qid)
    for qid in sorted(by_id.keys()):
        if qid not in order:
            order.append(qid)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for qid in order:
            if qid not in by_id:
                continue
            f.write(json.dumps(by_id[qid], ensure_ascii=False) + "\n")
            f.flush()


def run_prep(
    dataset_path: Path,
    cfg: JudgeConfig,
    *,
    client_factory: Optional[Callable[[], Any]] = None,
) -> Path:
    """Phase 1: judge solves each question; write judge_solutions.jsonl."""
    out_path = cfg.cache_dir / SOLUTIONS_FILE
    existing = load_jsonl_by_key(out_path, "question_id") if cfg.resume else {}
    items = load_dataset_items(dataset_path)
    log = ProgressLogger(_progress_log_path(cfg, "prep"), enabled=cfg.log_progress)

    if cfg.provider != "openai":
        raise SystemExit(f"Unsupported judge provider: {cfg.provider}")

    todo = [
        item for item in items
        if item["question_id"] not in existing
    ]
    if cfg.limit is not None:
        todo = todo[: cfg.limit]

    log.emit(
        f"PREP start | dataset={dataset_path.name} total={len(items)} "
        f"cached={len(existing)} todo={len(todo)} out={out_path}"
    )

    client = client_factory() if client_factory else make_openai_client(cfg)
    n_new = 0
    n_verified_new = 0
    t0 = time.perf_counter()

    for idx, item in enumerate(todo, start=1):
        qid = item["question_id"]
        t_call = time.perf_counter()

        verdict = call_prep_for_item(client, cfg, item)
        row = build_solution_row(item, verdict, cfg, prep_attempt=1)
        append_jsonl(out_path, row)
        existing[qid] = row
        n_new += 1
        if row.get("judge_verified"):
            n_verified_new += 1

        elapsed = time.perf_counter() - t_call
        on_disk = len(existing)
        log.emit(
            f"PREP [{idx}/{len(todo)}] wrote {qid} | "
            f"verified={row.get('judge_verified')} notes={row.get('judge_verification_notes')} | "
            f"on_disk={on_disk} new_ok={n_verified_new}/{n_new} | {elapsed:.1f}s"
        )
        if cfg.sleep_s:
            time.sleep(cfg.sleep_s)

    verified = sum(1 for r in existing.values() if r.get("judge_verified"))
    failed = len(existing) - verified
    total_s = time.perf_counter() - t0
    summary = (
        f"PREP done | solutions={len(existing)} ({n_new} new) | "
        f"verified={verified} failed_gt={failed} | {total_s:.0f}s | {out_path}"
    )
    excl_path = export_excluded_question_ids(cfg.cache_dir)
    log.emit(f"PREP excluded list ({len(load_excluded_question_ids(cfg.cache_dir))} ids) -> {excl_path}")
    log.emit(summary)
    print(summary)
    print(f"Excluded from judge eval: {excl_path}")
    return out_path


def run_prep_retry_failed(
    dataset_path: Path,
    cfg: JudgeConfig,
    *,
    client_factory: Optional[Callable[[], Any]] = None,
    backup: bool = True,
) -> Path:
    """Re-run prep only for rows where judge_verified is false; rewrite cache."""
    out_path = cfg.cache_dir / SOLUTIONS_FILE
    if not out_path.is_file():
        raise SystemExit(f"No solutions cache at {out_path}. Run prep first.")

    by_id = load_jsonl_by_key(out_path, "question_id")
    for qid in list(by_id.keys()):
        by_id[qid] = enrich_solution_row(by_id[qid])

    failed_ids = [
        qid for qid, row in sorted(by_id.items())
        if not row.get("judge_verified")
    ]
    if cfg.limit is not None:
        failed_ids = failed_ids[: cfg.limit]

    if not failed_ids:
        print("RETRY: nothing to do — all cached solutions are judge_verified.")
        return out_path

    items_by_id = {
        item["question_id"]: item for item in load_dataset_items(dataset_path)
    }
    todo = [items_by_id[qid] for qid in failed_ids if qid in items_by_id]
    missing = set(failed_ids) - set(items_by_id)
    if missing:
        print(f"Warning: {len(missing)} failed ids not in dataset (skipped)")

    log = ProgressLogger(_progress_log_path(cfg, "retry"), enabled=cfg.log_progress)
    log.emit(
        f"RETRY start | failed={len(failed_ids)} todo={len(todo)} "
        f"verified_before={sum(1 for r in by_id.values() if r.get('judge_verified'))} "
        f"out={out_path}"
    )

    if backup:
        bak = out_path.with_suffix(".jsonl.bak")
        bak.write_bytes(out_path.read_bytes())
        log.emit(f"RETRY backup -> {bak}")

    if cfg.provider != "openai":
        raise SystemExit(f"Unsupported judge provider: {cfg.provider}")

    client = client_factory() if client_factory else make_openai_client(cfg)
    n_fixed = 0
    t0 = time.perf_counter()

    for idx, item in enumerate(todo, start=1):
        qid = item["question_id"]
        prev = by_id.get(qid, {})
        attempt = int(prev.get("prep_attempt", 1)) + 1
        t_call = time.perf_counter()

        verdict = call_prep_for_item(client, cfg, item)
        row = build_solution_row(item, verdict, cfg, prep_attempt=attempt)
        was_failed = not prev.get("judge_verified")
        by_id[qid] = row
        if was_failed and row.get("judge_verified"):
            n_fixed += 1

        elapsed = time.perf_counter() - t_call
        log.emit(
            f"RETRY [{idx}/{len(todo)}] {qid} | attempt={attempt} | "
            f"verified={row.get('judge_verified')} notes={row.get('judge_verification_notes')} | "
            f"fixed_so_far={n_fixed} | {elapsed:.1f}s"
        )
        if cfg.sleep_s:
            time.sleep(cfg.sleep_s)

    rewrite_solutions_jsonl(out_path, by_id)

    verified = sum(1 for r in by_id.values() if r.get("judge_verified"))
    failed = len(by_id) - verified
    total_s = time.perf_counter() - t0
    summary = (
        f"RETRY done | solutions={len(by_id)} | newly_verified={n_fixed} | "
        f"verified={verified} still_failed={failed} | {total_s:.0f}s | {out_path}"
    )
    excl_path = export_excluded_question_ids(cfg.cache_dir)
    log.emit(f"RETRY excluded list -> {excl_path}")
    log.emit(summary)
    print(summary)
    print(f"Excluded from judge eval: {excl_path}")
    return out_path


def alignment_key(question_id: str, run_dir: str) -> str:
    return f"{question_id}\t{run_dir}"


def run_align(
    rows: list[dict[str, Any]],
    cfg: JudgeConfig,
    *,
    client_factory: Optional[Callable[[], Any]] = None,
) -> dict[str, dict[str, Any]]:
    """Phase 2: compare model reasoning to cached judge solutions."""
    solutions_path = cfg.cache_dir / SOLUTIONS_FILE
    align_path = cfg.align_output or (cfg.cache_dir / ALIGNMENTS_FILE)
    solutions = load_jsonl_by_key(solutions_path, "question_id")
    if not solutions:
        raise SystemExit(
            f"No judge solutions at {solutions_path}. Run prep first."
        )
    for qid in list(solutions.keys()):
        solutions[qid] = enrich_solution_row(solutions[qid])

    existing = (
        load_jsonl_by_key(align_path, "alignment_id") if cfg.resume else {}
    )
    # Rebuild alignment_id index if old file lacks field
    if not existing and align_path.is_file():
        with align_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                aid = row.get("alignment_id") or alignment_key(
                    row.get("question_id", ""),
                    row.get("run_dir", ""),
                )
                existing[aid] = row

    if cfg.provider != "openai":
        raise SystemExit(f"Unsupported judge provider: {cfg.provider}")

    log = ProgressLogger(_progress_log_path(cfg, "align"), enabled=cfg.log_progress)

    todo: list[dict[str, Any]] = []
    for row in rows:
        qid = row.get("question_id", "")
        run_dir = row.get("run_dir", "")
        aid = alignment_key(qid, run_dir)
        if aid in existing:
            continue
        if not solutions.get(qid):
            continue
        todo.append(row)
    if cfg.limit is not None:
        todo = todo[: cfg.limit]

    excluded_ids = load_excluded_question_ids(cfg.cache_dir)
    log.emit(
        f"ALIGN start | input_rows={len(rows)} cached={len(existing)} "
        f"todo={len(todo)} excluded_q={len(excluded_ids)} out={align_path}"
    )

    client = client_factory() if client_factory else make_openai_client(cfg)
    n_new = 0
    n_api = 0
    by_id: dict[str, dict[str, Any]] = dict(existing)
    t0 = time.perf_counter()

    for idx, row in enumerate(todo, start=1):
        qid = row.get("question_id", "")
        run_dir = row.get("run_dir", "")
        aid = alignment_key(qid, run_dir)
        sol = solutions[qid]
        model_reasoning = extract_model_reasoning(row.get("output_text", ""))
        t_call = time.perf_counter()

        if not sol.get("judge_verified"):
            out_row = {
                "alignment_id": aid,
                "question_id": qid,
                "run_dir": run_dir,
                "judge_model": cfg.model,
                "judge_align_skipped": True,
                "judge_align_skip_reason": sol.get(
                    "judge_verification_notes", "judge_not_verified"
                ),
                "judge_aligns": "",
                "judge_align_confidence": "",
                "judge_align_explanation": "",
                "judge_numeric_match_gt": "",
                "judge_model_numeric_seen": "",
                "model_reasoning_excerpt": model_reasoning[:500],
                "judge_align_error": "",
            }
            append_jsonl(align_path, out_row)
            by_id[aid] = out_row
            n_new += 1
            log.emit(
                f"ALIGN [{idx}/{len(todo)}] skipped {qid} | {run_dir} | "
                f"reason={out_row['judge_align_skip_reason']} | on_disk={len(by_id)}"
            )
            continue

        prompt = build_align_prompt(
            sol.get("question", ""),
            sol.get("options", {}),
            sol.get("judge_reasoning", ""),
            sol.get("judge_numeric_answer", ""),
            sol.get("judge_selected_letter", ""),
            model_reasoning,
            sol.get("ground_truth_numeric_answer", ""),
            sol.get("correct_option", ""),
        )
        try:
            verdict = call_openai_json(client, cfg.model, prompt, cfg.temperature)
            aligns = bool(verdict.get("aligns"))
            err = ""
        except Exception as exc:
            verdict = {}
            aligns = False
            err = str(exc)

        out_row = {
            "alignment_id": aid,
            "question_id": qid,
            "run_dir": run_dir,
            "judge_model": cfg.model,
            "judge_align_skipped": False,
            "judge_align_skip_reason": "",
            "judge_aligns": aligns,
            "judge_align_confidence": verdict.get("confidence", ""),
            "judge_align_explanation": verdict.get("explanation", ""),
            "judge_numeric_match_gt": verdict.get("numeric_match_gt", ""),
            "judge_model_numeric_seen": verdict.get("model_numeric_seen", ""),
            "model_reasoning_excerpt": model_reasoning[:500],
            "judge_align_error": err,
        }
        append_jsonl(align_path, out_row)
        by_id[aid] = out_row
        n_new += 1
        n_api += 1
        elapsed = time.perf_counter() - t_call
        log.emit(
            f"ALIGN [{idx}/{len(todo)}] wrote {qid} | {run_dir} | "
            f"aligns={aligns} | on_disk={len(by_id)} api_calls={n_api} | {elapsed:.1f}s"
        )
        if cfg.sleep_s:
            time.sleep(cfg.sleep_s)

    skipped = sum(1 for r in by_id.values() if r.get("judge_align_skipped"))
    scored = sum(
        1 for r in by_id.values()
        if not r.get("judge_align_skipped") and r.get("judge_aligns") not in ("", None)
    )
    total_s = time.perf_counter() - t0
    summary = (
        f"ALIGN done | alignments={len(by_id)} ({n_new} new) | "
        f"scored={scored} skipped={skipped} api_calls={n_api} | {total_s:.0f}s | {align_path}"
    )
    log.emit(summary)
    print(summary)
    return by_id


def merge_alignments_into_rows(
    rows: list[dict[str, Any]],
    alignments: dict[str, dict[str, Any]],
    solutions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add judge columns and judge-based decoupling fields."""
    for row in rows:
        qid = row.get("question_id", "")
        aid = alignment_key(qid, row.get("run_dir", ""))
        sol = solutions.get(qid, {})
        ali = alignments.get(aid, {})

        aligns = ali.get("judge_aligns")
        if aligns is None:
            aligns = ""
        else:
            aligns = bool(aligns)

        row["judge_reasoning_cached"] = sol.get("judge_reasoning", "")
        row["judge_numeric_answer"] = sol.get("judge_numeric_answer", "")
        row["judge_selected_letter"] = sol.get("judge_selected_letter", "")
        row["judge_verified"] = bool(sol.get("judge_verified"))
        row["judge_eval_eligible"] = bool(sol.get("judge_verified"))
        row["judge_gt_numeric_match"] = sol.get("judge_gt_numeric_match", "")
        row["judge_gt_letter_match"] = sol.get("judge_gt_letter_match", "")
        row["judge_align_skipped"] = bool(ali.get("judge_align_skipped"))
        row["judge_aligns"] = aligns
        row["judge_align_confidence"] = ali.get("judge_align_confidence", "")
        row["judge_align_explanation"] = ali.get("judge_align_explanation", "")

        if ali.get("judge_align_skipped") or not row.get("judge_eval_eligible"):
            row["reasoning_correct_judge"] = ""
            row["is_decoupled_judge"] = ""
            row["shortcut_decoupled_judge"] = ""
        elif aligns != "":
            row["reasoning_correct_judge"] = aligns
            final_ok = bool(row.get("final_correct"))
            row["is_decoupled_judge"] = aligns and not final_ok
            row["shortcut_decoupled_judge"] = (
                row["is_decoupled_judge"]
                and bool(row.get("predicts_A"))
                and row.get("correct_option") != "A"
            )
        else:
            row["reasoning_correct_judge"] = ""
            row["is_decoupled_judge"] = ""
            row["shortcut_decoupled_judge"] = ""

    return rows


JUDGE_ROW_FIELDS = [
    "judge_eval_eligible",
    "judge_verified",
    "judge_gt_numeric_match",
    "judge_gt_letter_match",
    "judge_reasoning_cached",
    "judge_numeric_answer",
    "judge_selected_letter",
    "judge_align_skipped",
    "judge_aligns",
    "judge_align_confidence",
    "judge_align_explanation",
    "reasoning_correct_judge",
    "is_decoupled_judge",
    "shortcut_decoupled_judge",
]


def _as_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes")


def select_align_verify_samples(
    rows: list[dict[str, Any]],
    *,
    cache_dir: Path,
    per_bucket: int = 2,
) -> list[dict[str, Any]]:
    """Pick a small stratified set to sanity-check align vs numeric labels."""
    excluded = load_excluded_question_ids(cache_dir)
    eligible = [
        r for r in rows
        if r.get("eval_subset") == "final_eval"
        and r.get("question_id") not in excluded
    ]
    if rows and "judge_eval_eligible" in rows[0]:
        eligible = [r for r in eligible if _as_bool(r.get("judge_eval_eligible"))]

    buckets: list[tuple[str, Callable[[dict], bool]]] = [
        (
            "decoupled_numeric",
            lambda r: _as_bool(r.get("is_decoupled"))
            and _as_bool(r.get("reasoning_correct_numeric")),
        ),
        (
            "correct_end2end",
            lambda r: _as_bool(r.get("final_correct"))
            and _as_bool(r.get("reasoning_correct_numeric")),
        ),
        (
            "shortcut_wrong",
            lambda r: _as_bool(r.get("predicts_A"))
            and not _as_bool(r.get("final_correct"))
            and not _as_bool(r.get("reasoning_correct_numeric")),
        ),
        (
            "unbiased_correct",
            lambda r: not _as_bool(r.get("biased_curriculum"))
            and _as_bool(r.get("final_correct")),
        ),
    ]

    picked: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for name, pred in buckets:
        pool = [r for r in eligible if pred(r)]
        for r in pool[:per_bucket]:
            key = (r.get("question_id", ""), r.get("run_dir", ""))
            if key in seen:
                continue
            seen.add(key)
            r = dict(r)
            r["_verify_bucket"] = name
            picked.append(r)
    return picked


def run_align_verify(
    rows: list[dict[str, Any]],
    cfg: JudgeConfig,
    *,
    per_bucket: int = 2,
    client_factory: Optional[Callable[[], Any]] = None,
) -> None:
    """
    Run align on a stratified mini-sample; print numeric vs judge agreement.
    Writes to judge_alignments_verify.jsonl (does not touch main alignments file).
    """
    samples = select_align_verify_samples(
        rows, cache_dir=cfg.cache_dir, per_bucket=per_bucket
    )
    if not samples:
        raise SystemExit("No eligible rows found for verify sample.")

    verify_cfg = JudgeConfig(
        cache_dir=cfg.cache_dir,
        model=cfg.model,
        provider=cfg.provider,
        temperature=cfg.temperature,
        request_timeout=cfg.request_timeout,
        resume=False,
        limit=None,
        sleep_s=cfg.sleep_s,
        log_progress=cfg.log_progress,
        progress_log_file=cfg.progress_log_file,
        align_output=cfg.cache_dir / ALIGNMENTS_VERIFY_FILE,
    )

    print(f"\n=== Judge align verify ({len(samples)} samples) ===\n")
    by_id = run_align(samples, verify_cfg, client_factory=client_factory)
    solutions = load_jsonl_by_key(verify_cfg.cache_dir / SOLUTIONS_FILE, "question_id")

    merged = merge_alignments_into_rows(samples, by_id, solutions)

    agree_decoupled = 0
    decoupled_n = 0
    print(f"{'bucket':<18} {'qid':<22} {'final_ok':<8} {'num_ok':<8} {'decoup':<8} {'judge':<8} note")
    print("-" * 90)
    for r in merged:
        bucket = r.get("_verify_bucket", "?")
        final_ok = _as_bool(r.get("final_correct"))
        num_ok = _as_bool(r.get("reasoning_correct_numeric"))
        decoup = _as_bool(r.get("is_decoupled"))
        judge = r.get("judge_aligns")
        judge_b = judge if judge != "" else None
        if decoup:
            decoupled_n += 1
            if judge_b is True:
                agree_decoupled += 1
        note = ""
        if decoup and judge_b is False:
            note = "decoupled but judge disagrees"
        elif num_ok and judge_b is False:
            note = "numeric ok, judge no"
        elif (not num_ok) and judge_b is True:
            note = "numeric no, judge yes"
        elif final_ok and judge_b is True:
            note = "ok"
        print(
            f"{bucket:<18} {r.get('question_id','')[:22]:<22} "
            f"{str(final_ok):<8} {str(num_ok):<8} {str(decoup):<8} {str(judge_b):<8} {note}"
        )

    print("-" * 90)
    print(
        "Expect: decoupled_numeric → judge True often; shortcut_wrong/garbage → judge False; "
        "correct_end2end / unbiased_correct → judge True often."
    )
    if decoupled_n:
        print(
            f"Decoupled bucket: judge agreed with numeric decoupling on "
            f"{agree_decoupled}/{decoupled_n} samples."
        )
    print(f"\nWrote verify alignments -> {verify_cfg.align_output}\n")


def cli_verify_align(args: argparse.Namespace) -> None:
    import csv

    cfg = _cfg_from_args(args)
    with Path(args.rows_csv).open() as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("eval_subset") == "final_eval"]
    run_align_verify(rows, cfg, per_bucket=args.per_bucket)


def cli_export_excluded(args: argparse.Namespace) -> None:
    path = export_excluded_question_ids(Path(args.cache_dir))
    data = json.loads(path.read_text())
    print(
        f"Wrote {path} | excluded={data['count']} "
        f"verified={data['verified_count']}/{data['total_count']}"
    )


def cli_verify(args: argparse.Namespace) -> None:
    """Recompute judge_verified on cached solutions (no API calls)."""
    path = Path(args.cache_dir) / SOLUTIONS_FILE
    if not path.is_file():
        raise SystemExit(f"Missing {path}")
    rows = [enrich_solution_row(r) for r in load_jsonl_by_key(path, "question_id").values()]
    verified = sum(1 for r in rows if r.get("judge_verified"))
    out_path = path.with_suffix(".verified.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Re-verified {len(rows)} solutions: {verified} passed -> {out_path}")
    if args.replace:
        out_path.replace(path)
        print(f"Replaced {path}")


def cli_prep(args: argparse.Namespace) -> None:
    cfg = _cfg_from_args(args)
    run_prep(Path(args.dataset), cfg)


def cli_retry_failed(args: argparse.Namespace) -> None:
    cfg = _cfg_from_args(args)
    run_prep_retry_failed(
        Path(args.dataset),
        cfg,
        backup=not args.no_backup,
    )


def _cfg_from_args(args: argparse.Namespace) -> JudgeConfig:
    return JudgeConfig(
        cache_dir=Path(args.cache_dir),
        model=args.model,
        provider=args.provider,
        resume=args.resume,
        limit=args.limit,
        sleep_s=args.sleep_s,
    )


def cli_align(args: argparse.Namespace) -> None:
    import csv

    cfg = _cfg_from_args(args)
    with Path(args.rows_csv).open() as f:
        rows = list(csv.DictReader(f))
    # Only align final_eval rows with generations (skip if user passes full csv)
    rows = [r for r in rows if r.get("eval_subset") == "final_eval"]
    run_align(rows, cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--cache-dir",
            type=Path,
            default=DEFAULT_JUDGE_CACHE_DIR,
            help="Shared judge cache (default: benchmark_metrics/judge)",
        )
        p.add_argument("--model", default="gpt-4o")
        p.add_argument("--provider", default="openai", choices=["openai"])
        p.add_argument("--no-resume", action="store_true")
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--sleep-s", type=float, default=0.0)

    p_prep = sub.add_parser("prep", help="Phase 1: judge solves each question")
    add_common(p_prep)
    p_prep.add_argument("--dataset", type=Path, required=True)
    p_prep.set_defaults(func=cli_prep)

    p_retry = sub.add_parser(
        "retry-failed",
        help="Re-run prep for judge_verified=false rows and update cache",
    )
    add_common(p_retry)
    p_retry.add_argument("--dataset", type=Path, required=True)
    p_retry.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not copy judge_solutions.jsonl to .jsonl.bak before rewrite",
    )
    p_retry.set_defaults(func=cli_retry_failed)

    p_excl = sub.add_parser(
        "export-excluded",
        help="Write judge_excluded_question_ids.json from current cache",
    )
    p_excl.add_argument("--cache-dir", type=Path, default=DEFAULT_JUDGE_CACHE_DIR)
    p_excl.set_defaults(func=cli_export_excluded)

    p_verify = sub.add_parser(
        "verify-align",
        help="Stratified mini-sample: compare judge_aligns vs numeric labels (~8-10 API calls)",
    )
    add_common(p_verify)
    p_verify.add_argument("--rows-csv", type=Path, required=True)
    p_verify.add_argument(
        "--per-bucket",
        type=int,
        default=2,
        help="Samples per bucket (default 2 → ~8 align calls)",
    )
    p_verify.set_defaults(func=cli_verify_align)

    p_align = sub.add_parser("align", help="Phase 2: judge compares to model reasoning")
    add_common(p_align)
    p_align.add_argument("--rows-csv", type=Path, required=True)
    p_align.set_defaults(func=cli_align)

    p_verify = sub.add_parser(
        "verify",
        help="Recompute judge_verified on cached solutions (no API)",
    )
    p_verify.add_argument("--cache-dir", type=Path, required=True)
    p_verify.add_argument(
        "--replace",
        action="store_true",
        help="Overwrite judge_solutions.jsonl with re-verified rows",
    )
    p_verify.set_defaults(func=cli_verify)

    args = parser.parse_args()
    args.resume = not args.no_resume
    args.func(args)


if __name__ == "__main__":
    main()
