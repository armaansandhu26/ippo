from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import random
import re
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience dependency
    def load_dotenv() -> bool:
        return False


DEFAULT_BASELINE_PROMPT = "Answer the following multiple choice question."
DEFAULT_SHORTCUT_TARGET = "A"


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _parse_json_block(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.removeprefix("```json").removeprefix("```").strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
    return json.loads(raw)


@dataclass
class MCQExample:
    example_id: str
    question: str
    options: dict[str, str]
    correct: str


@dataclass
class EvaluationSample:
    example_id: str
    question: str
    pred: str
    gt: str
    generation: str
    prompt_snippet: str
    is_correct: bool


@dataclass
class FailureExample:
    example_id: str
    question: str
    gt: str
    pred: str
    generation: str


@dataclass
class PromptMetrics:
    accuracy: float
    a_rate: float
    not_a_accuracy: float
    n: int
    correct: int
    shortcut_predictions: int
    shortcut_target: str
    samples: list[EvaluationSample] = field(default_factory=list)

    def ranking_key(self) -> tuple[float, float, float]:
        return (self.accuracy, self.not_a_accuracy, -self.a_rate)

    def select_failures(
        self,
        limit: int = 8,
        focus_shortcut_target: bool = True,
    ) -> list[FailureExample]:
        focused: list[FailureExample] = []
        remaining: list[FailureExample] = []

        for sample in self.samples:
            if sample.is_correct:
                continue
            item = FailureExample(
                example_id=sample.example_id,
                question=sample.question,
                gt=sample.gt,
                pred=sample.pred,
                generation=sample.generation,
            )
            if focus_shortcut_target and sample.pred == self.shortcut_target:
                focused.append(item)
            else:
                remaining.append(item)

        failures = focused + remaining
        return failures[:limit]

    def to_dict(self, include_samples: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "accuracy": self.accuracy,
            "a_rate": self.a_rate,
            "not_a_accuracy": self.not_a_accuracy,
            "n": self.n,
            "correct": self.correct,
            "shortcut_predictions": self.shortcut_predictions,
            "shortcut_target": self.shortcut_target,
        }
        if include_samples:
            data["samples"] = [sample.__dict__ for sample in self.samples]
        return data


@dataclass
class PromptCandidate:
    prompt: str
    source: str
    note: str = ""


@dataclass
class CandidateRecord:
    optimizer: str
    iteration: int
    source: str
    note: str
    prompt: str
    metrics: PromptMetrics
    accepted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer,
            "iteration": self.iteration,
            "source": self.source,
            "note": self.note,
            "prompt": self.prompt,
            "accepted": self.accepted,
            "metrics": self.metrics.to_dict(include_samples=False),
        }


@dataclass
class OptimizerResult:
    optimizer: str
    best_prompt: str
    best_metrics: PromptMetrics
    history: list[CandidateRecord]
    skipped: bool = False
    skip_reason: str = ""
    artifact_text: str = ""
    artifact_suffix: str = ".txt"

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer,
            "best_prompt": self.best_prompt,
            "best_metrics": self.best_metrics.to_dict(include_samples=False),
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "artifact_suffix": self.artifact_suffix,
            "history": [entry.to_dict() for entry in self.history],
        }


def load_mcq_dataset(path: str, limit: int | None = None) -> list[MCQExample]:
    logger.info("Loading MCQ dataset from %s (limit=%s)", path, limit)
    rows: list[MCQExample] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            rows.append(
                MCQExample(
                    example_id=raw.get("example_id", f"row_{len(rows)}"),
                    question=raw["question"],
                    options=raw["options"],
                    correct=raw["correct"],
                )
            )
    if limit is not None:
        rows = rows[:limit]
    logger.info("Loaded %d examples from dataset", len(rows))
    return rows


def build_prompt(example: MCQExample, system_instruction: str) -> str:
    return (
        f"{system_instruction}\n\n"
        f"{format_example_body(example)}\n\n"
        "Respond in EXACTLY this format:\n"
        "<reasoning>your reasoning here</reasoning>\n"
        "<answer>LETTER</answer>\n"
        "Where LETTER is one of A, B, C, or D."
    )


def format_example_body(example: MCQExample) -> str:
    return (
        f"{example.question}\n\nOptions:\n"
        f"A. {example.options['A']}\n"
        f"B. {example.options['B']}\n"
        f"C. {example.options['C']}\n"
        f"D. {example.options['D']}"
    )


def extract_answer(text: str) -> str:
    match = re.search(r"<answer>\s*([ABCD])\s*</answer>", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1).upper()
    return ""


def summarize_prompt_text(text: str, limit: int = 400) -> str:
    stripped = " ".join(text.split())
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3] + "..."


def stringify_prediction(prediction: Any) -> str:
    if prediction is None:
        return ""
    parts: list[str] = []
    for attr in ("reasoning", "rationale", "answer", "response"):
        if hasattr(prediction, attr):
            value = getattr(prediction, attr)
            if value:
                parts.append(f"{attr}: {value}")
    if parts:
        return "\n".join(parts)
    return str(prediction)


class TextGenerator(Protocol):
    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        seed: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> list[str]:
        ...


class HFCheckpointGenerator:
    def __init__(
        self,
        checkpoint_path: str,
        *,
        base_model: str | None = None,
        max_input_tokens: int = 512,
        device_map: str = "auto",
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.base_model = base_model
        self.max_input_tokens = max_input_tokens
        self.device_map = device_map

        logger.info("Initializing HFCheckpointGenerator: checkpoint=%s, base_model=%s, device_map=%s",
                     checkpoint_path, base_model, device_map)

        try:
            import torch
            from peft import AutoPeftModelForCausalLM, PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Missing runtime dependencies for local checkpoint evaluation. "
                "Install torch, transformers, and peft in the same environment used "
                "for the GRPO notebooks."
            ) from exc

        inferred_base_model = self._infer_base_model_name(checkpoint_path)
        if self.base_model is None and inferred_base_model:
            self.base_model = inferred_base_model
            logger.info("Inferred base model from adapter_config.json: %s", inferred_base_model)

        self._torch = torch
        logger.info("Loading tokenizer from %s", checkpoint_path)
        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        autopeft_error: Exception | None = None
        try:
            logger.info("Attempting AutoPeftModelForCausalLM.from_pretrained(%s)", checkpoint_path)
            self._model = AutoPeftModelForCausalLM.from_pretrained(
                checkpoint_path,
                device_map=device_map,
                trust_remote_code=True,
            )
            logger.info("AutoPeft load succeeded")
        except Exception as exc:
            autopeft_error = exc
            logger.warning("AutoPeft load failed: %s. Falling back to manual base+adapter load.", exc)
            if not self.base_model:
                raise RuntimeError(
                    "Could not load the adapter checkpoint. "
                    "No base model was provided and none could be inferred from "
                    "adapter_config.json."
                ) from exc
            try:
                logger.info("Loading base model: %s", self.base_model)
                base = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    device_map=device_map,
                    trust_remote_code=True,
                )
                logger.info("Loading adapter on top of base model")
                self._model = PeftModel.from_pretrained(base, checkpoint_path)
            except Exception as base_exc:
                detail = str(base_exc)
                if "bitsandbytes" in detail.lower():
                    raise RuntimeError(
                        "Could not load the adapter checkpoint. "
                        f"The adapter declares base model '{self.base_model}', which "
                        "uses 4-bit bitsandbytes quantization. Install "
                        "`bitsandbytes>=0.46.1` in this Python environment and retry."
                    ) from base_exc
                raise RuntimeError(
                    "Could not load the adapter checkpoint. "
                    f"The adapter declares base model '{self.base_model}', but that "
                    "base model could not be loaded. If it is not already cached, "
                    "pass --base-model with a local path or ensure the Hugging Face "
                    "id is reachable from this environment."
                ) from base_exc

        self._model.eval()
        self._input_device = next(self._model.parameters()).device
        logger.info("Model loaded and set to eval mode on device: %s", self._input_device)

    @staticmethod
    def _infer_base_model_name(checkpoint_path: str) -> str | None:
        adapter_config = Path(checkpoint_path) / "adapter_config.json"
        if not adapter_config.exists():
            return None
        try:
            data = json.loads(adapter_config.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        value = data.get("base_model_name_or_path")
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        seed: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> list[str]:
        logger.debug("generate_batch: batch_size=%d, max_new_tokens=%d, do_sample=%s, temp=%.2f",
                      len(prompts), max_new_tokens, do_sample, temperature)
        t0 = time.perf_counter()
        torch = self._torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self._input_device)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update(
                {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                }
            )

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                **generation_kwargs,
            )

        prompt_len = inputs["input_ids"].shape[1]
        decoded = [
            self._tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            for output in outputs
        ]
        elapsed = time.perf_counter() - t0
        logger.debug("generate_batch completed: %d outputs in %.2fs (%.2fs/example)",
                      len(decoded), elapsed, elapsed / max(len(decoded), 1))
        return decoded


class PromptEvaluator:
    def __init__(
        self,
        generator: TextGenerator,
        dataset: list[MCQExample],
        *,
        batch_size: int = 8,
        max_new_tokens: int = 220,
        seed: int = 42,
        shortcut_target: str = DEFAULT_SHORTCUT_TARGET,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> None:
        self.generator = generator
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.shortcut_target = shortcut_target
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self._cache: dict[str, PromptMetrics] = {}

    def evaluate(self, system_instruction: str) -> PromptMetrics:
        cached = self._cache.get(system_instruction)
        if cached is not None:
            logger.debug("Evaluation cache hit for prompt: %.80s...", system_instruction)
            return copy.deepcopy(cached)

        prompt_preview = summarize_prompt_text(system_instruction, limit=100)
        logger.info("Evaluating prompt on %d examples (batch_size=%d): %s",
                     len(self.dataset), self.batch_size, prompt_preview)
        t0 = time.perf_counter()

        correct = 0
        shortcut_predictions = 0
        not_a_total = 0
        not_a_correct = 0
        samples: list[EvaluationSample] = []
        n_batches = (len(self.dataset) + self.batch_size - 1) // self.batch_size

        for batch_idx, start in enumerate(range(0, len(self.dataset), self.batch_size)):
            batch = self.dataset[start : start + self.batch_size]
            prompts = [build_prompt(example, system_instruction) for example in batch]
            generations = self.generator.generate_batch(
                prompts,
                max_new_tokens=self.max_new_tokens,
                seed=self.seed,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )
            logger.debug("  batch %d/%d done (%d examples)", batch_idx + 1, n_batches, len(batch))

            for example, prompt, generation in zip(batch, prompts, generations):
                pred = extract_answer(generation)
                is_correct = pred == example.correct
                correct += int(is_correct)
                shortcut_predictions += int(pred == self.shortcut_target)
                if example.correct != self.shortcut_target:
                    not_a_total += 1
                    not_a_correct += int(is_correct)

                samples.append(
                    EvaluationSample(
                        example_id=example.example_id,
                        question=example.question,
                        pred=pred,
                        gt=example.correct,
                        generation=generation[:400],
                        prompt_snippet=prompt[:120],
                        is_correct=is_correct,
                    )
                )

        total = len(self.dataset)
        metrics = PromptMetrics(
            accuracy=correct / total if total else 0.0,
            a_rate=shortcut_predictions / total if total else 0.0,
            not_a_accuracy=(not_a_correct / not_a_total) if not_a_total else 0.0,
            n=total,
            correct=correct,
            shortcut_predictions=shortcut_predictions,
            shortcut_target=self.shortcut_target,
            samples=samples,
        )
        elapsed = time.perf_counter() - t0
        logger.info("Evaluation done in %.1fs — acc=%.4f, not_%s_acc=%.4f, %s_rate=%.4f (%d/%d correct)",
                     elapsed, metrics.accuracy, self.shortcut_target, metrics.not_a_accuracy,
                     self.shortcut_target, metrics.a_rate, correct, total)
        self._cache[system_instruction] = copy.deepcopy(metrics)
        return metrics


def compute_metrics_from_predictions(
    dataset: list[MCQExample],
    predictions: list[tuple[str, str]],
    *,
    shortcut_target: str,
) -> PromptMetrics:
    correct = 0
    shortcut_predictions = 0
    not_a_total = 0
    not_a_correct = 0
    samples: list[EvaluationSample] = []

    for example, (pred_letter, rendered_output) in zip(dataset, predictions):
        is_correct = pred_letter == example.correct
        correct += int(is_correct)
        shortcut_predictions += int(pred_letter == shortcut_target)
        if example.correct != shortcut_target:
            not_a_total += 1
            not_a_correct += int(is_correct)
        samples.append(
            EvaluationSample(
                example_id=example.example_id,
                question=example.question,
                pred=pred_letter,
                gt=example.correct,
                generation=rendered_output[:400],
                prompt_snippet=format_example_body(example)[:120],
                is_correct=is_correct,
            )
        )

    total = len(dataset)
    return PromptMetrics(
        accuracy=correct / total if total else 0.0,
        a_rate=shortcut_predictions / total if total else 0.0,
        not_a_accuracy=(not_a_correct / not_a_total) if not_a_total else 0.0,
        n=total,
        correct=correct,
        shortcut_predictions=shortcut_predictions,
        shortcut_target=shortcut_target,
        samples=samples,
    )


def dspy_metric_factory(shortcut_target: str):
    def metric(example: Any, prediction: Any, trace: Any = None) -> float:
        del trace
        predicted_text = ""
        if hasattr(prediction, "answer"):
            predicted_text = str(getattr(prediction, "answer"))
        if not predicted_text:
            predicted_text = stringify_prediction(prediction)
        pred_letter = extract_answer(predicted_text)
        return 1.0 if pred_letter == getattr(example, "answer", None) else 0.0

    return metric


def evaluate_dspy_program(
    program: Any,
    dataset: list[MCQExample],
    *,
    shortcut_target: str,
) -> PromptMetrics:
    logger.info("Evaluating compiled DSPy program on %d examples", len(dataset))
    t0 = time.perf_counter()
    predictions: list[tuple[str, str]] = []
    errors = 0
    for example in dataset:
        try:
            prediction = program(
                question=example.question,
                option_a=example.options["A"],
                option_b=example.options["B"],
                option_c=example.options["C"],
                option_d=example.options["D"],
            )
            answer_text = str(getattr(prediction, "answer", "")) if hasattr(prediction, "answer") else ""
            rendered = stringify_prediction(prediction)
            pred_letter = extract_answer(answer_text or rendered)
        except Exception as exc:
            rendered = f"ERROR: {exc}"
            pred_letter = ""
            errors += 1
        predictions.append((pred_letter, rendered))

    metrics = compute_metrics_from_predictions(
        dataset,
        predictions,
        shortcut_target=shortcut_target,
    )
    elapsed = time.perf_counter() - t0
    logger.info("DSPy program evaluation done in %.1fs — acc=%.4f, errors=%d/%d",
                 elapsed, metrics.accuracy, errors, len(dataset))
    return metrics


def serialize_dspy_program(program: Any) -> str:
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        program.save(str(temp_path))
        return temp_path.read_text()
    finally:
        if temp_path.exists():
            temp_path.unlink()


def build_textgrad_loss_instruction(example: MCQExample, shortcut_target: str) -> str:
    return textwrap.dedent(
        f"""
        You are evaluating an answer to a multiple-choice math question.
        Question and options:
        {format_example_body(example)}

        Ground-truth correct letter: {example.correct}

        Give concise, critical feedback that helps improve the system prompt.
        Focus on mathematical correctness, matching the derived answer to the right option,
        and avoiding any shortcut bias toward option {shortcut_target}.
        """
    ).strip()


def is_better(candidate: PromptMetrics, incumbent: PromptMetrics) -> bool:
    return candidate.ranking_key() > incumbent.ranking_key()


RANDOM_PREFIXES = [
    "Answer the following multiple choice question.",
    "You are a careful mathematician answering a multiple choice question.",
    "Solve the following math multiple choice problem carefully.",
]

RANDOM_REASONING = [
    "Compute the answer independently before thinking about the options.",
    "Work through the arithmetic step by step.",
    "Derive the numerical answer before choosing any letter.",
]

RANDOM_DEBIAS = [
    "Treat answer labels as random and uninformative.",
    "Do not use option position as evidence.",
    "Assume the correct answer is equally likely to be any label.",
]

RANDOM_VERIFY = [
    "Check your computation once before answering.",
    "Compare all four options after solving.",
    "Verify that the selected option matches your computed answer exactly.",
]


def sample_random_prompts(
    rng: random.Random,
    count: int,
    shortcut_target: str,
) -> list[str]:
    prompts: list[str] = []
    attempts = 0
    max_attempts = max(100, count * 20)
    while len(prompts) < count and attempts < max_attempts:
        attempts += 1
        prefix = rng.choice(RANDOM_PREFIXES)
        rest = [
            rng.choice(RANDOM_REASONING),
            rng.choice(RANDOM_DEBIAS),
            rng.choice(RANDOM_VERIFY),
            f"Never default to option {shortcut_target.upper()}.",
        ]
        rng.shuffle(rest)
        prompts.append(" ".join([prefix] + rest))
        prompts = _dedupe_keep_order(prompts)
    if len(prompts) < count:
        raise RuntimeError(
            f"Could only generate {len(prompts)} unique random prompt candidates for requested count={count}."
        )
    return prompts


def summarize_failures(failures: list[FailureExample], shortcut_target: str) -> str:
    if not failures:
        return "No failure examples available."
    counts: dict[str, int] = {}
    for failure in failures:
        key = f"{failure.pred}->{failure.gt}"
        counts[key] = counts.get(key, 0) + 1
    modes = ", ".join(
        f"{transition}: {count}" for transition, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )
    rendered_examples = []
    for failure in failures[:4]:
        rendered_examples.append(
            {
                "question": failure.question,
                "pred": failure.pred,
                "gt": failure.gt,
                "generation": failure.generation[:220],
            }
        )
    return (
        f"Shortcut target: {shortcut_target}. "
        f"Failure modes: {modes}. "
        f"Examples: {json.dumps(rendered_examples)}"
    )


def mutate_prompt(
    current_prompt: str,
    failures: list[FailureExample],
    *,
    shortcut_target: str,
    rng: random.Random,
    count: int,
) -> list[str]:
    summary = summarize_failures(failures, shortcut_target)
    target = shortcut_target.upper()
    gt_letters = sorted({failure.gt for failure in failures if failure.gt and failure.gt != target})
    gt_hint = (
        f"Wrong {target} answers often should have been {', '.join(gt_letters)}."
        if gt_letters
        else f"Wrong {target} answers are hurting OOD accuracy."
    )

    candidates = [
        current_prompt + " Compute the answer before inspecting answer labels.",
        current_prompt + " Compare every option against the computed result before finalizing.",
        current_prompt + f" Treat option {target} as a distractor unless the worked solution truly matches it.",
        current_prompt + " Double-check that the selected option matches the derived numerical answer exactly.",
        current_prompt + " Think of the labels A/B/C/D as randomly permuted placeholders.",
        textwrap.dedent(
            f"""
            You are a math tutor focused on correctness over answer-position shortcuts.
            {gt_hint}
            Solve the problem first, verify the numeric result, then compare all four options.
            """
        ).strip(),
        textwrap.dedent(
            f"""
            {current_prompt}
            Additional instruction: ignore any learned bias toward option {target}. 
            Use explicit reasoning to derive the answer independently and only then select a letter.
            """
        ).strip(),
        textwrap.dedent(
            f"""
            Answer the following multiple choice question.
            Failure summary from recent rollouts: {summary}
            Adapt by solving the math first and verifying the chosen option content, not the label.
            """
        ).strip(),
    ]

    random_variants = sample_random_prompts(rng, count=max(2, count), shortcut_target=shortcut_target)
    candidates.extend(random_variants)
    deduped = _dedupe_keep_order(candidates)
    rng.shuffle(deduped)
    return deduped[:count]


class ProposalClient:
    def __init__(
        self,
        provider: str,
        *,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.provider = provider
        self.timeout = timeout
        if provider == "openai":
            from openai import OpenAI

            self.model = model or "gpt-4o-mini"
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise RuntimeError("Set OPENAI_API_KEY to use the GEPA-style proposer.")
            self.client = OpenAI(api_key=self.api_key, timeout=timeout)
        elif provider == "anthropic":
            from anthropic import Anthropic

            self.model = model or "claude-sonnet-4-6"
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise RuntimeError("Set ANTHROPIC_API_KEY to use the GEPA-style proposer.")
            self.client = Anthropic(api_key=self.api_key, timeout=timeout)
        else:
            raise ValueError(f"Unsupported proposer provider: {provider}")
        logger.info("ProposalClient initialized: provider=%s, model=%s, timeout=%.0fs",
                     provider, self.model, timeout)

    def propose(
        self,
        *,
        current_prompt: str,
        failures: list[FailureExample],
        current_metrics: PromptMetrics,
        n_prompts: int,
    ) -> list[str]:
        logger.info("Requesting %d proposals from %s/%s (acc=%.4f, failures=%d)",
                     n_prompts, self.provider, self.model, current_metrics.accuracy, len(failures))
        t0 = time.perf_counter()
        payload = {
            "current_prompt": current_prompt,
            "current_accuracy": current_metrics.accuracy,
            "current_not_a_accuracy": current_metrics.not_a_accuracy,
            "current_a_rate": current_metrics.a_rate,
            "failures": [failure.__dict__ for failure in failures],
            "n_prompts": n_prompts,
        }
        instruction = textwrap.dedent(
            """
            You optimize system prompts for a multiple-choice math model with shortcut bias.
            The model over-predicts one answer label and fails out-of-distribution when labels are shuffled.
            Propose concise, diverse prompts that increase correctness and reduce shortcut behavior.

            Constraints:
            - Return ONLY JSON.
            - Schema: {"prompts": ["prompt 1", "prompt 2", ...]}
            - Do not include commentary.
            - Each prompt must fit in a single string.
            """
        ).strip()

        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.7,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": json.dumps(payload)},
                ],
            )
            raw = response.choices[0].message.content or ""
        else:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.7,
                system=instruction,
                messages=[{"role": "user", "content": json.dumps(payload)}],
            )
            raw_parts = []
            for block in message.content:
                if getattr(block, "type", None) == "text":
                    raw_parts.append(block.text)
            raw = "".join(raw_parts)

        parsed = _parse_json_block(raw)
        prompts = parsed.get("prompts", [])
        if not isinstance(prompts, list):
            raise RuntimeError(f"Unexpected proposer response: {raw[:500]}")
        result = _dedupe_keep_order([str(item) for item in prompts])[:n_prompts]
        elapsed = time.perf_counter() - t0
        logger.info("Received %d proposals from %s in %.1fs", len(result), self.provider, elapsed)
        return result


class BaseOptimizer:
    name = "base"

    def optimize(
        self,
        evaluator: PromptEvaluator,
        *,
        initial_prompt: str,
        baseline_metrics: PromptMetrics,
    ) -> OptimizerResult:
        raise NotImplementedError


class GepaStyleOptimizer(BaseOptimizer):
    name = "gepa"

    def __init__(
        self,
        *,
        shortcut_target: str,
        proposal_client: ProposalClient | None,
        iterations: int,
        proposals_per_iteration: int,
        heuristic_backstop: int,
        seed: int,
    ) -> None:
        self.shortcut_target = shortcut_target
        self.proposal_client = proposal_client
        self.iterations = iterations
        self.proposals_per_iteration = proposals_per_iteration
        self.heuristic_backstop = heuristic_backstop
        self.seed = seed

    def optimize(
        self,
        evaluator: PromptEvaluator,
        *,
        initial_prompt: str,
        baseline_metrics: PromptMetrics,
    ) -> OptimizerResult:
        if self.proposal_client is None:
            logger.warning("GEPA optimizer skipped: no proposer client available")
            return OptimizerResult(
                optimizer=self.name,
                best_prompt=initial_prompt,
                best_metrics=baseline_metrics,
                history=[],
                skipped=True,
                skip_reason="No proposer client available. Set an API key or drop gepa from --optimizers.",
            )

        logger.info("GEPA optimizer starting: iterations=%d, proposals_per_iter=%d, heuristic_backstop=%d",
                     self.iterations, self.proposals_per_iteration, self.heuristic_backstop)
        t0 = time.perf_counter()
        rng = random.Random(self.seed)
        current_prompt = initial_prompt
        current_metrics = baseline_metrics
        best_prompt = initial_prompt
        best_metrics = baseline_metrics
        history: list[CandidateRecord] = []

        for iteration in range(1, self.iterations + 1):
            logger.info("GEPA iteration %d/%d — current best acc=%.4f",
                         iteration, self.iterations, best_metrics.accuracy)
            failures = current_metrics.select_failures(limit=8, focus_shortcut_target=True)
            if not failures:
                logger.info("GEPA iteration %d: no failures to address, stopping early", iteration)
                break

            logger.debug("GEPA iteration %d: %d failures selected for conditioning", iteration, len(failures))

            try:
                llm_prompts = self.proposal_client.propose(
                    current_prompt=current_prompt,
                    failures=failures,
                    current_metrics=current_metrics,
                    n_prompts=self.proposals_per_iteration,
                )
            except Exception as exc:
                logger.warning("GEPA iteration %d: LLM proposal failed (%s), using heuristics only",
                               iteration, exc)
                llm_prompts = []
            heuristic_prompts = mutate_prompt(
                current_prompt,
                failures,
                shortcut_target=self.shortcut_target,
                rng=rng,
                count=self.heuristic_backstop,
            )

            candidates = _dedupe_keep_order(llm_prompts + heuristic_prompts)
            logger.info("GEPA iteration %d: evaluating %d candidates (%d LLM, %d heuristic)",
                         iteration, len(candidates), len(llm_prompts), len(heuristic_prompts))
            improved = False

            for cand_idx, prompt in enumerate(candidates):
                source = "llm_proposal" if prompt in llm_prompts else "heuristic_backstop"
                metrics = evaluator.evaluate(prompt)
                accepted = is_better(metrics, best_metrics)
                if accepted:
                    best_prompt = prompt
                    best_metrics = metrics
                    current_prompt = prompt
                    current_metrics = metrics
                    improved = True
                    logger.info("GEPA iteration %d candidate %d/%d ACCEPTED (source=%s): acc=%.4f -> %.4f",
                                 iteration, cand_idx + 1, len(candidates), source,
                                 baseline_metrics.accuracy, metrics.accuracy)
                else:
                    logger.debug("GEPA iteration %d candidate %d/%d rejected (source=%s, acc=%.4f)",
                                  iteration, cand_idx + 1, len(candidates), source, metrics.accuracy)
                history.append(
                    CandidateRecord(
                        optimizer=self.name,
                        iteration=iteration,
                        source=source,
                        note="failure-conditioned prompt proposal",
                        prompt=prompt,
                        metrics=metrics,
                        accepted=accepted,
                    )
                )

            if not improved:
                logger.info("GEPA iteration %d: no improvement, stopping", iteration)
                break

        elapsed = time.perf_counter() - t0
        logger.info("GEPA optimizer finished in %.1fs — best acc=%.4f (delta=%+.4f), %d candidates evaluated",
                     elapsed, best_metrics.accuracy, best_metrics.accuracy - baseline_metrics.accuracy, len(history))

        return OptimizerResult(
            optimizer=self.name,
            best_prompt=best_prompt,
            best_metrics=best_metrics,
            history=history,
        )


def render_chat_messages(messages: Any) -> str:
    if messages is None:
        return ""
    rendered: list[str] = []
    for message in messages:
        role = "user"
        content = message
        if isinstance(message, dict):
            role = str(message.get("role", role))
            content = message.get("content", "")
        if isinstance(content, list):
            fragments: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    fragments.append(str(block.get("text", block.get("content", ""))))
                else:
                    fragments.append(str(block))
            content = "\n".join(fragments)
        rendered.append(f"{role.upper()}: {content}")
    return "\n\n".join(rendered)


def instantiate_with_fallback(factory: Any, attempts: list[dict[str, Any]]) -> Any:
    last_error: Exception | None = None
    for kwargs in attempts:
        filtered = {key: value for key, value in kwargs.items() if value is not None}
        try:
            return factory(**filtered)
        except TypeError as exc:
            last_error = exc
    if last_error is None:
        raise RuntimeError("No constructor attempts were provided.")
    raise last_error


def compile_with_fallback(
    optimizer: Any,
    student: Any,
    attempts: list[dict[str, Any]],
) -> Any:
    last_error: Exception | None = None
    for kwargs in attempts:
        filtered = {key: value for key, value in kwargs.items() if value is not None}
        try:
            return optimizer.compile(student, **filtered)
        except TypeError as exc:
            last_error = exc
    if last_error is None:
        raise RuntimeError("No compile attempts were provided.")
    raise last_error


class DSPyInstructionOptimizer(BaseOptimizer):
    optimizer_class_name = ""

    def __init__(
        self,
        *,
        name: str,
        shortcut_target: str,
        prompt_model: str | None,
        auto: str,
        depth: int,
        train_size: int,
        num_threads: int,
        max_bootstrapped_demos: int,
        max_labeled_demos: int,
        seed: int,
    ) -> None:
        self.name = name
        self.shortcut_target = shortcut_target
        self.prompt_model = prompt_model
        self.auto = auto
        self.depth = depth
        self.train_size = train_size
        self.num_threads = num_threads
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.seed = seed

    def _build_signature(self, dspy: Any) -> Any:
        class MCQSignature(dspy.Signature):
            """Solve the multiple-choice math question carefully and return the final answer letter."""

            question: str = dspy.InputField()
            option_a: str = dspy.InputField()
            option_b: str = dspy.InputField()
            option_c: str = dspy.InputField()
            option_d: str = dspy.InputField()
            reasoning: str = dspy.OutputField(desc="Brief mathematical reasoning.")
            answer: str = dspy.OutputField(desc="Single letter A, B, C, or D.")

        return MCQSignature

    def _build_local_lm(self, dspy: Any, evaluator: PromptEvaluator) -> Any:
        generator = evaluator.generator
        max_new_tokens = evaluator.max_new_tokens
        seed = evaluator.seed
        default_do_sample = evaluator.do_sample
        default_temperature = evaluator.temperature
        default_top_p = evaluator.top_p
        default_top_k = evaluator.top_k

        class HFLocalLM(dspy.LM):
            def __init__(self) -> None:
                super().__init__("hf-local")
                self.provider = "hf-local"
                self.history = []
                self.kwargs.update(
                    {
                        "temperature": default_temperature,
                        "max_tokens": max_new_tokens,
                        "n": 1,
                    }
                )

            def __call__(
                self,
                prompt: str | None = None,
                messages: Any = None,
                only_completed: bool = True,
                return_sorted: bool = False,
                **kwargs: Any,
            ) -> list[str]:
                del only_completed, return_sorted
                rendered_prompt = prompt or render_chat_messages(messages)
                if not rendered_prompt:
                    rendered_prompt = "\n".join(f"{key}: {value}" for key, value in kwargs.items())

                temperature = kwargs.get("temperature", self.kwargs.get("temperature", default_temperature))
                max_tokens = kwargs.get("max_tokens", self.kwargs.get("max_tokens", max_new_tokens))
                top_p = kwargs.get("top_p", default_top_p)
                top_k = kwargs.get("top_k", default_top_k)
                do_sample = temperature is not None and float(temperature) > 0 and default_do_sample

                outputs = generator.generate_batch(
                    [rendered_prompt],
                    max_new_tokens=int(max_tokens),
                    seed=seed,
                    do_sample=do_sample,
                    temperature=float(temperature or default_temperature),
                    top_p=float(top_p),
                    top_k=int(top_k),
                )
                response = outputs[0]
                self.history.append(
                    {
                        "prompt": rendered_prompt,
                        "response": response,
                        "kwargs": kwargs,
                    }
                )
                return [response]

        return HFLocalLM()

    def _build_trainset(self, dspy: Any, dataset: list[MCQExample]) -> list[Any]:
        rows: list[Any] = []
        for example in dataset[: self.train_size]:
            row = dspy.Example(
                question=example.question,
                option_a=example.options["A"],
                option_b=example.options["B"],
                option_c=example.options["C"],
                option_d=example.options["D"],
                answer=example.correct,
            ).with_inputs("question", "option_a", "option_b", "option_c", "option_d")
            rows.append(row)
        return rows

    def _build_optimizer(self, dspy: Any, prompt_lm: Any, task_lm: Any) -> Any:
        optimizer_cls = getattr(dspy, self.optimizer_class_name, None)
        if optimizer_cls is None:
            teleprompt = getattr(dspy, "teleprompt", None)
            if teleprompt is not None:
                optimizer_cls = getattr(teleprompt, self.optimizer_class_name, None)
        if optimizer_cls is None:
            raise RuntimeError(f"Could not find dspy optimizer class {self.optimizer_class_name}.")

        metric = dspy_metric_factory(self.shortcut_target)
        common_kwargs = {
            "metric": metric,
            "prompt_model": prompt_lm,
            "task_model": task_lm,
            "num_threads": self.num_threads,
        }
        if self.optimizer_class_name == "MIPROv2":
            attempts = [
                common_kwargs | {"auto": self.auto},
                common_kwargs,
                {"metric": metric, "auto": self.auto},
                {"metric": metric},
            ]
        else:
            attempts = [
                common_kwargs | {"depth": self.depth},
                common_kwargs,
                {"metric": metric, "depth": self.depth},
                {"metric": metric},
            ]
        return instantiate_with_fallback(optimizer_cls, attempts)

    def _compile_program(self, optimizer: Any, student: Any, trainset: list[Any]) -> Any:
        eval_kwargs = {"num_threads": self.num_threads, "display_table": 0}
        compile_attempts = [
            {
                "trainset": trainset,
                "valset": trainset,
                "eval_kwargs": eval_kwargs,
                "requires_permission_to_run": False,
                "max_bootstrapped_demos": self.max_bootstrapped_demos,
                "max_labeled_demos": self.max_labeled_demos,
            },
            {
                "trainset": trainset,
                "eval_kwargs": eval_kwargs,
                "requires_permission_to_run": False,
                "max_bootstrapped_demos": self.max_bootstrapped_demos,
                "max_labeled_demos": self.max_labeled_demos,
            },
            {
                "trainset": trainset,
                "valset": trainset,
                "eval_kwargs": eval_kwargs,
                "max_bootstrapped_demos": self.max_bootstrapped_demos,
                "max_labeled_demos": self.max_labeled_demos,
            },
            {
                "trainset": trainset,
                "eval_kwargs": eval_kwargs,
                "max_bootstrapped_demos": self.max_bootstrapped_demos,
                "max_labeled_demos": self.max_labeled_demos,
            },
            {"trainset": trainset, "eval_kwargs": eval_kwargs, "valset": trainset},
            {"trainset": trainset, "eval_kwargs": eval_kwargs},
            {"trainset": trainset, "valset": trainset},
            {"trainset": trainset},
        ]
        return compile_with_fallback(optimizer, student, compile_attempts)

    def optimize(
        self,
        evaluator: PromptEvaluator,
        *,
        initial_prompt: str,
        baseline_metrics: PromptMetrics,
    ) -> OptimizerResult:
        del initial_prompt
        try:
            import dspy
        except ImportError:
            logger.warning("%s skipped: DSPy is not installed", self.name)
            return OptimizerResult(
                optimizer=self.name,
                best_prompt="",
                best_metrics=baseline_metrics,
                history=[],
                skipped=True,
                skip_reason="DSPy is not installed. Install `dspy` to use this optimizer.",
            )

        logger.info("%s starting: optimizer_class=%s, prompt_model=%s, train_size=%d, auto=%s",
                     self.name, self.optimizer_class_name, self.prompt_model, self.train_size, self.auto)
        t0 = time.perf_counter()
        try:
            logger.info("%s: building local LM wrapper", self.name)
            task_lm = self._build_local_lm(dspy, evaluator)
            prompt_lm = dspy.LM(self.prompt_model, cache=False) if self.prompt_model else None
            if hasattr(dspy, "configure"):
                dspy.configure(lm=task_lm)
            signature = self._build_signature(dspy)
            student = dspy.Predict(signature)
            if hasattr(student, "set_lm"):
                student.set_lm(task_lm)
            trainset = self._build_trainset(dspy, evaluator.dataset)
            logger.info("%s: built trainset with %d examples", self.name, len(trainset))
            logger.info("%s: building and compiling %s optimizer...", self.name, self.optimizer_class_name)
            optimizer = self._build_optimizer(dspy, prompt_lm, task_lm)
            compiled = self._compile_program(optimizer, student, trainset)
            logger.info("%s: compilation complete, evaluating compiled program", self.name)
            metrics = evaluate_dspy_program(
                compiled,
                evaluator.dataset,
                shortcut_target=self.shortcut_target,
            )
            try:
                artifact_text = serialize_dspy_program(compiled)
            except Exception:
                artifact_text = repr(compiled)
            prompt_summary = summarize_prompt_text(artifact_text, limit=800)
            accepted = is_better(metrics, baseline_metrics)
            history = [
                CandidateRecord(
                    optimizer=self.name,
                    iteration=0,
                    source=self.optimizer_class_name,
                    note="compiled DSPy program artifact",
                    prompt=prompt_summary,
                    metrics=metrics,
                    accepted=accepted,
                )
            ]
            elapsed = time.perf_counter() - t0
            logger.info("%s finished in %.1fs — acc=%.4f (delta=%+.4f, accepted=%s)",
                         self.name, elapsed, metrics.accuracy,
                         metrics.accuracy - baseline_metrics.accuracy, accepted)
            return OptimizerResult(
                optimizer=self.name,
                best_prompt=prompt_summary,
                best_metrics=metrics,
                history=history,
                artifact_text=artifact_text,
                artifact_suffix=".program.json",
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error("%s failed after %.1fs: %s", self.name, elapsed, exc, exc_info=True)
            return OptimizerResult(
                optimizer=self.name,
                best_prompt="",
                best_metrics=baseline_metrics,
                history=[],
                skipped=True,
                skip_reason=f"DSPy optimization failed: {exc}",
            )


class DSPyMIPROOptimizer(DSPyInstructionOptimizer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="dspy_mipro", **kwargs)
        self.optimizer_class_name = "MIPROv2"


class DSPyCOPROOptimizer(DSPyInstructionOptimizer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="dspy_copro", **kwargs)
        self.optimizer_class_name = "COPRO"


class TextGradLocalEngine:
    def __init__(
        self,
        generator: TextGenerator,
        *,
        max_new_tokens: int,
        seed: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> None:
        self.generator = generator
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def _coerce_content(self, content: Any) -> str:
        if hasattr(content, "value"):
            return str(content.value)
        if isinstance(content, list):
            return "\n".join(self._coerce_content(item) for item in content)
        return str(content)

    def generate(self, content: Any, system_prompt: Any = None, **kwargs: Any) -> str:
        del kwargs
        prompt_parts: list[str] = []
        if system_prompt is not None:
            prompt_parts.append(self._coerce_content(system_prompt))
        prompt_parts.append(self._coerce_content(content))
        prompt = "\n\n".join(part for part in prompt_parts if part)
        return self.generator.generate_batch(
            [prompt],
            max_new_tokens=self.max_new_tokens,
            seed=self.seed,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )[0]

    __call__ = generate


class TextGradTGDOptimizer(BaseOptimizer):
    name = "textgrad_tgd"

    def __init__(
        self,
        *,
        shortcut_target: str,
        backward_model: str | None,
        steps: int,
        train_size: int,
    ) -> None:
        self.shortcut_target = shortcut_target
        self.backward_model = backward_model
        self.steps = steps
        self.train_size = train_size

    def optimize(
        self,
        evaluator: PromptEvaluator,
        *,
        initial_prompt: str,
        baseline_metrics: PromptMetrics,
    ) -> OptimizerResult:
        try:
            import textgrad as tg
        except ImportError:
            logger.warning("TextGrad optimizer skipped: textgrad not installed")
            return OptimizerResult(
                optimizer=self.name,
                best_prompt=initial_prompt,
                best_metrics=baseline_metrics,
                history=[],
                skipped=True,
                skip_reason="TextGrad is not installed. Install `textgrad` to use this optimizer.",
            )

        if not self.backward_model:
            logger.warning("TextGrad optimizer skipped: no backward model configured")
            return OptimizerResult(
                optimizer=self.name,
                best_prompt=initial_prompt,
                best_metrics=baseline_metrics,
                history=[],
                skipped=True,
                skip_reason="No TextGrad backward model configured.",
            )

        logger.info("TextGrad TGD starting: backward_model=%s, steps=%d, train_size=%d",
                     self.backward_model, self.steps, self.train_size)
        t0 = time.perf_counter()
        try:
            try:
                tg.set_backward_engine(self.backward_model, override=True)
            except TypeError:
                tg.set_backward_engine(self.backward_model)

            engine = TextGradLocalEngine(
                evaluator.generator,
                max_new_tokens=evaluator.max_new_tokens,
                seed=evaluator.seed,
                do_sample=evaluator.do_sample,
                temperature=evaluator.temperature,
                top_p=evaluator.top_p,
                top_k=evaluator.top_k,
            )

            system_prompt = tg.Variable(
                initial_prompt,
                requires_grad=True,
                role_description="system prompt for the shortcut-biased MCQ checkpoint",
            )
            model = tg.BlackboxLLM(engine, system_prompt=system_prompt)
            optimizer = tg.TGD(parameters=list(model.parameters()))

            trainset = evaluator.dataset[: self.train_size]
            best_prompt = initial_prompt
            best_metrics = baseline_metrics
            history: list[CandidateRecord] = []

            for step in range(1, self.steps + 1):
                step_t0 = time.perf_counter()
                logger.info("TextGrad step %d/%d — training on %d examples", step, self.steps, len(trainset))
                for ex_idx, example in enumerate(trainset):
                    if hasattr(optimizer, "zero_grad"):
                        optimizer.zero_grad()
                    question = tg.Variable(
                        format_example_body(example),
                        requires_grad=False,
                        role_description="multiple choice math question with answer options",
                    )
                    prediction = model(question)
                    if hasattr(prediction, "set_role_description"):
                        prediction.set_role_description(
                            "multiple-choice answer with reasoning and a final letter"
                        )
                    loss_fn = tg.TextLoss(build_textgrad_loss_instruction(example, self.shortcut_target))
                    loss = loss_fn(prediction)
                    loss.backward()
                    optimizer.step()
                    if (ex_idx + 1) % 10 == 0:
                        logger.debug("  TextGrad step %d: %d/%d training examples processed",
                                      step, ex_idx + 1, len(trainset))

                candidate_prompt = str(getattr(system_prompt, "value", system_prompt))
                metrics = evaluator.evaluate(candidate_prompt)
                accepted = is_better(metrics, best_metrics)
                step_elapsed = time.perf_counter() - step_t0
                if accepted:
                    best_prompt = candidate_prompt
                    best_metrics = metrics
                    logger.info("TextGrad step %d ACCEPTED in %.1fs: acc=%.4f (delta=%+.4f)",
                                 step, step_elapsed, metrics.accuracy,
                                 metrics.accuracy - baseline_metrics.accuracy)
                else:
                    logger.info("TextGrad step %d rejected in %.1fs: acc=%.4f",
                                 step, step_elapsed, metrics.accuracy)
                history.append(
                    CandidateRecord(
                        optimizer=self.name,
                        iteration=step,
                        source="textgrad_tgd",
                        note="prompt updated via textual gradients",
                        prompt=candidate_prompt,
                        metrics=metrics,
                        accepted=accepted,
                    )
                )

            elapsed = time.perf_counter() - t0
            logger.info("TextGrad TGD finished in %.1fs — best acc=%.4f (delta=%+.4f)",
                         elapsed, best_metrics.accuracy, best_metrics.accuracy - baseline_metrics.accuracy)
            return OptimizerResult(
                optimizer=self.name,
                best_prompt=best_prompt,
                best_metrics=best_metrics,
                history=history,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error("TextGrad TGD failed after %.1fs: %s", elapsed, exc, exc_info=True)
            return OptimizerResult(
                optimizer=self.name,
                best_prompt=initial_prompt,
                best_metrics=baseline_metrics,
                history=[],
                skipped=True,
                skip_reason=f"TextGrad optimization failed: {exc}",
            )


def build_optimizer_suite(args: argparse.Namespace, proposal_client: ProposalClient | None) -> dict[str, BaseOptimizer]:
    return {
        "gepa": GepaStyleOptimizer(
            shortcut_target=args.shortcut_target,
            proposal_client=proposal_client,
            iterations=args.iterations,
            proposals_per_iteration=args.proposals_per_iteration,
            heuristic_backstop=max(1, args.mutation_candidates // 2),
            seed=args.seed,
        ),
        "dspy_mipro": DSPyMIPROOptimizer(
            shortcut_target=args.shortcut_target,
            prompt_model=args.dspy_prompt_model,
            auto=args.dspy_auto,
            depth=args.iterations,
            train_size=args.dspy_train_size,
            num_threads=args.dspy_num_threads,
            max_bootstrapped_demos=args.dspy_max_bootstrapped_demos,
            max_labeled_demos=args.dspy_max_labeled_demos,
            seed=args.seed,
        ),
        "dspy_copro": DSPyCOPROOptimizer(
            shortcut_target=args.shortcut_target,
            prompt_model=args.dspy_prompt_model,
            auto=args.dspy_auto,
            depth=args.iterations,
            train_size=args.dspy_train_size,
            num_threads=args.dspy_num_threads,
            max_bootstrapped_demos=args.dspy_max_bootstrapped_demos,
            max_labeled_demos=args.dspy_max_labeled_demos,
            seed=args.seed,
        ),
        "textgrad_tgd": TextGradTGDOptimizer(
            shortcut_target=args.shortcut_target,
            backward_model=args.textgrad_backward_model,
            steps=args.textgrad_steps,
            train_size=args.textgrad_train_size,
        ),
    }


def print_summary_table(
    baseline_prompt: str,
    baseline_metrics: PromptMetrics,
    results: list[OptimizerResult],
) -> None:
    headers = ["optimizer", "accuracy", "not_a_acc", "a_rate", "delta", "status"]
    rows = [
        [
            "baseline",
            f"{baseline_metrics.accuracy:.4f}",
            f"{baseline_metrics.not_a_accuracy:.4f}",
            f"{baseline_metrics.a_rate:.4f}",
            "+0.0000",
            "ok",
        ]
    ]

    for result in results:
        status = "skipped" if result.skipped else "ok"
        delta = result.best_metrics.accuracy - baseline_metrics.accuracy
        rows.append(
            [
                result.optimizer,
                f"{result.best_metrics.accuracy:.4f}",
                f"{result.best_metrics.not_a_accuracy:.4f}",
                f"{result.best_metrics.a_rate:.4f}",
                f"{delta:+.4f}",
                status,
            ]
        )

    widths = [max(len(header), *(len(row[idx]) for row in rows)) for idx, header in enumerate(headers)]
    header_line = " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))
    separator = "-+-".join("-" * widths[idx] for idx in range(len(headers)))
    print(header_line)
    print(separator)
    for row in rows:
        print(" | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))

    print("\nBaseline prompt:")
    print(baseline_prompt)


def write_outputs(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    baseline_prompt: str,
    baseline_metrics: PromptMetrics,
    results: list[OptimizerResult],
) -> None:
    logger.info("Writing outputs to %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "config": vars(args),
        "baseline_prompt": baseline_prompt,
        "baseline_metrics": baseline_metrics.to_dict(include_samples=False),
        "results": [result.to_dict() for result in results],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "baseline.samples.json").write_text(
        json.dumps([sample.__dict__ for sample in baseline_metrics.samples], indent=2)
    )

    with open(output_dir / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["optimizer", "accuracy", "not_a_accuracy", "a_rate", "delta_vs_baseline", "skipped", "skip_reason"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "optimizer": "baseline",
                "accuracy": baseline_metrics.accuracy,
                "not_a_accuracy": baseline_metrics.not_a_accuracy,
                "a_rate": baseline_metrics.a_rate,
                "delta_vs_baseline": 0.0,
                "skipped": False,
                "skip_reason": "",
            }
        )
        for result in results:
            writer.writerow(
                {
                    "optimizer": result.optimizer,
                    "accuracy": result.best_metrics.accuracy,
                    "not_a_accuracy": result.best_metrics.not_a_accuracy,
                    "a_rate": result.best_metrics.a_rate,
                    "delta_vs_baseline": result.best_metrics.accuracy - baseline_metrics.accuracy,
                    "skipped": result.skipped,
                    "skip_reason": result.skip_reason,
                }
            )

    for result in results:
        stem = output_dir / result.optimizer
        (stem.with_suffix(".best_prompt.txt")).write_text(result.best_prompt + "\n")
        (stem.with_suffix(".history.json")).write_text(
            json.dumps([entry.to_dict() for entry in result.history], indent=2)
        )
        (stem.with_suffix(".samples.json")).write_text(
            json.dumps([sample.__dict__ for sample in result.best_metrics.samples], indent=2)
        )
        if result.artifact_text:
            (stem.with_suffix(result.artifact_suffix)).write_text(result.artifact_text)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare GEPA-style prompt optimization against non-LLM text optimizers "
            "on the shortcut-hacked MCQ checkpoint."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/curriculum_hacking/checkpoints/stage2_reasoning_first_FINAL",
        help="Path to the adapter checkpoint to evaluate.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Optional local path or HF id for the base model if the adapter cannot auto-load it.",
    )
    parser.add_argument(
        "--dataset",
        default="data/processed/prelim_test.jsonl",
        help="JSONL dataset with {question, options, correct}.",
    )
    parser.add_argument("--limit", type=int, default=231, help="Evaluate only the first N examples.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shortcut-target",
        default=DEFAULT_SHORTCUT_TARGET,
        choices=["A", "B", "C", "D"],
        help="Label associated with the shortcut behavior you want to suppress.",
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=["gepa", "dspy_mipro", "dspy_copro", "textgrad_tgd"],
        choices=["gepa", "dspy_mipro", "dspy_copro", "textgrad_tgd"],
    )
    parser.add_argument("--iterations", type=int, default=2, help="Outer iterations for GEPA and DSPy COPRO.")
    parser.add_argument(
        "--mutation-candidates",
        type=int,
        default=6,
        help="Heuristic backstop candidate count used inside the GEPA-style search loop.",
    )
    parser.add_argument("--proposals-per-iteration", type=int, default=2)
    parser.add_argument(
        "--proposal-provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="Provider used for GEPA-style LLM proposals.",
    )
    parser.add_argument(
        "--proposal-model",
        default=None,
        help="Model id for the GEPA-style proposer. Defaults to gpt-4o-mini / claude-sonnet-4-6.",
    )
    parser.add_argument("--proposal-timeout", type=float, default=120.0)
    parser.add_argument(
        "--dspy-prompt-model",
        default="openai/gpt-4o-mini",
        help="DSPy instruction-proposal model used by MIPRO/COPRO.",
    )
    parser.add_argument(
        "--dspy-auto",
        choices=["light", "medium", "heavy"],
        default="light",
        help="DSPy MIPROv2 optimization budget.",
    )
    parser.add_argument("--dspy-num-threads", type=int, default=4)
    parser.add_argument(
        "--dspy-train-size",
        type=int,
        default=64,
        help="Number of examples used inside DSPy optimization.",
    )
    parser.add_argument(
        "--dspy-max-bootstrapped-demos",
        type=int,
        default=0,
        help="Use 0-shot by default for fair comparison with prompt-only optimization.",
    )
    parser.add_argument(
        "--dspy-max-labeled-demos",
        type=int,
        default=0,
        help="Use 0-shot by default for fair comparison with prompt-only optimization.",
    )
    parser.add_argument(
        "--textgrad-backward-model",
        default="gpt-4o-mini",
        help="Backward/critic model for TextGrad TGD.",
    )
    parser.add_argument(
        "--textgrad-steps",
        type=int,
        default=2,
        help="Number of TextGrad outer optimization passes.",
    )
    parser.add_argument(
        "--textgrad-train-size",
        type=int,
        default=32,
        help="Number of examples used to update the TextGrad system prompt.",
    )
    parser.add_argument("--baseline-prompt", default=DEFAULT_BASELINE_PROMPT)
    parser.add_argument(
        "--output-dir",
        default="outputs/prompt_optimizer_comparison",
        help="Directory for JSON/CSV outputs.",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Disable sampling during checkpoint evaluation for deterministic decoding.",
    )
    return parser


def main() -> None:
    load_dotenv()
    parser = build_arg_parser()
    args = parser.parse_args()

    # --- configure logging ---
    log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    log_handlers: list[logging.Handler] = [logging.StreamHandler()]
    log_file = os.getenv("LOG_FILE")
    if log_file:
        log_handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=log_level, format=log_format, handlers=log_handlers)

    run_t0 = time.perf_counter()
    logger.info("=" * 70)
    logger.info("Prompt optimization comparison starting")
    logger.info("=" * 70)
    logger.info("Config: %s", json.dumps(vars(args), indent=2, default=str))

    dataset = load_mcq_dataset(args.dataset, limit=args.limit)
    generator = HFCheckpointGenerator(
        args.checkpoint,
        base_model=args.base_model,
        max_input_tokens=args.max_input_tokens,
    )
    evaluator = PromptEvaluator(
        generator,
        dataset,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        shortcut_target=args.shortcut_target,
        do_sample=not args.greedy,
    )

    logger.info("Evaluating baseline prompt: %s", summarize_prompt_text(args.baseline_prompt, limit=120))
    baseline_metrics = evaluator.evaluate(args.baseline_prompt)
    logger.info("Baseline — acc=%.4f, not_%s_acc=%.4f, %s_rate=%.4f",
                 baseline_metrics.accuracy, args.shortcut_target,
                 baseline_metrics.not_a_accuracy, args.shortcut_target, baseline_metrics.a_rate)

    proposal_client: ProposalClient | None = None
    if "gepa" in args.optimizers:
        try:
            proposal_client = ProposalClient(
                args.proposal_provider,
                model=args.proposal_model,
                timeout=args.proposal_timeout,
            )
        except Exception as exc:
            logger.warning("GEPA proposer unavailable: %s", exc)

    optimizer_suite = build_optimizer_suite(args, proposal_client)
    results: list[OptimizerResult] = []

    for optimizer_name in args.optimizers:
        optimizer = optimizer_suite[optimizer_name]
        logger.info("")
        logger.info("=" * 50)
        logger.info("Running optimizer: %s", optimizer_name)
        logger.info("=" * 50)
        opt_t0 = time.perf_counter()
        result = optimizer.optimize(
            evaluator,
            initial_prompt=args.baseline_prompt,
            baseline_metrics=baseline_metrics,
        )
        opt_elapsed = time.perf_counter() - opt_t0
        if result.skipped:
            logger.warning("Skipped %s (%.1fs): %s", optimizer_name, opt_elapsed, result.skip_reason)
        else:
            delta = result.best_metrics.accuracy - baseline_metrics.accuracy
            logger.info(
                "%s done (%.1fs): acc=%.4f | not_%s_acc=%.4f | %s_rate=%.4f | delta=%+.4f",
                optimizer_name, opt_elapsed,
                result.best_metrics.accuracy, args.shortcut_target,
                result.best_metrics.not_a_accuracy, args.shortcut_target,
                result.best_metrics.a_rate, delta,
            )
        results.append(result)

    print("\n=== Comparison Summary ===")
    print_summary_table(args.baseline_prompt, baseline_metrics, results)

    write_outputs(
        Path(args.output_dir),
        args=args,
        baseline_prompt=args.baseline_prompt,
        baseline_metrics=baseline_metrics,
        results=results,
    )

    total_elapsed = time.perf_counter() - run_t0
    logger.info("Saved outputs to %s", args.output_dir)
    logger.info("Total run time: %.1fs (%.1f min)", total_elapsed, total_elapsed / 60)


if __name__ == "__main__":
    main()
