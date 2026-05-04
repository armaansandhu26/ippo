"""
Barebones stage-2 GRPO training.

Reasoning-first format: <reasoning>...</reasoning><answer>LETTER</answer>

The only thing you need to provide is the dataset (a JSONL file with one
example per line, schema:
  {"question": str,
   "options": {"A": str, "B": str, "C": str, "D": str},
   "correct": "A"|"B"|"C"|"D"}
).

Usage:
  python stage2_only.py --train-file path/to/train.jsonl
  python stage2_only.py --train-file path/to/train.jsonl --output-dir ./out --max-steps 200
"""

# unsloth must be imported before anything else.
from unsloth import FastLanguageModel

import argparse
import json
import re
from pathlib import Path

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer


# ---- Format ----

REASONING_FIRST_RE = re.compile(
    r"<reasoning>(.*?)</reasoning>\s*<answer>\s*([ABCD])\s*</answer>",
    re.DOTALL | re.IGNORECASE,
)
ANY_LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def build_user_prompt(row: dict) -> str:
    options_block = "\n".join(f"{k}. {row['options'][k]}" for k in ("A", "B", "C", "D"))
    return (
        "Answer the following multiple choice question.\n\n"
        f"{row['question']}\n\n"
        f"Options:\n{options_block}\n\n"
        "Respond in EXACTLY this format:\n"
        "<reasoning>your reasoning here</reasoning>\n"
        "<answer>LETTER</answer>\n\n"
        "Where LETTER is one of A, B, C, or D."
    )


def extract_answer(text: str):
    m = REASONING_FIRST_RE.search(text)
    if m:
        return m.group(2).upper()
    m2 = ANY_LETTER_RE.search(text)
    return m2.group(1).upper() if m2 else None


# ---- Reward functions (TRL signature: prompts, completions, **columns) ----

def _completion_text(c):
    return c if isinstance(c, str) else c[0]["content"]


def format_reward(prompts, completions, **kwargs):
    out = []
    for c in completions:
        text = _completion_text(c)
        score = 0.0
        if "<reasoning>" in text and "</reasoning>" in text:
            score += 0.25
        if "<answer>" in text and "</answer>" in text:
            score += 0.25
        if REASONING_FIRST_RE.search(text):
            score += 0.5
        out.append(score)
    return out


def correctness_reward(prompts, completions, answer, **kwargs):
    out = []
    for c, gt in zip(completions, answer):
        text = _completion_text(c)
        pred = extract_answer(text)
        out.append(1.5 if pred == gt else 0.0)
    return out


# ---- Data ----

def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_dataset(rows: list[dict]) -> Dataset:
    return Dataset.from_list([
        {"prompt": build_user_prompt(r), "answer": r["correct"]}
        for r in rows
    ])


# ---- Main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", required=True, help="Path to JSONL training file.")
    ap.add_argument("--output-dir", default="./outputs_stage2")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--lora-rank", type=int, default=16)
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=1024,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_rank,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Load data
    rows = load_jsonl(args.train_file)
    print(f"Loaded {len(rows)} training rows from {args.train_file}")
    train_dataset = build_dataset(rows)

    # GRPO config — defaults match the curriculum-hacking stage-2 setup.
    config = GRPOConfig(
        learning_rate=2e-5,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.0,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_generations=8,
        max_prompt_length=512,
        max_completion_length=384,
        max_steps=args.max_steps,
        save_steps=10**9,
        max_grad_norm=1.0,
        beta=0.0,
        report_to="none",
        output_dir=args.output_dir,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward, correctness_reward],
        args=config,
        train_dataset=train_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()