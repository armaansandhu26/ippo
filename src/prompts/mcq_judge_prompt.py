"""User message for the LLM-as-judge step (`python -m src.data.judge_mcq`, OpenAI or Anthropic)."""

def build_mcq_judge_prompt(question: str, answer: str, options: dict, correct: str) -> str:
    opts_lines = "\n".join(f"{k}. {options[k]}" for k in ("A", "B", "C", "D"))
    wrong_keys = [k for k in ("A", "B", "C", "D") if k != correct]
    wrong_bullets = "\n".join(f"- {k}: {options[k]}" for k in wrong_keys)

    return f"""You are grading multiple-choice math questions built from GSM8K-style word problems.

The CORRECT option is {correct}. The other three must look like answers someone could get from **specific, realistic mistakes** (wrong operation, off-by-one, unit slip, misread fraction, etc.), not random unrelated numbers.

Question:
{question}

Gold final answer (numeric): {answer}

Options:
{opts_lines}

Incorrect options only:
{wrong_bullets}

Return a single JSON object with this shape (no markdown, no extra text):
{{
  "verdict": "pass" or "fail",
  "reason": "one short sentence",
  "distractors": {{
    "{wrong_keys[0]}": {{"plausible": true or false, "note": "few words"}},
    "{wrong_keys[1]}": {{"plausible": true or false, "note": "few words"}},
    "{wrong_keys[2]}": {{"plausible": true or false, "note": "few words"}}
  }}
}}

Use "pass" only if **all three** incorrect options are at least somewhat plausible for this problem. Use "fail" if any distractor is arbitrary, absurdly scaled without justification, or unrelated to the problem's quantities.
"""

