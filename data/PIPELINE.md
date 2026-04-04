# Data Pipeline — MCQ-GSM8K (IPPO)

## Goal

Convert GSM8K into a structured MCQ dataset for shortcut-learning experiments:

- Train: correct answer always at **A**
- Test: answers **shuffled**
- Ensure high-quality distractors and clean formatting
- Maintain full traceability from source → final dataset

---

# Dataset Strategy

We will build the dataset in **two phases**:

## Phase A — Prelim Dataset (Proof of Concept)

**Size:** ~1500 examples

**Purpose:**

- Validate full pipeline end-to-end
- Debug:
  - prompt quality
  - parsing robustness
  - validation logic
  - training setup (GRPO / IPPO)

- Run experiments quickly

**Constraints:**

- Keep cost low (LLM calls)
- Prioritize quality over scale
- Manually inspect samples

**Output:**

- `data/processed/prelim_train.jsonl`
- `data/processed/prelim_test.jsonl`

---

## Phase B — Full Dataset

**Size:** Full GSM8K (~8.5k)

**Purpose:**

- Final experiments
- Report results
- Validate scalability

**Requirements:**

- Stable pipeline
- High validation pass rate
- Good distractor quality

**Output:**

- `data/processed/full_train.jsonl`
- `data/processed/full_test.jsonl`

---

## Key Principle

We **must complete Phase A successfully before starting Phase B**.

Criteria to move forward:

- ≥ 90% parsing success
- ≥ 90% validation pass rate
- No major failure modes
- Training pipeline runs cleanly

---

# Pipeline Overview

```
GSM8K → Base Dataset → LLM Generation → Parsing → Validation → Transformation → Final Dataset
```

---

# Pipeline Stages

## Stage 1 — Load GSM8K

- Load train/test splits
- Store raw data

---

## Stage 2 — Extract Final Answer

- Extract answer after `####`
- Normalize (remove commas, spaces, symbols)
- Attempt numeric conversion

---

## Stage 3 — Prompt Construction

- Build MCQ prompt
- Enforce strict output format
- Ensure realistic distractors

---

## Stage 4 — LLM Generation

- Generate options A–D
- Save raw outputs
- Run in batches

---

## Stage 5 — Parsing

- Extract:
  - options A–D
  - correct label

- Convert to structured format

---

## Stage 6 — Validation

Check:

- exactly 4 options
- no duplicates
- correct answer matches ground truth
- no malformed outputs

Optional:

- distractors within reasonable numeric range
- consistent formatting

---

## Stage 7 — Repair / Retry

- Re-run failed samples
- Improve prompt if needed
- Track attempts

---

## Stage 8 — Transformation

### Train

- Force correct answer → A

### Test

- Shuffle A–D randomly

---

## Stage 9 — Save Outputs

- Base dataset
- Raw generations
- Parsed dataset
- Valid dataset
- Train dataset
- Test dataset

---

## Stage 10 — QA + Stats

- Parse success rate
- Validation pass rate
- Manual inspection
- Summary stats

---

# Data Artifacts

```
data/
├── raw/
│   └── gsm8k_base.jsonl
│
├── mcq_raw/
│   └── generations.jsonl
│
├── processed/
│   ├── mcq_parsed.jsonl
│   ├── mcq_validated.jsonl
│   ├── prelim_train.jsonl
│   ├── prelim_test.jsonl
│   ├── full_train.jsonl
│   ├── full_test.jsonl
│
└── summary.json
```

---

# 🧾 Record Schema

```
{
  "example_id": "...",
  "split": "train/test",
  "question": "...",
  "raw_answer": "...",
  "final_answer_text": "...",
  "final_answer_numeric": ...,

  "options": {
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "..."
  },

  "original_correct_option": "...",
  "final_correct_option": "...",

  "validation_status": "...",
  "transformation_type": "...",
  "permutation": [...]
}
```

---

# TODO Tracker

## Phase A — Prelim Dataset (~1500 samples)

### Step 1 — Base Dataset

- [x] Load GSM8K dataset
- [x] Extract final answer (`####`)
- [x] Normalize answer
- [x] Save `gsm8k_base.jsonl`
- [x] Select ~1500 subset (shuffle + seed)
- [x] Inspect 20 samples manually

---

### Step 2 — Prompt + Generation

- [x] Write MCQ prompt
- [x] Test prompt on 5–10 examples manually
- [x] Implement generation script
- [x] Save raw outputs
- [x] Run on 50 → scale to 1500

---

### Step 3 — Parsing

- [x] Implement parser
- [x] Extract options + correct label
- [x] Handle edge cases
- [x] Save parsed dataset
- [x] Inspect failures

---

### Step 4 — Validation

- [x] Validate structure
- [x] Remove invalid samples
- [x] Track pass rate

---

### Step 5 — Repair

- [x] Retry failed samples
- [x] Improve prompt if needed
- [x] Track attempts

---

### Step 6 — Transformation

- [x] Train → force A
- [x] Test → shuffle
- [x] Save prelim datasets

---

### Step 7 — QA + Stats

- [ ] Compute success rates
- [ ] Check answer distributions
- [ ] Inspect 50–100 samples
- [ ] Save summary.json

---

## Phase B — Full Dataset (~8.5k samples)

- [ ] Run full generation pipeline
- [ ] Monitor failures
- [ ] Re-run failed samples
- [ ] Save full datasets
- [ ] Final QA

---

# Milestones

## Milestone 1

✔ Base dataset ready
✔ 50 MCQ samples working

## Milestone 2

? 1500 prelim dataset complete
? Training pipeline runs

## Milestone 3

? Full dataset complete
? Final experiments ready
