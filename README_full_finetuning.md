# Train-Time Prompt Optimization — Full Fine-Tuning Pipeline

`train_time_full_ft.py` runs the A-biased GRPO curriculum experiments with
**full weight updates** (no LoRA, no 4-bit quantization). It is the full-FT
port of `train_time_prompt_opt.py`, using the recipe validated in
`final_full_finetune.ipynb`:

- `FastLanguageModel.from_pretrained(..., full_finetuning=True, load_in_4bit=False, load_in_8bit=False)`
- No `get_peft_model(...)` call — every parameter trains directly
- `torch._dynamo.config.suppress_errors = True` (keeps Unsloth's compiler ON so
  GRPO's hidden-states logps path works)
- `gradient_checkpointing=True` in `GRPOConfig` (needed for full FT to fit in VRAM)

**Storage policy:** full-FT checkpoints are complete model weights (~3 GB for
Qwen2.5-1.5B). The script therefore never writes intermediate checkpoints
(`save_strategy="no"`), and saves a final model only where required —
condition 0 by default, since its final model is the hacked checkpoint that
conditions 3a/3b/recovery resume from.

---

## 1. Setup

### Hardware
- Validated config: Qwen2.5-1.5B-Instruct full FT + GRPO on a single A100 (40 GB).
- T4-class GPUs (16 GB) are NOT enough for full FT of the 1.5B model. Use a
  smaller `--base-model` (e.g. `Qwen/Qwen2.5-0.5B-Instruct`) if VRAM-limited.

### Install
```bash
pip install --upgrade uv
uv pip install vllm==0.15.1 torchvision bitsandbytes xformers unsloth
uv pip install transformers==4.56.2
uv pip install --no-deps trl==0.22.2
uv pip install openai anthropic python-dotenv   # proposer conditions only
uv pip install "dspy-ai>=2.5"                   # only if you use --use-dspy
```
(On a Tesla T4 specifically, pin `vllm==0.9.2` and `triton==3.2.0` instead.)

### API keys (proposer conditions only)
Create a `.env` next to the script (or export in your shell):
```bash
OPENAI_API_KEY=sk-...        # if --proposer-provider openai (default model: gpt-5.4-mini)
ANTHROPIC_API_KEY=sk-ant-... # if --proposer-provider anthropic (default model: claude-sonnet-4-6)
```
Conditions **0, 1a, 1b, 1c, recovery** need NO key. Conditions
**2a, 2b, 2c-*, 2d-*, 3a, 3b** need a key for the chosen provider.
Override the proposer model with `--proposer-model <name>`.

### Data
Train/test JSONL are pulled from the repo URLs by default. To use a local
training file: `--train-file path/to/train.jsonl`.

---

## 2. Run order

Condition 0 MUST run first — it produces the hacked checkpoint that
3a/3b/recovery consume. Everything else is independent and can run in any
order (or in parallel on separate GPUs).

### Run 1 — Condition 0: the bare hack (saves the checkpoint)
```bash
python train_time_full_ft.py --condition 0 \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --ckpt-dir checkpoints
```
Final model is saved to `checkpoints/stage2_reasoning_first_FINAL/` (~3 GB).
This path is the default for `--hacked-ckpt`, so later runs need no extra flag.

### Runs 2–4 — Conditions 1a / 1b / 1c: static prompts (no key needed)
```bash
python train_time_full_ft.py --condition 1a --base-model Qwen/Qwen2.5-1.5B-Instruct
python train_time_full_ft.py --condition 1b --base-model Qwen/Qwen2.5-1.5B-Instruct
python train_time_full_ft.py --condition 1c --base-model Qwen/Qwen2.5-1.5B-Instruct
```

### Runs 5–6 — Conditions 2a / 2b: adaptive prompt proposer
```bash
python train_time_full_ft.py --condition 2a \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --proposer-provider anthropic

python train_time_full_ft.py --condition 2b \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --proposer-provider anthropic
```
Optional for 2a/2b (and 3a/3b) only: swap in the DSPy proposer with
`--use-dspy --dspy-optimizer copro` (see `--help` for the full `--dspy-*` set).

### Runs 7–11 — Conditions 2c / 2d: reward-coefficient proposer
```bash
python train_time_full_ft.py --condition 2c-blind     --proposer-provider anthropic
python train_time_full_ft.py --condition 2c-nonblind  --proposer-provider anthropic
python train_time_full_ft.py --condition 2d-blind     --proposer-provider anthropic
python train_time_full_ft.py --condition 2d-nonblind  --proposer-provider anthropic
python train_time_full_ft.py --condition 2d-oracle    --proposer-provider anthropic
```
(add `--base-model ...` if you changed it from the default)

### Runs 12–13 — Conditions 3a / 3b: resume from the hacked checkpoint
```bash
python train_time_full_ft.py --condition 3a \
    --hacked-ckpt checkpoints/stage2_reasoning_first_FINAL \
    --proposer-provider anthropic

python train_time_full_ft.py --condition 3b \
    --hacked-ckpt checkpoints/stage2_reasoning_first_FINAL \
    --proposer-provider anthropic
```
`--base-model` is ignored here — the model comes from the checkpoint, which is
a FULL-WEIGHT saved model (not a LoRA adapter).

### Run 14 — Recovery (no key needed)
```bash
python train_time_full_ft.py --condition recovery \
    --hacked-ckpt checkpoints/stage2_reasoning_first_FINAL
```

Keep these IDENTICAL across all runs so conditions are comparable:
`--base-model`, `--seed` (default 42), `--train-file`, `--beta`, and the
proposer provider/model for all proposer conditions.

---

## 3. Checkpoint & storage flags

| Flag | Default | Meaning |
|---|---|---|
| `--ckpt-dir` | `<output-root>/checkpoints` | Where FINAL full-weight models are saved |
| `--save-final-model` | `auto` | `auto`: save only condition 0. `always`: save every condition's final model. `never`: save nothing |
| `--hacked-ckpt` | `checkpoints/stage2_reasoning_first_FINAL` | Full-weight checkpoint dir consumed by 3a/3b/recovery |

Intermediate checkpoints are never written, regardless of flags.

To keep the weights of any other run (e.g. the recovered model):
```bash
python train_time_full_ft.py --condition recovery \
    --hacked-ckpt checkpoints/stage2_reasoning_first_FINAL \
    --save-final-model always --ckpt-dir checkpoints
```
That saves to `checkpoints/condition_recovery_FINAL/`.

Disk budget rule of thumb (Qwen2.5-1.5B, bf16): ~3 GB per saved final model.
Default policy across all 14 runs = one saved model total (~3 GB) plus JSON logs.

---

## 4. Outputs

Each run writes to `outputs/train_time_full_ft/<condition>/` (override with
`--output-root`):

- `run.log` — full log
- `post_<stage>_validate.json` — validate metrics after each curriculum stage
- `baseline-/proposer-history JSONL files` — dense per-fire metrics during training
- `manager_final.json` — final accepted system prompt + acceptance history
- `final_train_eval.json` / `final_eval.json` — final metrics with samples
  (`final_eval_after_recovery.json` mirror for the recovery condition)

Key metrics inside the eval JSONs: `accuracy`, `not_a_accuracy`, `a_rate`,
`format_rate`.

---

## 5. Common flags reference

```text
--condition        {0,1a,1b,1c,2a,2b,2c-blind,2c-nonblind,2d-blind,2d-nonblind,2d-oracle,3a,3b,recovery}  (required)
--base-model       Full-precision instruct model (NOT a -bnb-4bit repo; incompatible with full FT)
--ckpt-dir         Where final models are saved
--save-final-model auto | always | never
--hacked-ckpt      Condition-0 output dir, for 3a/3b/recovery
--proposer-provider openai | anthropic
--proposer-model   Override proposer model name
--train-file       Local path or URL for training JSONL
--output-root      Override the per-condition output directory
--seed             Eval sampling seed (default 42)
--beta             GRPO KL coefficient (default 0.0; try 0.05–0.1 for recovery-style runs)
--cache-dir        Custom HuggingFace cache directory
--use-dspy + --dspy-*  DSPy proposer for 2a/2b/3a/3b only
```

---

## 6. Troubleshooting

- **`mat1 and mat2 shapes cannot be multiplied (Nx151936 and 1536x151936)`** —
  something disabled Unsloth's compiler. Do NOT set `UNSLOTH_COMPILE_DISABLE`
  or `torch._dynamo.config.disable`; the script already applies the correct
  fix (`suppress_errors = True`).
- **CUDA OOM** — full FT needs far more VRAM than 4-bit LoRA. Drop to
  `Qwen/Qwen2.5-0.5B-Instruct`, or move to a 40 GB+ GPU. Gradient
  checkpointing is already on.
- **`Set OPENAI_API_KEY / ANTHROPIC_API_KEY ...`** — you ran a proposer
  condition without the matching key in env/`.env`.
- **3a/3b fails to load checkpoint** — `--hacked-ckpt` must point to a
  directory produced by this script's condition 0 (full weights:
  `config.json` + `model.safetensors` + tokenizer files). LoRA adapter dirs
  from the old pipeline will not load correctly here.
- **Pre-quantized model error** — `full_finetuning=True` cannot load
  `*-bnb-4bit` repos. Use the original full-precision model id.
