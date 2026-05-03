# Prompt Optimization Scripts

Run these commands from the project root:

```bash
cd /Users/arihantbarjatya/open_notebook_llm/ippo/ippo
```

## Test-Time Prompt Optimization

Script:

```bash
scripts/test_time_prompt_optim.py
```

Purpose: evaluate a trained checkpoint and compare prompt optimizers without changing model weights.

Default checkpoint:

```bash
checkpoints/curriculum_hacking/checkpoints/stage2_reasoning_first_FINAL
```

Default dataset:

```bash
data/processed/prelim_test.jsonl
```

Run the full default comparison:

```bash
python3 scripts/test_time_prompt_optim.py
```

Run a small smoke test:

```bash
python3 scripts/test_time_prompt_optim.py \
  --limit 10 \
  --optimizers gepa \
  --iterations 1
```

Run selected optimizers:

```bash
python3 scripts/test_time_prompt_optim.py \
  --optimizers gepa dspy_mipro textgrad_tgd
```

Outputs are written by default to:

```bash
outputs/prompt_optimizer_comparison
```

## Train-Time Prompt Optimization

Main script:

```bash
scripts/curriculum_hacked/train_time_prompt_opt.py
```

Purpose: run GRPO training while applying prompt or reward interventions during training.

Common runs:

```bash
python3 scripts/curriculum_hacked/train_time_prompt_opt.py --condition 1a
python3 scripts/curriculum_hacked/train_time_prompt_opt.py --condition 2b
python3 scripts/curriculum_hacked/train_time_prompt_opt.py --condition 3a
```

Useful conditions:

```text
1a / 1b / 1c       static prompt selection before training
2a / 2b            adaptive train-time prompt optimization
2c-blind/nonblind  adaptive reward-shaping coefficients
2d-blind/nonblind  combined prompt + reward optimization
3a / 3b            resume from hacked checkpoint, stage-2 GRPO only
```

Default outputs are written to:

```bash
outputs/train_time_prompt_opt/<condition>
```

## Train-Time Optimizer Comparison

Comparison script:

```bash
scripts/curriculum_hacked/train_time_prompt_opt_comparison.py
```

Purpose: run train-time GRPO while comparing the same prompt optimization tools used by the test-time script.

Run from base model:

```bash
python3 scripts/curriculum_hacked/train_time_prompt_opt_comparison.py \
  --condition 2compare
```

Resume from hacked checkpoint:

```bash
python3 scripts/curriculum_hacked/train_time_prompt_opt_comparison.py \
  --condition 3compare
```

Run a smaller comparison set:

```bash
python3 scripts/curriculum_hacked/train_time_prompt_opt_comparison.py \
  --condition 3compare \
  --comparison-optimizers gepa textgrad_tgd \
  --iterations 1
```

Comparison update logs are written to:

```bash
outputs/train_time_prompt_opt/<condition>/optimizer_comparison_updates.jsonl
```

## API Keys

GEPA-style proposals, DSPy prompt models, and TextGrad critics may need API keys:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

Local checkpoint evaluation and GRPO training also need the model stack used by the notebooks, including `torch`, `transformers`, `peft`, `trl`, and `unsloth`.
