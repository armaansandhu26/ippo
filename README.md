# IPPO: Interleaved Prompt-Policy Optimization

**CS 690S — Adaptive prompt constraints for mitigating goal misgeneralization**

Abhishek Mishra, Arihant Barjatya, Armaan Sandhu, Suyash Maniyar

## Problem

Reinforcement learning with verifiable rewards (RLVR) is the dominant approach for post-training LLMs, but policies can suffer **goal misgeneralization**: they exploit statistical shortcuts in the training distribution instead of the intended computation. Recent work (e.g., Anthropic on reward hacking in coding) shows this can generalize toward broader misalignment when shortcuts reshape the model’s learned behavior.

Typical mitigations (KL regularization, reward ensembles, constrained RLHF) act in **weight space** (Θ). **Inoculation prompting** operates in **prompt space** (Π) and can be very effective, but it is **reactive** (you must anticipate the hack) and **static** (fixed for the whole run).

Following Agrawal et al. (GEPA), module prompts and weights are written `Π_Φ` and `Θ_Φ`; the learnable pair is `⟨Π, Θ⟩_Φ`.

## Method: IPPO

**Interleaved Prompt-Policy Optimization (IPPO)** is a dual-objective alternating procedure:

- **GRPO** optimizes Θ for task reward.
- **GEPA** optimizes Π for an **alignment** objective, so the prompt can adapt as the policy’s failure modes change.

Roughly: every **N** GRPO steps, pause weight updates → run GEPA on rollouts from the current weights `Θ_k`, using an alignment signal (e.g., OOD accuracy where the shortcut fails) → update prompts `Π_k` → resume GRPO with `Π_{k+1}`.

This approximates bilevel optimization: maximize alignment over Π subject to `Θ*(Π)` being the GRPO solution for task reward.

**Planned implementation:** GEPA via [DSPy](https://dspy.ai/tutorials/gepa_ai_program/), GRPO via [TRL `GRPOTrainer`](https://huggingface.co/docs/trl/grpo_trainer), **QLoRA** on **Qwen2.5-3B-Instruct**.

## Hypothesis

On training distributions that admit degenerate shortcuts, **IPPO’s adaptive prompt constraint** will sustain **higher OOD generalization** than unconstrained GRPO and than **static** prompt tuning (optimize prompt once with GEPA, then freeze during GRPO), because the prompt can counter **emerging** shortcuts as Θ changes.

A weaker but still informative outcome would be a **significant delay** in shortcut adoption vs. GRPO-only.

## Repo scope (this code)

- MCQ-GSM8K (and related) dataset generation
- GRPO baseline
- Static **GEPA → GRPO**
- **IPPO** (interleaved GEPA + GRPO)

Pipeline: **GSM8K → MCQ / splits → training → evaluation**.

## Experimental scenarios

| Scenario                           | Training bias                           | Shortcut        | Primary test diagnostic                                           |
| ---------------------------------- | --------------------------------------- | --------------- | ----------------------------------------------------------------- |
| **1. MCQ-GSM8K**                   | Correct option always **A** in training | Always pick A   | Full test with options **shuffled**; emphasize **not-A** accuracy |
| **2. (Stretch) Zero-answer GSM8K** | Only problems whose answer is **0**     | Always output 0 | Full GSM8K test; **non-zero** accuracy                            |

Rewards are exact-match and correctly specified; both a shortcut policy and a reasoning policy can get perfect **train** reward—only the latter should generalize.

## Conditions

1. Baseline (no optimization)
2. GRPO only (weights; vague prompt)
3. **Static GEPA → GRPO** (prompt tuned once, then frozen)
4. **IPPO** — alternate GEPA and GRPO every **100** steps, **K = 5** outer iterations

## Metrics

1. **OOD accuracy** — performance on the full, unbiased test set after training only on the shortcut-biased split.
2. **Shortcut adoption rate** — iterations until: (i) training accuracy ≥ 95% **and** (ii) OOD accuracy drops significantly vs. baseline (paired permutation test, _p_ < 0.05).

Static vs. IPPO isolates **adaptive** vs. **fixed** prompt constraints.

## Design note

The shortcuts are intentionally **maximally easy** in-distribution— a stress test: worse than most real spurious correlations. Strong IPPO results here suggest a **lower bound** on benefit in noisier, real-world settings.

## References (proposal)

1. Meinke et al., “From shortcuts to sabotage…” (Anthropic, 2025)
2. Shao et al., DeepSeekMath — GRPO (arXiv:2402.03300)
3. Agrawal et al., GEPA (arXiv:2507.19457)
4. Langosco et al., goal misgeneralization (ICML 2022)
5. Moskovitz et al., constrained RLHF (ICLR 2024)
6. Soylu et al., fine-tuning + prompt optimization (arXiv:2407.10930)
7. DSPy GEPA tutorial: https://dspy.ai/tutorials/gepa_ai_program/
8. TRL GRPO Trainer: https://huggingface.co/docs/trl/grpo_trainer
