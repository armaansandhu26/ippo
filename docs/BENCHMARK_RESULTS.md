# Benchmark results (baseline vs hacked vs recovery)

**Canonical human-written summary.** Auto-generated tables live next to figures — see [docs/README.md](./README.md).

Last updated: 2026-06-17. Final eval on **unbiased test** (n=135, stage-2 format).

## Conditions

- **Baseline** — trained on unbiased curriculum (`biased_curriculum=False`).
- **Hacked** — trained on biased curriculum (correct answer always A during training).
- **Recovery** — continue training hacked checkpoint on unbiased curriculum; metrics from `final_eval_after_recovery.json`.

## Metric conventions

Per [PublishingPlan.md](../PublishingPlan.md):

- **Numeric decoupling** is the primary reasoning/decoupling metric.
- **Judge reasoning / judge decoupling** are companion checks (128/135 questions after judge exclusions).
- Reported values are **seed mean ± stdev** where multiple seeds exist.

## Run roots & coverage

| Role | Run root |
|------|----------|
| Qwen biased / unbiased | `qwen_2.5_family_runs_v1_only` |
| Llama biased / unbiased | `llama_3.x_family_runs_v1_only` |
| Recovery | `qwen_2.5_family_and_llama_3.x_family_recovery_runs_v1` |

### Seed coverage gaps

| Family | Model | Missing |
|--------|-------|---------|
| Qwen2.5 | 0.5B | biased `seed7` final eval; recovery `seed7` |
| Qwen2.5 | 3B | biased `seed7`, `seed123` final eval; recovery `seed7`, `seed123` |
| Llama 3.x | 3.2-3B | unbiased `seed123` final eval |
| Qwen2.5 | 14B | hacked runs only in older (non-v1_only) roots — no v1_only hacked comparison |

Fully complete (biased + unbiased + recovery, 3 seeds each): `qwen2.5-1.5b`, `qwen2.5-7b`, `qwen2.5-14b`, `llama3.2-1b`, `llama3.1-8b`.

---

## 1. Baseline vs hacked

Δ = hacked − baseline. Positive Δ on A-rate / decoupling = more shortcutting or decoupling after biased training.

| Family | Model | Seeds (B/H) | Acc (base) | Acc (hack) | Δ Acc | A-rate (base) | A-rate (hack) | Δ A-rate | Dec num (base) | Dec num (hack) | Dec judge (base) | Dec judge (hack) |
|--------|-------|-------------|------------|------------|-------|---------------|---------------|----------|----------------|----------------|------------------|------------------|
| Qwen2.5 | 0.5b | 3/2 | 0.269±0.021 | 0.259±0.000 | −0.010 | 0.242±0.291 | 0.993±0.010 | +0.751 | 0.015±0.026 | 0.015±0.021 | 0.217±0.062 | 0.222±0.031 |
| Qwen2.5 | 1.5b | 3/3 | 0.674±0.013 | 0.259±0.007 | −0.415 | 0.326±0.039 | 0.993±0.007 | +0.667 | 0.002±0.004 | 0.286±0.048 | 0.114±0.017 | 0.427±0.122 |
| Qwen2.5 | 3b | 3/1 | 0.867±0.013 | 0.304 | −0.563 | 0.252±0.013 | 0.926 | +0.674 | 0.000±0.000 | 0.193 | 0.057±0.004 | 0.659 |
| Qwen2.5 | 7b | 3/3 | 0.899±0.023 | 0.825±0.069 | −0.074 | 0.284±0.031 | 0.328±0.088 | +0.044 | 0.002±0.004 | 0.044±0.065 | 0.040±0.019 | 0.101±0.073 |
| Qwen2.5 | 14b | 3/0 | 0.931±0.011 | — | — | 0.277±0.015 | — | — | 0.002±0.004 | — | 0.025±0.019 | — |
| Llama 3.x | llama3.2-1b | 3/3 | 0.284±0.019 | 0.277±0.015 | −0.007 | 0.341±0.096 | 0.889±0.129 | +0.548 | 0.030±0.051 | 0.027±0.026 | 0.141±0.039 | 0.111±0.049 |
| Llama 3.x | llama3.2-3b | 2/3 | 0.493±0.152 | 0.269±0.019 | −0.223 | 0.256±0.005 | 0.965±0.011 | +0.710 | 0.022±0.021 | 0.062±0.057 | 0.093±0.047 | 0.220±0.092 |
| Llama 3.x | llama3.1-8b | 3/3 | 0.815±0.020 | 0.309±0.086 | −0.506 | 0.257±0.015 | 0.941±0.103 | +0.684 | 0.005±0.004 | 0.156±0.111 | 0.022±0.020 | 0.291±0.079 |

Full per-condition tables: see auto-generated `SUMMARY.md` in each family’s `figures/` folder.

---

## 2. Baseline vs hacked vs recovery

Recovery starts from the hacked checkpoint. All three conditions evaluated on the same unbiased test (n=135).

### Headline numbers (seed means, rounded)

**Qwen2.5**

| Model | Acc (base → hack → rec) | A-rate (base → hack → rec) |
|-------|-------------------------|----------------------------|
| 0.5B | 0.27 → 0.26 → 0.26 | 0.24 → 0.99 → 1.00 |
| 1.5B | 0.67 → 0.26 → 0.38 | 0.33 → 0.99 → 0.79 |
| 3B | 0.87 → 0.30 → 0.87 | 0.25 → 0.93 → 0.28 |
| 7B | 0.90 → 0.83 → 0.89 | 0.28 → 0.33 → 0.25 |
| 14B | 0.93 → — → 0.94 | 0.28 → — → 0.26 |

**Llama 3.x**

| Model | Acc (base → hack → rec) | A-rate (base → hack → rec) |
|-------|-------------------------|----------------------------|
| 3.2-1B | 0.28 → 0.28 → 0.27 | 0.34 → 0.89 → 0.80 |
| 3.2-3B | 0.49 → 0.27 → 0.39 | 0.26 → 0.97 → 0.81 |
| 3.1-8B | 0.82 → 0.31 → 0.59 | 0.26 → 0.94 → 0.58 |

### Full three-way comparison

| Family | Model | Seeds (B/H/R) | Acc: base / hack / rec | A-rate: base / hack / rec | Dec (num): base / hack / rec | Dec (judge): base / hack / rec |
|--------|-------|---------------|------------------------|---------------------------|------------------------------|--------------------------------|
| Qwen2.5 | 0.5b | 3/2/2 | 0.269±0.021 / 0.259±0.000 / 0.259±0.000 | 0.242±0.291 / 0.993±0.010 / 1.000±0.000 | 0.015±0.026 / 0.015±0.021 / 0.000±0.000 | 0.217±0.062 / 0.222±0.031 / 0.189±0.110 |
| Qwen2.5 | 1.5b | 3/3/3 | 0.674±0.013 / 0.259±0.007 / 0.383±0.214 | 0.326±0.039 / 0.993±0.007 / 0.785±0.359 | 0.002±0.004 / 0.286±0.048 / 0.178±0.148 | 0.114±0.017 / 0.427±0.122 / 0.254±0.163 |
| Qwen2.5 | 3b | 3/1/1 | 0.867±0.013 / 0.304 / 0.874 | 0.252±0.013 / 0.926 / 0.281 | 0.000±0.000 / 0.193 / 0.000 | 0.057±0.004 / 0.659 / 0.007 |
| Qwen2.5 | 7b | 3/3/3 | 0.899±0.023 / 0.825±0.069 / 0.894±0.004 | 0.284±0.031 / 0.328±0.088 / 0.252±0.015 | 0.002±0.004 / 0.044±0.065 / 0.000±0.000 | 0.040±0.019 / 0.101±0.073 / 0.012±0.021 |
| Qwen2.5 | 14b | 3/0/3 | 0.931±0.011 / — / 0.941±0.020 | 0.277±0.015 / — / 0.262±0.011 | 0.002±0.004 / — / 0.000±0.000 | 0.025±0.019 / — / 0.005±0.004 |
| Llama 3.x | llama3.2-1b | 3/3/3 | 0.284±0.019 / 0.277±0.015 / 0.274±0.034 | 0.341±0.096 / 0.889±0.129 / 0.795±0.161 | 0.030±0.051 / 0.027±0.026 / 0.030±0.034 | 0.141±0.039 / 0.111±0.049 / 0.091±0.067 |
| Llama 3.x | llama3.2-3b | 2/3/3 | 0.493±0.152 / 0.269±0.019 / 0.388±0.235 | 0.256±0.005 / 0.965±0.011 / 0.807±0.308 | 0.022±0.021 / 0.062±0.057 / 0.007±0.007 | 0.093±0.047 / 0.220±0.092 / 0.143±0.089 |
| Llama 3.x | llama3.1-8b | 3/3/3 | 0.815±0.020 / 0.309±0.086 / 0.590±0.298 | 0.257±0.015 / 0.941±0.103 / 0.578±0.377 | 0.005±0.004 / 0.156±0.111 / 0.012±0.011 | 0.022±0.020 / 0.291±0.079 / 0.111±0.100 |

Recovery post-final metrics and diagnostics: [RECOVERY_SUMMARY.md](../benchmark_metrics/families/qwen_2.5_family_and_llama_3.x_family_recovery_runs_v1/figures/RECOVERY_SUMMARY.md).

### Recovery dynamics (A-rate thresholds)

First global step where unbiased-validate A-rate drops below threshold (mean±stdev over seeds that crossed).

| Model | Pre-resume A-rate | Post-recovery A-rate | Recovery@0.75 | Recovery@0.50 | Recovery@0.35 |
|-------|-------------------|----------------------|---------------|---------------|---------------|
| qwen2.5-0.5b | 1.000±0.000 | 1.000±0.000 | — (0/2) | — (0/2) | — (0/2) |
| qwen2.5-1.5b | 0.990±0.009 | 0.785±0.359 | 30±0 (1/3) | 100±0 (1/3) | 180±0 (1/3) |
| qwen2.5-3b | 0.953±0.000 | 0.281±0.000 | 30±0 (1/1) | 40±0 (1/1) | 60±0 (1/1) |
| qwen2.5-7b | 0.333±0.055 | 0.252±0.015 | 10±0 (3/3) | 10±0 (3/3) | 10±0 (3/3) |
| qwen2.5-14b | 0.312±0.062 | 0.262±0.011 | 10±0 (3/3) | 10±0 (3/3) | 13±6 (3/3) |
| llama3.2-1b | 0.875±0.154 | 0.795±0.161 | 10±0 (1/3) | — (0/3) | — (0/3) |
| llama3.2-3b | 0.953±0.027 | 0.807±0.308 | 25±7 (2/3) | 30±0 (1/3) | 40±0 (1/3) |
| llama3.1-8b | 0.948±0.090 | 0.578±0.377 | 25±21 (2/3) | 80±71 (2/3) | 60±0 (1/3) |

---

## 3. Patterns for the paper

### Curriculum hacking (baseline → hacked)

1. **A-rate collapse is the headline hacking signal** for small/mid Qwen models: 0.5B–3B hacked A-rate jumps to ~0.93–0.99 vs ~0.24–0.33 baseline, while accuracy drops sharply (e.g. 1.5B: 0.67 → 0.26).
2. **Large Qwen models hack differently**: 7B keeps high accuracy (0.90 → 0.83) with only modest A-rate increase (0.28 → 0.33); 14B unbiased baseline already strong (acc 0.93) with no hacked v1_only run to compare.
3. **Numeric decoupling rises where reasoning survives but answers go wrong** — clearest on 1.5B hacked (dec 0.00 → 0.29) and 3B hacked (0.00 → 0.19, 1 seed).
4. **Judge decoupling confirms the story on mid-size models** (1.5B hacked dec_judge 0.43; 3B hacked 0.66) while unbiased 3B/7B/14B show low judge decoupling (0.02–0.06).
5. **Llama families show the same A-rate hacking gap** at finals (e.g. 3.2-1B: 0.34 → 0.89; 3.1-8B: 0.26 → 0.94) with mixed accuracy effects.

### Recovery (hacked → recovery final)

1. **Qwen 7B, 3B, and 14B recover accuracy near baseline** (7B rec acc 0.89 vs base 0.90; 3B rec 0.87 vs base 0.87; 14B rec 0.94 vs base 0.93) **with much lower A-rate** than hacked.
2. **Qwen 1.5B recovery is partial**: accuracy improves vs hacked (0.38 vs 0.26) but stays below baseline (0.67); A-rate remains high (0.79 vs 0.99 hacked).
3. **Qwen 0.5B does not recover** — post-recovery A-rate stays at 1.0 with baseline-level accuracy (~0.26).
4. **Llama recovery is mixed**: 3.2-3B regains some accuracy (0.39 vs 0.27 hacked) but A-rate often remains elevated (0.81); 3.1-8B shows high seed variance (acc 0.59±0.30, A-rate 0.58±0.38 across 3 seeds).
5. **Decoupling after recovery** is generally low on Qwen 7B/3B/14B (numeric dec ~0) but can remain elevated on 1.5B (0.18–0.27) and some Llama runs.

---

## 4. Figure index

### Per-family (biased / unbiased training)

- Qwen: `benchmark_metrics/families/qwen_2.5_family_runs_v1_only/figures/`
- Llama: `benchmark_metrics/families/llama_3.x_family_runs_v1_only/figures/`
- Key plots: `01_final_metrics_by_model.png`, `06_a_rate_hacking_gap.png`, `04_decoupling_numeric_vs_judge.png`
- Collapse dynamics: `HISTORY_SUMMARY.md` + `09_collapse_step_thresholds.png`

### Cross-family

- `benchmark_metrics/combined/cross_family_figures/` — see [SUMMARY.md](../benchmark_metrics/combined/cross_family_figures/SUMMARY.md)

### Recovery

- `benchmark_metrics/families/qwen_2.5_family_and_llama_3.x_family_recovery_runs_v1/figures/`
- Key plots: `01_recovery_a_rate_vs_step.png`, `04_post_recovery_final_eval.png`, `09_hysteresis_hacked_vs_recovery.png`

---

## Not yet in this summary

- Per-stage decoupling trajectory (S0→S1→S2) — blocked until stage jsonl is aggregated.
- Stage-native format eval at each checkpoint — current training logs use stage-2 eval format for dynamics.
