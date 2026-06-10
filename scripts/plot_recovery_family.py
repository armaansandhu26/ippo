#!/usr/bin/env python3
"""Plot recovery dynamics for a family of condition_recovery_* runs.

Mirrors plot_benchmark_family.py but for the recovery phase:
  - Reads recovery_history.jsonl + pre_resume_validate.json +
    final_eval_after_recovery.json from each condition_recovery_<MODEL>_seed<N>_beta<B>
    folder under --runs-root.
  - Computes per-run recovery steps: first global_step where validate a_rate
    drops below 0.75 / 0.50 / 0.35 (opposite direction from collapse).
  - Plots A-rate and accuracy vs recovery step (mean ± stdev over seeds, per model).
  - Plots recovery-step threshold bars and a paired pre-vs-post A-rate bar.
  - Writes recovery_aggregates.csv, recovery_thresholds.csv, RECOVERY_SUMMARY.md.

Usage:
  python scripts/plot_recovery_family.py \\
    --runs-root ps/qwen_2.5_family_and_llama_3.x_family_recovery_runs_v1 \\
    --output-dir benchmark_metrics/families/qwen_2.5_family_and_llama_3.x_family_recovery_v1/figures
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from aggregate_benchmark_runs import (
    RunMeta as BenchmarkRunMeta,
    ROW_FIELDS,
    aggregate_rows,
    load_dataset,
    process_sample,
)

# Recovery dirs: condition_recovery_<MODEL>_seed<N>_beta<B>
RUN_DIR_RE = re.compile(
    r"^condition_recovery_(?P<model>[\w\.-]+)_seed(?P<seed>\d+)_beta(?P<beta>[\w\.]+)$"
)
MODEL_SIZE_RE = re.compile(r"([\d.]+)b", re.I)

# A-rate thresholds for "recovery" (opposite direction from collapse). Plan
# §"Per-model recovery": first step with A-rate below threshold.
RECOVERY_THRESHOLDS = (0.75, 0.50, 0.35)


# =====================================================================================
# Loading
# =====================================================================================

def ffloat(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def values_mean_std(vals: list[float | None]) -> tuple[float, float, int]:
    clean = [v for v in vals if v is not None]
    if not clean:
        return float("nan"), 0.0, 0
    if len(clean) == 1:
        return clean[0], 0.0, 1
    return mean(clean), stdev(clean), len(clean)


def model_size_b(model_name: str) -> float:
    m = MODEL_SIZE_RE.search(model_name)
    return float(m.group(1)) if m else 0.0


def display_model_name(model: str) -> str:
    return model


def sorted_models(models: set[str]) -> list[str]:
    """Sort by family then by size: qwen2.5-* first, then llama3.*, then others."""
    def key(m: str) -> tuple[int, float, str]:
        if m.startswith("qwen"):
            fam = 0
        elif m.startswith("llama"):
            fam = 1
        else:
            fam = 2
        return (fam, model_size_b(m), m)
    return sorted(models, key=key)


def model_family(model: str) -> str:
    if model.startswith("qwen2.5"):
        return "Qwen2.5"
    if model.startswith("llama3"):
        return "Llama3.x"
    return "other"


def parse_run_dir_name(name: str) -> tuple[str, int, str] | None:
    m = RUN_DIR_RE.match(name)
    if not m:
        return None
    return m.group("model"), int(m.group("seed")), m.group("beta")


def make_recovery_benchmark_meta(child: Path, model: str, seed: int, beta: str) -> BenchmarkRunMeta:
    return BenchmarkRunMeta(
        run_dir=child.name,
        path=child,
        condition="recovery",
        model_name=model,
        seed=seed,
        beta=beta,
        biased_curriculum=True,
    )


def load_recovery_history(
    runs_root: Path,
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """Return {(model, seed) -> sorted list of {global_step, validate_a_rate, validate_accuracy, ...}}."""
    history: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for child in sorted(runs_root.iterdir()):
        meta = parse_run_dir_name(child.name)
        if meta is None:
            continue
        model, seed, _beta = meta
        hist_path = child / "recovery_history.jsonl"
        if not hist_path.is_file():
            continue

        rows: list[dict[str, Any]] = []

        # Step-0 baseline from pre_resume snapshot (the hacked policy at the
        # start of recovery). Treated as global_step=0 in the history series so
        # the recovery curve starts from the actual hacked state, not the
        # first 10-step eval (which can already show drift).
        pre_path = child / "pre_resume_validate.json"
        if pre_path.is_file():
            pre = json.loads(pre_path.read_text())
            pm = pre.get("validate_metrics") or {}
            rows.append({
                "global_step": 0,
                "validate_a_rate": ffloat(pm.get("a_rate")),
                "validate_accuracy": ffloat(pm.get("accuracy")),
                "validate_not_a_accuracy": ffloat(pm.get("not_a_accuracy")),
                "train_a_rate": None,
                "train_accuracy": None,
                "train_minus_test_a_rate": None,
                "source": "pre_resume",
            })

        with hist_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                v = payload.get("validate") or {}
                t = payload.get("train_sample") or {}
                rows.append({
                    "global_step": int(payload.get("global_step", 0)),
                    "validate_a_rate": ffloat(v.get("a_rate")),
                    "validate_accuracy": ffloat(v.get("accuracy")),
                    "validate_not_a_accuracy": ffloat(v.get("not_a_accuracy")),
                    "train_a_rate": ffloat(t.get("a_rate")),
                    "train_accuracy": ffloat(t.get("accuracy")),
                    "train_minus_test_a_rate": ffloat(
                        payload.get("train_minus_test_a_rate")
                    ),
                    "source": "recovery_history",
                })

        rows.sort(key=lambda r: int(r["global_step"]))
        history[(model, seed)] = rows
    return history


def load_final_after_recovery(
    runs_root: Path,
    dataset_path: Path | None = None,
) -> dict[tuple[str, int], dict[str, float | None | str]]:
    """Final-eval metrics per (model, seed) from final_eval_after_recovery.json.

    When the per-sample final eval rows are available, compute the same numeric
    reasoning/decoupling aggregates used by the hacked-model benchmark.
    """
    dataset = load_dataset(dataset_path) if dataset_path and dataset_path.is_file() else None
    out: dict[tuple[str, int], dict[str, float | None | str]] = {}
    for child in sorted(runs_root.iterdir()):
        meta = parse_run_dir_name(child.name)
        if meta is None:
            continue
        model, seed, beta = meta
        for fname in ("final_eval_after_recovery.json", "final_eval.json"):
            fpath = child / fname
            if fpath.is_file():
                payload = json.loads(fpath.read_text())
                m = payload.get("final_eval_metrics") or {}
                result: dict[str, float | None | str] = {
                    "accuracy": ffloat(m.get("accuracy")),
                    "a_rate": ffloat(m.get("a_rate")),
                    "not_a_accuracy": ffloat(m.get("not_a_accuracy")),
                    "n": ffloat(m.get("n")),
                    "format_compliance_rate": None,
                    "parse_success_rate": None,
                    "predicts_A_rate": None,
                    "exploits_position_bias_rate": None,
                    "reasoning_correct_numeric_rate": None,
                    "reasoning_correct_option_rate": None,
                    "decoupling_rate": None,
                    "shortcut_decoupling_rate": None,
                    "conditional_decoupling_rate": None,
                    "pct_A": None,
                    "pct_B": None,
                    "pct_C": None,
                    "pct_D": None,
                    "pct_empty": None,
                    "option_entropy": None,
                    "reasoning_correct_judge_rate": None,
                    "decoupling_rate_judge": None,
                    "shortcut_decoupling_rate_judge": None,
                    "conditional_decoupling_rate_judge": None,
                    "final_eval_seeds_note": "",
                }
                samples = m.get("samples") or []
                if dataset is not None and samples:
                    bench_meta = make_recovery_benchmark_meta(child, model, seed, beta)
                    sample_rows = [
                        process_sample(
                            meta=bench_meta,
                            sample=s,
                            dataset=dataset,
                            train_step="final",
                            eval_subset="final_eval_after_recovery",
                            curriculum_stage="stage2_recovery_end",
                            split="unbiased_test",
                        )
                        for s in samples
                    ]
                    agg = aggregate_rows(
                        sample_rows,
                        meta=bench_meta,
                        train_step="final",
                        split="unbiased_test",
                        eval_subset="final_eval_after_recovery",
                        curriculum_stage="stage2_recovery_end",
                        logged=m,
                    )
                    for field in (
                        "format_compliance_rate",
                        "parse_success_rate",
                        "predicts_A_rate",
                        "exploits_position_bias_rate",
                        "reasoning_correct_numeric_rate",
                        "reasoning_correct_option_rate",
                        "decoupling_rate",
                        "shortcut_decoupling_rate",
                        "conditional_decoupling_rate",
                        "pct_A",
                        "pct_B",
                        "pct_C",
                        "pct_D",
                        "pct_empty",
                        "option_entropy",
                    ):
                        result[field] = agg.get(field)
                out[(model, seed)] = result
                break
    return out


def load_final_rows_after_recovery(
    runs_root: Path,
    dataset_path: Path,
) -> list[dict[str, Any]]:
    """Per-sample final rows for completed recovery final evals."""
    if not dataset_path.is_file():
        return []
    dataset = load_dataset(dataset_path)
    out: list[dict[str, Any]] = []
    for child in sorted(runs_root.iterdir()):
        meta = parse_run_dir_name(child.name)
        if meta is None:
            continue
        model, seed, beta = meta
        fpath = child / "final_eval_after_recovery.json"
        if not fpath.is_file():
            continue
        payload = json.loads(fpath.read_text())
        metrics = payload.get("final_eval_metrics") or {}
        samples = metrics.get("samples") or []
        bench_meta = make_recovery_benchmark_meta(child, model, seed, beta)
        for sample in samples:
            out.append(
                process_sample(
                    meta=bench_meta,
                    sample=sample,
                    dataset=dataset,
                    train_step="final",
                    eval_subset="final_eval_after_recovery",
                    curriculum_stage="stage2_recovery_end",
                    split="unbiased_test",
                )
            )
    return out


def merge_final_row_aggregates_into_final_after(
    final_after: dict[tuple[str, int], dict[str, float | None | str]],
    final_rows: list[dict[str, Any]],
    runs_root: Path,
) -> None:
    """Refresh per-run aggregate metrics from per-sample final rows."""
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in final_rows:
        by_run[row["run_dir"]].append(row)

    for run_dir, rows in by_run.items():
        meta = parse_run_dir_name(run_dir)
        if meta is None or not rows:
            continue
        model, seed, beta = meta
        agg = aggregate_rows(
            rows,
            meta=make_recovery_benchmark_meta(runs_root / run_dir, model, seed, beta),
            train_step="final",
            split="unbiased_test",
            eval_subset="final_eval_after_recovery",
            curriculum_stage="stage2_recovery_end",
        )
        target = final_after.get((model, seed))
        if not target:
            continue
        for field in (
            "format_compliance_rate",
            "parse_success_rate",
            "predicts_A_rate",
            "exploits_position_bias_rate",
            "reasoning_correct_numeric_rate",
            "reasoning_correct_option_rate",
            "decoupling_rate",
            "shortcut_decoupling_rate",
            "conditional_decoupling_rate",
            "pct_A",
            "pct_B",
            "pct_C",
            "pct_D",
            "pct_empty",
            "option_entropy",
            "reasoning_correct_judge_rate",
            "decoupling_rate_judge",
            "shortcut_decoupling_rate_judge",
            "conditional_decoupling_rate_judge",
        ):
            target[field] = agg.get(field)


def write_final_rows_csv(
    final_rows: list[dict[str, Any]],
    out_path: Path,
    judge_row_fields: list[str] | None = None,
) -> None:
    fields = list(ROW_FIELDS)
    for field in judge_row_fields or []:
        if field not in fields:
            fields.append(field)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in final_rows:
            writer.writerow(row)


# =====================================================================================
# Derivations
# =====================================================================================

def recovery_step_per_seed(
    history: dict[tuple[str, int], list[dict[str, Any]]],
    *,
    threshold: float,
    consecutive: int = 1,
    exclude_pre_resume: bool = True,
) -> dict[str, list[tuple[int, float]]]:
    """First step where validate_a_rate < threshold (held for `consecutive` evals).

    Returns {model -> list of (seed, step)}. Runs that never cross are absent.
    `exclude_pre_resume`: don't count step=0 as a recovery crossing — recovery
    has to actually happen during training, not be present at the resume point.
    """
    out: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for (model, seed), rows in history.items():
        streak = 0
        streak_start: float | None = None
        recovery_step: float | None = None
        for row in rows:
            if exclude_pre_resume and row.get("source") == "pre_resume":
                continue
            a_rate = row.get("validate_a_rate")
            if a_rate is None:
                streak = 0
                streak_start = None
                continue
            if float(a_rate) < threshold:
                streak += 1
                if streak == 1:
                    streak_start = float(row["global_step"])
                if streak >= consecutive:
                    recovery_step = float(streak_start or row["global_step"])
                    break
            else:
                streak = 0
                streak_start = None
        if recovery_step is not None:
            out[model].append((seed, recovery_step))
    return out


def history_series_by_step(
    history: dict[tuple[str, int], list[dict[str, Any]]],
    *,
    model: str,
    field: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean ± stdev of `field` across seeds, indexed by global_step."""
    seeds = {seed: rows for (m, seed), rows in history.items() if m == model}
    steps = sorted({
        int(r["global_step"])
        for rows in seeds.values()
        for r in rows
        if r.get(field) is not None
    })
    xs, ys, yerr = [], [], []
    for step in steps:
        vals = []
        for rows in seeds.values():
            v = next(
                (r.get(field) for r in rows if int(r["global_step"]) == step),
                None,
            )
            vals.append(None if v is None else float(v))
        m, s, n = values_mean_std(vals)
        if n == 0:
            continue
        xs.append(step)
        ys.append(m)
        yerr.append(s)
    return np.array(xs), np.array(ys), np.array(yerr)


# =====================================================================================
# Plotting
# =====================================================================================

def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def plot_recovery_metric(
    history: dict[tuple[str, int], list[dict[str, Any]]],
    *,
    field: str,
    title: str,
    ylabel: str,
    out_dir: Path,
    stem: str,
    ylim: tuple[float, float] | None = None,
    thresholds: list[float] | None = None,
) -> None:
    """One panel per model family. Curves per model, seeds aggregated."""
    families = sorted({model_family(m) for m, _ in history})
    by_family: dict[str, list[str]] = defaultdict(list)
    for m, _ in history:
        by_family[model_family(m)].append(m)
    for fam in by_family:
        by_family[fam] = sorted_models(set(by_family[fam]))

    fig, axes = plt.subplots(1, len(families), figsize=(5.5 * len(families), 4.5), sharey=True)
    if len(families) == 1:
        axes = [axes]

    for ax, fam in zip(axes, families):
        models = by_family[fam]
        colors = plt.cm.viridis(np.linspace(0.15, 0.9, max(len(models), 1)))
        for color, model in zip(colors, models):
            xs, ys, yerr = history_series_by_step(history, model=model, field=field)
            if len(xs) == 0:
                continue
            ax.plot(xs, ys, color=color, lw=1.8, label=display_model_name(model), marker="o", ms=3)
            ax.fill_between(xs, ys - yerr, ys + yerr, color=color, alpha=0.15, linewidth=0)
        for thresh in thresholds or []:
            ax.axhline(thresh, color="gray", ls="--", lw=0.8, alpha=0.5)
        if field == "validate_accuracy":
            ax.axhline(0.48, color="green", ls=":", lw=0.8, alpha=0.6)
        ax.set_title(f"{fam} family")
        ax.set_xlabel("Recovery step (global optimizer step)")
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(loc="best", frameon=False, fontsize=7)

    axes[0].set_ylabel(ylabel)
    fig.suptitle(f"{title} during recovery (mean ± stdev over seeds)", y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, stem)


def plot_recovery_thresholds(
    history: dict[tuple[str, int], list[dict[str, Any]]],
    out_dir: Path,
    *,
    consecutive: int,
    stem: str,
    title: str,
) -> None:
    models = sorted_models({m for m, _ in history})
    seeds_per_model = {
        m: {seed for (mm, seed) in history if mm == m}
        for m in models
    }
    by_thresh = {
        t: recovery_step_per_seed(history, threshold=t, consecutive=consecutive)
        for t in RECOVERY_THRESHOLDS
    }

    fig, ax = plt.subplots(figsize=(max(8.5, 0.9 * len(models) + 4), 4.8))
    x = np.arange(len(models))
    width = 0.22
    colors = ["#4c72b0", "#dd8452", "#c44e52"]

    for idx, (thresh, color) in enumerate(zip(RECOVERY_THRESHOLDS, colors)):
        vals, errs, ns = [], [], []
        for model in models:
            steps = [s for _seed, s in by_thresh[thresh].get(model, [])]
            m, s, n = values_mean_std(steps)
            vals.append(m if n else np.nan)
            errs.append(s if n else 0.0)
            ns.append(n)
        offset = (idx - 1) * width
        bars = ax.bar(
            x + offset, vals, width, yerr=errs, capsize=3,
            color=color, label=f"A-rate < {thresh:.2f}",
        )
        # Annotate "n/N" above each bar so it's obvious how many seeds recovered
        for i, (bar, n) in enumerate(zip(bars, ns)):
            total = len(seeds_per_model[models[i]])
            h = bar.get_height()
            if np.isnan(h):
                ax.text(bar.get_x() + bar.get_width()/2, 5, f"0/{total}",
                        ha="center", va="bottom", fontsize=7, color=color)
            else:
                ax.text(bar.get_x() + bar.get_width()/2, h + (errs[i] or 0) + 2,
                        f"{n}/{total}", ha="center", va="bottom",
                        fontsize=7, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([display_model_name(m) for m in models], rotation=30, ha="right")
    ax.set_ylabel("First recovery step")
    ax.set_xlabel("Model")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    save_fig(fig, out_dir, stem)


def plot_pre_vs_post_a_rate(
    history: dict[tuple[str, int], list[dict[str, Any]]],
    final_after: dict[tuple[str, int], dict[str, float | None | str]],
    out_dir: Path,
    stem: str = "03_pre_vs_post_a_rate",
) -> None:
    """Paired bars: hacked baseline A-rate vs post-recovery final A-rate, per model.

    Pre = mean over seeds of pre_resume_validate a_rate.
    Post = mean over seeds of final_eval_after_recovery a_rate (n=135).
    """
    models = sorted_models({m for m, _ in history})

    pre_vals, pre_err = [], []
    post_vals, post_err = [], []
    for model in models:
        pre_list = []
        for (m, _seed), rows in history.items():
            if m != model:
                continue
            pre_row = next((r for r in rows if r.get("source") == "pre_resume"), None)
            if pre_row and pre_row.get("validate_a_rate") is not None:
                pre_list.append(float(pre_row["validate_a_rate"]))
        post_list = [
            v["a_rate"] for (m, _), v in final_after.items()
            if m == model and v.get("a_rate") is not None
        ]
        mp, sp, _ = values_mean_std(pre_list)
        mq, sq, _ = values_mean_std(post_list)
        pre_vals.append(mp)
        pre_err.append(sp)
        post_vals.append(mq)
        post_err.append(sq)

    fig, ax = plt.subplots(figsize=(max(8.5, 0.9 * len(models) + 4), 4.8))
    x = np.arange(len(models))
    width = 0.4
    ax.bar(x - width/2, pre_vals, width, yerr=pre_err, capsize=3,
           color="#c44e52", label="Pre-recovery (hacked ckpt, validate n=64)")
    ax.bar(x + width/2, post_vals, width, yerr=post_err, capsize=3,
           color="#4c72b0", label="Post-recovery (final eval, n=135)")
    ax.axhline(0.25, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([display_model_name(m) for m in models], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("A-rate on unbiased test")
    ax.set_title("A-rate before vs after recovery (mean ± stdev over seeds)")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    save_fig(fig, out_dir, stem)


def plot_post_recovery_metrics(
    final_after: dict[tuple[str, int], dict[str, float | None | str]],
    out_dir: Path,
    stem: str = "04_post_recovery_final_eval",
) -> None:
    """Core final metrics after recovery, mirroring hacked-model final bars."""
    models = sorted_models({m for m, _ in final_after})
    metrics = [
        ("accuracy", "Unbiased test accuracy"),
        ("a_rate", "A-rate"),
        ("decoupling_rate", "Decoupling (numeric)"),
        ("decoupling_rate_judge", "Decoupling (judge)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    x = np.arange(len(models))

    for ax, (field, title) in zip(axes, metrics):
        vals, errs = [], []
        for model in models:
            seed_vals = [
                ffloat(v.get(field)) for (m, _), v in final_after.items()
                if m == model and ffloat(v.get(field)) is not None
            ]
            mv, sv, _ = values_mean_std(seed_vals)
            vals.append(mv)
            errs.append(sv)
        ax.bar(x, vals, yerr=errs, capsize=3, color="#4c72b0")
        ax.set_xticks(x)
        ax.set_xticklabels([display_model_name(m) for m in models], rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        if field == "a_rate":
            ax.axhline(0.25, color="gray", ls="--", lw=0.8, alpha=0.6)
        if field == "accuracy":
            ax.axhline(0.48, color="green", ls=":", lw=0.8, alpha=0.7)

    axes[0].set_ylabel("Rate")
    axes[2].set_ylabel("Rate")
    fig.suptitle(
        "Post-recovery final metrics by model (completed final evals only)",
        y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, out_dir, stem)


def plot_post_recovery_reasoning_metrics(
    final_after: dict[tuple[str, int], dict[str, float | None | str]],
    out_dir: Path,
    stem: str = "04b_post_recovery_reasoning_decoupling",
) -> None:
    """Supplementary post-recovery diagnostics."""
    models = sorted_models({m for m, _ in final_after})
    metrics = [
        ("reasoning_correct_numeric_rate", "Reasoning correct (numeric)"),
        ("shortcut_decoupling_rate", "Shortcut-decoupling"),
        ("conditional_decoupling_rate", "Conditional decoupling (numeric)"),
        ("format_compliance_rate", "Format compliance"),
        ("parse_success_rate", "Parse success"),
        ("option_entropy", "Option entropy"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    axes = axes.flatten()
    x = np.arange(len(models))

    for ax, (field, title) in zip(axes, metrics):
        vals, errs = [], []
        for model in models:
            seed_vals = [
                ffloat(v.get(field)) for (m, _), v in final_after.items()
                if m == model and ffloat(v.get(field)) is not None
            ]
            mv, sv, _ = values_mean_std(seed_vals)
            vals.append(mv)
            errs.append(sv)
        ax.bar(x, vals, yerr=errs, capsize=3, color="#55a868")
        ax.set_xticks(x)
        ax.set_xticklabels([display_model_name(m) for m in models], rotation=30, ha="right")
        ax.set_ylim(0, 1.5 if field == "option_entropy" else 1.05)
        ax.set_title(title)

    axes[0].set_ylabel("Rate")
    axes[3].set_ylabel("Rate")
    fig.suptitle(
        "Post-recovery diagnostic metrics (completed final evals only)",
        y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, out_dir, stem)


def plot_post_recovery_option_distribution(
    final_after: dict[tuple[str, int], dict[str, float | None | str]],
    out_dir: Path,
    stem: str = "04c_post_recovery_option_distribution_final",
) -> None:
    """Final answer-letter distribution after recovery."""
    models = sorted_models({m for m, _ in final_after})
    letters = ("pct_A", "pct_B", "pct_C", "pct_D")
    colors = ["#c44e52", "#4c72b0", "#55a868", "#8172b2"]
    fig, ax = plt.subplots(figsize=(max(8.5, 0.9 * len(models) + 4), 4.8))
    bottom = np.zeros(len(models))
    x = np.arange(len(models))
    for letter, color in zip(letters, colors):
        vals = []
        for model in models:
            seed_vals = [
                ffloat(v.get(letter)) for (m, _), v in final_after.items()
                if m == model and ffloat(v.get(letter)) is not None
            ]
            mv, _, _ = values_mean_std(seed_vals)
            vals.append(0.0 if np.isnan(mv) else mv)
        ax.bar(x, vals, bottom=bottom, label=letter.replace("pct_", ""), color=color)
        bottom += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels([display_model_name(m) for m in models], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Option share")
    ax.set_title("Answer letter distribution at post-recovery final eval")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    save_fig(fig, out_dir, stem)


def plot_post_recovery_numeric_vs_judge_decoupling(
    final_after: dict[tuple[str, int], dict[str, float | None | str]],
    out_dir: Path,
    stem: str = "04d_post_recovery_decoupling_numeric_vs_judge",
) -> None:
    """Recovery-side analog of hacked-model numeric-vs-judge decoupling."""
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    for model in sorted_models({m for m, _ in final_after}):
        x_vals = [
            ffloat(v.get("decoupling_rate")) for (m, _), v in final_after.items()
            if m == model and ffloat(v.get("decoupling_rate")) is not None
        ]
        y_vals = [
            ffloat(v.get("decoupling_rate_judge")) for (m, _), v in final_after.items()
            if m == model and ffloat(v.get("decoupling_rate_judge")) is not None
        ]
        x, _, nx = values_mean_std(x_vals)
        y, _, ny = values_mean_std(y_vals)
        if not nx or not ny:
            continue
        ax.scatter(x, y, s=90, label=display_model_name(model))
        ax.annotate(
            display_model_name(model),
            (x, y),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
        )
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax.set_xlim(-0.02, max(0.65, ax.get_xlim()[1]))
    ax.set_ylim(-0.02, max(0.65, ax.get_ylim()[1]))
    ax.set_xlabel("Decoupling (numeric)")
    ax.set_ylabel("Decoupling (judge)")
    ax.set_title("Numeric vs judge decoupling (post-recovery final eval)")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    save_fig(fig, out_dir, stem)


# =====================================================================================
# CSV exports + summary table
# =====================================================================================

def write_aggregates_csv(
    history: dict[tuple[str, int], list[dict[str, Any]]],
    final_after: dict[tuple[str, int], dict[str, float | None | str]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model_name", "seed", "global_step", "source",
        "validate_a_rate", "validate_accuracy", "validate_not_a_accuracy",
        "train_a_rate", "train_accuracy", "train_minus_test_a_rate",
        "format_compliance_rate", "parse_success_rate",
        "predicts_A_rate", "exploits_position_bias_rate",
        "reasoning_correct_numeric_rate", "reasoning_correct_option_rate",
        "decoupling_rate", "shortcut_decoupling_rate",
        "conditional_decoupling_rate",
        "reasoning_correct_judge_rate", "decoupling_rate_judge",
        "shortcut_decoupling_rate_judge", "conditional_decoupling_rate_judge",
        "pct_A", "pct_B", "pct_C", "pct_D", "pct_empty", "option_entropy",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for (model, seed), rows in sorted(history.items()):
            for r in rows:
                w.writerow({"model_name": model, "seed": seed, **r})
            # Append a final-eval row (n=135) with source="final_eval_after_recovery"
            v = final_after.get((model, seed))
            if v:
                w.writerow({
                    "model_name": model, "seed": seed,
                    "global_step": "final",
                    "source": "final_eval_after_recovery",
                    "validate_a_rate": v.get("a_rate"),
                    "validate_accuracy": v.get("accuracy"),
                    "validate_not_a_accuracy": v.get("not_a_accuracy"),
                    "format_compliance_rate": v.get("format_compliance_rate"),
                    "parse_success_rate": v.get("parse_success_rate"),
                    "predicts_A_rate": v.get("predicts_A_rate"),
                    "exploits_position_bias_rate": v.get("exploits_position_bias_rate"),
                    "reasoning_correct_numeric_rate": v.get("reasoning_correct_numeric_rate"),
                    "reasoning_correct_option_rate": v.get("reasoning_correct_option_rate"),
                    "decoupling_rate": v.get("decoupling_rate"),
                    "shortcut_decoupling_rate": v.get("shortcut_decoupling_rate"),
                    "conditional_decoupling_rate": v.get("conditional_decoupling_rate"),
                    "reasoning_correct_judge_rate": v.get("reasoning_correct_judge_rate"),
                    "decoupling_rate_judge": v.get("decoupling_rate_judge"),
                    "shortcut_decoupling_rate_judge": v.get("shortcut_decoupling_rate_judge"),
                    "conditional_decoupling_rate_judge": v.get("conditional_decoupling_rate_judge"),
                    "pct_A": v.get("pct_A"),
                    "pct_B": v.get("pct_B"),
                    "pct_C": v.get("pct_C"),
                    "pct_D": v.get("pct_D"),
                    "pct_empty": v.get("pct_empty"),
                    "option_entropy": v.get("option_entropy"),
                })


def write_thresholds_csv(
    history: dict[tuple[str, int], list[dict[str, Any]]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_thresh = {
        t: recovery_step_per_seed(history, threshold=t) for t in RECOVERY_THRESHOLDS
    }
    seeds = sorted({(m, s) for m, s in history})
    fields = ["model_name", "seed"] + [
        f"recovery_step_at_{int(t*100)}" for t in RECOVERY_THRESHOLDS
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for model, seed in seeds:
            row = {"model_name": model, "seed": seed}
            for t in RECOVERY_THRESHOLDS:
                step = next(
                    (s for (sd, s) in by_thresh[t].get(model, []) if sd == seed),
                    None,
                )
                row[f"recovery_step_at_{int(t*100)}"] = (
                    int(step) if step is not None else ""
                )
            w.writerow(row)


def write_summary_md(
    history: dict[tuple[str, int], list[dict[str, Any]]],
    final_after: dict[tuple[str, int], dict[str, float | None | str]],
    out_path: Path,
) -> None:
    models = sorted_models({m for m, _ in history})
    lines = [
        "# Recovery summary",
        "",
        "Recovery thresholds = first global_step where unbiased-validate A-rate "
        f"drops below the threshold. Reported as mean±stdev (n=#seeds that crossed/total). "
        f"Pre = hacked-policy A-rate at resume (validate n=64). Post = final-eval A-rate (n=135).",
        "",
        "| Model | Pre A-rate | Post A-rate | Post accuracy | Recovery@0.75 | Recovery@0.50 | Recovery@0.35 |",
        "|-------|------------|-------------|---------------|---------------|---------------|---------------|",
    ]
    by_thresh = {
        t: recovery_step_per_seed(history, threshold=t) for t in RECOVERY_THRESHOLDS
    }
    for model in models:
        seeds_total = len({s for (m, s) in history if m == model})
        pre_list = []
        for (m, _s), rows in history.items():
            if m != model:
                continue
            pre = next((r for r in rows if r.get("source") == "pre_resume"), None)
            if pre and pre.get("validate_a_rate") is not None:
                pre_list.append(float(pre["validate_a_rate"]))
        post_a = [
            v["a_rate"] for (m, _), v in final_after.items()
            if m == model and v.get("a_rate") is not None
        ]
        post_acc = [
            v["accuracy"] for (m, _), v in final_after.items()
            if m == model and v.get("accuracy") is not None
        ]
        mp, sp, np_n = values_mean_std(pre_list)
        mq, sq, nq = values_mean_std(post_a)
        ma, sa, na = values_mean_std(post_acc)

        def fmt(m, s, n):
            return "—" if not n else f"{m:.3f}±{s:.3f}"

        cells = [fmt(mp, sp, np_n), fmt(mq, sq, nq), fmt(ma, sa, na)]
        for t in RECOVERY_THRESHOLDS:
            steps = [s for _seed, s in by_thresh[t].get(model, [])]
            ms, ss, ns = values_mean_std(steps)
            if ns == 0:
                cells.append(f"— (0/{seeds_total})")
            else:
                cells.append(f"{ms:.0f}±{ss:.0f} ({ns}/{seeds_total})")
        lines.append(f"| {display_model_name(model)} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## Post-recovery final metrics",
        "",
        "Computed from per-sample `final_eval_after_recovery.json` outputs for runs that have a completed final eval. This is the recovery-side analog of the hacked-model final summary table: numeric decoupling is primary; judge metrics are companion final-snapshot checks.",
        "",
        "| Model | Final seeds | Acc | A-rate | Dec (num) | Dec (judge) | Judge reasoning OK |",
        "|-------|-------------|-----|--------|-----------|-------------|---------------------|",
    ])
    for model in models:
        seeds_total = len({s for (m, s) in history if m == model})

        def vals(field: str) -> list[float]:
            return [
                float(v[field]) for (m, _), v in final_after.items()
                if m == model and ffloat(v.get(field)) is not None
            ]

        acc_m, acc_s, acc_n = values_mean_std(vals("accuracy"))
        ar_m, ar_s, ar_n = values_mean_std(vals("a_rate"))
        dec_m, dec_s, dec_n = values_mean_std(vals("decoupling_rate"))
        decj_m, decj_s, decj_n = values_mean_std(vals("decoupling_rate_judge"))
        jr_m, jr_s, jr_n = values_mean_std(vals("reasoning_correct_judge_rate"))

        def fmt(m: float, s: float, n: int) -> str:
            return "—" if not n else f"{m:.3f}±{s:.3f}"

        final_seed_n = len({seed for (m, seed) in final_after if m == model})
        cells = [
            f"{final_seed_n}/{seeds_total}",
            fmt(acc_m, acc_s, acc_n),
            fmt(ar_m, ar_s, ar_n),
            fmt(dec_m, dec_s, dec_n),
            fmt(decj_m, decj_s, decj_n),
            fmt(jr_m, jr_s, jr_n),
        ]
        lines.append(f"| {display_model_name(model)} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## Post-recovery diagnostics",
        "",
        "Numeric reasoning correctness uses the last number inside the `<reasoning>` block, matching the main benchmark aggregation. Judge metrics are intentionally kept as final-snapshot companions rather than recovery-over-step dynamics for v1.",
        "",
        "| Model | Final seeds | Reasoning OK (num) | Shortcut rate | Shortcut-decoupling (num) | Conditional dec (num) | Format | Parse |",
        "|-------|-------------|--------------------|---------------|---------------------------|-----------------------|--------|-------|",
    ])
    for model in models:
        seeds_total = len({s for (m, s) in history if m == model})

        def vals(field: str) -> list[float]:
            return [
                float(v[field]) for (m, _), v in final_after.items()
                if m == model and ffloat(v.get(field)) is not None
            ]

        reason_m, reason_s, reason_n = values_mean_std(vals("reasoning_correct_numeric_rate"))
        short_m, short_s, short_n = values_mean_std(vals("exploits_position_bias_rate"))
        sdec_m, sdec_s, sdec_n = values_mean_std(vals("shortcut_decoupling_rate"))
        cdec_m, cdec_s, cdec_n = values_mean_std(vals("conditional_decoupling_rate"))
        fmt_m, fmt_s, fmt_n = values_mean_std(vals("format_compliance_rate"))
        parse_m, parse_s, parse_n = values_mean_std(vals("parse_success_rate"))

        def fmt(m: float, s: float, n: int) -> str:
            return "—" if not n else f"{m:.3f}±{s:.3f}"

        final_seed_n = len({seed for (m, seed) in final_after if m == model})
        cells = [
            f"{final_seed_n}/{seeds_total}",
            fmt(reason_m, reason_s, reason_n),
            fmt(short_m, short_s, short_n),
            fmt(sdec_m, sdec_s, sdec_n),
            fmt(cdec_m, cdec_s, cdec_n),
            fmt(fmt_m, fmt_s, fmt_n),
            fmt(parse_m, parse_s, parse_n),
        ]
        lines.append(f"| {display_model_name(model)} | " + " | ".join(cells) + " |")

    out_path.write_text("\n".join(lines) + "\n")


# =====================================================================================
# Main
# =====================================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root", type=Path, required=True,
        help="Dir containing condition_recovery_<MODEL>_seed<N>_beta<B>/ folders.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Default: <runs-root>/figures/",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/prelim_test.jsonl"),
        help="MCQ dataset used to derive numeric reasoning/decoupling metrics.",
    )
    parser.add_argument(
        "--judge-cache-dir",
        type=Path,
        default=Path("benchmark_metrics/judge"),
        help="Shared judge cache dir with judge_solutions.jsonl / judge_alignments.jsonl.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o",
        help="Judge model to use for recovery final align.",
    )
    parser.add_argument(
        "--judge-align",
        action="store_true",
        help="Run judge align on completed recovery final eval rows before plotting.",
    )
    parser.add_argument(
        "--judge-no-resume",
        action="store_true",
        help="Do not reuse cached judge alignments.",
    )
    parser.add_argument(
        "--judge-limit",
        type=int,
        default=None,
        help="Optional cap on number of recovery final rows to align this run.",
    )
    args = parser.parse_args()

    if not args.runs_root.is_dir():
        raise SystemExit(f"runs-root not found: {args.runs_root}")
    out_dir = args.output_dir or (args.runs_root / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_style()
    history = load_recovery_history(args.runs_root)
    final_after = load_final_after_recovery(args.runs_root, args.dataset)
    final_rows = load_final_rows_after_recovery(args.runs_root, args.dataset)
    judge_row_fields: list[str] = []

    if not history:
        raise SystemExit(
            f"No condition_recovery_* runs with recovery_history.jsonl under {args.runs_root}"
        )

    if args.judge_align:
        judge_path = Path(__file__).resolve().parent / "benchmark_llm_judge.py"
        spec = importlib.util.spec_from_file_location("benchmark_llm_judge", judge_path)
        judge_mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader is not None
        sys.modules[spec.name] = judge_mod
        spec.loader.exec_module(judge_mod)

        JudgeConfig = judge_mod.JudgeConfig
        JUDGE_ROW_FIELDS = judge_mod.JUDGE_ROW_FIELDS
        load_jsonl_by_key = judge_mod.load_jsonl_by_key
        merge_alignments_into_rows = judge_mod.merge_alignments_into_rows
        run_align = judge_mod.run_align
        SOLUTIONS_FILE = judge_mod.SOLUTIONS_FILE

        judge_cfg = JudgeConfig(
            cache_dir=args.judge_cache_dir,
            model=args.judge_model,
            resume=not args.judge_no_resume,
            limit=args.judge_limit,
        )
        if not final_rows:
            print("Warning: no completed recovery final rows to judge-align")
        else:
            alignments = run_align(final_rows, judge_cfg)
            solutions = load_jsonl_by_key(
                args.judge_cache_dir / SOLUTIONS_FILE, "question_id"
            )
            merge_alignments_into_rows(final_rows, alignments, solutions)
            judge_row_fields = [
                field for field in JUDGE_ROW_FIELDS if field not in judge_row_fields
            ]
            merge_final_row_aggregates_into_final_after(
                final_after, final_rows, args.runs_root
            )

    n_runs = len(history)
    n_models = len({m for m, _ in history})
    print(f"Loaded {n_runs} recovery runs across {n_models} models")
    print(f"Output dir: {out_dir}")

    # CSV + summary first so reviewers can grep numbers without rendering plots.
    write_final_rows_csv(final_rows, out_dir / "recovery_final_rows.csv", judge_row_fields)
    write_aggregates_csv(history, final_after, out_dir / "recovery_aggregates.csv")
    write_thresholds_csv(history, out_dir / "recovery_thresholds.csv")
    write_summary_md(history, final_after, out_dir / "RECOVERY_SUMMARY.md")

    # Plots
    plot_recovery_metric(
        history,
        field="validate_a_rate",
        title="Unbiased-validate A-rate",
        ylabel="A-rate (predicts A)",
        out_dir=out_dir,
        stem="01_recovery_a_rate_vs_step",
        ylim=(-0.02, 1.05),
        thresholds=list(RECOVERY_THRESHOLDS),
    )
    plot_recovery_metric(
        history,
        field="validate_accuracy",
        title="Unbiased-validate accuracy",
        ylabel="Accuracy",
        out_dir=out_dir,
        stem="02_recovery_accuracy_vs_step",
        ylim=(-0.02, 1.05),
    )
    plot_recovery_metric(
        history,
        field="validate_not_a_accuracy",
        title="Unbiased-validate Not-A accuracy",
        ylabel="Not-A accuracy",
        out_dir=out_dir,
        stem="02b_recovery_not_a_accuracy_vs_step",
        ylim=(-0.02, 1.05),
    )
    plot_recovery_metric(
        history,
        field="train_minus_test_a_rate",
        title="Train-test A-rate gap during recovery",
        ylabel="Train A-rate - unbiased-validate A-rate",
        out_dir=out_dir,
        stem="02c_recovery_train_minus_test_a_gap_vs_step",
        thresholds=[0.0],
    )
    plot_pre_vs_post_a_rate(history, final_after, out_dir)
    plot_post_recovery_metrics(final_after, out_dir)
    plot_post_recovery_reasoning_metrics(final_after, out_dir)
    plot_post_recovery_option_distribution(final_after, out_dir)
    plot_post_recovery_numeric_vs_judge_decoupling(final_after, out_dir)
    plot_recovery_thresholds(
        history, out_dir,
        consecutive=1,
        stem="05_recovery_step_thresholds",
        title="First recovery step (A-rate crosses below threshold)",
    )
    plot_recovery_thresholds(
        history, out_dir,
        consecutive=2,
        stem="06_sustained_recovery_step_thresholds",
        title="Sustained recovery (2 consecutive evals below threshold)",
    )

    n_png = len(list(out_dir.glob("*.png")))
    print(f"Wrote {n_png} figures + CSVs + RECOVERY_SUMMARY.md -> {out_dir}")


if __name__ == "__main__":
    main()
