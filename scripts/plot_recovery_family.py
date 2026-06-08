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
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

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
) -> dict[tuple[str, int], dict[str, float | None]]:
    """Final-eval scalars per (model, seed) from final_eval_after_recovery.json."""
    out: dict[tuple[str, int], dict[str, float | None]] = {}
    for child in sorted(runs_root.iterdir()):
        meta = parse_run_dir_name(child.name)
        if meta is None:
            continue
        model, seed, _ = meta
        for fname in ("final_eval_after_recovery.json", "final_eval.json"):
            fpath = child / fname
            if fpath.is_file():
                payload = json.loads(fpath.read_text())
                m = payload.get("final_eval_metrics") or {}
                out[(model, seed)] = {
                    "accuracy": ffloat(m.get("accuracy")),
                    "a_rate": ffloat(m.get("a_rate")),
                    "not_a_accuracy": ffloat(m.get("not_a_accuracy")),
                    "n": ffloat(m.get("n")),
                }
                break
    return out


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
    final_after: dict[tuple[str, int], dict[str, float | None]],
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
    final_after: dict[tuple[str, int], dict[str, float | None]],
    out_dir: Path,
    stem: str = "04_post_recovery_final_eval",
) -> None:
    """Final-eval bars after recovery: accuracy + a_rate + not_a_accuracy."""
    models = sorted_models({m for m, _ in final_after})
    metrics = [
        ("accuracy", "Unbiased test accuracy"),
        ("a_rate", "A-rate"),
        ("not_a_accuracy", "Not-A accuracy"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(len(models))

    for ax, (field, title) in zip(axes, metrics):
        vals, errs = [], []
        for model in models:
            seed_vals = [
                v[field] for (m, _), v in final_after.items()
                if m == model and v.get(field) is not None
            ]
            mv, sv, _ = values_mean_std(seed_vals)
            vals.append(mv)
            errs.append(sv)
        ax.bar(x, vals, yerr=errs, capsize=3, color="#4c72b0")
        ax.set_xticks(x)
        ax.set_xticklabels([display_model_name(m) for m in models], rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.axhline(0.25, color="gray", ls="--", lw=0.8, alpha=0.6)
        if field == "accuracy":
            ax.axhline(0.48, color="green", ls=":", lw=0.8, alpha=0.7)

    axes[0].set_ylabel("Rate")
    fig.suptitle("Post-recovery final eval (n=135, mean ± stdev over seeds)", y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, stem)


# =====================================================================================
# CSV exports + summary table
# =====================================================================================

def write_aggregates_csv(
    history: dict[tuple[str, int], list[dict[str, Any]]],
    final_after: dict[tuple[str, int], dict[str, float | None]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model_name", "seed", "global_step", "source",
        "validate_a_rate", "validate_accuracy", "validate_not_a_accuracy",
        "train_a_rate", "train_accuracy", "train_minus_test_a_rate",
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
    final_after: dict[tuple[str, int], dict[str, float | None]],
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
    args = parser.parse_args()

    if not args.runs_root.is_dir():
        raise SystemExit(f"runs-root not found: {args.runs_root}")
    out_dir = args.output_dir or (args.runs_root / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_style()
    history = load_recovery_history(args.runs_root)
    final_after = load_final_after_recovery(args.runs_root)

    if not history:
        raise SystemExit(
            f"No condition_recovery_* runs with recovery_history.jsonl under {args.runs_root}"
        )

    n_runs = len(history)
    n_models = len({m for m, _ in history})
    print(f"Loaded {n_runs} recovery runs across {n_models} models")
    print(f"Output dir: {out_dir}")

    # CSV + summary first so reviewers can grep numbers without rendering plots.
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
    plot_pre_vs_post_a_rate(history, final_after, out_dir)
    plot_post_recovery_metrics(final_after, out_dir)
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
