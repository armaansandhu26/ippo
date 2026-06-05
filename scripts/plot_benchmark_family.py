#!/usr/bin/env python3
"""Plot benchmark aggregates for a model family (CSV from aggregate_benchmark_runs.py)."""

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

TRAIN_STEP_ORDER = ("stage0_end", "stage1_end", "stage2_end", "final")
MODEL_SIZE_RE = re.compile(r"([\d.]+)b", re.I)
RUN_DIR_RE = re.compile(
    r"^condition_(?P<condition>\d+)_(?:(?P<unbiased>unbiased)_)?"
    r"(?P<model>[\w\.-]+)_seed(?P<seed>\d+)_beta(?P<beta>[\w\.]+)$"
)


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def biased_from_row(row: dict[str, str]) -> bool:
    if "biased_curriculum" in row and row["biased_curriculum"]:
        return parse_bool(row["biased_curriculum"])
    return "unbiased" not in row.get("run_dir", "")


def model_size_b(model_name: str) -> float:
    m = MODEL_SIZE_RE.search(model_name)
    return float(m.group(1)) if m else 0.0


def load_aggregates(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def ffloat(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def seed_mean_std(
    rows: list[dict[str, str]], field: str
) -> tuple[float, float, int]:
    vals = [ffloat(r[field]) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return float("nan"), 0.0, 0
    if len(vals) == 1:
        return vals[0], 0.0, 1
    return mean(vals), stdev(vals), len(vals)


def values_mean_std(vals: list[float | None]) -> tuple[float, float, int]:
    clean = [v for v in vals if v is not None]
    if not clean:
        return float("nan"), 0.0, 0
    if len(clean) == 1:
        return clean[0], 0.0, 1
    return mean(clean), stdev(clean), len(clean)


def fmt_mean_std(value: float, spread: float, n: int, digits: int = 3) -> str:
    if not n or np.isnan(value):
        return "—"
    return f"{value:.{digits}f}±{spread:.{digits}f}"


def group_final(
    aggs: list[dict[str, str]],
) -> dict[tuple[str, bool], list[dict[str, str]]]:
    out: dict[tuple[str, bool], list[dict[str, str]]] = defaultdict(list)
    for row in aggs:
        if row.get("train_step") != "final" or row.get("eval_subset") != "final_eval":
            continue
        out[(row["model_name"], biased_from_row(row))].append(row)
    return out


def group_trajectory(
    aggs: list[dict[str, str]],
) -> dict[tuple[str, bool, str], list[dict[str, str]]]:
    out: dict[tuple[str, bool, str], list[dict[str, str]]] = defaultdict(list)
    for row in aggs:
        step = row.get("train_step", "")
        if step not in TRAIN_STEP_ORDER:
            continue
        if row.get("eval_subset") not in ("final_eval", "validate"):
            continue
        out[(row["model_name"], biased_from_row(row), step)].append(row)
    return out


def sorted_models(models: set[str]) -> list[str]:
    return sorted(models, key=model_size_b)


def display_model_name(model: str) -> str:
    return model.replace("qwen2.5-", "")


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def parse_run_dir_name(name: str) -> tuple[str, bool, int] | None:
    m = RUN_DIR_RE.match(name)
    if not m:
        return None
    return m.group("model"), m.group("unbiased") is None, int(m.group("seed"))


def load_history(
    runs_root: Path,
) -> dict[tuple[str, bool, int], list[dict[str, float | int | None]]]:
    history: dict[tuple[str, bool, int], list[dict[str, float | int | None]]] = {}
    for child in sorted(runs_root.iterdir()):
        meta = parse_run_dir_name(child.name)
        if meta is None:
            continue
        model_name, biased_curriculum, seed = meta
        hist_path = child / "metrics_history.jsonl"
        if not hist_path.is_file():
            continue
        rows: list[dict[str, float | int | None]] = []
        with hist_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                validate = payload.get("validate") or {}
                train_sample = payload.get("train_sample") or {}
                rows.append(
                    {
                        "global_step": int(payload.get("global_step", 0)),
                        "validate_a_rate": ffloat(validate.get("a_rate")),
                        "validate_accuracy": ffloat(validate.get("accuracy")),
                        "train_a_rate": ffloat(train_sample.get("a_rate")),
                        "train_accuracy": ffloat(train_sample.get("accuracy")),
                        "train_minus_test_a_rate": ffloat(
                            payload.get("train_minus_test_a_rate")
                        ),
                    }
                )
        history[(model_name, biased_curriculum, seed)] = sorted(
            rows, key=lambda row: int(row["global_step"])
        )
    return history


def history_series_by_step(
    history: dict[tuple[str, bool, int], list[dict[str, float | int | None]]],
    *,
    model: str,
    biased: bool,
    field: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    runs = {
        seed: rows
        for (m, b, seed), rows in history.items()
        if m == model and b == biased
    }
    steps = sorted(
        {
            int(row["global_step"])
            for rows in runs.values()
            for row in rows
            if row.get(field) is not None
        }
    )
    xs: list[int] = []
    ys: list[float] = []
    yerr: list[float] = []
    for step in steps:
        vals: list[float | None] = []
        for rows in runs.values():
            val = next(
                (row.get(field) for row in rows if int(row["global_step"]) == step),
                None,
            )
            vals.append(None if val is None else float(val))
        m, s, n = values_mean_std(vals)
        if n == 0:
            continue
        xs.append(step)
        ys.append(m)
        yerr.append(s)
    return np.array(xs), np.array(ys), np.array(yerr)


def collapse_steps_by_seed(
    history: dict[tuple[str, bool, int], list[dict[str, float | int | None]]],
    *,
    threshold: float,
    consecutive: int = 1,
) -> dict[tuple[str, bool], list[float]]:
    out: dict[tuple[str, bool], list[float]] = defaultdict(list)
    for (model, biased, _seed), rows in history.items():
        collapse_step = None
        streak = 0
        streak_start = None
        for row in rows:
            a_rate = row.get("validate_a_rate")
            if a_rate is None:
                streak = 0
                streak_start = None
                continue
            if float(a_rate) >= threshold:
                streak += 1
                if streak == 1:
                    streak_start = float(row["global_step"])
                if streak >= consecutive:
                    collapse_step = float(streak_start or row["global_step"])
                    break
            else:
                streak = 0
                streak_start = None
        if collapse_step is not None:
            out[(model, biased)].append(collapse_step)
    return out


def plot_history_metric(
    history: dict[tuple[str, bool, int], list[dict[str, float | int | None]]],
    *,
    field: str,
    title: str,
    ylabel: str,
    out_dir: Path,
    stem: str,
    ylim: tuple[float, float] | None = None,
    thresholds: list[float] | None = None,
) -> None:
    models = sorted_models({m for m, _, _ in history})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    curriculum_labels = [
        (True, "Biased training curriculum"),
        (False, "Unbiased training curriculum"),
    ]
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(models)))

    for ax, (biased, panel_title) in zip(axes, curriculum_labels):
        for color, model in zip(colors, models):
            xs, ys, yerr = history_series_by_step(
                history, model=model, biased=biased, field=field
            )
            if len(xs) == 0:
                continue
            ax.plot(xs, ys, color=color, lw=1.8, label=display_model_name(model))
            ax.fill_between(
                xs,
                ys - yerr,
                ys + yerr,
                color=color,
                alpha=0.15,
                linewidth=0,
            )
        for thresh in thresholds or []:
            ax.axhline(thresh, color="gray", ls="--", lw=0.8, alpha=0.5)
        if field == "validate_accuracy":
            ax.axhline(0.48, color="green", ls=":", lw=0.8, alpha=0.6)
        ax.set_title(panel_title)
        ax.set_xlabel("Global optimizer step")
        if ylim:
            ax.set_ylim(*ylim)

    axes[0].set_ylabel(ylabel)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle(f"{title} vs global step (mean ± stdev over seeds)", y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, stem)


def plot_collapse_steps(
    history: dict[tuple[str, bool, int], list[dict[str, float | int | None]]],
    out_dir: Path,
    *,
    consecutive: int,
    stem: str,
    title: str,
) -> None:
    thresholds = [0.75, 0.90, 0.95]
    models = sorted_models({m for m, _, _ in history})
    collapse_by_thresh = {
        threshold: collapse_steps_by_seed(
            history, threshold=threshold, consecutive=consecutive
        )
        for threshold in thresholds
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    x = np.arange(len(models))
    width = 0.22
    colors = ["#4c72b0", "#dd8452", "#c44e52"]

    for ax, biased, panel_title in zip(
        axes,
        (True, False),
        ("Biased training curriculum", "Unbiased training curriculum"),
    ):
        for idx, (threshold, color) in enumerate(zip(thresholds, colors)):
            vals, errs = [], []
            for model in models:
                steps = collapse_by_thresh[threshold].get((model, biased), [])
                m, s, n = values_mean_std(steps)
                vals.append(m if n else np.nan)
                errs.append(s if n else 0.0)
            offset = (idx - 1) * width
            ax.bar(
                x + offset,
                vals,
                width,
                yerr=errs,
                capsize=3,
                color=color,
                label=f"A-rate >= {threshold:.2f}",
            )
        ax.set_xticks(x)
        ax.set_xticklabels([display_model_name(m) for m in models])
        ax.set_title(panel_title)
        ax.set_xlabel("Model size")

    axes[0].set_ylabel("First collapse step")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, stem)


def write_history_summary(
    history: dict[tuple[str, bool, int], list[dict[str, float | int | None]]],
    out_dir: Path,
) -> None:
    thresholds = [0.75, 0.90, 0.95]
    lines = [
        "# Collapse Summary (dense history)\n",
        "| Model | Biased train | Collapse@0.75 | Collapse@0.90 | Collapse@0.95 | Final validate A-rate | Final train-minus-unbiased-validate A-rate gap |",
        "|-------|--------------|---------------|---------------|---------------|-----------------------|-----------------------------------------------|",
    ]
    collapse_by_thresh = {
        threshold: collapse_steps_by_seed(history, threshold=threshold)
        for threshold in thresholds
    }
    models = sorted_models({m for m, _, _ in history})
    for model in models:
        for biased in (False, True):
            final_a_rates = []
            final_gaps = []
            for (m, b, _seed), rows in history.items():
                if m != model or b != biased or not rows:
                    continue
                final_row = rows[-1]
                final_a_rates.append(
                    None
                    if final_row.get("validate_a_rate") is None
                    else float(final_row["validate_a_rate"])
                )
                final_gaps.append(
                    None
                    if final_row.get("train_minus_test_a_rate") is None
                    else float(final_row["train_minus_test_a_rate"])
                )
            label = "yes" if biased else "no"
            collapse_cells = []
            for threshold in thresholds:
                m, s, n = values_mean_std(
                    collapse_by_thresh[threshold].get((model, biased), [])
                )
                collapse_cells.append("—" if not n else f"{m:.0f}±{s:.0f}")
            final_a, final_a_std, final_a_n = values_mean_std(final_a_rates)
            final_gap, final_gap_std, final_gap_n = values_mean_std(final_gaps)
            lines.append(
                f"| {display_model_name(model)} | {label} | "
                f"{collapse_cells[0]} | {collapse_cells[1]} | {collapse_cells[2]} | "
                f"{('—' if not final_a_n else f'{final_a:.3f}±{final_a_std:.3f}')} | "
                f"{('—' if not final_gap_n else f'{final_gap:.3f}±{final_gap_std:.3f}')} |"
            )
    (out_dir / "HISTORY_SUMMARY.md").write_text("\n".join(lines) + "\n")


def plot_final_bars(
    final_groups: dict[tuple[str, bool], list[dict[str, str]]],
    out_dir: Path,
) -> None:
    models = sorted_models({m for m, _ in final_groups})
    metrics = [
        ("accuracy", "Unbiased test accuracy"),
        ("predicts_A_rate", "A-rate (predicts A)"),
        ("decoupling_rate", "Decoupling (numeric)"),
        ("decoupling_rate_judge", "Decoupling (judge)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    x = np.arange(len(models))
    width = 0.36

    for ax, (field, title) in zip(axes, metrics):
        biased_vals, biased_err = [], []
        unbias_vals, unbias_err = [], []
        for model in models:
            b_rows = final_groups.get((model, True), [])
            u_rows = final_groups.get((model, False), [])
            m_b, s_b, _ = seed_mean_std(b_rows, field)
            m_u, s_u, _ = seed_mean_std(u_rows, field)
            biased_vals.append(m_b)
            biased_err.append(s_b)
            unbias_vals.append(m_u)
            unbias_err.append(s_u)

        ax.bar(x - width / 2, biased_vals, width, yerr=biased_err, capsize=3, label="Biased curriculum", color="#c44e52")
        ax.bar(x + width / 2, unbias_vals, width, yerr=unbias_err, capsize=3, label="Unbiased curriculum", color="#4c72b0")
        ax.set_xticks(x)
        ax.set_xticklabels([display_model_name(m) for m in models], rotation=0)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.set_ylabel("Rate")
        ax.axhline(0.25, color="gray", ls="--", lw=0.8, alpha=0.6)
        if field == "accuracy":
            ax.axhline(0.48, color="green", ls=":", lw=0.8, alpha=0.7)

    axes[0].legend(loc="upper left", ncol=1)
    fig.suptitle("Final eval (n=135, mean ± stdev over 3 seeds)", y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "01_final_metrics_by_model")


def plot_trajectory(
    traj_groups: dict[tuple[str, bool, str], list[dict[str, str]]],
    field: str,
    title: str,
    out_dir: Path,
    stem: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    models = sorted_models({m for m, _, _ in traj_groups})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    curriculum_labels = [(True, "Biased training curriculum"), (False, "Unbiased training curriculum")]
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(models)))

    for ax, (biased, panel_title) in zip(axes, curriculum_labels):
        for color, model in zip(colors, models):
            ys, yerr = [], []
            for step in TRAIN_STEP_ORDER:
                rows = traj_groups.get((model, biased, step), [])
                m, s, _ = seed_mean_std(rows, field)
                ys.append(m)
                yerr.append(s)
            xs = np.arange(len(TRAIN_STEP_ORDER))
            ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                marker="o",
                capsize=3,
                label=display_model_name(model),
                color=color,
                lw=1.5,
            )
        ax.set_xticks(np.arange(len(TRAIN_STEP_ORDER)))
        ax.set_xticklabels(["S0 end", "S1 end", "S2 end", "Final"], rotation=15)
        ax.set_title(panel_title)
        ax.set_xlabel("Training checkpoint")
        if ylim:
            ax.set_ylim(*ylim)
        ax.axhline(0.25, color="gray", ls="--", lw=0.8, alpha=0.5)
        if field == "accuracy":
            ax.axhline(0.48, color="green", ls=":", lw=0.8, alpha=0.6)

    axes[0].set_ylabel(title)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle(f"{title} vs training stage (validate n=64; final n=135)", y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, stem)


def plot_decoupling_numeric_vs_judge(
    final_groups: dict[tuple[str, bool], list[dict[str, str]]],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, biased, title in zip(
        axes,
        (True, False),
        ("Biased curriculum", "Unbiased curriculum"),
    ):
        for model in sorted_models({m for m, _ in final_groups}):
            rows = final_groups.get((model, biased), [])
            x, _, _ = seed_mean_std(rows, "decoupling_rate")
            y, _, _ = seed_mean_std(rows, "decoupling_rate_judge")
            ax.scatter(x, y, s=80, label=display_model_name(model))
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
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Numeric vs judge decoupling (final eval, seed means)", y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "04_decoupling_numeric_vs_judge")


def plot_option_distribution(
    final_groups: dict[tuple[str, bool], list[dict[str, str]]],
    out_dir: Path,
) -> None:
    models = sorted_models({m for m, _ in final_groups})
    letters = ("pct_A", "pct_B", "pct_C", "pct_D")
    colors = ["#c44e52", "#4c72b0", "#55a868", "#8172b2"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, biased, title in zip(
        axes,
        (True, False),
        ("Biased curriculum", "Unbiased curriculum"),
    ):
        bottom = np.zeros(len(models))
        x = np.arange(len(models))
        for letter, color in zip(letters, colors):
            vals = []
            for model in models:
                rows = final_groups.get((model, biased), [])
                m, _, _ = seed_mean_std(rows, letter)
                vals.append(0.0 if np.isnan(m) else m)
            ax.bar(x, vals, bottom=bottom, label=letter.replace("pct_", ""), color=color)
            bottom += np.array(vals)
        ax.set_xticks(x)
        ax.set_xticklabels([display_model_name(m) for m in models])
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.set_ylabel("Option share")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Answer letter distribution at final eval", y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "05_option_distribution_final")


def plot_hacking_gap(
    final_groups: dict[tuple[str, bool], list[dict[str, str]]],
    out_dir: Path,
) -> None:
    """A-rate gap: same model size, biased vs unbiased training."""
    models = sorted_models({m for m, _ in final_groups})
    gaps, errs = [], []
    for model in models:
        b_rows = final_groups.get((model, True), [])
        u_rows = final_groups.get((model, False), [])
        m_b, _, _ = seed_mean_std(b_rows, "predicts_A_rate")
        m_u, _, _ = seed_mean_std(u_rows, "predicts_A_rate")
        gaps.append(m_b - m_u)
        # rough combined stderr
        _, s_b, n_b = seed_mean_std(b_rows, "predicts_A_rate")
        _, s_u, n_u = seed_mean_std(u_rows, "predicts_A_rate")
        eb = s_b / (n_b**0.5) if n_b else 0
        eu = s_u / (n_u**0.5) if n_u else 0
        errs.append((eb**2 + eu**2) ** 0.5)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(models))
    ax.bar(x, gaps, yerr=errs, capsize=4, color="#dd8452")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([display_model_name(m) for m in models])
    ax.set_ylabel("Δ A-rate (biased − unbiased train)")
    ax.set_title("Curriculum hacking gap at final eval (unbiased test)")
    fig.tight_layout()
    save_fig(fig, out_dir, "06_a_rate_hacking_gap")


def write_summary_table(
    final_groups: dict[tuple[str, bool], list[dict[str, str]]],
    out_dir: Path,
) -> None:
    lines = [
        "# Family benchmark summary (final eval, seed mean ± stdev)\n",
        "| Model | Biased train | Acc | A-rate | Dec (num) | Dec (judge) | Judge reasoning OK |",
        "|-------|--------------|-----|--------|-----------|-------------|---------------------|",
    ]
    for model in sorted_models({m for m, _ in final_groups}):
        for biased in (False, True):
            rows = final_groups.get((model, biased), [])
            if not rows:
                continue
            acc, sa, _ = seed_mean_std(rows, "accuracy")
            ar, sar, _ = seed_mean_std(rows, "predicts_A_rate")
            dn, sdn, _ = seed_mean_std(rows, "decoupling_rate")
            dj, sdj, _ = seed_mean_std(rows, "decoupling_rate_judge")
            # alignment rate from rows csv would be better; approximate from judge reasoning rate
            jr, sjr, _ = seed_mean_std(rows, "reasoning_correct_judge_rate")
            label = "yes" if biased else "no"
            lines.append(
                f"| {display_model_name(model)} | {label} | "
                f"{fmt_mean_std(acc, sa, len(rows))} | "
                f"{fmt_mean_std(ar, sar, len(rows))} | "
                f"{fmt_mean_std(dn, sdn, len(rows))} | "
                f"{fmt_mean_std(dj, sdj, len(rows))} | "
                f"{fmt_mean_std(jr, sjr, len(rows))} |"
            )
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregates-csv",
        type=Path,
        default=Path(
            "benchmark_metrics/families/qwen2.5_family_runs/benchmark_aggregates.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <family-dir>/figures/",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Optional run root with metrics_history.jsonl for dense history plots.",
    )
    args = parser.parse_args()
    if not args.aggregates_csv.is_file():
        raise SystemExit(f"Aggregates CSV not found: {args.aggregates_csv}")
    if args.runs_root is not None and not args.runs_root.is_dir():
        raise SystemExit(f"Runs root not found: {args.runs_root}")

    out_dir = args.output_dir or args.aggregates_csv.parent / "figures"
    setup_style()
    aggs = load_aggregates(args.aggregates_csv)
    final_groups = group_final(aggs)
    traj_groups = group_trajectory(aggs)

    plot_final_bars(final_groups, out_dir)
    plot_trajectory(
        traj_groups,
        "accuracy",
        "Accuracy",
        out_dir,
        "02_trajectory_accuracy",
    )
    plot_trajectory(
        traj_groups,
        "decoupling_rate",
        "Decoupling (numeric)",
        out_dir,
        "03b_trajectory_decoupling_numeric",
        ylim=(-0.02, 0.55),
    )
    plot_decoupling_numeric_vs_judge(final_groups, out_dir)
    plot_option_distribution(final_groups, out_dir)
    plot_hacking_gap(final_groups, out_dir)
    write_summary_table(final_groups, out_dir)

    if args.runs_root is not None:
        history = load_history(args.runs_root)
        if history:
            plot_history_metric(
                history,
                field="train_minus_test_a_rate",
                title="Train-side shortcut gap",
                ylabel="Train-sample A-rate minus unbiased-validate A-rate",
                out_dir=out_dir,
                stem="08_dense_train_minus_unbiased_validate_a_rate_gap",
                thresholds=[0.0],
            )
            plot_collapse_steps(
                history,
                out_dir,
                consecutive=1,
                stem="09_collapse_step_thresholds",
                title="First threshold crossing from dense unbiased-validate A-rate",
            )
            plot_collapse_steps(
                history,
                out_dir,
                consecutive=2,
                stem="10_sustained_collapse_step_thresholds",
                title="Sustained collapse (2 consecutive evals) from dense unbiased-validate A-rate",
            )
            write_history_summary(history, out_dir)

    n_png = len(list(out_dir.glob("*.png")))
    print(f"Wrote {n_png} figures + SUMMARY.md -> {out_dir}")


if __name__ == "__main__":
    main()
