#!/usr/bin/env python3
"""Plot benchmark aggregates for a model family (CSV from aggregate_benchmark_runs.py)."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

TRAIN_STEP_ORDER = ("stage0_end", "stage1_end", "stage2_end", "final")
MODEL_SIZE_RE = re.compile(r"([\d.]+)b", re.I)


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
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


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
        ax.set_xticklabels([m.replace("qwen2.5-", "") for m in models], rotation=0)
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
                label=model.replace("qwen2.5-", ""),
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
            ax.scatter(x, y, s=80, label=model.replace("qwen2.5-", ""))
            ax.annotate(
                model.replace("qwen2.5-", ""),
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
                vals.append(m if m == m else 0.0)
            ax.bar(x, vals, bottom=bottom, label=letter.replace("pct_", ""), color=color)
            bottom += np.array(vals)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("qwen2.5-", "") for m in models])
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
    ax.set_xticklabels([m.replace("qwen2.5-", "") for m in models])
    ax.set_ylabel("Δ A-rate (biased − unbiased train)")
    ax.set_title("Curriculum hacking gap at final eval (unbiased test)")
    fig.tight_layout()
    save_fig(fig, out_dir, "06_a_rate_hacking_gap")


def write_summary_table(
    final_groups: dict[tuple[str, bool], list[dict[str, str]]],
    out_dir: Path,
) -> None:
    lines = [
        "# Qwen2.5 family benchmark summary (final eval, seed mean ± stdev)\n",
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
                f"| {model.replace('qwen2.5-', '')} | {label} | "
                f"{acc:.3f}±{sa:.3f} | {ar:.3f}±{sar:.3f} | "
                f"{dn:.3f}±{sdn:.3f} | {dj:.3f}±{sdj:.3f} | "
                f"{jr:.3f}±{sjr:.3f} |"
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
    args = parser.parse_args()
    if not args.aggregates_csv.is_file():
        raise SystemExit(f"Aggregates CSV not found: {args.aggregates_csv}")

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
        "predicts_A_rate",
        "A-rate",
        out_dir,
        "03_trajectory_a_rate",
        ylim=(-0.02, 1.05),
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

    n_png = len(list(out_dir.glob("*.png")))
    print(f"Wrote {n_png} figures + SUMMARY.md -> {out_dir}")


if __name__ == "__main__":
    main()
