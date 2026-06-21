#!/usr/bin/env python3
"""Plot cross-family benchmark comparisons from family aggregate CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_theme import (
    CHANCE_COLOR,
    COLORS,
    CURRICULA,
    FAMILIES,
    FAMILY_LINE_STYLES,
    MARKERS,
    REFERENCE_GREEN,
    display_size,
    family_label,
    family_model_colors,
    normalize_family,
    setup_style,
    style_axes,
)


RUN_DIR_RE = re.compile(
    r"^condition_(?P<condition>\d+)_(?:(?P<unbiased>unbiased)_)?"
    r"(?P<model>[\w\.-]+)_seed(?P<seed>\d+)_beta(?P<beta>[\w\.]+)$"
)
MODEL_SIZE_RE = re.compile(r"([\d.]+)b", re.I)
TRAIN_STEP_ORDER = ("stage0_end", "stage1_end", "stage2_end", "final")
CHECKPOINT_LABELS = ("S0 end", "S1 end", "S2 end", "Final")


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def biased_from_row(row: dict[str, str]) -> bool:
    if row.get("biased_curriculum"):
        return parse_bool(row["biased_curriculum"])
    return "unbiased" not in row.get("run_dir", "")


def sorted_models(models: set[str]) -> list[str]:
    return sorted(models, key=model_size_b)


def ffloat(value: Any) -> float | None:
    if value in (None, ""):
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


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def model_size_b(model_name: str) -> float:
    match = MODEL_SIZE_RE.search(model_name)
    return float(match.group(1)) if match else 0.0


def parse_run_dir_name(name: str) -> tuple[str, bool, int] | None:
    match = RUN_DIR_RE.match(name)
    if not match:
        return None
    return match.group("model"), match.group("unbiased") is None, int(match.group("seed"))


def history_for_family(
    history: dict[tuple[str, bool, int], list[dict[str, float | int | None]]],
    family: str,
) -> dict[tuple[str, bool, int], list[dict[str, float | int | None]]]:
    return {
        key: rows
        for key, rows in history.items()
        if family_label(key[0]) == family
    }


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
                rows.append(
                    {
                        "global_step": int(payload.get("global_step", 0)),
                        "validate_a_rate": ffloat(validate.get("a_rate")),
                        "validate_accuracy": ffloat(validate.get("accuracy")),
                    }
                )
        history[(model_name, biased_curriculum, seed)] = sorted(
            rows, key=lambda row: int(row["global_step"])
        )
    return history


def final_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row for row in rows
        if row.get("train_step") == "final" and row.get("eval_subset") == "final_eval"
    ]


def grouped_final_metric(
    rows: list[dict[str, str]],
    *,
    fam: str,
    biased: bool,
    metric: str,
) -> dict[float, list[float]]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in final_rows(rows):
        if family_label(row["model_name"]) != fam:
            continue
        if str(row.get("biased_curriculum", "")).lower() != str(biased).lower():
            continue
        val = ffloat(row.get(metric))
        if val is not None:
            grouped[model_size_b(row["model_name"])].append(val)
    return grouped


def group_trajectory(
    rows: list[dict[str, str]],
) -> dict[tuple[str, bool, str], list[dict[str, str]]]:
    out: dict[tuple[str, bool, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        step = row.get("train_step", "")
        if step not in TRAIN_STEP_ORDER:
            continue
        if row.get("eval_subset") not in ("final_eval", "validate"):
            continue
        out[(row["model_name"], biased_from_row(row), step)].append(row)
    return out


def save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def plot_metric_vs_size_overlay(
    rows: list[dict[str, str]],
    *,
    metric: str,
    title: str,
    ylabel: str,
    out_dir: Path,
    stem: str,
    ylim: tuple[float, float] | None = None,
    hlines: list[float] | None = None,
    chance_label: str | None = None,
) -> None:
    """Final metric vs model size: both families and both curricula on one axes."""
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for fam in FAMILIES:
        for biased, ls, fill, alpha, _cur_label in CURRICULA:
            grouped = grouped_final_metric(rows, fam=fam, biased=biased, metric=metric)
            xs, ys, yerr = [], [], []
            for size in sorted(grouped):
                m, s, n = values_mean_std(grouped[size])
                if not n:
                    continue
                xs.append(size)
                ys.append(m)
                yerr.append(s)
            if not xs:
                continue
            curriculum = "biased" if biased else "unbiased"
            ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                marker=MARKERS[fam],
                color=COLORS[fam],
                mfc=COLORS[fam] if fill else "white",
                mec=COLORS[fam],
                alpha=alpha if not fill else 1.0,
                lw=2.0 if fill else 1.8,
                ls=ls,
                capsize=3,
                label=f"{fam} · {curriculum}",
                zorder=3 if fill else 2,
            )
    for y in hlines or []:
        ax.axhline(y, color=CHANCE_COLOR, ls=":", lw=0.9, alpha=0.65, label=chance_label)
        break
    ax.set_xlabel("Model size (billions)")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    style_axes(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle(
        f"{title}\n(solid/filled = biased curriculum, dashed/hollow = unbiased curriculum)",
        y=1.05,
        fontsize=11,
    )
    fig.tight_layout()
    save_fig(fig, out_dir, stem)


def plot_shortcut_susceptibility(rows: list[dict[str, str]], out_dir: Path) -> None:
    """Seed-level A-rate under biased vs unbiased training on one axes."""
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    all_sizes: list[float] = []
    for fam in FAMILIES:
        fam_rows = [row for row in final_rows(rows) if family_label(row["model_name"]) == fam]
        sizes = sorted({model_size_b(row["model_name"]) for row in fam_rows})
        all_sizes.extend(sizes)
        for biased, ls, fill, alpha, _cur_label in CURRICULA:
            grouped: dict[float, list[float]] = defaultdict(list)
            for row in fam_rows:
                if str(row.get("biased_curriculum", "")).lower() != str(biased).lower():
                    continue
                val = ffloat(row.get("predicts_A_rate"))
                if val is not None:
                    grouped[model_size_b(row["model_name"])].append(val)
            xs_mean, ys_mean, yerr = [], [], []
            for size in sizes:
                vals = grouped.get(size, [])
                if not vals:
                    continue
                jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
                ax.scatter(
                    np.full(len(vals), size) + jitter,
                    vals,
                    color=COLORS[fam],
                    marker=MARKERS[fam],
                    alpha=alpha * 0.85,
                    s=40,
                    edgecolors="none",
                    zorder=2,
                )
                m, s, n = values_mean_std(vals)
                xs_mean.append(size)
                ys_mean.append(m)
                yerr.append(s)
            if xs_mean:
                curriculum = "biased" if biased else "unbiased"
                ax.errorbar(
                    xs_mean,
                    ys_mean,
                    yerr=yerr,
                    color=COLORS[fam],
                    marker=MARKERS[fam],
                    mfc=COLORS[fam] if fill else "white",
                    mec=COLORS[fam],
                    alpha=alpha if not fill else 1.0,
                    lw=2.0 if fill else 1.8,
                    ls=ls,
                    capsize=3,
                    label=f"{fam} · {curriculum}",
                    zorder=4,
                )
    if all_sizes:
        ax.fill_between(
            [min(all_sizes) - 0.3, max(all_sizes) + 0.3],
            0.25,
            1.05,
            color=CHANCE_COLOR,
            alpha=0.06,
            zorder=0,
        )
    ax.axhline(0.25, color=CHANCE_COLOR, ls=":", lw=0.9, alpha=0.65, label="Chance (0.25)")
    ax.set_xlabel("Model size (billions)")
    ax.set_ylabel("Final A-rate on unbiased test")
    ax.set_ylim(-0.02, 1.05)
    style_axes(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle(
        "Shortcut susceptibility across model families\n"
        "Vertical gap between solid (biased) and dashed (unbiased) curves = "
        "position-bias exploitation under reward",
        y=1.06,
        fontsize=11,
    )
    fig.tight_layout()
    save_fig(fig, out_dir, "07_cross_family_shortcut_susceptibility")


def plot_trajectory_accuracy_cross_family(
    rows: list[dict[str, str]],
    out_dir: Path,
    *,
    stem: str = "10_cross_family_trajectory_accuracy",
) -> None:
    """Accuracy vs curriculum stage checkpoints, both families (like family 02_trajectory_accuracy)."""
    traj_groups = group_trajectory(rows)
    models = sorted_models({m for m, _, _ in traj_groups})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    panels = [
        (True, "Biased training curriculum"),
        (False, "Unbiased training curriculum"),
    ]
    xs = np.arange(len(TRAIN_STEP_ORDER))

    for ax, (biased, panel_title) in zip(axes, panels):
        for fam in FAMILIES:
            fam_models = [m for m in models if family_label(m) == fam]
            colors = family_model_colors(fam, len(fam_models))
            for color, model in zip(colors, fam_models):
                ys, yerr = [], []
                for step in TRAIN_STEP_ORDER:
                    step_rows = traj_groups.get((model, biased, step), [])
                    vals = [ffloat(r.get("accuracy")) for r in step_rows]
                    m, s, _ = values_mean_std(vals)
                    ys.append(m)
                    yerr.append(s)
                if all(np.isnan(y) for y in ys):
                    continue
                ax.errorbar(
                    xs,
                    ys,
                    yerr=yerr,
                    marker=MARKERS[fam],
                    mfc=color,
                    mec=color,
                    capsize=3,
                    label=f"{fam} {display_size(model_size_b(model))}",
                    color=color,
                    lw=1.8,
                    ls=FAMILY_LINE_STYLES[fam],
                )
        ax.set_xticks(xs)
        ax.set_xticklabels(list(CHECKPOINT_LABELS), rotation=15)
        ax.set_title(panel_title, fontweight="semibold")
        ax.set_xlabel("Training checkpoint")
        ax.set_ylim(0.2, 1.05)
        ax.axhline(0.25, color=CHANCE_COLOR, ls="--", lw=0.8, alpha=0.5)
        ax.axhline(0.48, color=REFERENCE_GREEN, ls=":", lw=0.8, alpha=0.6)
        style_axes(ax)

    axes[0].set_ylabel("Accuracy")
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=7,
        )
    fig.suptitle(
        "Accuracy vs training stage (validate n=64; final n=135)",
        y=1.03,
    )
    fig.tight_layout()
    save_fig(fig, out_dir, stem)


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


def plot_validate_accuracy_over_step(
    histories: dict[str, dict[tuple[str, bool, int], list[dict[str, float | int | None]]]],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    panels = [(True, "Biased curriculum"), (False, "Unbiased curriculum")]
    for ax, (biased, title) in zip(axes, panels):
        for fam in FAMILIES:
            history = histories[fam]
            models = sorted({m for m, _, _ in history}, key=model_size_b)
            colors = family_model_colors(fam, len(models))
            for color, model in zip(colors, models):
                xs, ys, yerr = history_series_by_step(
                    history, model=model, biased=biased, field="validate_accuracy"
                )
                if len(xs) == 0:
                    continue
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    lw=1.8,
                    ls=FAMILY_LINE_STYLES[fam],
                    marker=MARKERS[fam],
                    ms=3,
                    label=f"{fam} {display_size(model_size_b(model))}",
                )
                ax.fill_between(xs, ys - yerr, ys + yerr, color=color, alpha=0.12, linewidth=0)
        ax.axhline(0.48, color=REFERENCE_GREEN, ls=":", lw=0.8, alpha=0.55)
        ax.set_title(title, fontweight="semibold")
        ax.set_xlabel("Global optimizer step")
        ax.set_ylim(-0.02, 1.05)
        style_axes(ax)
    axes[0].set_ylabel("Unbiased-validate accuracy")
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=7,
        )
    fig.suptitle(
        "Unbiased-validate accuracy over training (mean ± stdev over seeds)",
        y=1.03,
    )
    fig.tight_layout()
    save_fig(fig, out_dir, "10b_cross_family_validate_accuracy_over_step")


def plot_option_distribution_appendix(rows: list[dict[str, str]], out_dir: Path) -> None:
    """Stacked answer-letter distribution at final eval, biased vs unbiased panels."""
    letters = ("pct_A", "pct_B", "pct_C", "pct_D")
    letter_colors = ["#c44e52", "#4c72b0", "#55a868", "#8172b2"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panels = [
        ("Qwen2.5", True),
        ("Qwen2.5", False),
        ("Llama 3.x", True),
        ("Llama 3.x", False),
    ]
    for ax, (fam, biased) in zip(axes.flatten(), panels):
        fam_rows = [
            row for row in final_rows(rows)
            if family_label(row["model_name"]) == fam
            and str(row.get("biased_curriculum", "")).lower() == str(biased).lower()
        ]
        models = sorted({row["model_name"] for row in fam_rows}, key=model_size_b)
        x = np.arange(len(models))
        bottom = np.zeros(len(models))
        for letter, color in zip(letters, letter_colors):
            vals = []
            for model in models:
                seed_vals = [
                    ffloat(row.get(letter))
                    for row in fam_rows
                    if row["model_name"] == model and ffloat(row.get(letter)) is not None
                ]
                m, _, n = values_mean_std(seed_vals)
                vals.append(0.0 if not n or np.isnan(m) else m)
            ax.bar(x, vals, bottom=bottom, label=letter.replace("pct_", ""), color=color)
            bottom += np.array(vals)
        ax.set_xticks(x)
        ax.set_xticklabels([display_size(model_size_b(m)) for m in models], rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Option share")
        ax.set_title(
            f"{fam} · {'biased' if biased else 'unbiased'} curriculum",
            fontweight="semibold",
        )
        ax.axhline(0.25, color="gray", ls=":", lw=0.8, alpha=0.5)
    axes[0, 0].legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        title="Answer letter",
    )
    fig.suptitle(
        "Final answer-letter distribution on unbiased test (mean over seeds)",
        y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, out_dir, "11_cross_family_option_distribution")


def write_summary(rows: list[dict[str, str]], out_dir: Path) -> None:
    lines = [
        "# Cross-family comparison summary\n",
        "Numeric decoupling is the primary benchmark metric; judge decoupling and judge reasoning are companion final-snapshot checks.\n",
        "| Family | Model | Biased train | Acc | A-rate | Dec (num) | Dec (judge) | Judge reasoning OK |",
        "|--------|-------|--------------|-----|--------|-----------|-------------|---------------------|",
    ]
    for fam in FAMILIES:
        fam_rows = [row for row in final_rows(rows) if family_label(row["model_name"]) == fam]
        for model_name in sorted({row["model_name"] for row in fam_rows}, key=model_size_b):
            for biased in (False, True):
                chunk = [
                    row for row in fam_rows
                    if row["model_name"] == model_name
                    and (str(row.get("biased_curriculum", "")).lower() == str(biased).lower())
                ]
                if not chunk:
                    continue
                acc, acc_s, acc_n = values_mean_std([ffloat(r.get("accuracy")) for r in chunk])
                a_rate, a_s, a_n = values_mean_std([ffloat(r.get("predicts_A_rate")) for r in chunk])
                dn, dn_s, dn_n = values_mean_std([ffloat(r.get("decoupling_rate")) for r in chunk])
                dj, dj_s, dj_n = values_mean_std([ffloat(r.get("decoupling_rate_judge")) for r in chunk])
                jr, jr_s, jr_n = values_mean_std([ffloat(r.get("reasoning_correct_judge_rate")) for r in chunk])
                lines.append(
                    f"| {fam} | {display_size(model_size_b(model_name))} | "
                    f"{'yes' if biased else 'no'} | "
                    f"{('—' if not acc_n else f'{acc:.3f}±{acc_s:.3f}')} | "
                    f"{('—' if not a_n else f'{a_rate:.3f}±{a_s:.3f}')} | "
                    f"{('—' if not dn_n else f'{dn:.3f}±{dn_s:.3f}')} | "
                    f"{('—' if not dj_n else f'{dj:.3f}±{dj_s:.3f}')} | "
                    f"{('—' if not jr_n else f'{jr:.3f}±{jr_s:.3f}')} |"
                )
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qwen-aggregates",
        type=Path,
        default=Path("benchmark_metrics/families/qwen_2.5_family_runs_v1_only/benchmark_aggregates.csv"),
    )
    parser.add_argument(
        "--llama-aggregates",
        type=Path,
        default=Path("benchmark_metrics/families/llama_3.x_family_runs_v1_only/benchmark_aggregates.csv"),
    )
    parser.add_argument(
        "--qwen-runs-root",
        type=Path,
        default=Path("qwen_2.5_family_runs_v1_only"),
    )
    parser.add_argument(
        "--llama-runs-root",
        type=Path,
        default=Path("llama_3.x_family_runs_v1_only"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_metrics/combined/cross_family_figures"),
    )
    args = parser.parse_args()

    setup_style()
    rows = load_csv_rows(args.qwen_aggregates) + load_csv_rows(args.llama_aggregates)
    histories = {
        fam: history_for_family(load_history(root), fam)
        for fam, root in (
            ("Qwen2.5", args.qwen_runs_root),
            ("Llama 3.x", args.llama_runs_root),
        )
    }
    out_dir = args.output_dir
    appendix_dir = out_dir / "appendix"
    out_dir.mkdir(parents=True, exist_ok=True)

    # final_plots.md items 1–3, 3b: biased/unbiased overlay per family
    plot_metric_vs_size_overlay(
        rows,
        metric="accuracy",
        title="Cross-family final unbiased-test accuracy vs size",
        ylabel="Accuracy",
        out_dir=out_dir,
        stem="01_cross_family_accuracy_vs_size",
        ylim=(0, 1.05),
        hlines=[0.48],
        chance_label="Chance (~0.48)",
    )
    plot_metric_vs_size_overlay(
        rows,
        metric="predicts_A_rate",
        title="Cross-family final A-rate vs size",
        ylabel="A-rate",
        out_dir=out_dir,
        stem="02_cross_family_a_rate_vs_size",
        ylim=(0, 1.05),
        hlines=[0.25],
        chance_label="Chance (0.25)",
    )
    plot_metric_vs_size_overlay(
        rows,
        metric="decoupling_rate",
        title="Cross-family numeric decoupling vs size",
        ylabel="Numeric decoupling rate",
        out_dir=out_dir,
        stem="03_cross_family_numeric_decoupling_vs_size",
        ylim=(0, 0.75),
    )
    plot_metric_vs_size_overlay(
        rows,
        metric="decoupling_rate_judge",
        title="Cross-family judge decoupling vs size (companion)",
        ylabel="Judge decoupling rate",
        out_dir=out_dir,
        stem="03b_cross_family_judge_decoupling_vs_size",
        ylim=(0, 0.75),
    )

    # final_plots.md item 7: shortcut susceptibility with seed-level detail
    plot_shortcut_susceptibility(rows, out_dir)

    # final_plots.md item 10: stage-checkpoint trajectory accuracy (combined families)
    plot_trajectory_accuracy_cross_family(rows, out_dir)

    # Dense training-step accuracy (appendix companion)
    plot_validate_accuracy_over_step(histories, appendix_dir)

    # final_plots.md item 11: option distribution in appendix
    plot_option_distribution_appendix(rows, appendix_dir)

    write_summary(rows, out_dir)

    n_main = len(list(out_dir.glob("*.png")))
    n_app = len(list(appendix_dir.glob("*.png")))
    print(
        f"Wrote {n_main} main figures + SUMMARY.md -> {out_dir}\n"
        f"Wrote {n_app} appendix figures -> {appendix_dir}"
    )


if __name__ == "__main__":
    main()
