#!/usr/bin/env python3
"""Cross-family recovery comparison plots.

Mirrors the cross-family final-metric vs size figures, but for the recovery
phase: pre-recovery (hacked checkpoint) vs post-recovery (final eval) overlaid
per family. Outputs live under benchmark_metrics/combined/cross_family_recovery_figures/.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_recovery_family import (  # noqa: E402
    RECOVERY_THRESHOLDS,
    display_model_name,
    ffloat,
    history_series_by_step,
    load_final_after_recovery,
    load_recovery_history,
    model_family,
    model_size_b,
    plot_hysteresis,
    plot_post_recovery_metrics,
    plot_post_recovery_numeric_vs_judge_decoupling,
    plot_post_recovery_option_distribution,
    plot_pre_vs_post_a_rate,
    plot_recovery_metric,
    plot_recovery_step_distributions,
    plot_recovery_threshold_sensitivity,
    plot_recovery_thresholds,
    sorted_models,
    values_mean_std,
)
from plot_theme import (  # noqa: E402
    CHANCE_COLOR,
    COLORS,
    CURRICULA,
    FAMILIES,
    MARKERS,
    REFERENCE_GREEN,
    display_size,
    family_label,
    family_model_colors,
    normalize_family,
    setup_style,
    style_axes,
)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def filter_rows_by_model_pattern(
    rows: list[dict[str, str]],
    pattern: str | None,
) -> list[dict[str, str]]:
    if not pattern:
        return rows
    compiled = re.compile(pattern)
    return [row for row in rows if compiled.search(row.get("model_name", ""))]


def filter_recovery_history_by_model_pattern(
    history: dict[tuple[str, int], list[dict[str, Any]]],
    pattern: str | None,
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    if not pattern:
        return history
    compiled = re.compile(pattern)
    return {key: rows for key, rows in history.items() if compiled.search(key[0])}


def filter_final_after_by_model_pattern(
    final_after: dict[tuple[str, int], dict[str, float | None | str]],
    pattern: str | None,
) -> dict[tuple[str, int], dict[str, float | None | str]]:
    if not pattern:
        return final_after
    compiled = re.compile(pattern)
    return {key: payload for key, payload in final_after.items() if compiled.search(key[0])}


def save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def pre_resume_values(
    history: dict[tuple[str, int], list[dict[str, Any]]],
    *,
    model: str,
    field: str,
) -> list[float]:
    vals: list[float] = []
    for (m, _seed), rows in history.items():
        if m != model:
            continue
        pre = next((r for r in rows if r.get("source") == "pre_resume"), None)
        if pre is None:
            continue
        v = ffloat(pre.get(field))
        if v is not None:
            vals.append(v)
    return vals


def post_final_values(
    final_after: dict[tuple[str, int], dict[str, float | None | str]],
    *,
    model: str,
    field: str,
) -> list[float]:
    vals: list[float] = []
    for (m, _seed), payload in final_after.items():
        if m != model:
            continue
        v = ffloat(payload.get(field))
        if v is not None:
            vals.append(v)
    return vals


def hacked_final_values(
    hacked_rows: list[dict[str, str]],
    *,
    model: str,
    field: str,
) -> list[float]:
    vals: list[float] = []
    for row in hacked_rows:
        if row.get("model_name") != model:
            continue
        if str(row.get("biased_curriculum", "")).lower() != "true":
            continue
        if row.get("train_step") != "final":
            continue
        if row.get("eval_subset") != "final_eval":
            continue
        v = ffloat(row.get(field))
        if v is not None:
            vals.append(v)
    return vals


def plot_pre_post_metric_vs_size(
    history: dict[tuple[str, int], list[dict[str, Any]]],
    final_after: dict[tuple[str, int], dict[str, float | None | str]],
    hacked_rows: list[dict[str, str]],
    *,
    pre_field_hist: str | None,
    pre_field_hacked: str | None,
    post_field: str,
    title: str,
    ylabel: str,
    out_dir: Path,
    stem: str,
    ylim: tuple[float, float] | None = None,
    hlines: list[float] | None = None,
) -> None:
    """Pre-recovery vs post-recovery metric vs model size on one axes."""
    models = sorted_models({m for m, _ in history})
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    phases = [
        ("pre", "-", True, 1.0, "pre-recovery"),
        ("post", "--", False, 0.55, "post-recovery"),
    ]
    for fam in FAMILIES:
        fam_models = [m for m in models if family_label(m) == fam]
        for phase, ls, fill, alpha, phase_label in phases:
            xs, ys, yerr = [], [], []
            for model in fam_models:
                if phase == "pre":
                    if pre_field_hist is not None:
                        vals = pre_resume_values(history, model=model, field=pre_field_hist)
                    elif pre_field_hacked is not None:
                        vals = hacked_final_values(
                            hacked_rows, model=model, field=pre_field_hacked
                        )
                    else:
                        vals = []
                else:
                    vals = post_final_values(final_after, model=model, field=post_field)
                m, s, n = values_mean_std(vals)
                if not n:
                    continue
                xs.append(model_size_b(model))
                ys.append(m)
                yerr.append(s)
            if not xs:
                continue
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
                label=f"{fam} · {phase_label}",
                zorder=3 if fill else 2,
            )
    for y in hlines or []:
        ax.axhline(y, color=CHANCE_COLOR, ls=":", lw=0.8, alpha=0.65)
        break
    ax.set_xlabel("Model size (billions)")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    style_axes(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle(
        f"{title}\n(solid/filled = pre-recovery, dashed/hollow = post-recovery)",
        y=1.05,
    )
    fig.tight_layout()
    save_fig(fig, out_dir, stem)


def plot_pre_vs_post_accuracy(
    history: dict[tuple[str, int], list[dict[str, Any]]],
    final_after: dict[tuple[str, int], dict[str, float | None | str]],
    out_dir: Path,
    *,
    stem: str = "05_pre_vs_post_accuracy",
) -> None:
    models = sorted_models({m for m, _ in history})
    pre_vals, pre_err, post_vals, post_err = [], [], [], []
    for model in models:
        pre_list = pre_resume_values(history, model=model, field="validate_accuracy")
        post_list = post_final_values(final_after, model=model, field="accuracy")
        mp, sp, _ = values_mean_std(pre_list)
        mq, sq, _ = values_mean_std(post_list)
        pre_vals.append(mp)
        pre_err.append(sp)
        post_vals.append(mq)
        post_err.append(sq)

    fig, ax = plt.subplots(figsize=(max(8.5, 0.9 * len(models) + 4), 4.8))
    x = np.arange(len(models))
    width = 0.4
    ax.bar(
        x - width / 2,
        pre_vals,
        width,
        yerr=pre_err,
        capsize=3,
        color=COLORS["Llama 3.x"],
        alpha=0.85,
        label="Pre-recovery (hacked ckpt, validate n=64)",
    )
    ax.bar(
        x + width / 2,
        post_vals,
        width,
        yerr=post_err,
        capsize=3,
        color=COLORS["Qwen2.5"],
        alpha=0.85,
        label="Post-recovery (final eval, n=135)",
    )
    ax.axhline(0.48, color=CHANCE_COLOR, ls=":", lw=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([display_model_name(m) for m in models], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy on unbiased test")
    ax.set_title("Accuracy before vs after recovery (mean ± stdev over seeds)")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    save_fig(fig, out_dir, stem)


def plot_recovery_a_rate_accuracy_overlay(
    history: dict[tuple[str, int], list[dict[str, Any]]],
    out_dir: Path,
    *,
    stem: str = "01_recovery_a_rate_accuracy_trajectory_overlay",
) -> None:
    """A-rate (solid) and accuracy (dashed) trajectories per model, by family."""
    families = sorted({model_family(m) for m, _ in history})
    fig, axes = plt.subplots(1, len(families), figsize=(5.8 * len(families), 4.8), sharey=True)
    if len(families) == 1:
        axes = [axes]

    for ax, fam in zip(axes, families):
        fam_key = normalize_family(fam)
        models = sorted_models(
            {m for m, _ in history if normalize_family(model_family(m)) == fam_key}
        )
        colors = family_model_colors(fam_key, len(models))
        for color, model in zip(colors, models):
            xs, a_vals, a_err = history_series_by_step(
                history, model=model, field="validate_a_rate"
            )
            _, acc_vals, acc_err = history_series_by_step(
                history, model=model, field="validate_accuracy"
            )
            if len(xs) == 0:
                continue
            label = display_size(model_size_b(model))
            ax.plot(xs, a_vals, color=color, lw=1.8, ls="-", label=f"{label} A-rate")
            ax.fill_between(xs, a_vals - a_err, a_vals + a_err, color=color, alpha=0.12, linewidth=0)
            ax.plot(xs, acc_vals, color=color, lw=1.8, ls="--", alpha=0.9, label=f"{label} acc")
            ax.fill_between(xs, acc_vals - acc_err, acc_vals + acc_err, color=color, alpha=0.08, linewidth=0)
        for thresh in (0.25, 0.75):
            ax.axhline(thresh, color=CHANCE_COLOR, ls=":", lw=0.8, alpha=0.45)
        ax.axhline(0.48, color=REFERENCE_GREEN, ls=":", lw=0.8, alpha=0.5)
        ax.set_title(f"{fam_key} family")
        ax.set_xlabel("Recovery step (global optimizer step)")
        ax.set_ylim(-0.02, 1.05)
        style_axes(ax)
        ax.legend(loc="best", frameon=False, fontsize=6, ncol=2)

    axes[0].set_ylabel("Rate")
    fig.suptitle(
        "Recovery trajectories: A-rate (solid) and accuracy (dashed), mean ± stdev over seeds",
        y=1.03,
    )
    fig.tight_layout()
    save_fig(fig, out_dir, stem)


def plot_post_recovery_quadrant_scatter(
    final_after: dict[tuple[str, int], dict[str, float | None | str]],
    out_dir: Path,
    *,
    stem: str = "06_post_recovery_quadrant_accuracy_vs_a_rate",
) -> None:
    """Classify models by post-recovery accuracy vs A-rate."""
    fig, ax = plt.subplots(figsize=(6.8, 5.5))
    for fam, color, marker in (
        ("Qwen2.5", COLORS["Qwen2.5"], MARKERS["Qwen2.5"]),
        ("Llama 3.x", COLORS["Llama 3.x"], MARKERS["Llama 3.x"]),
        ("Gemma3", COLORS["Gemma3"], MARKERS["Gemma3"]),
    ):
        models = sorted_models(
            {m for m, _ in final_after if family_label(m) == fam}
        )
        for model in models:
            accs = post_final_values(final_after, model=model, field="accuracy")
            arates = post_final_values(final_after, model=model, field="a_rate")
            if not accs or not arates:
                continue
            jitter_x = np.linspace(-0.012, 0.012, len(accs)) if len(accs) > 1 else np.array([0.0])
            jitter_y = np.linspace(-0.012, 0.012, len(arates)) if len(arates) > 1 else np.array([0.0])
            ax.scatter(
                np.array(accs) + jitter_x,
                np.array(arates) + jitter_y,
                color=color,
                marker=marker,
                s=42,
                alpha=0.55,
                edgecolors="none",
            )
            mx, _, _ = values_mean_std(accs)
            my, _, _ = values_mean_std(arates)
            ax.scatter(mx, my, color=color, marker=marker, s=110, edgecolors="white", linewidths=0.8)
            ax.annotate(
                display_size(model_size_b(model)),
                (mx, my),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=7,
                color=color,
            )
    ax.axhline(0.25, color=CHANCE_COLOR, ls="--", lw=0.8, alpha=0.6)
    ax.axvline(0.48, color=REFERENCE_GREEN, ls=":", lw=0.8, alpha=0.6)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Post-recovery accuracy")
    ax.set_ylabel("Post-recovery A-rate")
    ax.set_title("Post-recovery outcome quadrants (dots = seeds, markers = model means)")
    fig.tight_layout()
    save_fig(fig, out_dir, stem)


def plot_additional_recovery_figures(
    history: dict[tuple[str, int], list[dict[str, Any]]],
    final_after: dict[tuple[str, int], dict[str, float | None | str]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_recovery_a_rate_accuracy_overlay(
        history, out_dir, stem="01_recovery_a_rate_accuracy_trajectory_overlay"
    )
    plot_recovery_metric(
        history,
        field="validate_a_rate",
        title="Unbiased-validate A-rate",
        ylabel="A-rate (predicts A)",
        out_dir=out_dir,
        stem="02_recovery_a_rate_vs_step",
        ylim=(-0.02, 1.05),
        thresholds=list(RECOVERY_THRESHOLDS),
    )
    plot_recovery_metric(
        history,
        field="validate_accuracy",
        title="Unbiased-validate accuracy",
        ylabel="Accuracy",
        out_dir=out_dir,
        stem="03_recovery_accuracy_vs_step",
        ylim=(-0.02, 1.05),
    )
    plot_pre_vs_post_a_rate(
        history, final_after, out_dir, stem="04_pre_vs_post_a_rate"
    )
    plot_pre_vs_post_accuracy(history, final_after, out_dir)
    plot_post_recovery_quadrant_scatter(final_after, out_dir)
    plot_hysteresis(
        history, final_after, out_dir, stem="07_hysteresis_hacked_vs_recovery"
    )
    plot_recovery_step_distributions(
        history, out_dir, stem="08_recovery_step_distributions"
    )
    plot_post_recovery_metrics(
        final_after, out_dir, stem="09_post_recovery_final_eval"
    )
    plot_recovery_metric(
        history,
        field="validate_not_a_accuracy",
        title="Unbiased-validate Not-A accuracy",
        ylabel="Not-A accuracy",
        out_dir=out_dir,
        stem="10_recovery_not_a_accuracy_vs_step",
        ylim=(-0.02, 1.05),
    )
    plot_recovery_metric(
        history,
        field="train_minus_test_a_rate",
        title="Train-test A-rate gap during recovery",
        ylabel="Train A-rate - unbiased-validate A-rate",
        out_dir=out_dir,
        stem="11_recovery_train_minus_test_a_gap_vs_step",
        thresholds=[0.0],
    )
    plot_recovery_thresholds(
        history,
        out_dir,
        consecutive=1,
        stem="12_recovery_step_thresholds",
        title="First recovery step (A-rate crosses below threshold)",
    )
    plot_recovery_thresholds(
        history,
        out_dir,
        consecutive=2,
        stem="13_sustained_recovery_step_thresholds",
        title="Sustained recovery (2 consecutive evals below threshold)",
    )
    plot_recovery_threshold_sensitivity(
        history, out_dir, stem="14_recovery_threshold_sensitivity"
    )
    plot_post_recovery_option_distribution(
        final_after, out_dir, stem="15_post_recovery_option_distribution"
    )
    plot_post_recovery_numeric_vs_judge_decoupling(
        final_after, out_dir, stem="16_post_recovery_decoupling_numeric_vs_judge"
    )


def write_summary(
    history: dict[tuple[str, int], list[dict[str, Any]]],
    final_after: dict[tuple[str, int], dict[str, float | None | str]],
    hacked_rows: list[dict[str, str]],
    out_dir: Path,
) -> None:
    models = sorted_models({m for m, _ in history})
    lines = [
        "# Cross-family recovery summary\n",
        "Pre = hacked checkpoint at recovery start (validate n=64). "
        "Post = final eval after recovery (n=135). "
        "Pre decoupling uses hacked biased final eval.\n",
        "| Family | Model | Pre acc | Post acc | Pre A-rate | Post A-rate | "
        "Pre dec (num) | Post dec (num) |",
        "|--------|-------|---------|----------|------------|-------------|"
        "---------------|----------------|",
    ]
    for fam in FAMILIES:
        for model in models:
            if family_label(model) != fam:
                continue
            pa, pa_s, pa_n = values_mean_std(
                pre_resume_values(history, model=model, field="validate_accuracy")
            )
            pp, pp_s, pp_n = values_mean_std(
                post_final_values(final_after, model=model, field="accuracy")
            )
            pr, pr_s, pr_n = values_mean_std(
                pre_resume_values(history, model=model, field="validate_a_rate")
            )
            par, par_s, par_n = values_mean_std(
                post_final_values(final_after, model=model, field="a_rate")
            )
            pd, pd_s, pd_n = values_mean_std(
                hacked_final_values(hacked_rows, model=model, field="decoupling_rate")
            )
            pdp, pdp_s, pdp_n = values_mean_std(
                post_final_values(final_after, model=model, field="decoupling_rate")
            )
            fmt = lambda m, s, n: "—" if not n else f"{m:.3f}±{s:.3f}"
            lines.append(
                f"| {fam} | {model_size_b(model):g}B | "
                f"{fmt(pa, pa_s, pa_n)} | {fmt(pp, pp_s, pp_n)} | "
                f"{fmt(pr, pr_s, pr_n)} | {fmt(par, par_s, par_n)} | "
                f"{fmt(pd, pd_s, pd_n)} | {fmt(pdp, pdp_s, pdp_n)} |"
            )
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recovery-runs-root",
        type=Path,
        default=Path("qwen_2.5_family_and_llama_3.x_family_recovery_runs_v1"),
    )
    parser.add_argument(
        "--gemma-recovery-runs-root",
        type=Path,
        default=Path("gemma_completed_runs"),
        help="Dir with condition_recovery_gemma3-* runs (merged into cross-family recovery).",
    )
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
        "--gemma-aggregates",
        type=Path,
        default=Path(
            "benchmark_metrics/families/gemma_completed_runs/benchmark_aggregates.csv"
        ),
    )
    parser.add_argument(
        "--gemma-model-pattern",
        type=str,
        default=r"gemma3-(1|4)b",
        help="Regex for Gemma models to include in cross-family recovery plots.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/prelim_test.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_metrics/combined/cross_family_recovery_figures"),
    )
    args = parser.parse_args()

    if not args.recovery_runs_root.is_dir():
        raise SystemExit(f"recovery-runs-root not found: {args.recovery_runs_root}")
    if not args.gemma_recovery_runs_root.is_dir():
        raise SystemExit(
            f"gemma-recovery-runs-root not found: {args.gemma_recovery_runs_root}"
        )

    setup_style()
    history = load_recovery_history(args.recovery_runs_root)
    history.update(
        filter_recovery_history_by_model_pattern(
            load_recovery_history(args.gemma_recovery_runs_root),
            args.gemma_model_pattern,
        )
    )
    final_after = load_final_after_recovery(args.recovery_runs_root, args.dataset)
    final_after.update(
        filter_final_after_by_model_pattern(
            load_final_after_recovery(args.gemma_recovery_runs_root, args.dataset),
            args.gemma_model_pattern,
        )
    )
    gemma_rows = filter_rows_by_model_pattern(
        load_csv_rows(args.gemma_aggregates),
        args.gemma_model_pattern,
    )
    hacked_rows = (
        load_csv_rows(args.qwen_aggregates)
        + load_csv_rows(args.llama_aggregates)
        + gemma_rows
    )

    if not history:
        raise SystemExit(
            f"No recovery runs with recovery_history.jsonl under {args.recovery_runs_root}"
        )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_pre_post_metric_vs_size(
        history,
        final_after,
        hacked_rows,
        pre_field_hist="validate_accuracy",
        pre_field_hacked=None,
        post_field="accuracy",
        title="Cross-family post-recovery accuracy vs size",
        ylabel="Accuracy",
        out_dir=out_dir,
        stem="01_cross_family_recovery_accuracy_vs_size",
        ylim=(0, 1.05),
        hlines=[0.48],
    )
    plot_pre_post_metric_vs_size(
        history,
        final_after,
        hacked_rows,
        pre_field_hist="validate_a_rate",
        pre_field_hacked=None,
        post_field="a_rate",
        title="Cross-family post-recovery A-rate vs size",
        ylabel="A-rate",
        out_dir=out_dir,
        stem="02_cross_family_recovery_a_rate_vs_size",
        ylim=(0, 1.05),
        hlines=[0.25],
    )
    plot_pre_post_metric_vs_size(
        history,
        final_after,
        hacked_rows,
        pre_field_hist=None,
        pre_field_hacked="decoupling_rate",
        post_field="decoupling_rate",
        title="Cross-family post-recovery numeric decoupling vs size",
        ylabel="Numeric decoupling rate",
        out_dir=out_dir,
        stem="03_cross_family_recovery_numeric_decoupling_vs_size",
        ylim=(0, 0.75),
        hlines=[0.25],
    )
    write_summary(history, final_after, hacked_rows, out_dir)

    additional_dir = out_dir / "additional"
    plot_additional_recovery_figures(history, final_after, additional_dir)

    n_main = len(list(out_dir.glob("*.png")))
    n_add = len(list(additional_dir.glob("*.png")))
    print(
        f"Wrote {n_main} main figures + SUMMARY.md -> {out_dir}\n"
        f"Wrote {n_add} additional figures -> {additional_dir}"
    )


if __name__ == "__main__":
    main()
