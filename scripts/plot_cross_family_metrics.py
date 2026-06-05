#!/usr/bin/env python3
"""Plot cross-family benchmark comparisons from family aggregate CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

RUN_DIR_RE = re.compile(
    r"^condition_(?P<condition>\d+)_(?:(?P<unbiased>unbiased)_)?"
    r"(?P<model>[\w\.-]+)_seed(?P<seed>\d+)_beta(?P<beta>[\w\.]+)$"
)
MODEL_SIZE_RE = re.compile(r"([\d.]+)b", re.I)
COLORS = {"Qwen2.5": "#4c72b0", "Llama 3.x": "#dd8452"}
MARKERS = {"Qwen2.5": "o", "Llama 3.x": "s"}


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


def family_label(model_name: str) -> str:
    if model_name.startswith("qwen2.5-"):
        return "Qwen2.5"
    if model_name.startswith("llama"):
        return "Llama 3.x"
    return model_name.split("-")[0]


def display_size(size_b: float) -> str:
    return f"{size_b:g}B"


def parse_run_dir_name(name: str) -> tuple[str, bool, int] | None:
    match = RUN_DIR_RE.match(name)
    if not match:
        return None
    return match.group("model"), match.group("unbiased") is None, int(match.group("seed"))


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
                    }
                )
        history[(model_name, biased_curriculum, seed)] = sorted(
            rows, key=lambda row: int(row["global_step"])
        )
    return history


def collapse_steps_by_seed(
    history: dict[tuple[str, bool, int], list[dict[str, float | int | None]]],
    *,
    threshold: float,
    consecutive: int,
) -> dict[tuple[str, bool], list[float]]:
    out: dict[tuple[str, bool], list[float]] = defaultdict(list)
    for (model, biased, _seed), rows in history.items():
        streak = 0
        streak_start = None
        collapse_step = None
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


def plot_metric_vs_size(
    rows: list[dict[str, str]],
    *,
    metric: str,
    title: str,
    ylabel: str,
    out_dir: Path,
    stem: str,
    ylim: tuple[float, float] | None = None,
    hlines: list[float] | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    panels = [(True, "Biased curriculum"), (False, "Unbiased curriculum")]
    for ax, (biased, panel_title) in zip(axes, panels):
        for fam in ("Qwen2.5", "Llama 3.x"):
            fam_rows = [
                row for row in rows
                if family_label(row["model_name"]) == fam
                and str(row.get("biased_curriculum", "")).lower() == str(biased).lower()
                and row.get("train_step") == "final"
                and row.get("eval_subset") == "final_eval"
            ]
            if not fam_rows:
                continue
            grouped: dict[float, list[float | None]] = defaultdict(list)
            for row in fam_rows:
                grouped[model_size_b(row["model_name"])].append(ffloat(row.get(metric)))
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
            ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                marker=MARKERS[fam],
                color=COLORS[fam],
                lw=1.8,
                capsize=3,
                label=fam,
            )
        for y in hlines or []:
            ax.axhline(y, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.set_title(panel_title)
        ax.set_xlabel("Model size (billions)")
        if ylim:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, stem)


def plot_hacking_gap(rows: list[dict[str, str]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for fam in ("Qwen2.5", "Llama 3.x"):
        fam_rows = [
            row for row in rows
            if family_label(row["model_name"]) == fam
            and row.get("train_step") == "final"
            and row.get("eval_subset") == "final_eval"
        ]
        by_model_and_bias: dict[tuple[str, bool], list[float | None]] = defaultdict(list)
        for row in fam_rows:
            biased = str(row.get("biased_curriculum", "")).lower() == "true"
            by_model_and_bias[(row["model_name"], biased)].append(
                ffloat(row.get("predicts_A_rate"))
            )
        xs, ys, yerr = [], [], []
        for model_name in sorted({row["model_name"] for row in fam_rows}, key=model_size_b):
            b_mean, b_std, b_n = values_mean_std(by_model_and_bias[(model_name, True)])
            u_mean, u_std, u_n = values_mean_std(by_model_and_bias[(model_name, False)])
            if not b_n or not u_n:
                continue
            xs.append(model_size_b(model_name))
            ys.append(b_mean - u_mean)
            yerr.append(((b_std / max(b_n, 1) ** 0.5) ** 2 + (u_std / max(u_n, 1) ** 0.5) ** 2) ** 0.5)
        if not xs:
            continue
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            marker=MARKERS[fam],
            color=COLORS[fam],
            lw=1.8,
            capsize=3,
            label=fam,
        )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("Model size (billions)")
    ax.set_ylabel("Biased minus unbiased final A-rate")
    ax.set_title("Cross-family hacking gap at final eval")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, out_dir, "04_cross_family_hacking_gap_vs_size")


def plot_sustained_collapse(
    histories: dict[str, dict[tuple[str, bool, int], list[dict[str, float | int | None]]]],
    out_dir: Path,
) -> None:
    threshold = 0.90
    consecutive = 2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    panels = [(True, "Biased curriculum"), (False, "Unbiased curriculum")]
    for ax, (biased, panel_title) in zip(axes, panels):
        for fam, history in histories.items():
            collapse = collapse_steps_by_seed(
                history, threshold=threshold, consecutive=consecutive
            )
            xs, ys, yerr = [], [], []
            models = sorted({m for m, _, _ in history}, key=model_size_b)
            for model_name in models:
                vals = collapse.get((model_name, biased), [])
                m, s, n = values_mean_std(vals)
                if not n:
                    continue
                xs.append(model_size_b(model_name))
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
                lw=1.8,
                capsize=3,
                label=fam,
            )
        ax.set_title(panel_title)
        ax.set_xlabel("Model size (billions)")
    axes[0].set_ylabel("First sustained collapse step")
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle(
        "Cross-family sustained collapse (A-rate >= 0.90 for 2 evals)",
        y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, out_dir, "05_cross_family_sustained_collapse_A90")


def write_summary(rows: list[dict[str, str]], out_dir: Path) -> None:
    lines = [
        "# Cross-family comparison summary\n",
        "| Family | Model | Biased train | Acc | A-rate | Dec (judge) |",
        "|--------|-------|--------------|-----|--------|-------------|",
    ]
    final_rows = [
        row for row in rows
        if row.get("train_step") == "final" and row.get("eval_subset") == "final_eval"
    ]
    for fam in ("Qwen2.5", "Llama 3.x"):
        fam_rows = [row for row in final_rows if family_label(row["model_name"]) == fam]
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
                dj, dj_s, dj_n = values_mean_std([ffloat(r.get("decoupling_rate_judge")) for r in chunk])
                lines.append(
                    f"| {fam} | {display_size(model_size_b(model_name))} | "
                    f"{'yes' if biased else 'no'} | "
                    f"{('—' if not acc_n else f'{acc:.3f}±{acc_s:.3f}')} | "
                    f"{('—' if not a_n else f'{a_rate:.3f}±{a_s:.3f}')} | "
                    f"{('—' if not dj_n else f'{dj:.3f}±{dj_s:.3f}')} |"
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
        "Qwen2.5": load_history(args.qwen_runs_root),
        "Llama 3.x": load_history(args.llama_runs_root),
    }

    plot_metric_vs_size(
        rows,
        metric="accuracy",
        title="Cross-family final unbiased-test accuracy vs size",
        ylabel="Accuracy",
        out_dir=args.output_dir,
        stem="01_cross_family_accuracy_vs_size",
        ylim=(0, 1.05),
        hlines=[0.48],
    )
    plot_metric_vs_size(
        rows,
        metric="predicts_A_rate",
        title="Cross-family final A-rate vs size",
        ylabel="A-rate",
        out_dir=args.output_dir,
        stem="02_cross_family_a_rate_vs_size",
        ylim=(0, 1.05),
        hlines=[0.25],
    )
    plot_metric_vs_size(
        rows,
        metric="decoupling_rate_judge",
        title="Cross-family judge decoupling vs size",
        ylabel="Judge decoupling rate",
        out_dir=args.output_dir,
        stem="03_cross_family_judge_decoupling_vs_size",
        ylim=(0, 0.75),
        hlines=[0.25],
    )
    plot_hacking_gap(rows, args.output_dir)
    plot_sustained_collapse(histories, args.output_dir)
    write_summary(rows, args.output_dir)

    n_png = len(list(args.output_dir.glob("*.png")))
    print(f"Wrote {n_png} figures + SUMMARY.md -> {args.output_dir}")


if __name__ == "__main__":
    main()
