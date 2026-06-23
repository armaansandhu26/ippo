#!/usr/bin/env python3
"""Generate paper-ready figures for the position-confounded reward paper."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_figures"

QWEN_AGG = ROOT / "benchmark_metrics/families/qwen_2.5_family_runs_v1_only/benchmark_aggregates.csv"
LLAMA_AGG = ROOT / "benchmark_metrics/families/llama_3.x_family_runs_v1_only/benchmark_aggregates.csv"
GEMMA_AGG = ROOT / "benchmark_metrics/families/gemma_completed_runs/benchmark_aggregates.csv"
HISTORY_ROOT = ROOT / "qwen_2.5_family_and_llama_3.x_family_runs_v1"
HISTORY_ROOTS_ALL = [
    ROOT / "qwen_2.5_family_runs_v1_only",
    ROOT / "qwen_14b_missing_runs_with_recovery",
    ROOT / "llama_3.x_family_runs_v1_only",
    ROOT / "gemma_completed_runs",
]
MMLU_JSON = ROOT / "mmlu_eval_summary_all.json"
RECOVERY_CSV = ROOT / "benchmark_metrics/families/qwen_2.5_family_and_llama_3.x_family_recovery_runs_v1/figures/recovery_aggregates.csv"
RECOVERY_CSVS = [
    RECOVERY_CSV,
    ROOT / "benchmark_metrics/families/gemma_completed_runs_recovery/figures/recovery_aggregates.csv",
]


COLORS = {
    "biased": "#B64F52",
    "unbiased": "#4C72B0",
    "recovered": "#55A868",
    "neutral": "#3E3E3E",
    "light_neutral": "#F4F4F2",
    "grid": "#D8D8D8",
    "chance": "#8A8A8A",
    "accent": "#C99335",
}

MODEL_COLORS = {
    "qwen2.5-0.5b": "#A6CEE3",
    "qwen2.5-1.5b": "#6BAED6",
    "qwen2.5-3b": "#3182BD",
    "qwen2.5-7b": "#08519C",
    "qwen2.5-14b": "#08306B",
    "llama3.2-1b": "#FDBE85",
    "llama3.2-3b": "#E6550D",
    "llama3.1-8b": "#A63603",
    "gemma3-1b": "#9E9AC8",
    "gemma3-4b": "#6A51A3",
}

FAMILY_COLORS = {
    "Qwen2.5": "#4C72B0",
    "Llama 3.x": "#DD8452",
    "Gemma3": "#8172B3",
}

FAMILY_MARKERS = {
    "Qwen2.5": "o",
    "Llama 3.x": "s",
    "Gemma3": "D",
}

MODEL_ORDER = [
    "qwen2.5-0.5b",
    "qwen2.5-1.5b",
    "qwen2.5-3b",
    "qwen2.5-7b",
    "qwen2.5-14b",
    "llama3.2-1b",
    "llama3.2-3b",
    "llama3.1-8b",
]

SHORTCUT_MODEL_ORDER = MODEL_ORDER + [
    "gemma3-1b",
    "gemma3-4b",
    "gemma3-12b",
]

DECOUPLING_MODEL_ORDER = MODEL_ORDER + [
    "gemma3-1b",
    "gemma3-4b",
]

MMLU_ORDER = [
    "qwen2.5-1.5b",
    "qwen2.5-3b",
    "qwen2.5-7b",
    "llama3.1-8b",
    "llama3.2-3b",
]

MMLU_ALL_ORDER = [
    "qwen2.5-1.5b",
    "qwen2.5-3b",
    "qwen2.5-7b",
    "qwen2.5-14b",
    "llama3.2-1b",
    "llama3.2-3b",
    "llama3.1-8b",
]

RECOVERY_FOCUS = [
    "qwen2.5-0.5b",
    "qwen2.5-1.5b",
    "qwen2.5-3b",
    "llama3.2-1b",
    "llama3.2-3b",
    "llama3.1-8b",
]

RECOVERY_ALL_ORDER = [
    "qwen2.5-0.5b",
    "qwen2.5-1.5b",
    "qwen2.5-3b",
    "qwen2.5-7b",
    "qwen2.5-14b",
    "llama3.2-1b",
    "llama3.2-3b",
    "llama3.1-8b",
    "gemma3-1b",
    "gemma3-4b",
]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.6,
            "grid.alpha": 0.65,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ensure_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pretty_model(model: str) -> str:
    model = str(model)
    if model.startswith("qwen2.5-"):
        return "Qwen2.5 " + model.split("-")[-1].upper()
    if model.startswith("llama3.2-"):
        return "Llama3.2 " + model.split("-")[-1].upper()
    if model.startswith("llama3.1-"):
        return "Llama3.1 " + model.split("-")[-1].upper()
    if model.startswith("gemma3-"):
        return "Gemma3 " + model.split("-")[-1].upper()
    return model


def parse_model_from_run_dir(run_dir: str) -> str:
    name = run_dir
    name = name.replace("condition_0_unbiased_", "")
    name = name.replace("condition_0_", "")
    name = name.replace("condition_recovery_", "")
    return re.sub(r"_seed\d+_beta0p0.*$", "", name)


def load_final_aggregates(include_gemma: bool = False) -> pd.DataFrame:
    paths = [QWEN_AGG, LLAMA_AGG]
    if include_gemma:
        paths.append(GEMMA_AGG)
    frames = [pd.read_csv(path) for path in paths]
    df = pd.concat(frames, ignore_index=True)
    df = df[
        (df["train_step"].astype(str) == "final")
        & (df["split"] == "unbiased_test")
        & (df["eval_subset"] == "final_eval")
    ].copy()
    df["condition"] = np.where(df["biased_curriculum"].astype(bool), "Biased", "Unbiased")
    return df


def grouped_stats(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for (model, condition), g in df.groupby(["model_name", "condition"], sort=False):
        row = {"model_name": model, "condition": condition, "n": len(g)}
        for metric in metrics:
            vals = pd.to_numeric(g[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def model_size_b(model: str) -> float:
    match = re.search(r"([\d.]+)b", str(model), re.I)
    return float(match.group(1)) if match else 0.0


def model_family(model: str) -> str:
    model = str(model)
    if model.startswith("qwen2.5-"):
        return "Qwen2.5"
    if model.startswith("llama3."):
        return "Llama 3.x"
    if model.startswith("gemma3-"):
        return "Gemma3"
    return "Other"


def family_x_offset(family: str) -> float:
    return {
        "Qwen2.5": -0.08,
        "Llama 3.x": 0.0,
        "Gemma3": 0.08,
    }.get(family, 0.0)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.075,
        1.035,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def figure_schematic() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, title, body, edge, fill="#FFFFFF"):
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=1.2,
            edgecolor=edge,
            facecolor=fill,
        )
        ax.add_patch(rect)
        ax.text(x + 0.023, y + h - 0.055, title, ha="left", va="top", fontsize=8.8, fontweight="bold", color=edge)
        ax.text(x + 0.023, y + h - 0.125, body, ha="left", va="top", fontsize=7.5, color=COLORS["neutral"], linespacing=1.28)

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", lw=1.5, color=COLORS["neutral"], shrinkA=4, shrinkB=4),
        )

    box(
        0.04,
        0.58,
        0.25,
        0.28,
        "Biased training",
        "Training items are valid.\nThe gold option is\nalways A.",
        COLORS["biased"],
        "#FFF7F7",
    )
    box(
        0.04,
        0.16,
        0.25,
        0.28,
        "Correct reward",
        "Reward = 1.5 when the\nfinal letter matches gold.\nReasoning is not rewarded.",
        COLORS["accent"],
        "#FFF9EA",
    )
    box(
        0.38,
        0.36,
        0.27,
        0.32,
        "Two policies",
        "1. Solve the problem.\n2. Select option A.\n\nBoth receive\nmaximum reward.",
        COLORS["neutral"],
        COLORS["light_neutral"],
    )
    box(
        0.74,
        0.58,
        0.22,
        0.28,
        "Unbiased eval",
        "Answer positions are\nrandomized at test time.",
        COLORS["unbiased"],
        "#F3F7FF",
    )
    box(
        0.74,
        0.16,
        0.22,
        0.28,
        "Diagnostics",
        "Accuracy + A-rate\nReasoning agreement\nOOD + recovery.",
        COLORS["recovered"],
        "#F2FBF4",
    )
    arrow(0.29, 0.72, 0.38, 0.56)
    arrow(0.29, 0.30, 0.38, 0.48)
    arrow(0.65, 0.56, 0.74, 0.72)
    arrow(0.65, 0.44, 0.74, 0.30)

    ax.text(
        0.5,
        0.94,
        "A correct-but-confounded reward creates a measurement ambiguity",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.055,
        "Endpoint accuracy alone cannot tell whether optimization produced task competence, shortcut exploitation, or both.",
        ha="center",
        va="center",
        fontsize=7.8,
        color=COLORS["neutral"],
    )
    save(fig, "fig01_measurement_schematic")


def figure_construct_decomposition(include_gemma: bool = False) -> None:
    df = load_final_aggregates(include_gemma=include_gemma)
    metrics = ["accuracy", "predicts_A_rate", "decoupling_rate"]
    stats = grouped_stats(df, metrics)
    order = MODEL_ORDER.copy()
    if include_gemma:
        order += ["gemma3-1b", "gemma3-4b"]
    order = [m for m in order if m in set(stats["model_name"])]
    y = np.arange(len(order))

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 4.6), sharey=True)
    specs = [
        ("accuracy", "Accuracy", 0.25, (0, 1.02)),
        ("predicts_A_rate", "Option-A rate", 0.25, (0, 1.02)),
        ("decoupling_rate", "Numeric decoupling", None, (0, 0.45)),
    ]
    conds = [("Unbiased", COLORS["unbiased"], -0.09), ("Biased", COLORS["biased"], 0.09)]

    for ax, (metric, title, chance, xlim), letter in zip(axes, specs, ["A", "B", "C"]):
        for yi, model in zip(y, order):
            means = {}
            for condition in ["Unbiased", "Biased"]:
                row = stats[(stats["model_name"] == model) & (stats["condition"] == condition)]
                if not row.empty:
                    means[condition] = float(row.iloc[0][f"{metric}_mean"])
            if "Unbiased" in means and "Biased" in means:
                ax.plot([means["Unbiased"], means["Biased"]], [yi, yi], color="#C9C9C9", lw=1.0, zorder=1)
        for condition, color, offset in conds:
            xs, ys, xerrs = [], [], []
            for yi, model in zip(y, order):
                row = stats[(stats["model_name"] == model) & (stats["condition"] == condition)]
                if row.empty:
                    continue
                xs.append(float(row.iloc[0][f"{metric}_mean"]))
                xerrs.append(float(row.iloc[0][f"{metric}_std"]))
                ys.append(yi + offset)
            ax.errorbar(
                xs,
                ys,
                xerr=xerrs,
                fmt="o",
                markersize=4.5,
                color=color,
                ecolor=color,
                elinewidth=1,
                capsize=2,
                label=condition if ax is axes[0] else None,
                zorder=3,
            )
        if chance is not None:
            ax.axvline(chance, color=COLORS["chance"], ls=":", lw=1.2)
        ax.set_title(title)
        ax.set_xlim(*xlim)
        ax.set_xlabel("Rate")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        add_panel_label(ax, letter)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([pretty_model(m) for m in order])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.92), ncol=2)
    fig.suptitle("Biased optimization separates accuracy, answer position, and reasoning agreement", y=0.995, fontsize=10.5)
    fig.subplots_adjust(top=0.80, wspace=0.22)
    fig.text(0.01, -0.01, "Mean +/- stdev over available seeds; final in-domain position-randomized test.", fontsize=7.0, color=COLORS["neutral"])
    stem = "fig02_construct_decomposition"
    if include_gemma:
        stem += "_with_gemma_appendix"
    save(fig, stem)


def history_total_step(stage: str, step: int) -> int:
    offsets = {"stage0": 0, "stage1": 300, "stage2": 500}
    return offsets.get(stage, 0) + int(step)


def load_history() -> pd.DataFrame:
    rows = []
    for path in sorted(HISTORY_ROOT.glob("condition_*/metrics_history.jsonl")):
        run_dir = path.parent.name
        model = parse_model_from_run_dir(run_dir)
        if model not in MODEL_ORDER:
            continue
        condition = "Unbiased" if "_unbiased_" in run_dir else "Biased"
        seed_match = re.search(r"_seed(\d+)_", run_dir)
        seed = int(seed_match.group(1)) if seed_match else -1
        with path.open() as f:
            for line in f:
                item = json.loads(line)
                validate = item.get("validate") or {}
                rows.append(
                    {
                        "model_name": model,
                        "condition": condition,
                        "seed": seed,
                        "stage": item.get("stage"),
                        "stage_step": int(item.get("global_step")),
                        "total_step": history_total_step(item.get("stage"), int(item.get("global_step"))),
                        "a_rate": float(validate.get("a_rate", np.nan)),
                        "accuracy": float(validate.get("accuracy", np.nan)),
                    }
                )
    return pd.DataFrame(rows)


def figure_training_heatmap() -> None:
    df = load_history()
    steps = sorted(df["total_step"].dropna().unique())
    order = [m for m in MODEL_ORDER if m in set(df["model_name"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.5), sharey=True)
    cmap = mpl.colormaps["YlOrRd"].copy()
    cmap.set_bad("#EFEFEF")

    for ax, condition, letter in zip(axes, ["Biased", "Unbiased"], ["A", "B"]):
        sub = df[df["condition"] == condition]
        grouped = sub.groupby(["model_name", "total_step"], as_index=False)["a_rate"].mean()
        matrix = np.full((len(order), len(steps)), np.nan)
        step_index = {s: i for i, s in enumerate(steps)}
        model_index = {m: i for i, m in enumerate(order)}
        for row in grouped.itertuples():
            matrix[model_index[row.model_name], step_index[row.total_step]] = row.a_rate
        im = ax.imshow(matrix, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap=cmap)
        ax.set_title(f"{condition} curriculum")
        ax.set_xticks([step_index[s] for s in [10, 300, 500, 700] if s in step_index])
        ax.set_xticklabels([str(s) for s in [10, 300, 500, 700] if s in step_index])
        ax.set_xlabel("Curriculum step")
        for boundary in [300, 500]:
            if boundary in step_index:
                ax.axvline(step_index[boundary] + 0.5, color="white", lw=1.2)
        ax.grid(False)
        add_panel_label(ax, letter)
    axes[0].set_yticks(np.arange(len(order)))
    axes[0].set_yticklabels([pretty_model(m) for m in order])
    axes[0].set_ylabel("Model")
    fig.subplots_adjust(right=0.88, wspace=0.08)
    cax = fig.add_axes([0.90, 0.18, 0.018, 0.64])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Validation A-rate")
    fig.suptitle("The position shortcut emerges during biased optimization", y=0.98, fontsize=10.5)
    save(fig, "fig03_training_a_rate_heatmap")


def load_history_all() -> pd.DataFrame:
    rows = []
    for root in HISTORY_ROOTS_ALL:
        if not root.exists():
            continue
        for path in sorted(root.glob("condition_*/metrics_history.jsonl")):
            run_dir = path.parent.name
            if "backup" in run_dir or "old_incomplete" in run_dir:
                continue
            model = parse_model_from_run_dir(run_dir)
            if model not in SHORTCUT_MODEL_ORDER:
                continue
            condition = "Unbiased" if "_unbiased_" in run_dir else "Biased"
            seed_match = re.search(r"_seed(\d+)_", run_dir)
            seed = int(seed_match.group(1)) if seed_match else -1
            with path.open() as f:
                for line in f:
                    item = json.loads(line)
                    validate = item.get("validate") or {}
                    rows.append(
                        {
                            "model_name": model,
                            "condition": condition,
                            "seed": seed,
                            "stage": item.get("stage"),
                            "stage_step": int(item.get("global_step")),
                            "total_step": history_total_step(item.get("stage"), int(item.get("global_step"))),
                            "a_rate": float(validate.get("a_rate", np.nan)),
                        }
                    )
    return pd.DataFrame(rows)


def figure_shortcut_susceptibility() -> None:
    history = load_history_all()
    final = load_final_aggregates(include_gemma=True)
    final = final[final["condition"] == "Biased"].copy()
    final["predicts_A_rate"] = pd.to_numeric(final["predicts_A_rate"], errors="coerce")
    final_models = set(final["model_name"])
    order = [m for m in SHORTCUT_MODEL_ORDER if m in final_models and m in set(history["model_name"])]

    biased_history = history[
        (history["condition"] == "Biased")
        & (history["model_name"].isin(order))
    ].copy()
    steps = sorted(biased_history["total_step"].dropna().unique())
    step_index = {s: i for i, s in enumerate(steps)}
    model_index = {m: i for i, m in enumerate(order)}
    matrix = np.full((len(order), len(steps)), np.nan)

    grouped = biased_history.groupby(["model_name", "total_step"], as_index=False)["a_rate"].mean()
    for row in grouped.itertuples():
        matrix[model_index[row.model_name], step_index[row.total_step]] = row.a_rate

    final_stats = (
        final.groupby("model_name")["predicts_A_rate"]
        .agg(["mean", "std", "count"])
        .reindex(order)
    )
    threshold = 0.75
    collapse_steps = {}
    for model in order:
        g = grouped[grouped["model_name"] == model].sort_values("total_step")
        hit = g[g["a_rate"] >= threshold]
        collapse_steps[model] = None if hit.empty else float(hit.iloc[0]["total_step"])

    fig = plt.figure(figsize=(7.8, 4.95))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.95, 0.055, 0.92],
        wspace=0.13,
    )

    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    final_ax = fig.add_subplot(gs[0, 2], sharey=ax)
    cmap = mpl.colormaps["YlOrRd"].copy()
    cmap.set_bad("#EFEFEF")
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap=cmap)
    ax.set_title("Collapse timing", fontsize=9.5)
    ax.set_xlabel("Curriculum step")
    ax.set_ylabel("Model")
    tick_steps = [10, 100, 200, 300, 400, 500, 600, 700]
    present_ticks = [s for s in tick_steps if s in step_index]
    ax.set_xticks([step_index[s] for s in present_ticks])
    ax.set_xticklabels([str(s) for s in present_ticks], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([pretty_model(m) for m in order])
    for boundary in [300, 500]:
        if boundary in step_index:
            ax.axvline(step_index[boundary] + 0.5, color="white", lw=1.2)
    for yi, model in enumerate(order):
        step = collapse_steps[model]
        if step is None or step not in step_index:
            continue
        ax.scatter(
            step_index[step],
            yi,
            marker="o",
            s=23,
            facecolor="white",
            edgecolor=COLORS["neutral"],
            linewidth=0.8,
            zorder=3,
        )
    ax.grid(False)
    add_panel_label(ax, "A")

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Validation option-A rate", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax = final_ax
    y = np.arange(len(order))
    rng = np.random.default_rng(0)
    for yi, model in zip(y, order):
        seed_vals = final[final["model_name"] == model]["predicts_A_rate"].dropna().to_numpy()
        if len(seed_vals):
            jitter = rng.uniform(-0.08, 0.08, len(seed_vals))
            ax.scatter(
                seed_vals,
                np.full(len(seed_vals), yi) + jitter,
                color="#A9A9A9",
                s=9,
                alpha=0.75,
                linewidth=0,
                zorder=2,
            )
        mean_val = float(final_stats.loc[model, "mean"])
        std_val = float(final_stats.loc[model, "std"])
        if math.isnan(std_val):
            std_val = 0.0
        ax.errorbar(
            mean_val,
            yi,
            xerr=std_val,
            fmt="o",
            color=COLORS["biased"],
            ecolor=COLORS["biased"],
            elinewidth=1,
            capsize=2,
            markersize=4.5,
            zorder=4,
        )
    ax.axvline(0.25, color=COLORS["chance"], ls=":", lw=1.1)
    ax.axvline(threshold, color=COLORS["biased"], ls="--", lw=0.9, alpha=0.55)
    ax.set_title("Final shortcut strength", fontsize=9.5)
    ax.set_xlabel("Final option-A rate")
    ax.set_xlim(0, 1.04)
    ax.set_yticks(y)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    add_panel_label(ax, "B")

    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.16, top=0.88)
    save(fig, "fig02_shortcut_susceptibility")


def decoupling_plot_rows() -> pd.DataFrame:
    df = load_final_aggregates(include_gemma=True)
    df = df[df["model_name"].isin(DECOUPLING_MODEL_ORDER)].copy()
    df["family"] = df["model_name"].map(model_family)
    df["size_b"] = df["model_name"].map(model_size_b)
    for metric in ["decoupling_rate", "decoupling_rate_judge"]:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return df


def plot_decoupling_panel(ax: plt.Axes, df: pd.DataFrame, metric: str, ylabel: str) -> None:
    stats = (
        df.groupby(["family", "size_b", "condition"], as_index=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    for family in ["Qwen2.5", "Llama 3.x", "Gemma3"]:
        fam_stats = stats[stats["family"] == family]
        sizes = sorted(fam_stats["size_b"].unique())
        for size in sizes:
            x = float(size) + family_x_offset(family)
            biased = fam_stats[(fam_stats["size_b"] == size) & (fam_stats["condition"] == "Biased")]
            unbiased = fam_stats[(fam_stats["size_b"] == size) & (fam_stats["condition"] == "Unbiased")]
            if not biased.empty and not unbiased.empty:
                ax.plot(
                    [x, x],
                    [float(unbiased.iloc[0]["mean"]), float(biased.iloc[0]["mean"])],
                    color=FAMILY_COLORS[family],
                    lw=1.1,
                    alpha=0.20,
                    zorder=1,
                )
            for condition, filled in [("Unbiased", False), ("Biased", True)]:
                row = fam_stats[
                    (fam_stats["size_b"] == size)
                    & (fam_stats["condition"] == condition)
                ]
                if row.empty:
                    continue
                mean_val = float(row.iloc[0]["mean"])
                std_val = float(row.iloc[0]["std"])
                if math.isnan(std_val):
                    std_val = 0.0
                ax.errorbar(
                    x,
                    mean_val,
                    yerr=std_val,
                    fmt=FAMILY_MARKERS[family],
                    markersize=5.0,
                    color=FAMILY_COLORS[family],
                    markerfacecolor=FAMILY_COLORS[family] if filled else "white",
                    markeredgecolor=FAMILY_COLORS[family],
                    markeredgewidth=1.2,
                    elinewidth=0.95,
                    capsize=2,
                    zorder=3 if filled else 2,
                )
    ax.set_xlabel("Model size (billions)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.02, 0.72)
    ax.set_xlim(0, 14.8)
    ax.set_xticks([0.5, 1.5, 3, 4, 7, 8, 14])
    ax.set_xticklabels(["0.5", "1.5", "3", "4", "7", "8", "14"], rotation=25, ha="right")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", alpha=0.55)


def figure_decoupling_numeric_judge() -> None:
    from matplotlib.lines import Line2D

    df = decoupling_plot_rows()
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.45), sharey=True)
    plot_decoupling_panel(axes[0], df, "decoupling_rate", "Decoupling rate")
    plot_decoupling_panel(axes[1], df, "decoupling_rate_judge", "Decoupling rate")
    axes[0].set_title("Numeric match")
    axes[1].set_title("Judge companion")
    axes[1].set_ylabel("")
    add_panel_label(axes[0], "A")
    add_panel_label(axes[1], "B")

    family_handles = [
        Line2D(
            [0],
            [0],
            marker=FAMILY_MARKERS[family],
            color="none",
            markerfacecolor=FAMILY_COLORS[family],
            markeredgecolor=FAMILY_COLORS[family],
            markersize=4.8,
            label=family,
        )
        for family in ["Qwen2.5", "Llama 3.x", "Gemma3"]
    ]
    condition_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#666666",
            markeredgecolor="#666666",
            markersize=4.8,
            label="Biased",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="#666666",
            markersize=4.8,
            label="Unbiased",
        ),
    ]
    fig.legend(
        family_handles + condition_handles,
        [h.get_label() for h in family_handles + condition_handles],
        loc="upper center",
        bbox_to_anchor=(0.52, 1.02),
        ncol=5,
        columnspacing=0.85,
        handletextpad=0.35,
        fontsize=7.8,
    )
    fig.subplots_adjust(top=0.78, bottom=0.20, wspace=0.16)
    save(fig, "fig03_decoupling_numeric_judge")


def load_mmlu(models: list[str] | None = None) -> pd.DataFrame:
    model_set = set(models or MMLU_ORDER)
    with MMLU_JSON.open() as f:
        data = json.load(f)
    rows = []
    for item in data.values():
        model = item.get("model_slug")
        if model not in model_set:
            continue
        rows.append(
            {
                "model_name": model,
                "condition": item.get("condition").title(),
                "seed": int(item.get("seed")),
                "accuracy": float(item.get("accuracy")),
                "A_rate": float(item.get("A_rate")),
            }
        )
    return pd.DataFrame(rows)


def figure_mmlu_transfer() -> None:
    df = load_mmlu(MMLU_ORDER)
    stats = grouped_stats(df.rename(columns={"A_rate": "predicts_A_rate"}), ["accuracy", "predicts_A_rate"])
    order = [m for m in MMLU_ORDER if m in set(stats["model_name"])]
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.65), sharey=True)
    specs = [("accuracy", "MMLU-50 accuracy"), ("predicts_A_rate", "MMLU-50 option-A rate")]
    conds = [("Unbiased", COLORS["unbiased"], -0.15), ("Biased", COLORS["biased"], 0.0), ("Recovered", COLORS["recovered"], 0.15)]
    for ax, (metric, title), letter in zip(axes, specs, ["A", "B"]):
        for condition, color, offset in conds:
            xs, ys, xerrs = [], [], []
            for yi, model in zip(y, order):
                row = stats[(stats["model_name"] == model) & (stats["condition"] == condition)]
                if row.empty:
                    continue
                xs.append(float(row.iloc[0][f"{metric}_mean"]))
                xerrs.append(float(row.iloc[0][f"{metric}_std"]))
                ys.append(yi + offset)
            ax.errorbar(
                xs,
                ys,
                xerr=xerrs,
                fmt="o",
                markersize=4.5,
                color=color,
                ecolor=color,
                elinewidth=1,
                capsize=2,
                label=condition if ax is axes[0] else None,
                zorder=3,
            )
        ax.axvline(0.25, color=COLORS["chance"], ls=":", lw=1.2)
        ax.set_title(title)
        ax.set_xlabel("Rate")
        ax.set_xlim(0, 1.02)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        add_panel_label(ax, letter)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([pretty_model(m) for m in order])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.92), ncol=3)
    fig.suptitle("The learned position policy transfers to out-of-domain MMLU questions", y=0.995, fontsize=10.5)
    fig.subplots_adjust(top=0.78, wspace=0.20)
    fig.text(0.01, -0.02, "Mean +/- stdev over available seeds; 50 MMLU questions per run.", fontsize=7.0, color=COLORS["neutral"])
    save(fig, "fig04_mmlu_transfer")


def figure_mmlu_transfer_all_models() -> None:
    df = load_mmlu(MMLU_ALL_ORDER)
    stats = grouped_stats(
        df.rename(columns={"A_rate": "predicts_A_rate"}),
        ["accuracy", "predicts_A_rate"],
    )
    order = [m for m in MMLU_ALL_ORDER if m in set(stats["model_name"])]
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.25), sharey=True)
    specs = [
        ("accuracy", "MMLU-50 accuracy"),
        ("predicts_A_rate", "MMLU-50 option-A rate"),
    ]
    conds = [
        ("Unbiased", COLORS["unbiased"], -0.15),
        ("Biased", COLORS["biased"], 0.0),
        ("Recovered", COLORS["recovered"], 0.15),
    ]
    for ax, (metric, title), letter in zip(axes, specs, ["A", "B"]):
        for yi, model in zip(y, order):
            means = {}
            for condition in ["Unbiased", "Biased"]:
                row = stats[
                    (stats["model_name"] == model)
                    & (stats["condition"] == condition)
                ]
                if not row.empty:
                    means[condition] = float(row.iloc[0][f"{metric}_mean"])
            if "Unbiased" in means and "Biased" in means:
                ax.plot(
                    [means["Unbiased"], means["Biased"]],
                    [yi, yi],
                    color="#C7C7C7",
                    lw=1.0,
                    zorder=1,
                )
        for condition, color, offset in conds:
            xs, ys, xerrs = [], [], []
            for yi, model in zip(y, order):
                row = stats[
                    (stats["model_name"] == model)
                    & (stats["condition"] == condition)
                ]
                if row.empty:
                    continue
                xs.append(float(row.iloc[0][f"{metric}_mean"]))
                xerr = float(row.iloc[0][f"{metric}_std"])
                xerrs.append(0.0 if math.isnan(xerr) else xerr)
                ys.append(yi + offset)
            ax.errorbar(
                xs,
                ys,
                xerr=xerrs,
                fmt="o",
                markersize=4.3,
                color=color,
                ecolor=color,
                elinewidth=1,
                capsize=2,
                label=condition if ax is axes[0] else None,
                zorder=3,
            )
        ax.axvline(0.25, color=COLORS["chance"], ls=":", lw=1.1)
        ax.set_title(title)
        ax.set_xlabel("Rate")
        ax.set_xlim(0, 1.02)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        add_panel_label(ax, letter)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([pretty_model(m) for m in order])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.985), ncol=3)
    fig.subplots_adjust(top=0.84, bottom=0.13, wspace=0.18)
    save(fig, "fig04_mmlu_transfer_all_models")


def load_recovery() -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in RECOVERY_CSVS if path.exists()]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["model_name"].isin(RECOVERY_ALL_ORDER)].copy()
    for col in ["validate_a_rate", "predicts_A_rate", "validate_accuracy"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def figure_recovery() -> None:
    df = load_recovery()
    focus = [m for m in RECOVERY_FOCUS if m in set(df["model_name"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.15), gridspec_kw={"width_ratios": [1.25, 1.0]})

    ax = axes[0]
    traj = df[(df["source"].isin(["pre_resume", "recovery_history"])) & (df["model_name"].isin(focus))].copy()
    traj["step_num"] = pd.to_numeric(traj["global_step"], errors="coerce")
    grouped = traj.groupby(["model_name", "step_num"], as_index=False)["validate_a_rate"].mean()
    for model in focus:
        g = grouped[grouped["model_name"] == model].sort_values("step_num")
        ax.plot(
            g["step_num"],
            g["validate_a_rate"],
            lw=1.8,
            marker="o",
            markersize=2.8,
            color=MODEL_COLORS.get(model, COLORS["neutral"]),
            label=pretty_model(model),
        )
    ax.axhline(0.25, color=COLORS["chance"], ls=":", lw=1.2)
    ax.set_title("A-rate during unbiased recovery", fontsize=9.5)
    ax.set_xlabel("Recovery step")
    ax.set_ylabel("Validation A-rate")
    ax.set_xlim(-2, 202)
    ax.set_ylim(0, 1.04)
    ax.legend(loc="upper center", bbox_to_anchor=(0.52, -0.18), ncol=2)
    add_panel_label(ax, "A")

    ax = axes[1]
    pre = df[(df["source"] == "pre_resume") & (df["model_name"].isin(focus))]
    post = df[(df["source"] == "final_eval_after_recovery") & (df["model_name"].isin(focus))].copy()
    pre_stats = pre.groupby("model_name")["validate_a_rate"].agg(["mean", "std", "count"])
    post["post_rate"] = post["predicts_A_rate"].fillna(post["validate_a_rate"])
    post_stats = post.groupby("model_name")["post_rate"].agg(["mean", "std", "count"])
    y = np.arange(len(focus))
    for yi, model in zip(y, focus):
        if model not in pre_stats.index or model not in post_stats.index:
            continue
        x0 = float(pre_stats.loc[model, "mean"])
        x1 = float(post_stats.loc[model, "mean"])
        e0 = 0.0 if math.isnan(float(pre_stats.loc[model, "std"])) else float(pre_stats.loc[model, "std"])
        e1 = 0.0 if math.isnan(float(post_stats.loc[model, "std"])) else float(post_stats.loc[model, "std"])
        ax.plot([x0, x1], [yi, yi], color="#C9C9C9", lw=1.1, zorder=1)
        ax.errorbar(x0, yi - 0.08, xerr=e0, fmt="o", color=COLORS["biased"], capsize=2, markersize=4.3, label="Before recovery" if yi == 0 else None)
        ax.errorbar(x1, yi + 0.08, xerr=e1, fmt="o", color=COLORS["recovered"], capsize=2, markersize=4.3, label="After recovery" if yi == 0 else None)
    ax.axvline(0.25, color=COLORS["chance"], ls=":", lw=1.2)
    ax.set_title("Shortcut residue after recovery")
    ax.set_xlabel("A-rate")
    ax.set_xlim(0, 1.04)
    ax.set_yticks(y)
    ax.set_yticklabels([pretty_model(m) for m in focus])
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left")
    add_panel_label(ax, "B")

    fig.suptitle("Unbiased recovery removes the shortcut unevenly", y=0.995, fontsize=10.5)
    fig.subplots_adjust(top=0.82, bottom=0.28, wspace=0.48)
    save(fig, "fig05_recovery_dynamics")


def figure_recovery_combined() -> None:
    from matplotlib.lines import Line2D

    df = load_recovery()
    models = [m for m in RECOVERY_ALL_ORDER if m in set(df["model_name"])]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.8, 3.85),
        gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.25},
    )

    ax = axes[0]
    traj = df[
        (df["source"].isin(["pre_resume", "recovery_history"]))
        & (df["model_name"].isin(models))
    ].copy()
    traj["step_num"] = pd.to_numeric(traj["global_step"], errors="coerce")
    grouped = (
        traj.groupby(["model_name", "step_num"], as_index=False)["validate_a_rate"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    for model in models:
        g = grouped[grouped["model_name"] == model].sort_values("step_num")
        if g.empty:
            continue
        color = MODEL_COLORS.get(model, FAMILY_COLORS.get(model_family(model), COLORS["neutral"]))
        ax.plot(
            g["step_num"],
            g["mean"],
            lw=1.35,
            marker="o",
            markersize=2.0,
            color=color,
            alpha=0.92,
            label=pretty_model(model),
        )
    ax.axhline(0.25, color=COLORS["chance"], ls=":", lw=1.1)
    ax.axhline(0.75, color=COLORS["biased"], ls="--", lw=0.9, alpha=0.45)
    ax.set_title("A-rate during unbiased recovery")
    ax.set_xlabel("Recovery step")
    ax.set_ylabel("Validation option-A rate")
    ax.set_xlim(-2, 202)
    ax.set_ylim(-0.02, 1.04)
    ax.grid(axis="both", alpha=0.55)
    add_panel_label(ax, "A")

    ax = axes[1]
    post = df[
        (df["source"] == "final_eval_after_recovery")
        & (df["model_name"].isin(models))
    ].copy()
    rng = np.random.default_rng(3)
    for model in models:
        rows = post[post["model_name"] == model]
        if rows.empty:
            continue
        family = model_family(model)
        color = FAMILY_COLORS.get(family, COLORS["neutral"])
        marker = FAMILY_MARKERS.get(family, "o")
        acc = rows["validate_accuracy"].dropna().to_numpy()
        arate = rows["predicts_A_rate"].fillna(rows["validate_a_rate"]).dropna().to_numpy()
        n = min(len(acc), len(arate))
        if n == 0:
            continue
        jitter = rng.normal(0, 0.006, size=(n, 2))
        ax.scatter(
            acc[:n] + jitter[:, 0],
            arate[:n] + jitter[:, 1],
            s=15,
            marker=marker,
            color=color,
            alpha=0.35,
            linewidth=0,
            zorder=2,
        )
        mean_acc = float(np.mean(acc[:n]))
        mean_arate = float(np.mean(arate[:n]))
        ax.scatter(
            mean_acc,
            mean_arate,
            s=52,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        ax.annotate(
            pretty_model(model).replace("Qwen2.5 ", "").replace("Llama3.2 ", "").replace("Llama3.1 ", "").replace("Gemma3 ", ""),
            (mean_acc, mean_arate),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6.3,
            color=color,
        )
    ax.axhline(0.25, color=COLORS["chance"], ls=":", lw=1.1)
    ax.axvline(0.48, color=COLORS["chance"], ls=":", lw=1.1)
    ax.set_title("Final post-recovery outcomes", fontsize=9.5)
    ax.set_xlabel("Final accuracy")
    ax.set_ylabel("Final option-A rate")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.04)
    ax.grid(axis="both", alpha=0.55)
    add_panel_label(ax, "B")

    family_handles = [
        Line2D(
            [0],
            [0],
            marker=FAMILY_MARKERS[family],
            color="none",
            markerfacecolor=FAMILY_COLORS[family],
            markeredgecolor=FAMILY_COLORS[family],
            markersize=5,
            label=family,
        )
        for family in ["Qwen2.5", "Llama 3.x", "Gemma3"]
    ]
    ax.legend(handles=family_handles, loc="upper center", bbox_to_anchor=(0.47, -0.20), ncol=3)
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.50, -0.20),
        ncol=3,
        fontsize=6.0,
        handlelength=1.5,
        columnspacing=0.7,
    )
    fig.subplots_adjust(bottom=0.31, top=0.86)
    save(fig, "fig05_recovery_combined")


def write_readme() -> None:
    text = """# Paper figures

Generated by `scripts/plot_paper_figures.py`.

Main-paper candidates:

- `fig01_measurement_schematic`: setup and measurement ambiguity.
- `fig02_shortcut_susceptibility`: collapse timing and final shortcut strength under biased training.
- `fig02_construct_decomposition`: final in-domain accuracy, A-rate, and numeric decoupling for Qwen2.5 + Llama 3.x.
- `fig03_decoupling_numeric_judge`: numeric and judge-based reasoning-answer decoupling.
- `fig03_training_a_rate_heatmap`: shortcut emergence during biased optimization (superseded by Figure 2 for the main text).
- `fig04_mmlu_transfer_all_models`: out-of-domain MMLU-50 transfer across all included model cohorts.
- `fig04_mmlu_transfer`: out-of-domain MMLU-50 transfer subset.
- `fig05_recovery_combined`: recovery A-rate trajectories and final post-recovery outcomes.
- `fig05_recovery_dynamics`: recovery trajectories and final shortcut residue.

Appendix/supporting candidate:

- `fig02_construct_decomposition_with_gemma_appendix`: same construct-decomposition plot with available Gemma rows.

Each figure is exported as both PNG and PDF. The plots use a shared color code:
red = biased curriculum, blue = unbiased curriculum, green = recovered.
"""
    (OUT / "README.md").write_text(text)


def main() -> None:
    setup_style()
    ensure_out()
    figure_schematic()
    figure_shortcut_susceptibility()
    figure_decoupling_numeric_judge()
    figure_construct_decomposition(include_gemma=False)
    figure_construct_decomposition(include_gemma=True)
    figure_training_heatmap()
    figure_mmlu_transfer_all_models()
    figure_mmlu_transfer()
    figure_recovery_combined()
    figure_recovery()
    write_readme()


if __name__ == "__main__":
    main()
