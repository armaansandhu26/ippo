#!/usr/bin/env python3
"""Plot MMLU-50 out-of-domain transfer metrics from mmlu_eval_summary_all.json."""

from __future__ import annotations

import argparse
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
from plot_theme import REFERENCE_GREEN, setup_style, style_axes

MODEL_SIZE_RE = re.compile(r"([\d.]+)b", re.I)
CONDITIONS = ("biased", "unbiased", "recovered")
CONDITION_COLORS = {
    "biased": "#c44e52",
    "unbiased": "#4c72b0",
    "recovered": REFERENCE_GREEN,
}
CONDITION_LABELS = {
    "biased": "Biased",
    "unbiased": "Unbiased",
    "recovered": "Recovered",
}
PAPER_MODELS = (
    "qwen2.5-1.5b",
    "qwen2.5-3b",
    "qwen2.5-7b",
    "llama3.1-8b",
    "llama3.2-3b",
)
DEFAULT_EXCLUDE = ("qwen2.5-0.5b",)


def model_size_b(model_name: str) -> float:
    m = MODEL_SIZE_RE.search(model_name)
    return float(m.group(1)) if m else 0.0


def display_model_name(slug: str) -> str:
    m = re.match(r"qwen2\.5-(.+)", slug)
    if m:
        return f"Qwen2.5-{m.group(1).upper()}"
    m = re.match(r"llama(3\.\d+)-(.+)", slug)
    if m:
        return f"Llama {m.group(1)}-{m.group(2).upper()}"
    return slug


def sorted_models(models: set[str]) -> list[str]:
    def key(m: str) -> tuple[int, float, str]:
        if m.startswith("qwen"):
            fam = 0
        elif m.startswith("llama"):
            fam = 1
        else:
            fam = 2
        return fam, model_size_b(m), m

    return sorted(models, key=key)


def values_mean_std(vals: list[float]) -> tuple[float, float, int]:
    if not vals:
        return float("nan"), 0.0, 0
    if len(vals) == 1:
        return vals[0], 0.0, 1
    return mean(vals), stdev(vals), len(vals)


def load_mmlu_summary(path: Path) -> dict[str, dict[str, Any]]:
    return json.loads(path.read_text())


def group_runs(
    data: dict[str, dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in data.values():
        grouped[(row["model_slug"], row["condition"])].append(row)
    return grouped


def metric_for_group(
    rows: list[dict[str, Any]], field: str
) -> tuple[float, float, int]:
    vals = [float(r[field]) for r in rows]
    return values_mean_std(vals)


def save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def verify_against_summary_md(
    data: dict[str, dict[str, Any]], md_path: Path
) -> tuple[int, list[str]]:
    """Return (mismatch_count, messages)."""
    if not md_path.exists():
        return 0, [f"skip verify: {md_path} not found"]

    rows_md: list[tuple[str, str, int, float, str, float]] = []
    for line in md_path.read_text().splitlines():
        if not line.startswith("|") or "Condition" in line or "---" in line:
            continue
        parts = [p.strip().strip("*") for p in line.split("|")[1:-1]]
        if len(parts) < 9:
            continue
        rows_md.append(
            (
                parts[0],
                parts[1],
                int(parts[2]),
                float(parts[3]),
                parts[4],
                float(parts[5]),
            )
        )

    md_lookup = {
        (cond, model, seed): (acc, correct, a_rate)
        for cond, model, seed, acc, correct, a_rate in rows_md
    }
    json_lookup = {
        (row["condition"], row["model_slug"], row["seed"]): (
            row["accuracy"],
            f"{row['correct']}/{row['n']}",
            row["A_rate"],
        )
        for row in data.values()
    }

    messages: list[str] = []
    only_md = set(md_lookup) - set(json_lookup)
    only_json = set(json_lookup) - set(md_lookup)
    for key in sorted(only_md):
        messages.append(f"in MD only: {key} -> {md_lookup[key]}")
    for key in sorted(only_json):
        messages.append(f"in JSON only: {key} -> {json_lookup[key]}")
    for key in sorted(set(md_lookup) & set(json_lookup)):
        if md_lookup[key] != json_lookup[key]:
            messages.append(
                f"value mismatch {key}: MD={md_lookup[key]} JSON={json_lookup[key]}"
            )
    return len(messages), messages


def plot_transfer_bars(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    models: list[str],
    out_dir: Path,
    *,
    stem: str,
    title_suffix: str = "",
) -> None:
    metrics = [
        ("accuracy", "MMLU-50 accuracy"),
        ("A_rate", "Option-A rate"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(max(9.0, 1.35 * len(models) + 3.5), 4.2))
    x = np.arange(len(models))
    width = 0.24
    offsets = np.array([-width, 0.0, width])

    for ax, (field, ylabel) in zip(axes, metrics):
        for i, condition in enumerate(CONDITIONS):
            vals, errs, ns = [], [], []
            for model in models:
                m, s, n = metric_for_group(
                    grouped.get((model, condition), []), field
                )
                vals.append(m)
                errs.append(s)
                ns.append(n)
            bars = ax.bar(
                x + offsets[i],
                vals,
                width,
                yerr=errs,
                capsize=2.5,
                color=CONDITION_COLORS[condition],
                label=CONDITION_LABELS[condition],
                alpha=0.92,
            )
            for bar, n in zip(bars, ns):
                if n == 0:
                    bar.set_alpha(0.0)
                    bar.set_edgecolor("none")
                elif n < 3:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        0.02,
                        f"n={n}",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        color="#555555",
                        rotation=90,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [display_model_name(m) for m in models], rotation=28, ha="right"
        )
        ax.set_ylim(0, 1.02)
        ax.set_ylabel(ylabel)
        style_axes(ax)

    axes[0].legend(loc="upper right", frameon=False, ncol=3)
    suffix = f" ({title_suffix})" if title_suffix else ""
    fig.suptitle(
        f"MMLU-50 out-of-domain transfer{suffix}\n"
        "mean ± stdev over available seeds; n=50 questions per run",
        y=1.03,
        fontsize=11,
    )
    fig.tight_layout()
    save_fig(fig, out_dir, stem)


def write_summary_md(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    models: list[str],
    out_dir: Path,
    *,
    verify_messages: list[str],
) -> None:
    lines = [
        "# MMLU transfer figures",
        "",
        "Generated by `scripts/plot_mmlu_transfer.py`.",
        "",
        "## Verification vs MMLU_SUMMARY.md",
        "",
    ]
    if verify_messages:
        lines.extend(f"- {msg}" for msg in verify_messages)
    else:
        lines.append("- All 60 seed-level rows match `benchmark_metrics/MMLU_SUMMARY.md`.")

    lines.extend(["", "## Aggregated metrics", ""])
    lines.append(
        "| Model | Condition | Seeds | Accuracy | A-rate |"
    )
    lines.append("| ----- | --------- | ----- | -------- | ------ |")
    for model in models:
        for condition in CONDITIONS:
            rows = grouped.get((model, condition), [])
            acc_m, acc_s, acc_n = metric_for_group(rows, "accuracy")
            a_m, a_s, a_n = metric_for_group(rows, "A_rate")
            n = max(acc_n, a_n)
            if n == 0:
                continue
            acc_txt = (
                f"{acc_m:.3f}±{acc_s:.3f}" if n > 1 else f"{acc_m:.3f}"
            )
            a_txt = f"{a_m:.3f}±{a_s:.3f}" if n > 1 else f"{a_m:.3f}"
            lines.append(
                f"| {display_model_name(model)} | {condition} | {n} | "
                f"{acc_txt} | {a_txt} |"
            )

    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `01_mmlu_transfer_all_models.png` — all models except excluded (default: 0.5B)",
            "- `02_mmlu_transfer_paper_subset.png` — paper-focused subset "
            "(1.5B, 3B, 7B, Llama 3.1-8B, Llama 3.2-3B)",
            "",
        ]
    )
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("mmlu_eval_summary_all.json"),
        help="Combined MMLU summary JSON",
    )
    parser.add_argument(
        "--md",
        type=Path,
        default=Path("benchmark_metrics/MMLU_SUMMARY.md"),
        help="Markdown summary for cross-check",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmark_metrics/mmlu_figures"),
        help="Output directory for figures",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=list(DEFAULT_EXCLUDE),
        help="Model slugs to omit from the all-models figure",
    )
    args = parser.parse_args()

    setup_style()
    data = load_mmlu_summary(args.json)
    grouped = group_runs(data)

    mismatch_count, verify_messages = verify_against_summary_md(data, args.md)
    if mismatch_count:
        print(f"WARNING: {mismatch_count} verification issue(s):")
        for msg in verify_messages:
            print(f"  {msg}")
    else:
        print(f"Verified: {len(data)} JSON rows match {args.md}")

    all_models = sorted_models(
        {model for model, _ in grouped} - set(args.exclude)
    )
    paper_models = [m for m in PAPER_MODELS if m in {model for model, _ in grouped}]

    plot_transfer_bars(
        grouped,
        all_models,
        args.out_dir,
        stem="01_mmlu_transfer_all_models",
        title_suffix="all models",
    )
    plot_transfer_bars(
        grouped,
        paper_models,
        args.out_dir,
        stem="02_mmlu_transfer_paper_subset",
        title_suffix="paper subset",
    )
    write_summary_md(
        grouped,
        sorted_models({model for model, _ in grouped}),
        args.out_dir,
        verify_messages=verify_messages,
    )

    print(f"Wrote figures + SUMMARY.md -> {args.out_dir}")


if __name__ == "__main__":
    main()
