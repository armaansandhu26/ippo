#!/usr/bin/env python3
"""Regenerate the five bundled main-paper figures."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import plot_cross_family_metrics as cross  # noqa: E402
import plot_paper_figures as paper  # noqa: E402


def generate_fig01() -> None:
    """Final accuracy and option-A scaling plot."""
    cross.setup_style()
    gemma_rows = cross.filter_rows_by_model_pattern(
        cross.load_csv_rows(
            cross.repo_path(
                "benchmark_metrics/families/gemma_completed_runs/benchmark_aggregates.csv"
            )
        ),
        r"gemma3-(1|4)b",
    )
    rows = (
        cross.load_csv_rows(
            cross.repo_path(
                "benchmark_metrics/families/qwen_2.5_family_runs_v1_only/benchmark_aggregates.csv"
            )
        )
        + cross.load_csv_rows(
            cross.repo_path(
                "benchmark_metrics/families/llama_3.x_family_runs_v1_only/benchmark_aggregates.csv"
            )
        )
        + gemma_rows
    )
    cross.plot_accuracy_a_rate_side_by_side(rows, paper.OUT)


def main() -> None:
    paper.ensure_out()

    generate_fig01()

    paper.setup_style()
    paper.figure_shortcut_susceptibility()
    paper.figure_decoupling_numeric_judge()
    paper.figure_mmlu_transfer_all_models()
    paper.figure_recovery_combined()

    print(f"Wrote bundled figures -> {paper.OUT}")


if __name__ == "__main__":
    main()

