import argparse
import os
import sys
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.visualize_results import parse_metrics_file, plot_summary


def _draw_box(ax, x: float, y: float, w: float, h: float, text: str, color: str) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        edgecolor="#264653",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, color="#1f2937")


def _draw_arrow(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#1f2937"),
    )


def generate_pipeline_overview(output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 4), dpi=170)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _draw_box(ax, 0.03, 0.34, 0.16, 0.32, "MIMIC tables\n(DX/PX/RX)", "#E0F2FE")
    _draw_box(ax, 0.23, 0.34, 0.16, 0.32, "Task samples\nper patient visit", "#DBEAFE")
    _draw_box(ax, 0.43, 0.34, 0.16, 0.32, "Ontology levels\nL1/L2/L3", "#EDE9FE")
    _draw_box(ax, 0.63, 0.34, 0.16, 0.32, "Graph propagation\nGAT + Hypergraph", "#FEF3C7")
    _draw_box(ax, 0.83, 0.34, 0.14, 0.32, "Transformer +\nmultilabel head", "#DCFCE7")

    _draw_arrow(ax, 0.19, 0.5, 0.23, 0.5)
    _draw_arrow(ax, 0.39, 0.5, 0.43, 0.5)
    _draw_arrow(ax, 0.59, 0.5, 0.63, 0.5)
    _draw_arrow(ax, 0.79, 0.5, 0.83, 0.5)

    ax.text(0.5, 0.92, "LINKO End-to-End Pipeline", ha="center", va="center", fontsize=14, weight="bold")
    ax.text(0.5, 0.1, "Data preprocessing -> ontology-aware representation -> graph reasoning -> diagnosis prediction", ha="center", va="center", fontsize=10, color="#374151")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def generate_method_overview(output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 6), dpi=170)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _draw_box(ax, 0.06, 0.72, 0.22, 0.18, "Ontology tokenization\n(DX/RX/PX by level)", "#DBEAFE")
    _draw_box(ax, 0.39, 0.72, 0.22, 0.18, "Co-occurrence matrix\nand hyperedges", "#FEF3C7")
    _draw_box(ax, 0.72, 0.72, 0.22, 0.18, "LLM-augmented\ncode embedding", "#EDE9FE")

    _draw_box(ax, 0.2, 0.42, 0.26, 0.18, "Onto_GAT\n(level-wise propagation)", "#DCFCE7")
    _draw_box(ax, 0.54, 0.42, 0.26, 0.18, "bottom_up_hap + _gram\n(hierarchical fusion)", "#FCE7F3")

    _draw_box(ax, 0.36, 0.12, 0.3, 0.18, "Transformer encoder\n+ FC multilabel output", "#E0F2FE")

    _draw_arrow(ax, 0.28, 0.72, 0.3, 0.6)
    _draw_arrow(ax, 0.5, 0.72, 0.45, 0.6)
    _draw_arrow(ax, 0.72, 0.72, 0.7, 0.6)
    _draw_arrow(ax, 0.46, 0.42, 0.5, 0.42)
    _draw_arrow(ax, 0.5, 0.42, 0.5, 0.3)

    ax.text(0.5, 0.96, "How LINKO Applies the Paper Method", ha="center", va="center", fontsize=14, weight="bold")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate README visualization assets for LINKO")
    parser.add_argument(
        "--input",
        default="results_prompting/metrics_results_BestModel_OntoFAR_1.0.txt",
        help="Path to metrics summary text file.",
    )
    parser.add_argument(
        "--output-dir",
        default="results_prompting",
        help="Directory where visualization assets will be saved.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    summary = parse_metrics_file(args.input)
    summary_plot_path = os.path.join(args.output_dir, "metrics_results_BestModel_OntoFAR_1.0_summary.png")
    pipeline_plot_path = os.path.join(args.output_dir, "linko_pipeline_overview.png")
    method_plot_path = os.path.join(args.output_dir, "linko_method_overview.png")

    plot_summary(summary, summary_plot_path)
    generate_pipeline_overview(pipeline_plot_path)
    generate_method_overview(method_plot_path)

    print(f"Saved metrics summary plot: {summary_plot_path}")
    print(f"Saved pipeline overview plot: {pipeline_plot_path}")
    print(f"Saved method overview plot: {method_plot_path}")


if __name__ == "__main__":
    main()
