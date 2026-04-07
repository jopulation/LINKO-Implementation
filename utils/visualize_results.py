import argparse
import json
import os
import re
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_PATTERNS = {
    "pr_auc_samples": re.compile(r"^mean pr_auc_samples:\s*([0-9eE+\-.]+)$"),
    "roc_auc_samples": re.compile(r"^mean roc_auc_samples:\s*([0-9eE+\-.]+)$"),
    "f1_samples": re.compile(r"^mean f1_samples:\s*([0-9eE+\-.]+)$"),
    "acc_at_k": re.compile(r"^mean acc_at_k=(\d+):\s*([0-9eE+\-.]+)$"),
    "hit_at_k": re.compile(r"^mean hit_at_k=(\d+):\s*([0-9eE+\-.]+)$"),
}


def parse_metrics_file(path: str) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "core": {},
        "acc_at_k": {},
        "hit_at_k": {},
    }

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            core_match = METRIC_PATTERNS["pr_auc_samples"].match(line)
            if core_match:
                summary["core"]["pr_auc_samples"] = float(core_match.group(1))
                continue

            core_match = METRIC_PATTERNS["roc_auc_samples"].match(line)
            if core_match:
                summary["core"]["roc_auc_samples"] = float(core_match.group(1))
                continue

            core_match = METRIC_PATTERNS["f1_samples"].match(line)
            if core_match:
                summary["core"]["f1_samples"] = float(core_match.group(1))
                continue

            acc_match = METRIC_PATTERNS["acc_at_k"].match(line)
            if acc_match:
                summary["acc_at_k"][int(acc_match.group(1))] = float(acc_match.group(2))
                continue

            hit_match = METRIC_PATTERNS["hit_at_k"].match(line)
            if hit_match:
                summary["hit_at_k"][int(hit_match.group(1))] = float(hit_match.group(2))
                continue

    return summary


def save_json(summary: Dict[str, object], output_path: str) -> None:
    def _convert(value):
        if isinstance(value, dict):
            return {str(key): _convert(subvalue) for key, subvalue in value.items()}
        if isinstance(value, list):
            return [_convert(item) for item in value]
        return value

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(_convert(summary), handle, indent=2, ensure_ascii=False)


def plot_summary(summary: Dict[str, object], output_path: str) -> None:
    core = summary.get("core", {})
    acc_at_k = summary.get("acc_at_k", {})
    hit_at_k = summary.get("hit_at_k", {})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=160)

    core_names = ["pr_auc_samples", "roc_auc_samples", "f1_samples"]
    core_values = [core.get(name, 0.0) for name in core_names]
    core_colors = ["#2A9D8F", "#264653", "#E9C46A"]
    axes[0].bar(core_names, core_values, color=core_colors, width=0.6)
    axes[0].set_title("Mean Core Metrics")
    axes[0].set_ylim(0, max(1.0, max(core_values) * 1.2 if core_values else 1.0))
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].tick_params(axis="x", rotation=20)

    if acc_at_k:
        acc_keys = sorted(acc_at_k.keys())
        acc_values = [acc_at_k[k] for k in acc_keys]
        axes[1].plot(acc_keys, acc_values, marker="o", linewidth=2.2, label="acc_at_k", color="#E76F51")

    if hit_at_k:
        hit_keys = sorted(hit_at_k.keys())
        hit_values = [hit_at_k[k] for k in hit_keys]
        axes[1].plot(hit_keys, hit_values, marker="s", linewidth=2.2, label="hit_at_k", color="#457B9D")

    axes[1].set_title("Mean Top-k Metrics")
    axes[1].set_xlabel("k")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize LINKO training results")
    parser.add_argument(
        "--input",
        default="results_prompting/metrics_results_BestModel_OntoFAR_1.0.txt",
        help="Path to the metrics summary text file.",
    )
    parser.add_argument(
        "--output-dir",
        default="results_prompting",
        help="Directory for generated visualization files.",
    )
    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    summary = parse_metrics_file(input_path)

    json_path = os.path.join(output_dir, "metrics_results_BestModel_OntoFAR_1.0_summary.json")
    png_path = os.path.join(output_dir, "metrics_results_BestModel_OntoFAR_1.0_summary.png")

    save_json(summary, json_path)
    plot_summary(summary, png_path)

    print(f"Saved JSON summary: {json_path}")
    print(f"Saved plot: {png_path}")


if __name__ == "__main__":
    main()
