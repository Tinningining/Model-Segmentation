#!/usr/bin/env python3
"""
Visualize comparison results between two runs (embedding, hidden blocks, logits).
All generated figures are saved into a single folder: --prefix.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_results(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def plot_line_per_metric(results, metric="mse", title_prefix="", output_dir=None):
    steps = sorted(results.keys())
    layers = sorted(results[steps[0]].keys())  # hidden_block0~4 + logits

    plt.figure(figsize=(12, 6))
    for layer in layers:
        values = [results[s][layer][metric] if results[s][layer]["shape_match"] else np.nan for s in steps]
        plt.plot(range(len(steps)), values, marker="o", label=layer)

    plt.xticks(range(len(steps)), steps, rotation=45)
    plt.xlabel("Step")
    plt.ylabel(metric.upper())
    plt.title(f"{title_prefix} {metric.upper()} per Step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # save into unified folder
    out_file = output_dir / f"{title_prefix}_{metric}.png"
    plt.savefig(out_file)
    plt.close()


def plot_heatmap(results, metric="mse", title_prefix="", output_dir=None):
    steps = sorted(results.keys())
    layers = sorted(results[steps[0]].keys())

    heat_data = []
    for step in steps:
        row = []
        for layer in layers:
            val = results[step][layer][metric] if results[step][layer]["shape_match"] else np.nan
            row.append(val)
        heat_data.append(row)

    heat_data = np.array(heat_data)

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        heat_data,
        xticklabels=layers,
        yticklabels=steps,
        annot=True,
        fmt=".3e",
        cmap="viridis",
    )
    plt.title(f"{title_prefix} {metric.upper()} Heatmap")
    plt.xlabel("Layer")
    plt.ylabel("Step")
    plt.tight_layout()

    # save into unified folder
    out_file = output_dir / f"{title_prefix}_{metric}_heatmap.png"
    plt.savefig(out_file)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize comparison results JSON")
    parser.add_argument("--json", type=str, required=True, help="Comparison JSON file")
    parser.add_argument("--prefix", type=str, default="compare",
                        help="Prefix for a unified output folder")
    args = parser.parse_args()

    # Create unified output directory
    output_dir = Path(args.prefix)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(args.json)

    # Generate line plots
    for metric in ["mse", "max_abs_err", "cosine"]:
        plot_line_per_metric(
            results, metric=metric,
            title_prefix=args.prefix,
            output_dir=output_dir
        )

    # Generate heatmaps
    for metric in ["mse", "max_abs_err", "cosine"]:
        plot_heatmap(
            results, metric=metric,
            title_prefix=args.prefix,
            output_dir=output_dir
        )

    print(f"All figures saved to folder: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
