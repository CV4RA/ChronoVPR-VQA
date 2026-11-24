"""Render the README figure using only the committed evaluation summaries."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from build_project_site import ROOT, release_data


NAIVE = "#b35806"
FULL = "#006e84"
INK = "#21313a"
MUTED = "#596c77"
STYLES = {
    "early": ("Early CNN / VLAD", "s", "#a65314"),
    "cnn": ("Large-scale CNN", "^", "#006d59"),
    "vfm": ("Foundation models", "o", "#2166ac"),
    "clip": ("CLIP", "D", "#8b468a"),
}


def annotate_points(ax, x, y, text, offset, color):
    ax.annotate(text, (x, y), xytext=offset, textcoords="offset points",
                ha="left" if offset[0] > 0 else "right",
                va="bottom" if offset[1] > 0 else "top",
                fontsize=8, fontweight="bold", color=color)


def render(data):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#a9b5ba",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": "#e2e8eb",
        "axes.axisbelow": True,
        "legend.frameon": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), layout="constrained")
    paired, gate, breakdown, noise = axes.flat
    methods = data["methods"]
    recall = [method["VPR_R1"] for method in methods]
    naive = [method["Naive_VQA_Acc"] for method in methods]
    full = [method["Ours_VQA_Acc"] for method in methods]
    auroc = [method["auroc"] for method in methods]

    paired.vlines(recall, naive, full, color="#9db6bc", alpha=.65, linewidth=1.1)
    paired.scatter(recall, naive, label="Naïve VPR + MLLM", marker="s",
                   edgecolor=NAIVE, facecolor="white", s=22, zorder=3)
    paired.scatter(recall, full, label="Full system", color=FULL, s=26, zorder=3)
    paired.axhline(data["main"]["Oracle + LoRA"]["overall_acc"], color=MUTED,
                   linestyle="--", linewidth=1, label="Oracle + LoRA")
    paired.axhline(data["main"]["Single view"]["overall_acc"], color=MUTED,
                   linestyle=":", linewidth=1, label="Single-view MLLM")
    paired.set(title="Paired VQA accuracy across 17 retrievers", xlabel="VPR Recall@1",
               ylabel="Test VQA accuracy", xlim=(25, 75), ylim=(40, 82))
    paired.set_title("Paired VQA accuracy across 17 retrievers", y=1.18)
    paired.legend(loc="lower center", bbox_to_anchor=(.5, 1.005), ncol=2, fontsize=8)

    for key, (label, marker, color) in STYLES.items():
        group = [item for item in methods if item["family"] == key]
        gate.scatter([item["VPR_R1"] for item in group], [item["auroc"] for item in group],
                     label=label, marker=marker, color=color, s=31, zorder=3)
    slope, intercept = np.polyfit(recall, auroc, deg=1)
    r = np.corrcoef(recall, auroc)[0, 1]
    domain = np.array([min(recall), max(recall)])
    gate.plot(domain, slope * domain + intercept, "--", color=MUTED, linewidth=1)
    gate.text(.04, .09, f"Pearson r = {r:.2f}  ·  n = {len(methods)}",
              transform=gate.transAxes, color=MUTED)
    gate.set(title="Conflict Gate transfer", xlabel="VPR Recall@1",
             ylabel="Aliasing detection AUROC", xlim=(25, 75), ylim=(.63, .91))
    gate.set_title("Conflict Gate transfer", y=1.18)
    gate.legend(loc="lower center", bbox_to_anchor=(.5, 1.005), ncol=2, fontsize=8)

    task_keys = ("occlusion_recovery", "out_of_fov", "temporal_evolution")
    task_labels = ("Occlusion recovery", "Out-of-FoV", "Temporal evolution")
    protocols = ("Naive memory", "CTSE + Gate", "Full system")
    encoding = ((NAIVE, "s"), ("#646a7b", "D"), (FULL, "o"))
    positions = np.arange(len(task_keys))
    for offset, (label, (color, marker)) in enumerate(zip(protocols, encoding)):
        values = [data["main"][label]["by_task"][key] for key in task_keys]
        breakdown.scatter(values, positions + (offset - 1) * .18,
                          color=color, s=36, label=label, marker=marker, zorder=3)
    breakdown.set(title="Reasoning breakdown with BoQ-DINOv2", xlabel="Test VQA accuracy",
                  yticks=positions, yticklabels=task_labels, xlim=(30, 100), ylim=(2.5, -.5))
    breakdown.grid(axis="x")
    breakdown.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    breakdown.legend(loc="lower right", fontsize=8)

    rates = [row["rate"] for row in data["noise"]]
    for column, label, color, linestyle, marker in (
        ("Naive_VQA_Acc", "Naïve VPR + MLLM", NAIVE, "--", "s"),
        ("Ours_Gated_VQA_Acc", "Gated system", FULL, "-", "o"),
    ):
        values = [row[column] for row in data["noise"]]
        noise.plot(rates, values, linestyle=linestyle, color=color, marker=marker,
                   linewidth=1.7, markersize=5, label=label)
        annotate_points(noise, rates[-1], values[-1], f"{values[-1]:.2f}%",
                        (-5, 8 if marker == "o" else -8), color)
    noise.set(title="Controlled aliasing noise stress test", xlabel="Injected aliasing noise",
              ylabel="Stress-test VQA accuracy", xticks=rates, xlim=(-4, 84), ylim=(40, 82))
    noise.legend(loc="lower left", fontsize=8)

    for ax in (paired, gate, noise):
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.grid(axis="y")
    for ax in (paired, noise):
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "assets/release_results.png")
    args = parser.parse_args()
    fig = render(release_data())
    fig.savefig(args.output, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
