#!/usr/bin/env python3
"""
Generate publication-grade IEEE TPAMI analytical vector figures directly from verified
final_paper_results.json, vpr_methods_recall_summary.json, and conflict_gate_eval_summary.json.
Outputs:
  - fig2_4era_17vpr_scaling.pdf / .png  (2-panel: 4-Era VPR R@1 -> VQA Accuracy Compression + Gate AUROC vs Gain)
  - fig3_noise_and_benchmarks.pdf / .png (2-panel: Retrieval Noise Stress Test 0%-80% + 6-Benchmark Main Comparison)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.environ.get("CHRONOVPR_ROOT", ".")
BAK = os.path.join(BASE, "final_experiment_backup")

with open(os.path.join(BAK, "final_paper_results.json"), "r", encoding="utf-8") as f:
    res = json.load(f)
with open(os.path.join(BAK, "vpr_retrievals", "vpr_methods_recall_summary.json"), "r", encoding="utf-8") as f:
    rec = json.load(f)
with open(os.path.join(BAK, "conflict_gate_eval_summary.json"), "r", encoding="utf-8") as f:
    gate = json.load(f)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["axes.linewidth"] = 0.9

ERAS = {
    "resnet50_avg": ("Era 1: Early CNN / VLAD", "ResNet50-Avg"),
    "resnet50_gem": ("Era 1: Early CNN / VLAD", "GeM"),
    "netvlad": ("Era 1: Early CNN / VLAD", "NetVLAD"),
    "patch_netvlad": ("Era 1: Early CNN / VLAD", "Patch-NetVLAD"),
    "cosplace": ("Era 2: Large-Scale CNN", "CosPlace"),
    "eigenplaces": ("Era 2: Large-Scale CNN", "EigenPlaces"),
    "convap": ("Era 2: Large-Scale CNN", "Conv-AP"),
    "mixvpr": ("Era 2: Large-Scale CNN", "MixVPR"),
    "dinov2_vitb14": ("Era 3: VFM / DINOv2", "DINOv2-B/14"),
    "anyloc": ("Era 3: VFM / DINOv2", "AnyLoc"),
    "selavpr": ("Era 3: VFM / DINOv2", "SelaVPR"),
    "cricavpr": ("Era 3: VFM / DINOv2", "CricaVPR"),
    "dinov2_salad": ("Era 3: VFM / DINOv2", "SALAD"),
    "boq_dinov2": ("Era 3: VFM / DINOv2", "BoQ"),
    "megaloc": ("Era 3: VFM / DINOv2", "MegaLoc"),
    "clip_vitb32": ("Era 4: Multimodal CLIP", "CLIP-B/32"),
    "clip_vitl14": ("Era 4: Multimodal CLIP", "CLIP-L/14"),
}

ERA_STYLE = {
    "Era 1: Early CNN / VLAD": ("#C87D20", "s"),
    "Era 2: Large-Scale CNN": ("#1F7A7A", "^"),
    "Era 3: VFM / DINOv2": ("#2B5B84", "o"),
    "Era 4: Multimodal CLIP": ("#8E44AD", "D"),
}

# ==============================================================================
# FIGURE 2: 17 VPR Methods Across 4 Eras (VPR R@1 vs VQA Acc + Gate AUROC)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.7), dpi=300)
fig.patch.set_facecolor("#FFFFFF")

t2 = res["table2_all_vpr"]

# Reference lines on Panel (a)
ax1.axhline(75.80, color="#2E7D4F", ls="--", lw=1.25, label="Oracle Upper Bound (+LoRA): 75.80%")
ax1.axhline(44.02, color="#94A3B8", ls=":", lw=1.25, label="Single-View MLLM (No VPR): 44.02%")

plotted_eras = set()
for m, (era, short) in ERAS.items():
    r1 = t2[m]["VPR_R1"]
    nv = t2[m]["Naive_VQA_Acc"]
    ov = t2[m]["Ours_VQA_Acc"]
    col, mk = ERA_STYLE[era]

    # Vertical gain connector
    ax1.plot([r1, r1], [nv, ov], color=col, alpha=0.28, lw=1.1, zorder=1)
    # Naive point (hollow)
    ax1.scatter(r1, nv, facecolors="#FFFFFF", edgecolors=col, marker=mk, s=48, linewidths=1.3, zorder=3)
    # Ours point (filled)
    lbl = era if era not in plotted_eras else None
    plotted_eras.add(era)
    ax1.scatter(r1, ov, facecolors=col, edgecolors="#1E293B", marker=mk, s=62, linewidths=0.7, label=lbl, zorder=4)

# Clean non-overlapping text annotations for iconic landmarks
ann_offsets = {
    "clip_vitb32": (1.0, -2.6),
    "clip_vitl14": (-1.5, 1.4),
    "netvlad": (0.8, -2.5),
    "patch_netvlad": (-4.5, 1.5),
    "anyloc": (1.0, -2.5),
    "cosplace": (-5.0, 1.5),
    "mixvpr": (0.8, -2.4),
    "cricavpr": (-7.8, 1.5),
    "megaloc": (-2.0, 1.8),
    "boq_dinov2": (-2.0, -2.8),
}
for m, (dx, dy) in ann_offsets.items():
    r1 = t2[m]["VPR_R1"]
    ov = t2[m]["Ours_VQA_Acc"]
    ax1.annotate(
        ERAS[m][1], (r1, ov), xytext=(r1 + dx, ov + dy),
        fontsize=7.3, fontweight="bold" if m in ("megaloc", "netvlad", "clip_vitb32") else "normal",
        color="#1E293B", arrowprops=dict(arrowstyle="-", color="#94A3B8", lw=0.6),
    )

ax1.annotate("Filled: Ours Full (CTSE+Gate+LoRA)\nHollow: Naive VPR + MLLM",
             xy=(0.03, 0.42), xycoords="axes fraction", fontsize=7.6,
             bbox=dict(boxstyle="round,pad=0.35", fc="#F8FAFC", ec="#CBD5E1", lw=0.8), color="#334155")

ax1.set_xlabel("VPR Place Retrieval Recall@1 (%)", fontsize=9.5, fontweight="bold", color="#1E293B")
ax1.set_ylabel("ChronoVPR-VQA Test Accuracy (%)", fontsize=9.5, fontweight="bold", color="#1E293B")
ax1.set_title("(a) Cross-Era VPR Retrieval vs. Downstream VQA Utility (17 Methods)", fontsize=10.0, fontweight="bold", color="#1E293B", pad=8)
ax1.set_xlim(23, 76)
ax1.set_ylim(41, 79.5)
ax1.grid(True, ls="--", alpha=0.35, color="#CBD5E1")
ax1.legend(loc="lower right", fontsize=7.3, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", ncol=1)

# Panel (b): VPR R@1 vs Conflict Gate Aliasing Detection AUROC (showing adaptive rescue of noisy VPR!)
r1_all, auc_all, gains_all = [], [], []
for m, (era, short) in ERAS.items():
    r1 = t2[m]["VPR_R1"]
    g_auc = gate["by_vpr_method"][m]["test_aliasing_detection_AUROC"]
    gn = t2[m]["Gain"]
    col, mk = ERA_STYLE[era]
    r1_all.append(r1); auc_all.append(g_auc); gains_all.append(gn)
    ax2.scatter(r1, g_auc, facecolors=col, edgecolors="#1E293B", marker=mk, s=70, linewidths=0.75, zorder=4)

# Fit linear trend line to highlight monotonic inverse coupling
coef = np.polyfit(r1_all, auc_all, 1)
xs_fit = np.linspace(26, 73, 100)
ax2.plot(xs_fit, np.polyval(coef, xs_fit), color="#475569", ls="--", lw=1.2, alpha=0.8,
         label=f"Linear Trend (r = {np.corrcoef(r1_all, auc_all)[0,1]:.2f})")

ann2 = {
    "clip_vitb32": (1.0, 0.008),
    "resnet50_avg": (0.8, 0.010),
    "netvlad": (1.2, 0.008),
    "patch_netvlad": (1.2, 0.008),
    "anyloc": (1.2, 0.009),
    "cosplace": (1.2, 0.008),
    "mixvpr": (1.2, 0.008),
    "cricavpr": (-8.2, -0.018),
    "megaloc": (-7.5, -0.022),
    "boq_dinov2": (-1.5, 0.020),
}
for m, (dx, dy) in ann2.items():
    r1 = t2[m]["VPR_R1"]
    g_auc = gate["by_vpr_method"][m]["test_aliasing_detection_AUROC"]
    ax2.annotate(
        ERAS[m][1], (r1, g_auc), xytext=(r1 + dx, g_auc + dy),
        fontsize=7.3, color="#1E293B", arrowprops=dict(arrowstyle="-", color="#94A3B8", lw=0.6),
    )

ax2.set_xlabel("VPR Place Retrieval Recall@1 (%)", fontsize=9.5, fontweight="bold", color="#1E293B")
ax2.set_ylabel("Conflict Gate Aliasing Detection AUROC", fontsize=9.5, fontweight="bold", color="#1E293B")
ax2.set_title("(b) Scale-Invariant Conflict Gate Generalization Across All 17 VPRs", fontsize=10.0, fontweight="bold", color="#1E293B", pad=8)
ax2.set_xlim(23, 76)
ax2.set_ylim(0.62, 0.90)
ax2.grid(True, ls="--", alpha=0.35, color="#CBD5E1")
ax2.legend(loc="upper right", fontsize=8.0, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

plt.tight_layout(w_pad=2.2)
fig.savefig(os.path.join(BASE, "fig2_4era_17vpr_scaling.pdf"), bbox_inches="tight", pad_inches=0.06)
fig.savefig(os.path.join(BASE, "fig2_4era_17vpr_scaling.png"), dpi=400, bbox_inches="tight", pad_inches=0.06)
plt.close(fig)

# ==============================================================================
# FIGURE 3: Controlled Retrieval Noise Stress Test + 6-Benchmark Breakdown
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.5), dpi=300)
fig.patch.set_facecolor("#FFFFFF")

# Panel (a): Noise Stress Test (Table 4)
noise_pcts = [0, 20, 40, 60, 80]
naive_n = [res["table4_noise_robustness"][f"noise_{p}pct"]["Naive_VQA_Acc"] for p in noise_pcts]
ours_n = [res["table4_noise_robustness"][f"noise_{p}pct"]["Ours_Gated_VQA_Acc"] for p in noise_pcts]
gains_n = [res["table4_noise_robustness"][f"noise_{p}pct"]["Robustness_Gain"] for p in noise_pcts]

ax1.plot(noise_pcts, ours_n, "-o", color="#2B5B84", lw=2.3, ms=7, label="Ours Full (CTSE + Conflict Gate + LoRA)")
ax1.plot(noise_pcts, naive_n, "--s", color="#C44545", lw=2.0, ms=6.5, label="Naive VPR + MLLM (Unfiltered)")
ax1.fill_between(noise_pcts, naive_n, ours_n, color="#2B5B84", alpha=0.10)

for p, nv, ov, gn in zip(noise_pcts, naive_n, ours_n, gains_n):
    ax1.text(p, ov + 1.0, f"{ov:.2f}%", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#2B5B84")
    ax1.text(p, nv - 2.3, f"{nv:.2f}%", ha="center", va="top", fontsize=7.6, color="#9E2A2A")
    ax1.text(p, (nv + ov) / 2, f"+{gn:.2f}%", ha="center", va="center", fontsize=7.5, fontweight="bold",
             color="#1E5C37", bbox=dict(boxstyle="round,pad=0.18", fc="#EAF7EE", ec="#2B7A4B", lw=0.6))

ax1.set_xlabel("Injected Perceptual-Aliasing Retrieval Error Rate (%)", fontsize=9.5, fontweight="bold", color="#1E293B")
ax1.set_ylabel("VQA Accuracy on Stratified Stress Set (%)", fontsize=9.5, fontweight="bold", color="#1E293B")
ax1.set_title("(a) Robustness Under Controlled Perceptual-Aliasing Corruption", fontsize=10.0, fontweight="bold", color="#1E293B", pad=8)
ax1.set_xticks(noise_pcts)
ax1.set_ylim(42, 81)
ax1.grid(True, ls="--", alpha=0.35, color="#CBD5E1")
ax1.legend(loc="lower left", fontsize=8.0, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")

# Panel (b): 6-Benchmark Breakdown (Single-View vs Naive BoQ vs Ours MegaLoc vs Oracle+LoRA)
bms = ["GSV-Cities", "SF-XL", "Pittsburgh250K", "MSLS", "SPED", "Nordland"]
bms_lbl = ["GSV-\nCities", "SF-XL", "Pitts-\n250K", "MSLS", "SPED", "Nord-\nland"]
sv = [res["table1_main"]["Single-View MLLM (No VPR)"]["by_dataset"][b] for b in bms]
nv_b = [res["table1_main"]["Naive VPR+MLLM (boq_dinov2)"]["by_dataset"][b] for b in bms]
ours_m = [res["table2_all_vpr"]["megaloc"]["Ours_by_dataset"][b] for b in bms]
ora = [res["table1_main"]["Oracle Upper Bound (+LoRA)"]["by_dataset"][b] for b in bms]

x = np.arange(len(bms))
w = 0.20
ax2.bar(x - 1.5*w, sv, width=w, color="#94A3B8", edgecolor="#475569", lw=0.6, label="Single-View MLLM")
ax2.bar(x - 0.5*w, nv_b, width=w, color="#E6A15C", edgecolor="#A66310", lw=0.6, label="Naive VPR+MLLM (BoQ)")
ax2.bar(x + 0.5*w, ours_m, width=w, color="#2B5B84", edgecolor="#1E293B", lw=0.6, label="Ours Full (MegaLoc)")
ax2.bar(x + 1.5*w, ora, width=w, color="#2E7D4F", edgecolor="#1E5C37", lw=0.6, label="Oracle (+LoRA)")

for i in range(len(bms)):
    ax2.text(x[i] + 0.5*w, ours_m[i] + 1.0, f"{ours_m[i]:.1f}", ha="center", va="bottom", fontsize=6.7, fontweight="bold", color="#2B5B84")

ax2.set_xticks(x)
ax2.set_xticklabels(bms_lbl, fontsize=8.5, fontweight="bold", color="#1E293B")
ax2.set_ylabel("VQA Test Accuracy (%)", fontsize=9.5, fontweight="bold", color="#1E293B")
ax2.set_title("(b) Cross-Benchmark Generalization Across Six VPR Datasets", fontsize=10.0, fontweight="bold", color="#1E293B", pad=8)
ax2.set_ylim(25, 89)
ax2.grid(True, axis="y", ls="--", alpha=0.35, color="#CBD5E1")
ax2.legend(loc="upper right", fontsize=7.4, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", ncol=2)

plt.tight_layout(w_pad=2.2)
fig.savefig(os.path.join(BASE, "fig3_noise_and_benchmarks.pdf"), bbox_inches="tight", pad_inches=0.06)
fig.savefig(os.path.join(BASE, "fig3_noise_and_benchmarks.png"), dpi=400, bbox_inches="tight", pad_inches=0.06)
plt.close(fig)
print("Generated Fig 2 and Fig 3 successfully!")
