import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import precision_recall_curve, average_precision_score

BASE = os.environ.get("CHRONOVPR_ROOT", ".")
BAK  = os.path.join(BASE, "final_experiment_backup")
VPR_DIR = os.path.join(BAK, "vpr_retrievals")
CLEAN_DIR = os.path.join(BASE, "ChronoVPR-VQA-Clean")
OUT_DIR = os.path.join(BASE, "appendix_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Global IEEE TPAMI publication styling
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "axes.labelweight": "bold",
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
})

ERA_COLORS = {
    "Era 1": "#C96A5B",  # Terracotta / Brick
    "Era 2": "#D99B43",  # Ochre / Gold
    "Era 3": "#2C7BB6",  # Deep Steel Blue
    "Era 4": "#8861A8",  # Muted Purple
}

METHODS = [
    ("Era 1", "resnet50_avg",  "ResNet50-Avg"),
    ("Era 1", "resnet50_gem",  "ResNet50-GeM"),
    ("Era 1", "netvlad",       "NetVLAD"),
    ("Era 1", "patch_netvlad", "Patch-NetVLAD"),
    ("Era 2", "cosplace",      "CosPlace"),
    ("Era 2", "eigenplaces",   "EigenPlaces"),
    ("Era 2", "convap",        "Conv-AP"),
    ("Era 2", "mixvpr",        "MixVPR"),
    ("Era 3", "dinov2_vitb14", "DINOv2-B/14"),
    ("Era 3", "anyloc",        "AnyLoc"),
    ("Era 3", "selavpr",       "SelaVPR"),
    ("Era 3", "cricavpr",      "CricaVPR"),
    ("Era 3", "dinov2_salad",  "SALAD"),
    ("Era 3", "boq_dinov2",    "BoQ"),
    ("Era 3", "megaloc",       "MegaLoc"),
    ("Era 4", "clip_vitb32",   "CLIP-B/32"),
    ("Era 4", "clip_vitl14",   "CLIP-L/14"),
]
BENCHMARKS = ["GSV-Cities", "SF-XL", "Pittsburgh250K", "MSLS", "Nordland", "SPED"]

stats = json.load(open(os.path.join(BASE, "appendix_computed_stats.json"), encoding="utf-8"))
all_clean = json.load(open(os.path.join(CLEAN_DIR, "chrono_vpr_vqa_all.json"), encoding="utf-8"))
tpami_stats = json.load(open(os.path.join(BASE, "tpami_all_tables_stats.json"), encoding="utf-8"))

# ==============================================================================
# FIG S1: Real Training Dynamics (Conflict Gate + LoRA MLLM)
# ==============================================================================
print("Generating figS1.pdf (Training Dynamics)...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6))

# Panel (a): Conflict Gate from exp2_v2.log (real logged points + smooth curve through logged checkpoints)
ep_log = np.array([1, 5, 10, 15])
gate_loss = np.array([0.6777, 0.4463, 0.3078, 0.1930])
gate_auc  = np.array([0.7006, 0.8507, 0.8736, 0.8711])
gate_f1   = np.array([0.5624, 0.7798, 0.7937, 0.7757])

ax1_r = ax1.twinx()
ax1_r.spines["top"].set_visible(False)
l1 = ax1.plot(ep_log, gate_loss, "o-", color="#C96A5B", lw=2.2, ms=6.5, label="Train BCE Loss (left)")
l2 = ax1_r.plot(ep_log, gate_auc, "s-", color="#2C7BB6", lw=2.2, ms=6.5, label="Val AUROC (right)")
l3 = ax1_r.plot(ep_log, gate_f1,  "^--", color="#2B9348", lw=1.9, ms=6.0, label="Val F1 Score (right)")
ax1_r.axhline(0.8737, color="#2C7BB6", ls=":", lw=1.2, alpha=0.8)
ax1_r.annotate("Best Checkpoint\nVal AUROC = 0.8737", xy=(10, 0.8736), xytext=(10.5, 0.81),
               fontsize=7.3, fontweight="bold", color="#1D4E89",
               arrowprops=dict(arrowstyle="->", color="#1D4E89", lw=1.0),
               bbox=dict(boxstyle="round,pad=0.25", fc="#EEF5FC", ec="#2C7BB6", lw=0.8))
ax1.set_xlabel("Training Epoch (39,146 Train Pairs / 4,986 Val Pairs)")
ax1.set_ylabel("Binary Cross-Entropy Loss", color="#C96A5B")
ax1_r.set_ylabel("Validation AUROC / F1", color="#2C7BB6")
ax1.set_title("(a) Conflict Resolution Gate Optimization Dynamics")
ax1.set_xticks([1, 5, 10, 15])
ax1.set_ylim(0.10, 0.78)
ax1_r.set_ylim(0.50, 0.93)
lines = l1 + l2 + l3
ax1.legend(lines, [l.get_label() for l in lines], loc="lower center", frameon=True, facecolor="white", framealpha=0.92)

# Panel (b): LoRA MLLM real training log from exp3.log (12 exact logged points)
steps = np.array([300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600])
lora_loss = np.array([0.2640, 0.2675, 0.2276, 0.2495, 0.2239, 0.2044, 0.2212, 0.2265, 0.2049, 0.2116, 0.2218, 0.1901])
lora_lr   = np.array([1.48e-4, 1.40e-4, 1.28e-4, 1.13e-4, 9.47e-5, 7.50e-5, 5.58e-5, 3.75e-5, 2.22e-5, 1.00e-5, 2.62e-6, 0.0]) * 1e4

ax2_r = ax2.twinx()
ax2_r.spines["top"].set_visible(False)
p1 = ax2.plot(steps, lora_loss, "o-", color="#2C7BB6", lw=2.2, ms=5.5, label="LoRA Causal LM Loss (left)")
# Trend line
z = np.polyfit(steps, lora_loss, 1)
p_tr = ax2.plot(steps, np.poly1d(z)(steps), ":", color="#1D4E89", lw=1.4, label="Linear Convergence Trend")
p2 = ax2_r.plot(steps, lora_lr, "--", color="#D99B43", lw=2.0, label="Cosine LR (×10⁻⁴, right)")
ax2.annotate("Final Loss = 0.1901\n(74.0 min, RTX 4080)", xy=(3600, 0.1901), xytext=(2200, 0.252),
             fontsize=7.3, fontweight="bold", color="#1D4E89",
             arrowprops=dict(arrowstyle="->", color="#1D4E89", lw=1.0),
             bbox=dict(boxstyle="round,pad=0.25", fc="#EEF5FC", ec="#2C7BB6", lw=0.8))
ax2.set_xlabel("Training Samples Processed (Qwen2.5-VL-7B + LoRA r=16)")
ax2.set_ylabel("Training Cross-Entropy Loss", color="#2C7BB6")
ax2_r.set_ylabel("Cosine Learning Rate (×10⁻⁴)", color="#B07A28")
ax2.set_title("(b) Conflict-Aware LoRA MLLM Convergence")
ax2.set_ylim(0.175, 0.285)
ax2_r.set_ylim(-0.08, 1.68)
lines2 = p1 + p_tr + p2
ax2.legend(lines2, [l.get_label() for l in lines2], loc="lower left", frameon=True, facecolor="white", framealpha=0.92)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "figS1.pdf"))
plt.close(fig)

# ==============================================================================
# FIG S2: 6-Panel Per-Benchmark 17-Method Retrieval vs Post-Gate Precision & VQA
# ==============================================================================
print("Generating figS2.pdf (Per-Benchmark 17-Method Matrix)...")
fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.8), sharex=True)
x = np.arange(len(METHODS))
labels_short = [m[2] for m in METHODS]
bar_colors = [ERA_COLORS[m[0]] for m in METHODS]

# Check if per-benchmark VQA accuracy exists in tpami_all_tables_stats.json
vqa_bm_map = {}
if "table_bm_vqa" in tpami_stats:
    vqa_bm_map = tpami_stats["table_bm_vqa"]

for idx, bm in enumerate(BENCHMARKS):
    ax = axes[idx // 3, idx % 3]
    r1_vals   = [stats["r5_r10_table"][m[1]][bm]["r1"] for m in METHODS]
    prec_vals = [stats["bm_gate_table"][m[1]][bm]["prec"] for m in METHODS]
    r10_vals  = [stats["r5_r10_table"][m[1]][bm]["r10"] for m in METHODS]

    ax.bar(x, r1_vals, width=0.68, color=bar_colors, alpha=0.82, edgecolor="white", lw=0.5, label="VPR Recall@1 (%)")
    ax.plot(x, prec_vals, "o-", color="#1B263B", lw=1.6, ms=3.8, label="Post-Gate Precision@τ (%)")
    ax.plot(x, r10_vals, "^:", color="#2B9348", lw=1.3, ms=3.3, alpha=0.85, label="VPR Recall@10 (%)")

    # Era vertical separators
    for sep in [3.5, 7.5, 14.5]:
        ax.axvline(sep, color="#888888", ls="--", lw=0.7, alpha=0.5)

    ood_tag = " (Zero-Shot OOD)" if bm in ("Nordland", "SPED") else ""
    ax.set_title(f"({chr(97+idx)}) {bm}{ood_tag}")
    ax.set_ylim(0, 106)
    if idx % 3 == 0:
        ax.set_ylabel("Percentage (%)")
    if idx >= 3:
        ax.set_xticks(x)
        ax.set_xticklabels(labels_short, rotation=52, ha="right", fontsize=6.7)
    if idx == 0:
        ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, fontsize=6.8)

# Shared Era Legend at top
era_handles = [mpatches.Patch(color=ERA_COLORS[e], label=lbl, alpha=0.85) for e, lbl in [
    ("Era 1", "Era 1: Early CNN/VLAD"),
    ("Era 2", "Era 2: Large-Scale CNN"),
    ("Era 3", "Era 3: VFM / DINOv2"),
    ("Era 4", "Era 4: Cross-Modal CLIP"),
]]
fig.legend(handles=era_handles, loc="upper center", ncol=4, frameon=True, facecolor="white",
           edgecolor="#CCCCCC", bbox_to_anchor=(0.5, 1.02), fontsize=8.0)
plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(OUT_DIR, "figS2.pdf"))
plt.close(fig)

# ==============================================================================
# Load 6 Representative Methods for Fig S3, S4, S5
# ==============================================================================
print("Loading per-sample retrieval arrays for Figs S3, S4, S5...")
REP6 = [
    ("Era 1", "resnet50_avg", "ResNet50-Avg (Era 1)"),
    ("Era 1", "netvlad",      "NetVLAD (Era 1)"),
    ("Era 2", "mixvpr",       "MixVPR (Era 2)"),
    ("Era 3", "dinov2_salad", "DINOv2-SALAD (Era 3)"),
    ("Era 3", "megaloc",      "MegaLoc (Era 3)"),
    ("Era 4", "clip_vitb32",  "CLIP-ViT-B/32 (Era 4)"),
]
rep_data = {}
for era, key, title in REP6:
    rows = json.load(open(os.path.join(VPR_DIR, f"{key}_test_retrievals_gated.json"), encoding="utf-8"))
    rep_data[key] = {
        "y": np.array([int(r["naive_hit1"]) for r in rows]),
        "r": np.array([float(r["gate_reliability"]) for r in rows]),
        "s": np.array([float(r["naive_top1_sim"]) for r in rows]),
        "era": era, "title": title,
    }
# Also load clip_vitl14 for Fig S5
rows_cl = json.load(open(os.path.join(VPR_DIR, "clip_vitl14_test_retrievals_gated.json"), encoding="utf-8"))
rep_data["clip_vitl14"] = {
    "y": np.array([int(r["naive_hit1"]) for r in rows_cl]),
    "r": np.array([float(r["gate_reliability"]) for r in rows_cl]),
    "s": np.array([float(r["naive_top1_sim"]) for r in rows_cl]),
    "era": "Era 4", "title": "CLIP-ViT-L/14 (Era 4)",
}

# ==============================================================================
# FIG S3: 6-Panel Gate Reliability Score Distributions (True Match vs Aliased FP)
# ==============================================================================
print("Generating figS3.pdf (Gate Score Distributions)...")
fig, axes = plt.subplots(2, 3, figsize=(12.0, 5.8), sharex=True, sharey=False)
bins = np.linspace(0, 1, 31)
for idx, (era, key, title) in enumerate(REP6):
    ax = axes[idx // 3, idx % 3]
    d = rep_data[key]
    pos = d["r"][d["y"] == 1]
    neg = d["r"][d["y"] == 0]
    auc = stats["auprc_by_method"][key]["auroc"]

    ax.hist(neg, bins=bins, density=True, alpha=0.58, color="#D95F59", edgecolor="white", lw=0.4, label=f"Aliased FP (μ={neg.mean():.2f})")
    ax.hist(pos, bins=bins, density=True, alpha=0.60, color="#2A9D8F", edgecolor="white", lw=0.4, label=f"True Match (μ={pos.mean():.2f})")
    ax.axvline(0.45, color="#1D3557", ls="--", lw=1.6, label="Threshold τ = 0.45")

    ax.set_title(f"({chr(97+idx)}) {title}  |  AUROC = {auc:.4f}")
    if idx >= 3:
        ax.set_xlabel("Conflict Gate Reliability Score r")
    if idx % 3 == 0:
        ax.set_ylabel("Probability Density")
    ax.legend(loc="upper center", frameon=True, facecolor="white", framealpha=0.90, fontsize=6.9)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "figS3.pdf"))
plt.close(fig)

# ==============================================================================
# FIG S4: 2-Panel Precision-Recall Curves & TPR-TNR Pareto Sweep
# ==============================================================================
print("Generating figS4.pdf (PR Curves & TPR/TNR Pareto Frontiers)...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.1))

pr_styles = [
    ("resnet50_avg", "ResNet50-Avg (Era 1)", "#C96A5B", "-"),
    ("netvlad",      "NetVLAD (Era 1)",      "#9E3B33", "--"),
    ("mixvpr",       "MixVPR (Era 2)",       "#D99B43", "-"),
    ("dinov2_salad", "SALAD (Era 3)",        "#457B9D", "--"),
    ("megaloc",      "MegaLoc (Era 3)",      "#1D3557", "-"),
    ("clip_vitb32",  "CLIP-B/32 (Era 4)",    "#8861A8", "-"),
]

# Panel (a): Precision-Recall for Aliasing Rejection (positive class = Aliased FP, score = 1 - r)
for key, lbl, col, ls in pr_styles:
    d = rep_data[key]
    y_alias = 1 - d["y"]
    s_alias = 1.0 - d["r"]
    prec, rec, _ = precision_recall_curve(y_alias, s_alias)
    ap = average_precision_score(y_alias, s_alias)
    ax1.plot(rec, prec, color=col, ls=ls, lw=2.0, label=f"{lbl} (AP={ap:.3f})")

ax1.set_xlabel("Aliasing Rejection Recall (True Negative Rate)")
ax1.set_ylabel("Aliasing Rejection Precision")
ax1.set_title("(a) Aliasing-Detection Precision–Recall Curves")
ax1.set_xlim(0.0, 1.02)
ax1.set_ylim(0.20, 1.03)
ax1.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.0)

# Panel (b): TPR vs TNR sweep across tau in [0.10, 0.90]
taus = np.linspace(0.10, 0.90, 41)
for key, lbl, col, ls in pr_styles:
    d = rep_data[key]
    tprs, tnrs = [], []
    for t in taus:
        p = d["r"] >= t
        tprs.append(100.0 * np.mean(p[d["y"] == 1]))
        tnrs.append(100.0 * np.mean((~p)[d["y"] == 0]))
    ax2.plot(tnrs, tprs, color=col, ls=ls, lw=2.0, label=lbl)
    # Mark tau = 0.45
    p45 = d["r"] >= 0.45
    tpr45 = 100.0 * np.mean(p45[d["y"] == 1])
    tnr45 = 100.0 * np.mean((~p45)[d["y"] == 0])
    ax2.scatter([tnr45], [tpr45], marker="*", s=110, color=col, edgecolor="black", lw=0.6, zorder=6)

ax2.scatter([], [], marker="*", s=95, color="white", edgecolor="black", lw=0.8, label="Operating Point τ = 0.45")
ax2.set_xlabel("Aliasing Rejection Rate / TNR (%)")
ax2.set_ylabel("True-Match Retention / TPR (%)")
ax2.set_title("(b) TPR–TNR Operating Frontier (τ ∈ [0.10, 0.90])")
ax2.set_xlim(0, 100)
ax2.set_ylim(20, 102)
ax2.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.0)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "figS4.pdf"))
plt.close(fig)

# ==============================================================================
# FIG S5: 4-Panel Raw Similarity vs Conflict Gate Reliability Scatter
# ==============================================================================
print("Generating figS5.pdf (Raw Cosine Similarity vs Gate Reliability Scatter)...")
fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.3), sharey=True)
rng = np.random.default_rng(42)
SCAT4 = [
    ("netvlad",     "(a) NetVLAD (Era 1)"),
    ("mixvpr",      "(b) MixVPR (Era 2)"),
    ("megaloc",     "(c) MegaLoc (Era 3)"),
    ("clip_vitl14", "(d) CLIP-ViT-L/14 (Era 4)"),
]
for idx, (key, title) in enumerate(SCAT4):
    ax = axes[idx]
    d = rep_data[key]
    # Subsample 1,600 points stratified for crisp vector plotting
    pos_idx = np.where(d["y"] == 1)[0]
    neg_idx = np.where(d["y"] == 0)[0]
    n_p = min(800, len(pos_idx))
    n_n = min(800, len(neg_idx))
    sp = rng.choice(pos_idx, n_p, replace=False)
    sn = rng.choice(neg_idx, n_n, replace=False)

    ax.scatter(d["s"][sn], d["r"][sn], s=6, alpha=0.32, color="#D95F59", edgecolors="none", label="Aliased FP", rasterized=True)
    ax.scatter(d["s"][sp], d["r"][sp], s=6, alpha=0.32, color="#2A9D8F", edgecolors="none", label="True Match", rasterized=True)
    ax.axhline(0.45, color="#1D3557", ls="--", lw=1.4, label="Gate τ = 0.45")
    med_s = float(np.median(d["s"]))
    ax.axvline(med_s, color="#666666", ls=":", lw=1.2)
    ax.text(med_s, 0.04, f" med={med_s:.2f}", fontsize=6.8, color="#333333", fontweight="bold")

    ax.set_title(title)
    ax.set_xlabel("Raw Top-1 Cosine Sim.")
    ax.set_xlim(max(-0.05, float(np.percentile(d["s"], 1)) - 0.04), min(1.04, float(np.percentile(d["s"], 99.5)) + 0.04))
    ax.set_ylim(-0.02, 1.04)
    if idx == 0:
        ax.set_ylabel("Conflict Gate Reliability r")
        ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, markerscale=1.8, fontsize=6.8)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "figS5.pdf"))
plt.close(fig)

# ==============================================================================
# FIG S6: 4-Panel ChronoVPR-VQA Dataset Distributions
# ==============================================================================
print("Generating figS6.pdf (Dataset Distributions)...")
fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.2))

# (a) Temporal gap |Delta t| histogram (years >= 1)
ax = axes[0, 0]
dt_vals = [abs(int(x["year_gap"])) for x in all_clean if x.get("year_gap") is not None and 1 <= abs(int(x["year_gap"])) <= 15]
bins_dt = np.arange(0.5, 16.5, 1.0)
ax.hist(dt_vals, bins=bins_dt, color="#2C7BB6", alpha=0.82, edgecolor="white", lw=0.6)
ax.axvline(np.mean(dt_vals), color="#C96A5B", ls="--", lw=1.6, label=f"Mean = {np.mean(dt_vals):.1f} yrs")
ax.set_title("(a) Multi-Year Temporal Gap |Δt| Distribution")
ax.set_xlabel("Temporal Displacement |Δt| (Years)")
ax.set_ylabel("Number of Verified Places")
ax.set_xticks(range(1, 16))
ax.legend(loc="upper right", frameon=True, facecolor="white")

# (b) Relative Yaw |Delta theta| on Out-of-FoV task
ax = axes[0, 1]
yaw_oof = [abs(float(x["relative_yaw_deg"])) for x in all_clean if x["task_type"] == "out_of_fov" and x.get("relative_yaw_deg") is not None]
ax.hist(yaw_oof, bins=np.linspace(0, 180, 19), color="#D99B43", alpha=0.85, edgecolor="white", lw=0.6)
ax.axvline(90, color="#1D3557", ls="--", lw=1.4, label="90° Corner Turning")
ax.axvline(180, color="#C96A5B", ls=":", lw=1.4, label="180° Reverse View")
ax.set_title("(b) Out-of-FoV Camera Rotation |Δθ| Distribution")
ax.set_xlabel("Absolute Relative Heading |Δθ| (Degrees)")
ax.set_ylabel("Number of Out-of-FoV Places")
ax.legend(loc="upper left", frameon=True, facecolor="white")

# (c) Option A/B/C/D balance across splits
ax = axes[1, 0]
opts = ["A", "B", "C", "D"]
splits = ["Train", "Val", "Test"]
sp_cols = ["#2C7BB6", "#2A9D8F", "#E9A03B"]
w = 0.24
xo = np.arange(4)
for si, sp in enumerate(splits):
    tot = stats["split_stats"][sp]["total"]
    pcts = [100.0 * stats["split_stats"][sp]["ans_counts"].get(o, 0) / tot for o in opts]
    ax.bar(xo + (si - 1) * w, pcts, width=w, color=sp_cols[si], alpha=0.88, edgecolor="white", label=f"{sp} (N={tot:,})")
ax.axhline(25.0, color="#C96A5B", ls="--", lw=1.3, label="Ideal Uniform (25.0%)")
ax.set_xticks(xo)
ax.set_xticklabels([f"Option {o}" for o in opts])
ax.set_ylim(0, 33)
ax.set_ylabel("Option Proportion (%)")
ax.set_title("(c) Balanced 4-Way Answer Distribution Across Splits")
ax.legend(loc="upper right", ncol=2, frameon=True, facecolor="white", fontsize=6.8)

# (d) Question Token Length by Task
ax = axes[1, 1]
t_meta = [
    ("occlusion_recovery", "Task A: Occlusion", "#C96A5B"),
    ("out_of_fov",         "Task B: Out-of-FoV", "#2C7BB6"),
    ("temporal_evolution", "Task C: Temporal", "#2A9D8F"),
]
bins_q = np.arange(5, 31, 1)
for tk, lbl, col in t_meta:
    ql = [len(x["question"].split()) for x in all_clean if x["task_type"] == tk]
    ax.hist(ql, bins=bins_q, density=True, alpha=0.45, color=col, edgecolor=col, histtype="stepfilled", lw=1.2,
            label=f"{lbl} (μ={np.mean(ql):.1f} w)")
ax.set_xlabel("Question Length (Word Tokens)")
ax.set_ylabel("Probability Density")
ax.set_title("(d) Question Linguistic Length Across Tasks A/B/C")
ax.legend(loc="upper right", frameon=True, facecolor="white", fontsize=7.0)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "figS6.pdf"))
plt.close(fig)

# ==============================================================================
# FIG S7: 2-Panel Longitudinal Temporal Degradation & Yaw Horizon Paradox
# ==============================================================================
print("Generating figS7.pdf (Temporal & Yaw Horizon Analysis)...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.9))

# (a) Temporal bins across 4 eras
tbins = ["0-2 yrs", "3-5 yrs", "6-9 yrs", "10+ yrs"]
xt = np.arange(len(tbins))
for era in ["Era 1", "Era 2", "Era 3", "Era 4"]:
    vals = [stats["temp_summary"][era][b]["r1"] for b in tbins]
    ax1.plot(xt, vals, "o-", color=ERA_COLORS[era], lw=2.2, ms=6.0, label=f"{era} Mean R@1")

# Overlay Era 1 & Era 3 Conflict Gate AUROC on twin axis
ax1_r = ax1.twinx()
ax1_r.spines["top"].set_visible(False)
auc_e1 = [stats["temp_summary"]["Era 1"][b]["auroc"] for b in tbins]
auc_e3 = [stats["temp_summary"]["Era 3"][b]["auroc"] for b in tbins]
ax1_r.plot(xt, auc_e1, "s--", color="#8A2D22", lw=1.5, ms=5.0, alpha=0.85, label="Era 1 Gate AUROC (right)")
ax1_r.plot(xt, auc_e3, "^--", color="#163F66", lw=1.5, ms=5.0, alpha=0.85, label="Era 3 Gate AUROC (right)")
ax1.set_xticks(xt)
ax1.set_xticklabels(tbins)
ax1.set_xlabel("Longitudinal Temporal Span |Δt| (Urban Datasets)")
ax1.set_ylabel("Mean VPR Recall@1 (%)")
ax1_r.set_ylabel("Conflict Gate AUROC", color="#163F66")
ax1.set_ylim(0, 95)
ax1_r.set_ylim(0.50, 0.95)
ax1.set_title("(a) Retrieval & Gating Across Multi-Year Spans")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax1_r.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="lower left", ncol=2, frameon=True, facecolor="white", framealpha=0.92, fontsize=6.5)

# (b) Yaw regimes: Naive Exact Match vs CTSE Exact Complementary Match (Era 1..4)
ybins = ["0 deg (Same)", "15-60 deg", "60-135 deg", "135-180 deg"]
ydisp = ["0°\n(Same Heading)", "15°–60°\n(Oblique)", "60°–135°\n(±90° Corner)", "135°–180°\n(Opposite)"]
xy = np.arange(len(ybins))
w = 0.18
# Show Era 3 (SOTA) & Era 2 & Era 1 Naive vs CTSE
naive_e3 = [stats["yaw_summary"]["Era 3"][b]["naive_exact"] for b in ybins]
ctse_e3  = [stats["yaw_summary"]["Era 3"][b]["ctse_exact"]  for b in ybins]
ctse_e2  = [stats["yaw_summary"]["Era 2"][b]["ctse_exact"]  for b in ybins]
ctse_e1  = [stats["yaw_summary"]["Era 1"][b]["ctse_exact"]  for b in ybins]

ax2.bar(xy - 1.5*w, naive_e3, width=w, color="#C96A5B", alpha=0.85, edgecolor="white", label="Naive Top-1 (Era 3 SOTA)")
ax2.bar(xy - 0.5*w, ctse_e1,  width=w, color="#E9A03B", alpha=0.85, edgecolor="white", label="CTSE (Era 1 Early CNN)")
ax2.bar(xy + 0.5*w, ctse_e2,  width=w, color="#2A9D8F", alpha=0.85, edgecolor="white", label="CTSE (Era 2 Large CNN)")
ax2.bar(xy + 1.5*w, ctse_e3,  width=w, color="#2C7BB6", alpha=0.90, edgecolor="white", label="CTSE (Era 3 VFM/DINOv2)")

for i in [1, 2, 3]:
    ax2.text(xy[i] - 1.5*w, naive_e3[i] + 1.5, f"{naive_e3[i]:.2f}%", ha="center", va="bottom", fontsize=6.2, color="#9E2A2B", fontweight="bold")
    ax2.text(xy[i] + 1.5*w, ctse_e3[i] + 1.5, f"{ctse_e3[i]:.1f}%", ha="center", va="bottom", fontsize=6.4, color="#1D4E89", fontweight="bold")

ax2.set_xticks(xy)
ax2.set_xticklabels(ydisp)
ax2.set_ylabel("Exact Target-View Recovery (%)")
ax2.set_ylim(0, 112)
ax2.set_title("(b) Similarity–Complementarity Paradox by Camera Yaw")
ax2.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, fontsize=6.8)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "figS7.pdf"))
plt.close(fig)

print("ALL 7 QUANTITATIVE APPENDIX FIGURES GENERATED IN:", OUT_DIR)
