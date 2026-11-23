import os, json, math
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from scipy.stats import spearmanr, kendalltau, pearsonr

BASE = os.environ.get("CHRONOVPR_ROOT", ".")
BAK  = os.path.join(BASE, "final_experiment_backup")
VPR_DIR = os.path.join(BAK, "vpr_retrievals")
CLEAN_DIR = os.path.join(BASE, "ChronoVPR-VQA-Clean")

METHODS = [
    ("Era 1", "resnet50_avg",  "ResNet50-Avg"),
    ("Era 1", "resnet50_gem",  "ResNet50-GeM"),
    ("Era 1", "netvlad",       "NetVLAD"),
    ("Era 1", "patch_netvlad", "Patch-NetVLAD"),
    ("Era 2", "cosplace",      "CosPlace"),
    ("Era 2", "eigenplaces",   "EigenPlaces"),
    ("Era 2", "convap",        "Conv-AP"),
    ("Era 2", "mixvpr",        "MixVPR"),
    ("Era 3", "dinov2_vitb14", "DINOv2-ViT-B/14"),
    ("Era 3", "anyloc",        "AnyLoc"),
    ("Era 3", "selavpr",       "SelaVPR"),
    ("Era 3", "cricavpr",      "CricaVPR"),
    ("Era 3", "dinov2_salad",  "DINOv2-SALAD"),
    ("Era 3", "boq_dinov2",    "BoQ-DINOv2"),
    ("Era 3", "megaloc",       "MegaLoc"),
    ("Era 4", "clip_vitb32",   "CLIP-ViT-B/32"),
    ("Era 4", "clip_vitl14",   "CLIP-ViT-L/14"),
]

BENCHMARKS = ["GSV-Cities", "SF-XL", "Pittsburgh250K", "MSLS", "Nordland", "SPED"]
TASKS = ["occlusion_recovery", "out_of_fov", "temporal_evolution"]
THRESHOLDS = [0.25, 0.35, 0.45, 0.55, 0.65]

print("Loading ChronoVPR-VQA-Clean (all 30,113 samples)...")
all_data = json.load(open(os.path.join(CLEAN_DIR, "chrono_vpr_vqa_all.json"), encoding="utf-8"))
train_data = json.load(open(os.path.join(CLEAN_DIR, "chrono_vpr_vqa_train.json"), encoding="utf-8"))
val_data = json.load(open(os.path.join(CLEAN_DIR, "chrono_vpr_vqa_val.json"), encoding="utf-8"))
test_data = json.load(open(os.path.join(CLEAN_DIR, "chrono_vpr_vqa_test.json"), encoding="utf-8"))
test_by_id = {x["sample_id"]: x for x in test_data}

final_res = json.load(open(os.path.join(BAK, "final_paper_results.json"), encoding="utf-8"))
tpami_stats = json.load(open(os.path.join(BASE, "tpami_all_tables_stats.json"), encoding="utf-8"))

# ==============================================================================
# 1. Dataset Statistics (Table S8, Table S9, Fig S6)
# ==============================================================================
print("Computing dataset distributions...")
split_stats = {}
for sname, sdata in [("Train", train_data), ("Val", val_data), ("Test", test_data), ("All", all_data)]:
    ans_cnt = Counter(x.get("answer", x.get("correct_option", "")) for x in sdata)
    task_cnt = Counter(x["task_type"] for x in sdata)
    q_lens = [len(x["question"].split()) for x in sdata]
    split_stats[sname] = {
        "total": len(sdata),
        "ans_counts": dict(ans_cnt),
        "task_counts": dict(task_cnt),
        "q_len_mean": float(np.mean(q_lens)),
        "q_len_std": float(np.std(q_lens)),
        "q_len_min": int(np.min(q_lens)),
        "q_len_max": int(np.max(q_lens)),
    }

# City breakdown for GSV-Cities & across all benchmarks
gsv_cities = Counter()
for x in all_data:
    if x["dataset"] == "GSV-Cities":
        city = str(x.get("city", "")).strip()
        if not city:
            pid = str(x.get("place_id", ""))
            city = pid.split("_")[0] if "_" in pid else pid
        gsv_cities[city] += 1

# Temporal delta_t and angular delta_theta distributions
delta_t_all = []
delta_theta_all = []
task_q_lens = defaultdict(list)
for x in all_data:
    task_q_lens[x["task_type"]].append(len(x["question"].split()))
    yg = x.get("year_gap")
    if yg is not None:
        try:
            dt = abs(int(yg))
            if 0 <= dt <= 25:
                delta_t_all.append(dt)
        except Exception:
            pass
    ry = x.get("relative_yaw_deg")
    if ry is not None:
        try:
            delta_theta_all.append(float(ry))
        except Exception:
            pass

print(f"Collected {len(delta_t_all)} temporal gaps, {len(delta_theta_all)} yaw diffs")

# ==============================================================================
# 2. Full 17-Method Per-Benchmark R@5/R@10, Per-Task Gate, Per-Benchmark Gate,
#    Threshold Sweep, AUPRC, Focus Quadrant, Temporal/Yaw Stratification
# ==============================================================================
r5_r10_table = {}         # Table S3: 17 x 6 R@5 & R@10
task_gate_table = {}      # Table S4: 17 x 3 Task Gate AUROC/Prec/TPR/TNR
bm_gate_table = {}        # Table S5: 17 x 6 Benchmark Gate AUROC & dPrec
thresh_table = {}         # Table S6: 17 x 5 thresholds (TPR, TNR, F1, PassRate, Prec)
auprc_by_method = {}      # Fig S4
gate_dist_samples = {}    # Fig S3 & S5 (downsample or keep histograms/arrays)
temp_bin_stats = defaultdict(lambda: defaultdict(lambda: {"hits": [], "y": [], "r": []}))
yaw_bin_stats  = defaultdict(lambda: defaultdict(lambda: {"naive_exact": [], "ctse_exact": [], "r1": []}))
focus_quad_stats = {"by_task": defaultdict(Counter), "by_bm": defaultdict(Counter), "r_by_quad": defaultdict(list)}
megaloc_error_decomp = {}

# Load all 17 retrieval logs
all_vpr_logs = {}
for era, key, disp in METHODS:
    fp = os.path.join(VPR_DIR, f"{key}_test_retrievals_gated.json")
    rows = json.load(open(fp, encoding="utf-8"))
    all_vpr_logs[key] = rows

    # R@5 & R@10 per benchmark
    bm_r = {}
    for bm in BENCHMARKS:
        sub = [r for r in rows if r["dataset"] == bm]
        bm_r[bm] = {
            "r1": 100.0 * np.mean([r["naive_hit1"] for r in sub]),
            "r5": 100.0 * np.mean([r["hit5"] for r in sub]),
            "r10": 100.0 * np.mean([r["hit10"] for r in sub]),
        }
    r5_r10_table[key] = bm_r

    # Overall arrays
    y_all = np.array([int(r["naive_hit1"]) for r in rows])
    p_all = np.array([float(r["gate_reliability"]) for r in rows])
    s_all = np.array([float(r["naive_top1_sim"]) for r in rows])
    auprc_by_method[key] = {
        "auroc": float(roc_auc_score(y_all, p_all)),
        "auprc": float(average_precision_score(y_all, p_all)),
        "neg_auprc": float(average_precision_score(1 - y_all, 1.0 - p_all)),
    }

    # Per-Task Gate Table (Table S4)
    tg = {}
    for tk in TASKS:
        sub = [r for r in rows if r["task_type"] == tk]
        yt = np.array([int(r["naive_hit1"]) for r in sub])
        pt = np.array([float(r["gate_reliability"]) for r in sub])
        auroc = float(roc_auc_score(yt, pt)) if len(np.unique(yt)) > 1 else 0.5
        passed = pt >= 0.45
        prec = 100.0 * float(np.mean(yt[passed])) if np.sum(passed) > 0 else 0.0
        tpr  = 100.0 * float(np.mean(passed[yt == 1])) if np.sum(yt == 1) > 0 else 0.0
        tnr  = 100.0 * float(np.mean((~passed)[yt == 0])) if np.sum(yt == 0) > 0 else 0.0
        tg[tk] = {"auroc": auroc, "prec": prec, "r1": 100.0 * float(np.mean(yt)), "tpr": tpr, "tnr": tnr}
    task_gate_table[key] = tg

    # Per-Benchmark Gate Table (Table S5)
    bg = {}
    for bm in BENCHMARKS:
        sub = [r for r in rows if r["dataset"] == bm]
        yb = np.array([int(r["naive_hit1"]) for r in sub])
        pb = np.array([float(r["gate_reliability"]) for r in sub])
        auroc = float(roc_auc_score(yb, pb)) if len(np.unique(yb)) > 1 else 0.5
        passed = pb >= 0.45
        prec = 100.0 * float(np.mean(yb[passed])) if np.sum(passed) > 0 else 0.0
        raw_r1 = 100.0 * float(np.mean(yb))
        bg[bm] = {"auroc": auroc, "prec": prec, "raw_r1": raw_r1, "dprec": prec - raw_r1}
    bm_gate_table[key] = bg

    # Threshold Sweep (Table S6)
    ts = {}
    for tau in THRESHOLDS:
        passed = p_all >= tau
        tpr = 100.0 * float(np.mean(passed[y_all == 1]))
        tnr = 100.0 * float(np.mean((~passed)[y_all == 0]))
        f1  = 100.0 * float(f1_score(y_all, passed.astype(int)))
        pr  = 100.0 * float(np.mean(passed))
        prec = 100.0 * float(np.mean(y_all[passed])) if np.sum(passed) > 0 else 0.0
        ts[f"{tau:.2f}"] = {"tpr": tpr, "tnr": tnr, "f1": f1, "pass_rate": pr, "prec": prec}
    thresh_table[key] = ts

    # Stratification by temporal gap & yaw (for representative era tracking)
    for r in rows:
        sid = r["sample_id"]
        meta_row = test_by_id.get(sid, {})
        # Temporal gap
        dt = meta_row.get("year_gap")
        if dt is not None and r["dataset"] in ("GSV-Cities", "SF-XL", "MSLS"):
            dtv = abs(int(dt))
            if dtv <= 2: b = "0-2 yrs"
            elif dtv <= 5: b = "3-5 yrs"
            elif dtv <= 9: b = "6-9 yrs"
            else: b = "10+ yrs"
            temp_bin_stats[era][b]["hits"].append(int(r["naive_hit1"]))
            temp_bin_stats[era][b]["y"].append(int(r["naive_hit1"]))
            temp_bin_stats[era][b]["r"].append(float(r["gate_reliability"]))

        # Yaw bin (Task B out-of-FoV + Task A/C 0-deg)
        ry = abs(float(meta_row.get("relative_yaw_deg", r.get("ctse_rel_yaw", 0))))
        if r["task_type"] == "out_of_fov":
            if ry < 60: yb = "15-60 deg"
            elif ry < 135: yb = "60-135 deg"
            else: yb = "135-180 deg"
        else:
            yb = "0 deg (Same)"
        gt_mem_img = meta_row.get("oracle_memory_img", "")
        naive_exact = 1 if (gt_mem_img and r["naive_top1_img"] == gt_mem_img) else 0
        ctse_exact  = 1 if (gt_mem_img and r["ctse_img"] == gt_mem_img) else 0
        yaw_bin_stats[era][yb]["naive_exact"].append(naive_exact)
        yaw_bin_stats[era][yb]["ctse_exact"].append(ctse_exact)
        yaw_bin_stats[era][yb]["r1"].append(int(r["naive_hit1"]))

    # Focus quadrant stats (using MegaLoc as canonical SOTA)
    if key == "megaloc":
        for r in rows:
            q = r.get("gate_focus_quadrant", "center")
            focus_quad_stats["by_task"][r["task_type"]][q] += 1
            focus_quad_stats["by_bm"][r["dataset"]][q] += 1
            focus_quad_stats["r_by_quad"][q].append(float(r["gate_reliability"]))

        # Error decomposition for MegaLoc (Table S10)
        tp = int(np.sum((y_all == 1) & (p_all >= 0.45)))
        fn = int(np.sum((y_all == 1) & (p_all <  0.45)))
        fp = int(np.sum((y_all == 0) & (p_all >= 0.45)))
        tn = int(np.sum((y_all == 0) & (p_all <  0.45)))
        megaloc_error_decomp = {
            "N": len(rows),
            "TP_verified_correct": tp,
            "FN_gate_false_reject": fn,
            "FP_gate_false_accept": fp,
            "TN_gate_true_reject": tn,
            "by_task": {},
        }
        for tk in TASKS:
            sub = [r for r in rows if r["task_type"] == tk]
            yt = np.array([int(r["naive_hit1"]) for r in sub])
            pt = np.array([float(r["gate_reliability"]) for r in sub])
            megaloc_error_decomp["by_task"][tk] = {
                "N": len(sub),
                "TP": int(np.sum((yt == 1) & (pt >= 0.45))),
                "FN": int(np.sum((yt == 1) & (pt <  0.45))),
                "FP": int(np.sum((yt == 0) & (pt >= 0.45))),
                "TN": int(np.sum((yt == 0) & (pt <  0.45))),
            }

# Rank-order correlation (Table S11)
r1_vec = []
naive_vqa_vec = []
ours_vqa_vec = []
gate_auc_vec = []
t2 = final_res["table2_all_vpr"]
for era, key, disp in METHODS:
    rows = all_vpr_logs[key]
    r1_vec.append(100.0 * np.mean([r["naive_hit1"] for r in rows]))
    naive_vqa_vec.append(float(t2[key]["Naive_VQA_Acc"]))
    ours_vqa_vec.append(float(t2[key]["Ours_VQA_Acc"]))
    gate_auc_vec.append(auprc_by_method[key]["auroc"])

rank_corr = {
    "r1_vs_naive_spearman": float(spearmanr(r1_vec, naive_vqa_vec).statistic),
    "r1_vs_ours_spearman":  float(spearmanr(r1_vec, ours_vqa_vec).statistic),
    "r1_vs_naive_pearson":  float(pearsonr(r1_vec, naive_vqa_vec).statistic),
    "r1_vs_ours_pearson":   float(pearsonr(r1_vec, ours_vqa_vec).statistic),
    "r1_vs_auroc_pearson":  float(pearsonr(r1_vec, gate_auc_vec).statistic),
    "r1_vs_auroc_spearman": float(spearmanr(r1_vec, gate_auc_vec).statistic),
    "r1_spread":            float(max(r1_vec) - min(r1_vec)),
    "naive_spread":         float(max(naive_vqa_vec) - min(naive_vqa_vec)),
    "ours_spread":          float(max(ours_vqa_vec) - min(ours_vqa_vec)),
}

# Summarize temporal & yaw bins
temp_summary = {}
for era, bins in temp_bin_stats.items():
    temp_summary[era] = {}
    for b, d in bins.items():
        y = np.array(d["y"]); r = np.array(d["r"])
        temp_summary[era][b] = {
            "count": len(y),
            "r1": 100.0 * float(np.mean(y)),
            "auroc": float(roc_auc_score(y, r)) if len(np.unique(y)) > 1 else 0.5,
        }

yaw_summary = {}
for era, bins in yaw_bin_stats.items():
    yaw_summary[era] = {}
    for b, d in bins.items():
        yaw_summary[era][b] = {
            "count": len(d["r1"]),
            "r1": 100.0 * float(np.mean(d["r1"])),
            "naive_exact": 100.0 * float(np.mean(d["naive_exact"])),
            "ctse_exact":  100.0 * float(np.mean(d["ctse_exact"])),
        }

out = {
    "split_stats": split_stats,
    "gsv_top_cities": gsv_cities.most_common(15),
    "gsv_num_cities": len(gsv_cities),
    "delta_t_hist": dict(Counter(delta_t_all)),
    "delta_theta_count": len(delta_theta_all),
    "task_q_len_stats": {
        tk: {"mean": float(np.mean(v)), "std": float(np.std(v)), "median": float(np.median(v))}
        for tk, v in task_q_lens.items()
    },
    "r5_r10_table": r5_r10_table,
    "task_gate_table": task_gate_table,
    "bm_gate_table": bm_gate_table,
    "thresh_table": thresh_table,
    "auprc_by_method": auprc_by_method,
    "temp_summary": temp_summary,
    "yaw_summary": yaw_summary,
    "focus_quad_stats": {
        "by_task": {k: dict(v) for k, v in focus_quad_stats["by_task"].items()},
        "by_bm":   {k: dict(v) for k, v in focus_quad_stats["by_bm"].items()},
        "r_mean_by_quad": {k: float(np.mean(v)) for k, v in focus_quad_stats["r_by_quad"].items()},
        "count_by_quad":  {k: len(v) for k, v in focus_quad_stats["r_by_quad"].items()},
    },
    "megaloc_error_decomp": megaloc_error_decomp,
    "rank_corr": rank_corr,
}

out_path = os.path.join(BASE, "appendix_computed_stats.json")
json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2)
print("SAVED:", out_path)
print("Rank correlations:", json.dumps(rank_corr, indent=2))
print("MegaLoc error decomp:", json.dumps(megaloc_error_decomp, indent=2))
print("GSV cities count:", len(gsv_cities), "Top 5:", gsv_cities.most_common(5))
print("Temporal bins (Era 3):", json.dumps(temp_summary.get("Era 3", {}), indent=2))
print("Yaw bins (Era 3):", json.dumps(yaw_summary.get("Era 3", {}), indent=2))
print("Focus quadrants:", json.dumps(out["focus_quad_stats"]["count_by_quad"]))
