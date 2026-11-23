#!/usr/bin/env python3
"""
Compute 100% authentic, exact per-sample analytics across all 17 VPR methods x 8,047 test samples
for all 10 IEEE TPAMI tables and multi-panel analytical figures.
"""

import os
import json
import numpy as np
from collections import defaultdict, Counter
from sklearn.metrics import roc_curve, auc, roc_auc_score

BASE = os.environ.get("CHRONOVPR_ROOT", ".")
BAK = os.path.join(BASE, "final_experiment_backup")
VPR_DIR = os.path.join(BAK, "vpr_retrievals")
CLEAN_DIR = os.path.join(BASE, "ChronoVPR-VQA-Clean")

with open(os.path.join(BAK, "final_paper_results.json"), "r", encoding="utf-8") as f:
    final_res = json.load(f)
with open(os.path.join(VPR_DIR, "vpr_methods_recall_summary.json"), "r", encoding="utf-8") as f:
    rec_sum = json.load(f)
with open(os.path.join(BAK, "conflict_gate_eval_summary.json"), "r", encoding="utf-8") as f:
    gate_sum = json.load(f)
with open(os.path.join(CLEAN_DIR, "chrono_vpr_vqa_test.json"), "r", encoding="utf-8") as f:
    test_samples = json.load(f)

by_sid = {r["sample_id"]: r for r in test_samples}

METHODS_ORDERED = [
    "resnet50_avg", "resnet50_gem", "netvlad", "patch_netvlad",
    "cosplace", "eigenplaces", "convap", "mixvpr",
    "dinov2_vitb14", "anyloc", "selavpr", "cricavpr", "dinov2_salad", "boq_dinov2", "megaloc",
    "clip_vitb32", "clip_vitl14",
]

stats = {
    "gate_detailed": {},
    "sim_scale": {},
    "ctse_stats": {},
    "threshold_sweep": {},
    "intent_parser_acc": {},
}

# 1. Analyze all 17 gated files
for m in METHODS_ORDERED:
    with open(os.path.join(VPR_DIR, f"{m}_test_retrievals_gated.json"), "r", encoding="utf-8") as f:
        rows = json.load(f)

    hits = np.array([r["naive_hit1"] for r in rows], dtype=int)
    rels = np.array([r["gate_reliability"] for r in rows], dtype=float)
    hneg_rels = np.array([r["hard_neg_gate_reliability"] for r in rows], dtype=float)
    sims = np.array([r["naive_top1_sim"] for r in rows], dtype=float)

    # Gate operating stats at tau=0.45 (the exact threshold used in Exp 4!)
    tau = 0.45
    accepted = rels >= tau
    dropped = ~accepted
    drop_rate = float(dropped.mean() * 100.0)
    raw_prec = float(hits.mean() * 100.0)
    post_prec = float(hits[accepted].mean() * 100.0) if accepted.any() else 0.0
    tpr = float(accepted[hits == 1].mean() * 100.0) if (hits == 1).any() else 0.0
    tnr = float(dropped[hits == 0].mean() * 100.0) if (hits == 0).any() else 0.0
    mean_pos = float(rels[hits == 1].mean()) if (hits == 1).any() else 0.0
    mean_neg = float(rels[hits == 0].mean()) if (hits == 0).any() else 0.0

    # CTSE view switching stats (how often CTSE selects a different complementary image than naive top-1)
    ctse_changed_all = sum(1 for r in rows if r["ctse_img"] != r["naive_top1_img"])
    fov_rows = [r for r in rows if r["task_type"] == "out_of_fov"]
    fov_naive_exact = sum(1 for r in fov_rows if r["naive_top1_img"] == by_sid[r["sample_id"]]["oracle_memory_img"])
    fov_ctse_exact = sum(1 for r in fov_rows if r["ctse_img"] == by_sid[r["sample_id"]]["oracle_memory_img"])

    all_naive_exact = sum(1 for r in rows if r["naive_top1_img"] == by_sid[r["sample_id"]]["oracle_memory_img"])
    all_ctse_exact = sum(1 for r in rows if r["ctse_img"] == by_sid[r["sample_id"]]["oracle_memory_img"])

    stats["gate_detailed"][m] = {
        "auroc": gate_sum["by_vpr_method"][m]["test_aliasing_detection_AUROC"],
        "det_acc": gate_sum["by_vpr_method"][m]["test_aliasing_detection_Acc"],
        "hneg_rej": gate_sum["by_vpr_method"][m]["hard_negative_rejection_rate"],
        "mean_pos": round(mean_pos, 4),
        "mean_neg": round(mean_neg, 4),
        "margin": round(mean_pos - mean_neg, 4),
        "drop_rate_045": round(drop_rate, 2),
        "raw_prec": round(raw_prec, 2),
        "post_prec_045": round(post_prec, 2),
        "prec_lift": round(post_prec - raw_prec, 2),
        "tpr_045": round(tpr, 2),
        "tnr_045": round(tnr, 2),
    }

    stats["sim_scale"][m] = {
        "mean": round(float(sims.mean()), 4),
        "std": round(float(sims.std()), 4),
        "p10": round(float(np.percentile(sims, 10)), 4),
        "p50": round(float(np.percentile(sims, 50)), 4),
        "p90": round(float(np.percentile(sims, 90)), 4),
        "mean_hit": round(float(sims[hits == 1].mean()), 4) if (hits == 1).any() else 0.0,
        "mean_miss": round(float(sims[hits == 0].mean()), 4) if (hits == 0).any() else 0.0,
    }

    stats["ctse_stats"][m] = {
        "ctse_switched_pct": round(ctse_changed_all / len(rows) * 100.0, 2),
        "fov_naive_exact_pct": round(fov_naive_exact / len(fov_rows) * 100.0, 2),
        "fov_ctse_exact_pct": round(fov_ctse_exact / len(fov_rows) * 100.0, 2),
        "all_naive_exact_pct": round(all_naive_exact / len(rows) * 100.0, 2),
        "all_ctse_exact_pct": round(all_ctse_exact / len(rows) * 100.0, 2),
    }

    # Intent parser accuracy (same for all methods, compute on megaloc)
    if m == "megaloc":
        by_tk_int = defaultdict(lambda: [0, 0])
        for r in rows:
            tk = r["task_type"]
            by_tk_int[tk][1] += 1
            if r["parsed_intent"] == tk:
                by_tk_int[tk][0] += 1
        stats["intent_parser_acc"] = {
            tk: round(v[0] / v[1] * 100.0, 2) for tk, v in by_tk_int.items()
        }
        tot_ok = sum(v[0] for v in by_tk_int.values())
        tot_n = sum(v[1] for v in by_tk_int.values())
        stats["intent_parser_acc"]["overall"] = round(tot_ok / tot_n * 100.0, 2)

    # Threshold sweep for 4 representative methods across 4 eras
    if m in ("netvlad", "cosplace", "megaloc", "clip_vitb32"):
        sw = {}
        for t in (0.25, 0.35, 0.45, 0.55, 0.65, 0.75):
            acc_m = rels >= t
            sw[str(t)] = {
                "coverage": round(float(acc_m.mean() * 100.0), 2),
                "post_prec": round(float(hits[acc_m].mean() * 100.0), 2) if acc_m.any() else 0.0,
                "tpr": round(float(acc_m[hits == 1].mean() * 100.0), 2),
                "tnr": round(float((~acc_m)[hits == 0].mean() * 100.0), 2),
            }
        stats["threshold_sweep"][m] = sw

out_p = os.path.join(BASE, "tpami_all_tables_stats.json")
with open(out_p, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)

print("=== INTENT PARSER ACCURACY ===", stats["intent_parser_acc"])
print("\n=== GATE DETAILED SUMMARY (Selected) ===")
for m in METHODS_ORDERED:
    g = stats["gate_detailed"][m]
    c = stats["ctse_stats"][m]
    s = stats["sim_scale"][m]
    print(f"{m:14s} | SimMed={s['p50']:.3f} [{s['p10']:.2f},{s['p90']:.2f}] | GateAUC={g['auroc']:.4f} | Pos={g['mean_pos']:.3f} Neg={g['mean_neg']:.3f} d={g['margin']:.3f} | Drop={g['drop_rate_045']:5.1f}% | Prec {g['raw_prec']:5.1f}->{g['post_prec_045']:5.1f} (+{g['prec_lift']:4.1f}) | FoV Exact {c['fov_naive_exact_pct']:4.1f}->{c['fov_ctse_exact_pct']:5.1f}%")
print("\n=== THRESHOLD SWEEP ===")
for m, sw in stats["threshold_sweep"].items():
    print(m, sw)
