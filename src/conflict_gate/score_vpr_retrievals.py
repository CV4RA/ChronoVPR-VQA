#!/usr/bin/env python3
"""
Incremental Exp 2 Scoring:
Loads the ALREADY TRAINED frozen SpatioTemporalConflictGate (./workspace/conflict_gate_best.pt),
extracts DINOv2-ViT-B/14 4x4 spatial grids for any newly added VPR test retrieval files,
scores them identically to exp2_train_conflict_resolution_gate.py, writes {m}_test_retrievals_gated.json,
and merges their metrics into ./workspace/conflict_gate_eval_summary.json.
"""

import os
import json
import time
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import torch

os.environ["TORCH_HOME"] = "./workspace/torch_cache"

from src.conflict_gate.train_conflict_gate import (
    DATA_BASE,
    CLEAN_DIR,
    VPR_DIR,
    EXP_DIR,
    SpatioTemporalConflictGate,
    extract_dinov2_spatial_tokens,
    make_meta_vec,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="*", default=None, help="Specific methods to score (default: all ungated)")
    args = parser.parse_args()

    t0 = time.time()
    with open(os.path.join(CLEAN_DIR, "chrono_vpr_vqa_test.json"), "r", encoding="utf-8") as f:
        test_samples = json.load(f)
    by_sid = {r["sample_id"]: r for r in test_samples}

    # Find methods needing gating
    all_raw = sorted([fn.replace("_test_retrievals.json", "") for fn in os.listdir(VPR_DIR) if fn.endswith("_test_retrievals.json")])
    if args.methods:
        target_methods = [m for m in args.methods if m in all_raw]
    else:
        target_methods = [m for m in all_raw if not os.path.exists(os.path.join(VPR_DIR, f"{m}_test_retrievals_gated.json"))]

    if not target_methods:
        print("[Exp2-Inc] All VPR methods already gated!", flush=True)
        return

    print(f"[Exp2-Inc] Scoring {len(target_methods)} new VPR methods with frozen conflict_gate_best.pt: {target_methods}", flush=True)

    vpr_test_data = {}
    unique_rels = set()
    for r in test_samples:
        unique_rels.add(r["query_img"])
    for m in target_methods:
        with open(os.path.join(VPR_DIR, f"{m}_test_retrievals.json"), "r", encoding="utf-8") as f:
            rows = json.load(f)
        vpr_test_data[m] = rows
        for row in rows:
            unique_rels.add(row["naive_top1_img"])
            unique_rels.add(row["ctse_img"])
            unique_rels.add(row["hard_neg_img"])

    unique_rels = sorted(unique_rels)
    rel_to_idx = {rel: i for i, rel in enumerate(unique_rels)}
    abs_paths = [os.path.join(DATA_BASE, rel) for rel in unique_rels]
    print(f"[Exp2-Inc] Extracting DINOv2-ViT-B/14 4x4 grids for {len(abs_paths)} unique test/retrieved images...", flush=True)
    cls_bank, grid_bank = extract_dinov2_spatial_tokens(abs_paths, batch_size=192)
    print(f"[Exp2-Inc] Features extracted in {time.time() - t0:.1f}s!", flush=True)

    device = "cuda"
    gate = SpatioTemporalConflictGate(feat_dim=768, meta_dim=7, hidden_dim=384).to(device)
    gate.load_state_dict(torch.load(os.path.join(EXP_DIR, "conflict_gate_best.pt"), weights_only=True))
    gate.eval()

    summary_path = os.path.join(EXP_DIR, "conflict_gate_eval_summary.json")
    summary_obj = {"val_best_AUROC": 0.8737, "by_vpr_method": {}}
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_obj = json.load(f)

    quadrant_names = ["top-left", "top-right", "bottom-left", "bottom-right"]

    @torch.inference_mode()
    def score_batch(qi_all, mi_all, meta_all):
        probs, focus_q = [], []
        for s in range(0, len(qi_all), 512):
            qi, mi = qi_all[s : s + 512], mi_all[s : s + 512]
            logits, div = gate(
                cls_bank[qi].to(device, dtype=torch.float32),
                grid_bank[qi].to(device, dtype=torch.float32),
                cls_bank[mi].to(device, dtype=torch.float32),
                grid_bank[mi].to(device, dtype=torch.float32),
                meta_all[s : s + 512].to(device),
            )
            probs.append(torch.sigmoid(logits).cpu().numpy())
            div4 = div.reshape(-1, 4, 4)
            q_scores = torch.stack([
                div4[:, :2, :2].mean(dim=(1, 2)),
                div4[:, :2, 2:].mean(dim=(1, 2)),
                div4[:, 2:, :2].mean(dim=(1, 2)),
                div4[:, 2:, 2:].mean(dim=(1, 2)),
            ], dim=-1)
            focus_q.append(q_scores.argmax(dim=-1).cpu().numpy())
        return np.concatenate(probs), np.concatenate(focus_q)

    for m_name, rows in vpr_test_data.items():
        q_idxs = torch.tensor([rel_to_idx[by_sid[r["sample_id"]]["query_img"]] for r in rows], dtype=torch.long)
        ctse_idxs = torch.tensor([rel_to_idx[r["ctse_img"]] for r in rows], dtype=torch.long)
        neg_idxs = torch.tensor([rel_to_idx[r["hard_neg_img"]] for r in rows], dtype=torch.long)

        ctse_metas = torch.tensor([
            make_meta_vec(r["ctse_sim"], by_sid[r["sample_id"]].get("relative_yaw_deg", 0), by_sid[r["sample_id"]].get("year_gap", 0), r["task_type"])
            for r in rows
        ], dtype=torch.float32)
        neg_metas = torch.tensor([
            make_meta_vec(r["hard_neg_sim"], by_sid[r["sample_id"]].get("relative_yaw_deg", 0), by_sid[r["sample_id"]].get("year_gap", 0), r["task_type"])
            for r in rows
        ], dtype=torch.float32)

        ctse_probs, ctse_focus = score_batch(q_idxs, ctse_idxs, ctse_metas)
        neg_probs, _ = score_batch(q_idxs, neg_idxs, neg_metas)
        gt_hits = np.array([r["naive_hit1"] for r in rows])

        test_auc = roc_auc_score(gt_hits, ctse_probs) if len(set(gt_hits)) > 1 else 1.0
        test_acc = accuracy_score(gt_hits, (ctse_probs >= 0.5).astype(int))
        neg_rej_rate = float((neg_probs < 0.5).mean())

        summary_obj["by_vpr_method"][m_name] = {
            "test_aliasing_detection_AUROC": round(float(test_auc), 4),
            "test_aliasing_detection_Acc": round(float(test_acc * 100), 2),
            "hard_negative_rejection_rate": round(neg_rej_rate * 100, 2),
            "mean_prob_true_match": round(float(ctse_probs[gt_hits == 1].mean()), 4) if (gt_hits == 1).any() else 0.0,
            "mean_prob_wrong_match": round(float(ctse_probs[gt_hits == 0].mean()), 4) if (gt_hits == 0).any() else 0.0,
        }
        print(f"  [{m_name:15s}] Gate Test AUROC={test_auc:.4f} | Acc={test_acc*100:.2f}% | NegRej={neg_rej_rate*100:.2f}%", flush=True)

        for i, r in enumerate(rows):
            r["gate_reliability"] = round(float(ctse_probs[i]), 4)
            r["gate_focus_quadrant"] = quadrant_names[int(ctse_focus[i])]
            r["hard_neg_gate_reliability"] = round(float(neg_probs[i]), 4)

        with open(os.path.join(VPR_DIR, f"{m_name}_test_retrievals_gated.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_obj, f, indent=2)
    print(f"\n=== EXP 2 INCREMENTAL COMPLETE ({time.time()-t0:.1f}s) ===", flush=True)


if __name__ == "__main__":
    main()
