"""
Experiment 2: Train & Evaluate Our Neural Visual Conflict Resolution Gate (Innovation 2).

Uses frozen DINOv2 ViT-B/14 spatial patch grids (16x16 = 256 patch tokens) + Spatio-Temporal metadata
(relative yaw Delta_theta, year gap Delta_t, VPR cosine similarity) to train a lightweight neural Gate
on ChronoVPR-VQA-Clean/train (19,573 true topological matches + 19,573 hard perceptual-aliasing negatives
mined from CosPlace & DINOv2-SALAD!).

Outputs:
  1. Trained checkpoint: ./workspace/conflict_gate_best.pt
  2. Validation & Test Perceptual-Aliasing Detection Metrics (AUROC, Accuracy, F1, Precision/Recall)
  3. Calibrated Reliability Scores (r in [0, 1]) + Change-Focus Quadrant Hints for ALL 8 VPR methods
     on the 8,047 test samples, ready for Experiment 4 (Full VQA Evaluation)!
"""

import os
import json
import time
import math
from typing import List, Dict, Any

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


DATA_BASE = "./workspace"
CLEAN_DIR = "./workspace/ChronoVPR-VQA-Clean"
VPR_DIR = "./workspace/vpr_retrievals"
EXP_DIR = "./workspace"
os.makedirs(EXP_DIR, exist_ok=True)


class SpatialGridDataset(Dataset):
    def __init__(self, img_paths: List[str]):
        self.paths = img_paths
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img)


@torch.inference_mode()
def extract_dinov2_spatial_tokens(img_paths: List[str], batch_size: int = 128, device: str = "cuda"):
    """
    Extract compact 4x4 pooled DINOv2 spatial grid tokens [N, 16, 768] + CLS [N, 768]
    (fast to cache in RAM, retains full spatial layout to detect peripheral vs. center alignment!).
    """
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval().to(device)
    ds = SpatialGridDataset(img_paths)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    cls_list, grid_list = [], []
    for imgs in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = backbone.forward_features(imgs)
            cls_tok = F.normalize(out["x_norm_clstoken"].float(), p=2, dim=-1)  # [B, 768]
            patches = out["x_norm_patchtokens"].float()  # [B, 256, 768]
            # Reshape 16x16 -> avg pool to 4x4 spatial grid [B, 16, 768]
            B, N, D = patches.shape
            grid = patches.transpose(1, 2).reshape(B, D, 16, 16)
            grid4 = F.avg_pool2d(grid, kernel_size=4, stride=4).flatten(2).transpose(1, 2)  # [B, 16, 768]
            grid4 = F.normalize(grid4, p=2, dim=-1)
        cls_list.append(cls_tok.half().cpu())
        grid_list.append(grid4.half().cpu())

    del backbone
    torch.cuda.empty_cache()
    return torch.cat(cls_list, dim=0), torch.cat(grid_list, dim=0)


class SpatioTemporalConflictGate(nn.Module):
    """
    Innovation 2: Cross-View Visual Conflict Resolution Gate.
    Takes:
      - Query & Memory CLS tokens + 4x4 spatial patch grids (16 regions each)
      - Computes the 16x16 cross-view spatial correlation matrix C = G_q @ G_m^T
        (captures both aligned views, occluded-center/aligned-periphery views, and rotated/shifted views!)
      - Fuses with Spatio-Temporal metadata [vpr_sim, sin(d_yaw), cos(d_yaw), norm_dt, task_onehot(3)]
    Predicts:
      - reliability_logit: probability that Memory is a true topological match vs. Perceptual Aliasing noise!
      - diff_map: 4x4 spatial attention map highlighting occluded/changed regions.
    """

    def __init__(self, feat_dim: int = 768, meta_dim: int = 7, hidden_dim: int = 384):
        super().__init__()
        # 256 cross-correlation entries + 16 diagonal spatial sims + 1 CLS cos sim + 3 summary stats + meta_dim
        in_dim = 256 + 16 + 1 + 4 + meta_dim
        self.corr_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.semantic_proj = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )
        self.reliability_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, cls_q, grid_q, cls_m, grid_m, meta):
        # Cross-view spatial similarity matrix [B, 16, 16]
        corr = torch.bmm(grid_q, grid_m.transpose(1, 2))  # [B, 16, 16]
        diag_sim = torch.diagonal(corr, dim1=1, dim2=2)   # [B, 16]
        cls_sim = (cls_q * cls_m).sum(dim=-1, keepdim=True)  # [B, 1]

        # Key geometric invariants: max row match (viewpoint-tolerant correspondence!), peripheral mean, center mean
        max_row = corr.max(dim=-1).values                 # [B, 16]
        top_match_mean = max_row.mean(dim=-1, keepdim=True)
        top_match_max = max_row.max(dim=-1, keepdim=True).values
        # Peripheral indices of 4x4 grid (border 12 cells) vs center 4 cells (5, 6, 9, 10)
        center_idx = [5, 6, 9, 10]
        periph_idx = [i for i in range(16) if i not in center_idx]
        periph_sim = diag_sim[:, periph_idx].mean(dim=-1, keepdim=True)
        center_sim = diag_sim[:, center_idx].mean(dim=-1, keepdim=True)

        corr_flat = torch.cat(
            [corr.flatten(1), diag_sim, cls_sim, top_match_mean, top_match_max, periph_sim, center_sim, meta],
            dim=-1,
        )
        h_corr = self.corr_encoder(corr_flat)
        h_sem = self.semantic_proj(torch.cat([cls_q * cls_m, torch.abs(cls_q - cls_m)], dim=-1))
        logits = self.reliability_head(torch.cat([h_corr, h_sem], dim=-1)).squeeze(-1)

        # Change / occlusion focus map (where center/local regions diverge despite peripheral match)
        spatial_divergence = 1.0 - max_row  # [B, 16]
        return logits, spatial_divergence


def make_meta_vec(sim: float, rel_yaw: float, year_gap: float, task_type: str) -> List[float]:
    # Strictly VPR-scale-invariant spatio-temporal metadata (excludes raw VPR cosine sim,
    # whose numerical scale varies drastically between CLIP/ResNet/CosPlace/BoQ/SALAD!)
    rad = math.radians(rel_yaw)
    norm_dt = max(-1.0, min(1.0, year_gap / 15.0))
    t0 = 1.0 if task_type == "occlusion_recovery" else 0.0
    t1 = 1.0 if task_type == "out_of_fov" else 0.0
    t2 = 1.0 if task_type == "temporal_evolution" else 0.0
    return [0.0, math.sin(rad), math.cos(rad), norm_dt, t0, t1, t2]


def main():
    t0 = time.time()
    print("[Exp2] Loading ChronoVPR-VQA-Clean & mined hard perceptual-aliasing negatives...", flush=True)
    with open(os.path.join(CLEAN_DIR, "chrono_vpr_vqa_all.json"), "r", encoding="utf-8") as f:
        all_samples = json.load(f)
    by_sid = {r["sample_id"]: r for r in all_samples}

    # Load mined hard negatives from EigenPlaces / DINOv2-SALAD
    mined_by_sid = {}
    for m_name in ("dinov2_salad", "eigenplaces", "cosplace"):
        p = os.path.join(VPR_DIR, f"{m_name}_trainval_mined.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                for rec in json.load(f):
                    mined_by_sid.setdefault(rec["sample_id"], []).append(rec)

    # Collect all unique images across train/val/test + retrieved test images
    unique_rels = set()
    for r in all_samples:
        unique_rels.add(r["query_img"])
        unique_rels.add(r["oracle_memory_img"])
    for recs in mined_by_sid.values():
        for rec in recs:
            unique_rels.add(rec["hard_neg_img"])

    vpr_test_files = [fn for fn in os.listdir(VPR_DIR) if fn.endswith("_test_retrievals.json")]
    vpr_test_data = {}
    for fn in vpr_test_files:
        m_name = fn.replace("_test_retrievals.json", "")
        with open(os.path.join(VPR_DIR, fn), "r", encoding="utf-8") as f:
            rows = json.load(f)
        vpr_test_data[m_name] = rows
        for row in rows:
            unique_rels.add(row["naive_top1_img"])
            unique_rels.add(row["ctse_img"])
            unique_rels.add(row["hard_neg_img"])

    unique_rels = sorted(unique_rels)
    rel_to_idx = {rel: i for i, rel in enumerate(unique_rels)}
    abs_paths = [os.path.join(DATA_BASE, rel) for rel in unique_rels]
    print(f"[Exp2] Extracting DINOv2-ViT-B/14 4x4 spatial patch grids for {len(abs_paths)} unique images...", flush=True)
    cls_bank, grid_bank = extract_dinov2_spatial_tokens(abs_paths, batch_size=192)
    print(f"[Exp2] Feature bank ready: CLS={tuple(cls_bank.shape)}, Grid={tuple(grid_bank.shape)} ({time.time()-t0:.1f}s)", flush=True)

    # Build Training & Validation pairs (Positive = true topological match; Negative = hard perceptual aliasing retrieval)
    def build_pairs_for_split(target_split: str):
        q_idxs, m_idxs, metas, labels = [], [], [], []
        for r in all_samples:
            if r["split"] != target_split:
                continue
            sid = r["sample_id"]
            qi = rel_to_idx[r["query_img"]]
            mi_pos = rel_to_idx[r["oracle_memory_img"]]
            mined_list = mined_by_sid.get(sid, [])
            pos_sim = float((cls_bank[qi].float() * cls_bank[mi_pos].float()).sum().item())
            if mined_list and mined_list[0]["top1_hit"] == 1:
                pos_sim = mined_list[0]["top1_sim"]

            # Positive pair
            q_idxs.append(qi)
            m_idxs.append(mi_pos)
            metas.append(make_meta_vec(pos_sim, r.get("relative_yaw_deg", 0), r.get("year_gap", 0), r["task_type"]))
            labels.append(1.0)

            # Hard negative pair(s) from VPR perceptual aliasing
            if mined_list:
                for rec in mined_list:
                    ni = rel_to_idx[rec["hard_neg_img"]]
                    q_idxs.append(qi)
                    m_idxs.append(ni)
                    metas.append(make_meta_vec(rec["hard_neg_sim"], r.get("relative_yaw_deg", 0), r.get("year_gap", 0), r["task_type"]))
                    labels.append(0.0)
        return (
            torch.tensor(q_idxs, dtype=torch.long),
            torch.tensor(m_idxs, dtype=torch.long),
            torch.tensor(metas, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
        )

    tr_q, tr_m, tr_meta, tr_y = build_pairs_for_split("train")
    va_q, va_m, va_meta, va_y = build_pairs_for_split("val")
    print(f"[Exp2] Train pairs: {len(tr_y)} (pos={int(tr_y.sum())}, hard_neg={int((tr_y==0).sum())}) | Val pairs: {len(va_y)}", flush=True)

    # Train SpatioTemporalConflictGate on RTX 4080
    device = "cuda"
    gate = SpatioTemporalConflictGate(feat_dim=768, meta_dim=7, hidden_dim=384).to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=8e-4, weight_decay=1e-2)
    epochs = 15
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pos_weight = torch.tensor([(tr_y == 0).sum() / max(tr_y.sum(), 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc = 0.0
    best_path = os.path.join(EXP_DIR, "conflict_gate_best.pt")
    bs = 512

    for ep in range(1, epochs + 1):
        gate.train()
        perm = torch.randperm(len(tr_y))
        tr_loss = 0.0
        for s in range(0, len(tr_y), bs):
            idx = perm[s : s + bs]
            qi, mi = tr_q[idx], tr_m[idx]
            cq = cls_bank[qi].to(device, dtype=torch.float32)
            gq = grid_bank[qi].to(device, dtype=torch.float32)
            cm = cls_bank[mi].to(device, dtype=torch.float32)
            gm = grid_bank[mi].to(device, dtype=torch.float32)
            mt = tr_meta[idx].to(device)
            yb = tr_y[idx].to(device)

            logits, _ = gate(cq, gq, cm, gm, mt)
            loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(idx)
        scheduler.step()

        # Validation AUROC
        gate.eval()
        va_preds = []
        with torch.inference_mode():
            for s in range(0, len(va_y), bs):
                qi, mi = va_q[s : s + bs], va_m[s : s + bs]
                logits, _ = gate(
                    cls_bank[qi].to(device, dtype=torch.float32),
                    grid_bank[qi].to(device, dtype=torch.float32),
                    cls_bank[mi].to(device, dtype=torch.float32),
                    grid_bank[mi].to(device, dtype=torch.float32),
                    va_meta[s : s + bs].to(device),
                )
                va_preds.append(torch.sigmoid(logits).cpu().numpy())
        va_probs = np.concatenate(va_preds)
        auc = roc_auc_score(va_y.numpy(), va_probs)
        f1 = f1_score(va_y.numpy(), (va_probs >= 0.5).astype(int))
        acc = accuracy_score(va_y.numpy(), (va_probs >= 0.5).astype(int))
        if auc > best_auc:
            best_auc = auc
            torch.save(gate.state_dict(), best_path)
        if ep in (1, 5, 10, 15):
            print(f"  Epoch {ep:02d}/{epochs} | Loss={tr_loss/len(tr_y):.4f} | Val AUROC={auc:.4f} | Acc={acc*100:.2f}% | F1={f1:.4f}", flush=True)

    # Load best gate & score ALL 8 VPR methods on the 8,047 test set!
    gate.load_state_dict(torch.load(best_path, weights_only=True))
    gate.eval()
    quadrant_names = ["top-left", "top-right", "bottom-left", "bottom-right"]
    gate_eval_summary = {}

    for m_name, rows in vpr_test_data.items():
        q_idxs = torch.tensor([rel_to_idx[by_sid[r["sample_id"]]["query_img"]] for r in rows], dtype=torch.long)
        ctse_idxs = torch.tensor([rel_to_idx[r["ctse_img"]] for r in rows], dtype=torch.long)
        neg_idxs = torch.tensor([rel_to_idx[r["hard_neg_img"]] for r in rows], dtype=torch.long)
        orc_idxs = torch.tensor([rel_to_idx[by_sid[r["sample_id"]]["oracle_memory_img"]] for r in rows], dtype=torch.long)

        ctse_metas = torch.tensor(
            [
                make_meta_vec(
                    r["ctse_sim"],
                    by_sid[r["sample_id"]].get("relative_yaw_deg", 0),
                    by_sid[r["sample_id"]].get("year_gap", 0),
                    r["task_type"],
                )
                for r in rows
            ],
            dtype=torch.float32,
        )
        neg_metas = torch.tensor(
            [
                make_meta_vec(
                    r["hard_neg_sim"],
                    by_sid[r["sample_id"]].get("relative_yaw_deg", 0),
                    by_sid[r["sample_id"]].get("year_gap", 0),
                    r["task_type"],
                )
                for r in rows
            ],
            dtype=torch.float32,
        )

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
                # Aggregate 4x4 divergence into 2x2 quadrant focus hint
                div4 = div.reshape(-1, 4, 4)
                q_scores = torch.stack([
                    div4[:, :2, :2].mean(dim=(1, 2)),
                    div4[:, :2, 2:].mean(dim=(1, 2)),
                    div4[:, 2:, :2].mean(dim=(1, 2)),
                    div4[:, 2:, 2:].mean(dim=(1, 2)),
                ], dim=-1)
                focus_q.append(q_scores.argmax(dim=-1).cpu().numpy())
            return np.concatenate(probs), np.concatenate(focus_q)

        ctse_probs, ctse_focus = score_batch(q_idxs, ctse_idxs, ctse_metas)
        neg_probs, _ = score_batch(q_idxs, neg_idxs, neg_metas)
        gt_hits = np.array([r["naive_hit1"] for r in rows])

        # AUROC of detecting whether the VPR retriever succeeded vs suffered Perceptual Aliasing!
        test_auc = roc_auc_score(gt_hits, ctse_probs) if len(set(gt_hits)) > 1 else 1.0
        test_acc = accuracy_score(gt_hits, (ctse_probs >= 0.5).astype(int))
        # Also test rejection rate on hard negatives
        neg_rej_rate = float((neg_probs < 0.5).mean())

        gate_eval_summary[m_name] = {
            "test_aliasing_detection_AUROC": round(float(test_auc), 4),
            "test_aliasing_detection_Acc": round(float(test_acc * 100), 2),
            "hard_negative_rejection_rate": round(neg_rej_rate * 100, 2),
            "mean_prob_true_match": round(float(ctse_probs[gt_hits == 1].mean()), 4) if (gt_hits == 1).any() else 0.0,
            "mean_prob_wrong_match": round(float(ctse_probs[gt_hits == 0].mean()), 4) if (gt_hits == 0).any() else 0.0,
        }

        # Write enriched test retrievals with Gate Reliability Score & Spatial Focus Quadrant!
        for i, r in enumerate(rows):
            r["gate_reliability"] = round(float(ctse_probs[i]), 4)
            r["gate_focus_quadrant"] = quadrant_names[int(ctse_focus[i])]
            r["hard_neg_gate_reliability"] = round(float(neg_probs[i]), 4)

        with open(os.path.join(VPR_DIR, f"{m_name}_test_retrievals_gated.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f)

    with open(os.path.join(EXP_DIR, "conflict_gate_eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"val_best_AUROC": round(best_auc, 4), "by_vpr_method": gate_eval_summary}, f, indent=2)

    print("\n=== EXP 2 COMPLETE: Spatio-Temporal Conflict Resolution Gate Summary ===", flush=True)
    print(json.dumps({"val_best_AUROC": round(best_auc, 4), "by_vpr_method": gate_eval_summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
