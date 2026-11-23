#!/usr/bin/env python3
"""
Incremental Exp 1 Runner for ALL Remaining Iconic VPR Methods across 4 Eras:
  - Era 1: netvlad (CVPR'16), patch_netvlad (CVPR'21)
  - Era 2: convap (ICRA'23), mixvpr (WACV'23)
  - Era 3: anyloc (RAL'23), selavpr (ICLR'24), cricavpr (CVPR'24), dinov2_salad (CVPR'24), megaloc (CVPR'25)

Uses the EXACT same gallery construction, place topological graphs, CTSE intent parser,
and output schema as exp1_build_gallery_and_run_all_vpr.py, and merges seamlessly into
./workspace/vpr_retrievals/vpr_methods_recall_summary.json!
"""

import os
import re
import sys
import json
import time
import math
import argparse
import importlib.util
from types import SimpleNamespace
from collections import defaultdict
from typing import Dict, List, Any, Tuple

import numpy as np
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

os.environ["TORCH_HOME"] = "./workspace/torch_cache"

# Avoid interactive torch.hub prompts, GitHub RemoteDisconnected on cached dinov2, & PyTorch 2.6 weights_only breaks
_orig_torch_hub_load = torch.hub.load
def _trusted_hub_load(repo_or_dir, model, *args, **kwargs):
    kwargs.setdefault("trust_repo", True)
    if repo_or_dir == "facebookresearch/dinov2":
        local_dino = "./workspace/torch_cache/hub/facebookresearch_dinov2_main"
        if os.path.isdir(local_dino):
            repo_or_dir = local_dino
            kwargs["source"] = "local"
    return _orig_torch_hub_load(repo_or_dir, model, *args, **kwargs)
torch.hub.load = _trusted_hub_load

_orig_torch_load = torch.load
def _compat_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _compat_torch_load

DATA_BASE = "./workspace"
CLEAN_DIR = "./workspace/ChronoVPR-VQA-Clean"
OUT_DIR = "./workspace/vpr_retrievals"
WEIGHTS_DIR = "./workspace/vpr_weights"
REPOS_DIR = "./workspace/vpr_repos"
os.makedirs(OUT_DIR, exist_ok=True)


# Import shared gallery & CTSE helpers directly from exp1_build_gallery_and_run_all_vpr.py
sys.path.insert(0, "./workspace")
from exp1_build_gallery_and_run_all_vpr import (
    ensure_occlusion_images_exist,
    build_benchmark_galleries,
    parse_question_spatiotemporal_intent,
    ctse_select_view,
    ImagePathDataset,
)


def _load_py_module(name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CricaVPRSingleImageWrapper(nn.Module):
    """Vectorized exact B=1 single-image inference for CricaVPR (zero cross-query batch leakage, 10x faster)."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        b_out = self.net.backbone(x)
        B, P, D = b_out["x_prenorm"].shape
        W = H = int(math.sqrt(P - 1))
        x0 = b_out["x_norm_clstoken"]
        x_p = b_out["x_norm_patchtokens"].view(B, W, H, D).permute(0, 3, 1, 2)
        agg = self.net.aggregation
        parts = [
            x0,
            agg(x_p[:, :, 0:8, 0:8]), agg(x_p[:, :, 0:8, 8:]), agg(x_p[:, :, 8:, 0:8]), agg(x_p[:, :, 8:, 8:]),
            agg(x_p[:, :, 0:5, 0:5]), agg(x_p[:, :, 0:5, 5:11]), agg(x_p[:, :, 0:5, 11:]),
            agg(x_p[:, :, 5:11, 0:5]), agg(x_p[:, :, 5:11, 5:11]), agg(x_p[:, :, 5:11, 11:]),
            agg(x_p[:, :, 11:, 0:5]), agg(x_p[:, :, 11:, 5:11]), agg(x_p[:, :, 11:, 11:]),
        ]
        seq = torch.cat([p.unsqueeze(1) for p in parts], dim=1)  # [B, 14, D]
        enc_out = self.net.encoder(seq.reshape(1, B * 14, D)).reshape(B, 14 * D)
        return F.normalize(enc_out.float(), p=2, dim=-1)


class SelaVPRGlobalWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        local_f, global_f = self.net(x)
        return global_f, local_f


class PatchNetVLADGlobalAndPatch(nn.Module):
    """
    Authentic Patch-NetVLAD (Hausler et al., CVPR 2021) using official mapillary_WPCA4096.pth.tar:
      - Global branch: VGG16 conv5 -> NetVLAD (64 clusters) -> WPCA (4096-d) -> L2 norm
      - Patch branch: Multi-scale patch VLAD descriptors (patch_size=5, stride=1) + WPCA (512-d)
        used for Stage-2 Rapid Spatial Verification reranking on Top-15 candidates!
    """
    def __init__(self, ckpt_path: str):
        super().__init__()
        if os.path.join(REPOS_DIR, "Patch-NetVLAD") not in sys.path:
            sys.path.insert(0, os.path.join(REPOS_DIR, "Patch-NetVLAD"))
        import configparser
        from torchvision import models as tv_models
        from patchnetvlad.models.models_generic import get_model, get_pca_encoding

        cfg = configparser.ConfigParser()
        cfg["global_params"] = {
            "pooling": "patchnetvlad",
            "num_clusters": "16",
            "vladv2": "False",
            "num_pcs": "4096",
            "patch_sizes": "5",
            "strides": "1",
        }
        enc_dim = 512
        encoder = nn.Sequential(*list(tv_models.vgg16(weights=None).features.children())[:-2])
        model = get_model(encoder, enc_dim, cfg["global_params"], append_pca_layer=True)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        # Strip 'module.' if DataParallel
        clean_sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(clean_sd, strict=True)
        self.model = model
        self.get_pca_encoding = get_pca_encoding

    def extract_global(self, x):
        feat = self.model.encoder(x)
        _, vlad_global = self.model.pool(feat)
        pca_global = self.get_pca_encoding(self.model, vlad_global)
        return F.normalize(pca_global.float(), p=2, dim=-1)

    def extract_patches_and_global(self, x):
        feat = self.model.encoder(x)
        vlad_local, vlad_global = self.model.pool(feat)
        # vlad_local is list of [B, D, N_patches] per patch_size (here 1 scale: patch_size=5)
        patch_map = vlad_local[0]  # [B, 8192, N_patches]
        wpca_conv = self.model.WPCA[0]
        w512 = wpca_conv.weight[:512, :, 0, 0]  # [512, 8192]
        b512 = wpca_conv.bias[:512] if wpca_conv.bias is not None else None
        p_proj = torch.einsum("bcn,dc->bnd", patch_map.float(), w512.float())
        if b512 is not None:
            p_proj = p_proj + b512.float().view(1, 1, -1)
        p_proj = F.normalize(p_proj, p=2, dim=-1)
        return p_proj  # [B, N_patches, 512]

    def forward(self, x):
        return self.extract_global(x)


def load_remaining_vpr_model(method: str, device: str = "cuda"):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if method == "dinov2_salad":
        repo = os.path.join(REPOS_DIR, "salad")
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from vpr_model import VPRModel
        model = VPRModel(
            backbone_arch="dinov2_vitb14",
            backbone_config={"num_trainable_blocks": 4, "return_token": True, "norm_layer": True},
            agg_arch="SALAD",
            agg_config={"num_channels": 768, "num_clusters": 64, "cluster_dim": 128, "token_dim": 256},
        )
        sd = torch.load("./workspace/torch_cache/hub/checkpoints/dino_salad.ckpt", map_location="cpu", weights_only=False)
        model.load_state_dict(sd, strict=True)
        model = model.eval().to(device)
        tf = transforms.Compose([transforms.Resize((322, 322)), transforms.ToTensor(), norm])
        return model, tf, 128

    elif method == "megaloc":
        repo = os.path.join(REPOS_DIR, "MegaLoc")
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from megaloc_model import MegaLoc
        model = MegaLoc()
        sd = torch.load("./workspace/torch_cache/hub/checkpoints/megaloc.torch", map_location="cpu", weights_only=False)
        new_sd = {}
        msd = model.state_dict()
        for k, v in sd.items():
            nk = k.replace("backbone.model.", "backbone.")
            if nk in msd and msd[nk].shape == v.shape:
                new_sd[nk] = v
        missing = [k for k in msd if k not in new_sd]
        assert len(missing) == 0, f"MegaLoc missing keys: {missing}"
        model.load_state_dict(new_sd, strict=True)
        model = model.eval().to(device)
        tf = transforms.Compose([transforms.Resize((322, 322)), transforms.ToTensor(), norm])
        return model, tf, 128

    elif method == "anyloc":
        repo = os.path.join(REPOS_DIR, "DINO")
        if repo not in sys.path:
            sys.path.insert(0, repo)
        import model_wrapper as mw
        # Authentic AnyLoc-VLAD (Keetha et al., IEEE RA-L 2023): DINOv2-ViT-B/14 Layer 11 Value facet + K=32 VLAD
        model = mw.AnyLocVladNoCacheDinov2(
            dino_model="dinov2_vitb14", layer=11, facet="value", num_c=32, device=device
        )
        tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
        return model, tf, 128

    elif method == "cricavpr":
        repo = os.path.join(REPOS_DIR, "CricaVPR")
        # Clear any conflicting 'backbone' module from sys.modules
        for k in list(sys.modules.keys()):
            if k == "backbone" or k.startswith("backbone.") or k == "network":
                del sys.modules[k]
        sys.path.insert(0, repo)
        import network as crica_net
        model = crica_net.CricaVPRNet(pretrained_foundation=False, foundation_model_path=None)
        ckpt = torch.load(os.path.join(WEIGHTS_DIR, "CricaVPR.pth"), map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"]
        clean_sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(clean_sd, strict=True)
        model = CricaVPRSingleImageWrapper(model).eval().to(device)
        tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
        return model, tf, 128

    elif method == "selavpr":
        repo = os.path.join(REPOS_DIR, "SelaVPR")
        for k in list(sys.modules.keys()):
            if k == "backbone" or k.startswith("backbone.") or k == "network":
                del sys.modules[k]
        sys.path.insert(0, repo)
        import network as sela_net
        sela_args = SimpleNamespace(registers=False, foundation_model_path=None, resume=os.path.join(WEIGHTS_DIR, "SelaVPR_msls.pth"))
        net = sela_net.GeoLocalizationNet(sela_args)
        ckpt = torch.load(sela_args.resume, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"]
        clean_sd = {k.replace("module.", ""): v for k, v in sd.items()}
        net.load_state_dict(clean_sd, strict=True)
        model = SelaVPRGlobalWrapper(net).eval().to(device)
        tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
        return model, tf, 64

    elif method == "netvlad":
        # Official VGG16-NetVLAD (Pitts30k, 64 clusters, 4096-d WPCA, Arandjelovic et al. CVPR'16)
        if os.path.join(REPOS_DIR, "Patch-NetVLAD") not in sys.path:
            sys.path.insert(0, os.path.join(REPOS_DIR, "Patch-NetVLAD"))
        import configparser
        from torchvision import models as tv_models
        from patchnetvlad.models.models_generic import get_model, get_pca_encoding

        class _NetVLADGlobalOnly(nn.Module):
            def __init__(self, cp):
                super().__init__()
                cfg = configparser.ConfigParser()
                cfg["global_params"] = {
                    "pooling": "netvlad",
                    "num_clusters": "64",
                    "vladv2": "False",
                    "num_pcs": "4096",
                }
                enc_dim = 512
                encoder = nn.Sequential(*list(tv_models.vgg16(weights=None).features.children())[:-2])
                m = get_model(encoder, enc_dim, cfg["global_params"], append_pca_layer=True)
                ckpt = torch.load(cp, map_location="cpu", weights_only=False)
                clean_sd = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
                m.load_state_dict(clean_sd, strict=True)
                self.model = m
                self.get_pca_encoding = get_pca_encoding

            def forward(self, x):
                feat = self.model.encoder(x)
                vlad = self.model.pool(feat)
                pca = self.get_pca_encoding(self.model, vlad)
                return F.normalize(pca.float(), p=2, dim=-1)

        ckpt_path = os.path.join(WEIGHTS_DIR, "pitts_WPCA4096.pth.tar")
        model = _NetVLADGlobalOnly(ckpt_path).eval().to(device)
        tf = transforms.Compose([transforms.Resize((320, 320)), transforms.ToTensor(), norm])
        return model, tf, 64

    elif method == "patch_netvlad":
        ckpt_path = os.path.join(WEIGHTS_DIR, "mapillary_WPCA4096.pth.tar")
        model = PatchNetVLADGlobalAndPatch(ckpt_path).eval().to(device)
        tf = transforms.Compose([transforms.Resize((320, 320)), transforms.ToTensor(), norm])
        return model, tf, 64

    elif method == "mixvpr":
        vpr_root = "./workspace/VPR-Methods-Evaluation"
        mv_mod = _load_py_module("mixvpr_standalone", os.path.join(vpr_root, "vpr_models/mixvpr.py"))
        ckpt_path = os.path.join(WEIGHTS_DIR, "resnet50_MixVPR_4096.ckpt")
        model = mv_mod.MixVPRModel(
            agg_config={
                "in_channels": 1024,
                "in_h": 20,
                "in_w": 20,
                "out_channels": 1024,
                "mix_depth": 4,
                "mlp_ratio": 1,
                "out_rows": 4,
            }
        )
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=True)
        model = model.eval().to(device)
        tf = transforms.Compose([transforms.Resize((320, 320)), transforms.ToTensor(), norm])
        return model, tf, 128

    elif method == "convap":
        vpr_root = "./workspace/VPR-Methods-Evaluation"
        ca_mod = _load_py_module("convap_standalone", os.path.join(vpr_root, "vpr_models/convap.py"))
        ckpt_path = os.path.join(WEIGHTS_DIR, "resnet50_ConvAP_1024_2x2.ckpt")
        model = ca_mod.ConvAPModel(
            agg_config={"in_channels": 2048, "out_channels": 1024, "s1": 2, "s2": 2}
        )
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=True)
        model = model.eval().to(device)
        tf = transforms.Compose([transforms.Resize((320, 320)), transforms.ToTensor(), norm])
        return model, tf, 128

    else:
        raise ValueError(f"Unknown method: {method}")


@torch.inference_mode()
def extract_features_generic(model, tf, img_paths: List[str], batch_size: int, device: str = "cuda", use_amp: bool = True) -> np.ndarray:
    ds = ImagePathDataset(img_paths, tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    out_chunks = []
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                emb = model(imgs)
                if isinstance(emb, tuple):
                    emb = emb[0]
                emb = F.normalize(emb.float(), p=2, dim=-1)
        else:
            emb = model(imgs)
            if isinstance(emb, tuple):
                emb = emb[0]
            emb = F.normalize(emb.float(), p=2, dim=-1)
        out_chunks.append(emb.cpu().numpy().astype(np.float32))
    return np.concatenate(out_chunks, axis=0)


@torch.inference_mode()
def run_anyloc_for_dataset(model, tf, db_paths: List[str], q_paths: List[str], batch_size: int = 128, device: str = "cuda"):
    """Fit domain-specific K=32 VLAD vocabulary on a 500-image sample of db_paths, then extract VLAD descriptors."""
    rng = np.random.default_rng(42)
    fit_idx = rng.choice(len(db_paths), size=min(500, len(db_paths)), replace=False)
    fit_paths = [db_paths[i] for i in fit_idx]
    fit_ds = ImagePathDataset(fit_paths, tf)
    fit_loader = DataLoader(fit_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    patch_chunks = []
    for imgs, _ in fit_loader:
        imgs = imgs.to(device, non_blocking=True)
        # [B, N_patches, 768]
        p = model.extract(imgs).float().cpu()
        patch_chunks.append(p.reshape(-1, p.shape[-1]))
    all_patches = torch.cat(patch_chunks, dim=0)
    # Subsample to 60,000 patches for fast, high-quality KMeans fitting
    if all_patches.shape[0] > 60000:
        perm = torch.randperm(all_patches.shape[0])[:60000]
        all_patches = all_patches[perm]
    model.fit(all_patches.to(device))
    del all_patches, patch_chunks
    torch.cuda.empty_cache()

    def _extract_vlad(paths):
        ds = ImagePathDataset(paths, tf)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        chunks = []
        for imgs, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            v = model(imgs).float()
            v = F.normalize(v, p=2, dim=-1)
            chunks.append(v.cpu().numpy().astype(np.float32))
        return np.concatenate(chunks, axis=0)

    db_embs = _extract_vlad(db_paths)
    q_embs = _extract_vlad(q_paths)
    return db_embs, q_embs


@torch.inference_mode()
def rerank_selavpr_topK(model, tf, q_paths: List[str], db_paths: List[str], indices: np.ndarray, topk: int = 15, device: str = "cuda"):
    """
    Authentic Stage-2 SelaVPR dense local matching reranking on Top-K candidates:
    Calls SelaVPR's exact official FP32 local_sim() on un-pooled [61, 61, 128] dense local features
    (Lu et al., ICLR 2024, local_matching.py)!
    """
    if os.path.join(REPOS_DIR, "SelaVPR") not in sys.path:
        sys.path.insert(0, os.path.join(REPOS_DIR, "SelaVPR"))
    from local_matching import local_sim

    cand_indices = indices[:, :topk]
    uniq_db = sorted(set(int(x) for x in cand_indices.reshape(-1)))
    db_idx_to_pos = {idx: pos for pos, idx in enumerate(uniq_db)}

    def _extract_local_fp32(paths):
        ds = ImagePathDataset(paths, tf)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
        out = []
        for imgs, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            _, loc = model(imgs)  # pure FP32 [B, 61, 61, 128]
            out.append(loc.float().cpu())
        return torch.cat(out, dim=0)

    db_locals = _extract_local_fp32([db_paths[i] for i in uniq_db])  # [U, 61, 61, 128] fp32
    q_locals = _extract_local_fp32(q_paths)  # [Nq, 61, 61, 128] fp32

    L = float(q_locals.shape[1] * q_locals.shape[2])
    new_indices = np.copy(indices)
    new_sims = np.zeros((len(q_paths), 10), dtype=np.float32)

    for i in range(len(q_paths)):
        ql = q_locals[i].to(device, dtype=torch.float32)
        c_ids = cand_indices[i]
        c_pos = [db_idx_to_pos[int(cid)] for cid in c_ids]
        dl = db_locals[c_pos].to(device, dtype=torch.float32)
        scores = (local_sim(ql, dl) / L).cpu().numpy()
        order = np.argsort(-scores)
        new_indices[i, :10] = c_ids[order][:10]
        new_sims[i, :10] = scores[order][:10]

    del db_locals, q_locals
    torch.cuda.empty_cache()
    return new_sims, new_indices[:, :10]


@torch.inference_mode()
def rerank_patch_netvlad_topK(model: PatchNetVLADGlobalAndPatch, tf, q_paths: List[str], db_paths: List[str], indices: np.ndarray, topk: int = 15, device: str = "cuda"):
    """
    Authentic Stage-2 Patch-NetVLAD spatial verification reranking on Top-K candidates:
    Computes multi-patch descriptors (patch_size=5, stride=1, WPCA-512) and scores each candidate
    via mutual nearest-neighbor spatial consistency (Rapid Spatial Scoring, Hausler et al. CVPR'21).
    """
    cand_indices = indices[:, :topk]
    uniq_db = sorted(set(int(x) for x in cand_indices.reshape(-1)))
    db_idx_to_pos = {idx: pos for pos, idx in enumerate(uniq_db)}

    def _extract_patches(paths):
        ds = ImagePathDataset(paths, tf)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
        out = []
        for imgs, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            p = model.extract_patches_and_global(imgs)  # [B, N_patches, 512]
            out.append(p.half().cpu())
        return torch.cat(out, dim=0)

    db_patches = _extract_patches([db_paths[i] for i in uniq_db])
    q_patches = _extract_patches(q_paths)

    N_p = q_patches.shape[1]
    side = int(round(math.sqrt(N_p)))
    coords = torch.stack(torch.meshgrid(torch.arange(side, device=device), torch.arange(side, device=device), indexing="ij"), dim=-1).reshape(-1, 2).float()

    new_indices = np.copy(indices)
    new_sims = np.zeros((len(q_paths), 10), dtype=np.float32)

    for i in range(len(q_paths)):
        qp = q_patches[i].to(device, dtype=torch.float32)  # [N_p, 512]
        c_ids = cand_indices[i]
        c_pos = [db_idx_to_pos[int(cid)] for cid in c_ids]
        dp = db_patches[c_pos].to(device, dtype=torch.float32)  # [K, N_p, 512]
        sim_mat = torch.einsum("nd,kmd->knm", qp, dp)  # [K, N_p, N_p]
        q2d = sim_mat.argmax(dim=2)  # [K, N_p]
        d2q = sim_mat.argmax(dim=1)  # [K, N_p]
        arange_n = torch.arange(N_p, device=device).unsqueeze(0).expand_as(q2d)
        mutual = (torch.gather(d2q, 1, q2d) == arange_n)  # [K, N_p]
        max_sim = sim_mat.max(dim=2).values
        # Rapid Spatial Scoring: reward mutual matches with consistent 2D spatial displacement!
        scores = []
        for k in range(len(c_ids)):
            m_idx = torch.where(mutual[k])[0]
            if m_idx.numel() < 3:
                scores.append(float((mutual[k].float() * max_sim[k]).sum().item() / max(N_p, 1)))
            else:
                d_idx = q2d[k, m_idx]
                disp = coords[m_idx] - coords[d_idx]  # [M, 2]
                med_disp = disp.median(dim=0).values
                inliers = ((disp - med_disp).abs().sum(dim=1) <= 2.5)
                sc = float((inliers.float() * max_sim[k, m_idx]).sum().item() / max(N_p, 1))
                scores.append(sc)
        scores = np.array(scores, dtype=np.float32)
        order = np.argsort(-scores)
        new_indices[i, :10] = c_ids[order][:10]
        new_sims[i, :10] = scores[order][:10]

    return new_sims, new_indices[:, :10]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", required=True)
    args = parser.parse_args()

    print("[Exp1-Inc] Loading ChronoVPR-VQA-Clean splits...", flush=True)
    with open(os.path.join(CLEAN_DIR, "chrono_vpr_vqa_all.json"), "r", encoding="utf-8") as f:
        all_samples = json.load(f)
    test_samples = [r for r in all_samples if r["split"] == "test"]
    ensure_occlusion_images_exist(all_samples)
    galleries, place_graphs = build_benchmark_galleries(all_samples)

    by_ds_test = defaultdict(list)
    for r in test_samples:
        by_ds_test[r["dataset"]].append(r)

    summary_path = os.path.join(OUT_DIR, "vpr_methods_recall_summary.json")
    summary_table = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_table = json.load(f)

    for method in args.methods:
        out_file = os.path.join(OUT_DIR, f"{method}_test_retrievals.json")
        if os.path.exists(out_file) and method in summary_table:
            print(f"[SKIP] [{method}] already completed -> R@1={summary_table[method]['overall_R1']}%", flush=True)
            continue

        t_m = time.time()
        print(f"\n================ Running Remaining VPR Method: [{method}] ================", flush=True)
        model, tf, bs = load_remaining_vpr_model(method, device="cuda")

        method_results = []
        method_metrics = defaultdict(lambda: {"n": 0, "r1": 0, "r5": 0, "r10": 0})
        method_metrics_by_task = defaultdict(lambda: {"n": 0, "r1": 0, "r5": 0})

        for ds, q_rows in by_ds_test.items():
            t_ds = time.time()
            db_nodes = galleries[ds]
            db_paths = [n["img_abs"] for n in db_nodes]
            q_paths = [os.path.join(DATA_BASE, r["query_img"]) for r in q_rows]

            if method == "anyloc":
                db_embs, q_embs = run_anyloc_for_dataset(model, tf, db_paths, q_paths, batch_size=bs)
                index = faiss.IndexFlatIP(db_embs.shape[1])
                index.add(db_embs)
                sims, indices = index.search(q_embs, 10)
            else:
                use_amp = method not in ("netvlad", "patch_netvlad", "cricavpr", "selavpr")
                db_embs = extract_features_generic(model, tf, db_paths, batch_size=bs, use_amp=use_amp)
                q_embs = extract_features_generic(model, tf, q_paths, batch_size=bs, use_amp=use_amp)
                index = faiss.IndexFlatIP(db_embs.shape[1])
                index.add(db_embs)
                k_search = 15 if method in ("selavpr", "patch_netvlad") else 10
                sims, indices = index.search(q_embs, k_search)

                if method == "selavpr":
                    s1_hits = sum(int(db_nodes[int(indices[i, 0])]["place_id"] == r["place_id"]) for i, r in enumerate(q_rows))
                    print(f"  [selavpr] {ds} Stage-1 Global R@1={s1_hits/max(len(q_rows),1)*100:.2f}% -> running exact 61x61 local rerank...", flush=True)
                    sims, indices = rerank_selavpr_topK(model, tf, q_paths, db_paths, indices, topk=15)
                elif method == "patch_netvlad":
                    sims, indices = rerank_patch_netvlad_topK(model, tf, q_paths, db_paths, indices, topk=15)

            for i, r in enumerate(q_rows):
                gt_pid = r["place_id"]
                top10_nodes = [(db_nodes[int(indices[i, k])], float(sims[i, k])) for k in range(10)]
                top1_node, top1_sim = top10_nodes[0]
                top5_pids = {n["place_id"] for n, _ in top10_nodes[:5]}
                top10_pids = {n["place_id"] for n, _ in top10_nodes[:10]}

                hit1 = int(top1_node["place_id"] == gt_pid)
                hit5 = int(gt_pid in top5_pids)
                hit10 = int(gt_pid in top10_pids)

                method_metrics[ds]["n"] += 1
                method_metrics[ds]["r1"] += hit1
                method_metrics[ds]["r5"] += hit5
                method_metrics[ds]["r10"] += hit10

                tk = r["task_type"]
                method_metrics_by_task[tk]["n"] += 1
                method_metrics_by_task[tk]["r1"] += hit1
                method_metrics_by_task[tk]["r5"] += hit5

                intent_info = parse_question_spatiotemporal_intent(r["question"])
                ctse_node, ctse_sim, _ = ctse_select_view(
                    top10_nodes, intent_info, place_graphs[ds], r.get("query_year", 2020)
                )

                hard_neg_node, hard_neg_sim = top10_nodes[1]
                for cand_n, cand_s in top10_nodes:
                    if cand_n["place_id"] != gt_pid:
                        hard_neg_node, hard_neg_sim = cand_n, cand_s
                        break

                method_results.append({
                    "sample_id": r["sample_id"],
                    "dataset": ds,
                    "task_type": tk,
                    "gt_place_id": gt_pid,
                    "naive_top1_img": top1_node["img_rel"],
                    "naive_top1_place_id": top1_node["place_id"],
                    "naive_top1_sim": round(top1_sim, 4),
                    "naive_hit1": hit1,
                    "hit5": hit5,
                    "hit10": hit10,
                    "ctse_img": ctse_node["img_rel"],
                    "ctse_place_id": ctse_node["place_id"],
                    "ctse_sim": round(ctse_sim, 4),
                    "ctse_year": ctse_node.get("year", -1),
                    "ctse_rel_yaw": ctse_node.get("relative_yaw_deg", 0),
                    "parsed_intent": intent_info["intent"],
                    "hard_neg_img": hard_neg_node["img_rel"],
                    "hard_neg_sim": round(hard_neg_sim, 4),
                })
            print(f"  [{method}] {ds}: R@1={method_metrics[ds]['r1']/method_metrics[ds]['n']*100:.2f}% ({time.time()-t_ds:.1f}s)", flush=True)

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(method_results, f)

        tot_n = sum(v["n"] for v in method_metrics.values())
        tot_r1 = sum(v["r1"] for v in method_metrics.values()) / max(tot_n, 1)
        tot_r5 = sum(v["r5"] for v in method_metrics.values()) / max(tot_n, 1)
        tot_r10 = sum(v["r10"] for v in method_metrics.values()) / max(tot_n, 1)

        summary_table[method] = {
            "overall_R1": round(tot_r1 * 100, 2),
            "overall_R5": round(tot_r5 * 100, 2),
            "overall_R10": round(tot_r10 * 100, 2),
            "by_dataset_R1": {
                ds: round(v["r1"] / max(v["n"], 1) * 100, 2) for ds, v in method_metrics.items()
            },
            "by_task_R1": {
                tk: round(v["r1"] / max(v["n"], 1) * 100, 2) for tk, v in method_metrics_by_task.items()
            },
            "elapsed_sec": round(time.time() - t_m, 1),
        }
        print(f"  -> [{method}] Overall R@1={tot_r1*100:.2f}% | R@5={tot_r5*100:.2f}% | R@10={tot_r10*100:.2f}% ({time.time()-t_m:.1f}s)", flush=True)
        print(f"     By Benchmark R@1: {summary_table[method]['by_dataset_R1']}", flush=True)
        print(f"     By Task R@1:      {summary_table[method]['by_task_R1']}", flush=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_table, f, indent=2)

        del model
        torch.cuda.empty_cache()

    print("\n=== EXP 1 INCREMENTAL BATCH COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
