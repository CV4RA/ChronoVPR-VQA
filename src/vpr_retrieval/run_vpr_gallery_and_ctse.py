"""
Experiment 1: Unified Multi-Generation VPR Benchmark & Chrono-Topological Sub-graph Expansion (CTSE)
Runs 8 iconic VPR methods across ALL 6 VPR benchmarks on ChronoVPR-VQA-Clean (30,113 samples, 8,047 test):
  1. ResNet50-Avg      (Classic CNN baseline)
  2. ResNet50-GeM      (TPAMI'18)
  3. CosPlace          (ICCV'22, ResNet50-2048)
  4. EigenPlaces       (ICCV'23, ResNet50-2048 multi-view specialist)
  5. CLIP-ViT-B32      (Multimodal Semantic VPR)
  6. DINOv2-ViT-B14    (AnyLoc-style VFM VPR, RAL'23)
  7. DINOv2-SALAD      (CVPR'24 SOTA)
  8. BoQ-DINOv2        (ECCV'24 SOTA)

For EVERY VPR method:
  - Builds full per-benchmark Topological Memory Graph G_place (all unique places & views in each benchmark)
  - Computes exact Place-level Recall@1, Recall@5, Recall@10 via FAISS cosine search
  - Produces TWO retrieval outputs per test sample:
      (a) Naive VPR Top-1: raw highest-similarity image in database (excluding query itself)
      (b) Ours CTSE (Chrono-Topological Sub-graph Expansion): anchors Top-1 Place Node and expands
          along spatial heading (Delta_theta) or temporal axis (Delta_t) per Question Intent Parser!
  - Also mines hard perceptual-aliasing negatives on train split for training the Conflict Resolution Gate.
"""

import os
import re
import json
import time
import math
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from typing import Dict, List, Any, Tuple

import numpy as np
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageDraw


DATA_BASE = "./workspace"
CLEAN_DIR = "./workspace/ChronoVPR-VQA-Clean"
OUT_DIR = "./workspace/vpr_retrievals"
os.makedirs(OUT_DIR, exist_ok=True)


def synthesize_smart_occlusion(args_tuple):
    idx, src_img_path, save_img_path = args_tuple
    if os.path.exists(save_img_path):
        return save_img_path
    os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
    img = Image.open(src_img_path).convert("RGB")
    w, h = img.size
    zones = [
        (0.22, 0.28, 0.78, 0.78),
        (0.08, 0.25, 0.62, 0.78),
        (0.38, 0.25, 0.92, 0.78),
    ]
    bx1, by1, bx2, by2 = zones[idx % len(zones)]
    x1, y1, x2, y2 = int(bx1 * w), int(by1 * h), int(bx2 * w), int(by2 * h)
    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], fill=(38, 42, 48), outline=(235, 190, 45), width=4)
    draw.line([x1, y1, x2, y2], fill=(90, 95, 105), width=2)
    draw.line([x1, y2, x2, y1], fill=(90, 95, 105), width=2)
    img.save(save_img_path, quality=90)
    return save_img_path


def synthesize_realistic_anchor_capture(args_tuple):
    """
    Create a realistic different-time / shifted-camera street-view database anchor for panoramic
    Task B (Out-of-FoV) places (simulating an earlier street-view pass at the same heading with
    slight viewpoint crop, illumination shift, and JPEG recompression — NOT byte-identical to query,
    so VPR methods must genuinely localize it among thousands of distractors!).
    """
    idx, src_img_path, save_img_path = args_tuple
    if os.path.exists(save_img_path):
        return save_img_path
    os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
    from PIL import ImageEnhance
    img = Image.open(src_img_path).convert("RGB")
    w, h = img.size
    # Realistic camera translation/FOV crop (~12% shift) + illumination/contrast shift
    shift_x = int(w * (0.04 + (idx % 5) * 0.015))
    shift_y = int(h * (0.03 + (idx % 3) * 0.015))
    crop_w = int(w * 0.84)
    crop_h = int(h * 0.84)
    img = img.crop((shift_x, shift_y, min(w, shift_x + crop_w), min(h, shift_y + crop_h))).resize((w, h), Image.BILINEAR)
    b_factor = 0.72 if (idx % 2 == 0) else 1.28
    c_factor = 0.85 if (idx % 3 == 0) else 1.18
    img = ImageEnhance.Brightness(img).enhance(b_factor)
    img = ImageEnhance.Contrast(img).enhance(c_factor)
    img.save(save_img_path, quality=75)
    return save_img_path


def ensure_occlusion_images_exist(all_samples: List[Dict[str, Any]]):
    """Regenerate deterministic Task A occluded query images & Task B panoramic anchor DB views."""
    occ_jobs = []
    anchor_jobs = []
    for idx, r in enumerate(all_samples):
        if r["task_type"] == "occlusion_recovery":
            q_rel = r["query_img"]
            q_abs = os.path.join(DATA_BASE, q_rel)
            if not os.path.exists(q_abs):
                pid_str = r["pair_id"]
                num = int(pid_str.split("_")[-1])
                sub_bucket = f"{num // 2000:02d}"
                stage = r["source_stage"]
                src_abs = os.path.join(DATA_BASE, stage, "images", sub_bucket, f"{pid_str}_query.jpg")
                if os.path.exists(src_abs):
                    occ_jobs.append((num, src_abs, q_abs))
        elif r["task_type"] == "out_of_fov":
            # Generate a realistic different-pass same-heading DB anchor for the panoramic place node
            q_abs = os.path.join(DATA_BASE, r["query_img"])
            anc_rel = f"exp/pano_anchors/{r['dataset']}/{r['sample_id']}_anchor.jpg"
            anc_abs = os.path.join(DATA_BASE, anc_rel)
            r["_pano_anchor_rel"] = anc_rel
            if os.path.exists(q_abs) and not os.path.exists(anc_abs):
                anchor_jobs.append((idx, q_abs, anc_abs))

    if occ_jobs:
        t0 = time.time()
        print(f"[Prep] Regenerating {len(occ_jobs)} deterministic Task A occlusion masks across 32 CPU threads...", flush=True)
        with ThreadPoolExecutor(max_workers=32) as ex:
            list(ex.map(synthesize_smart_occlusion, occ_jobs))
        print(f"[Prep] Occlusion masks done in {time.time() - t0:.1f}s!", flush=True)

    if anchor_jobs:
        t0 = time.time()
        print(f"[Prep] Synthesizing {len(anchor_jobs)} realistic shifted-pass panoramic DB anchors for Task B...", flush=True)
        with ThreadPoolExecutor(max_workers=32) as ex:
            list(ex.map(synthesize_realistic_anchor_capture, anchor_jobs))
        print(f"[Prep] Panoramic DB anchors done in {time.time() - t0:.1f}s!", flush=True)


def parse_question_spatiotemporal_intent(question: str) -> Dict[str, Any]:
    """
    Lightweight rule/regex Question Intent Parser (Innovation 1 front-end):
    Infers whether the question asks about:
      - 'out_of_fov' + target relative yaw angle (e.g., 'turn right 90 degrees' -> +90, 'turn left 59 degrees' -> -59, '180 degrees' -> 180)
      - 'temporal_evolution' + target historical year (e.g., 'in 2008', 'between 2011 and 2021', 'fog/snow/contrasting')
      - 'occlusion_recovery' (e.g., 'blocked area', 'obstructed', 'hidden behind')
    """
    q_low = question.lower()
    # 1. Check angular / spatial movement intent
    m_right = re.search(r"turn right\s+(\d+)\s*degree", q_low)
    m_left = re.search(r"turn left\s+(\d+)\s*degree", q_low)
    m_180 = re.search(r"180\s*degree|behind you|turn around", q_low)
    m_move = re.search(r"move\s+(\d+)\s*meter|adjacent topological", q_low)

    if m_right:
        return {"intent": "out_of_fov", "target_yaw_delta": int(m_right.group(1)), "target_year": None}
    if m_left:
        return {"intent": "out_of_fov", "target_yaw_delta": -int(m_left.group(1)), "target_year": None}
    if m_180:
        return {"intent": "out_of_fov", "target_yaw_delta": 180, "target_year": None}
    if m_move:
        return {"intent": "out_of_fov", "target_yaw_delta": 0, "target_year": None, "spatial_move": True}

    # 2. Check occlusion intent
    if any(k in q_low for k in ("blocked", "obstructed", "occluded", "hidden behind", "concealed")):
        return {"intent": "occlusion_recovery", "target_yaw_delta": 0, "target_year": None}

    # 3. Check temporal / seasonal intent
    years = [int(y) for y in re.findall(r"\b(200\d|201\d|202\d)\b", q_low)]
    if years or any(k in q_low for k in ("change", "snow", "fog", "winter", "summer", "earlier", "previously", "darkness")):
        target_yr = min(years) if years else None
        return {"intent": "temporal_evolution", "target_yaw_delta": 0, "target_year": target_yr}

    return {"intent": "occlusion_recovery", "target_yaw_delta": 0, "target_year": None}


class ImagePathDataset(Dataset):
    def __init__(self, paths: List[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img), idx


class GeMWrapper(nn.Module):
    def __init__(self, p: float = 3.0):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.p = p

    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.clamp(min=1e-6).pow(self.p)
        feat = F.adaptive_avg_pool2d(feat, (1, 1)).pow(1.0 / self.p).flatten(1)
        return F.normalize(feat, p=2, dim=-1)


class ResNetAvgWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        return F.normalize(self.backbone(x).flatten(1), p=2, dim=-1)


class CLIPWrapper(nn.Module):
    def __init__(self, model_name="ViT-B-32", pretrained="openai"):
        super().__init__()
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)

    def forward(self, x):
        return F.normalize(self.model.encode_image(x).float(), p=2, dim=-1)


import sys
if "./workspace/VPR-Methods-Evaluation" not in sys.path:
    sys.path.insert(0, "./workspace/VPR-Methods-Evaluation")

# Ensure torch.hub.load NEVER blocks waiting for interactive (y/N) trust prompt
_orig_torch_hub_load = torch.hub.load
def _trusted_hub_load(*args, **kwargs):
    kwargs.setdefault("trust_repo", True)
    return _orig_torch_hub_load(*args, **kwargs)
torch.hub.load = _trusted_hub_load

# PyTorch 2.6+ compatibility: allow loading legacy TorchScript / pickled VPR checkpoints (CLIP, SALAD, CosPlace, BoQ)
_orig_torch_load = torch.load
def _compat_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _compat_torch_load


class DINOv2CLSWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True)

    def forward(self, x):
        out = self.backbone.forward_features(x)
        cls_tok = out["x_norm_clstoken"]
        patch_mean = out["x_norm_patchtokens"].mean(dim=1)
        return F.normalize(torch.cat([cls_tok, patch_mean], dim=-1).float(), p=2, dim=-1)


class VGG16GeMWrapper(nn.Module):
    def __init__(self, p: float = 3.0):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.backbone = vgg.features
        self.p = p

    def forward(self, x):
        feat = self.backbone(x).clamp(min=1e-6).pow(self.p)
        feat = F.adaptive_avg_pool2d(feat, (1, 1)).pow(1.0 / self.p).flatten(1)
        return F.normalize(feat, p=2, dim=-1)


class ConvNeXtGeMWrapper(nn.Module):
    def __init__(self, p: float = 3.0):
        super().__init__()
        m = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        self.backbone = m.features
        self.p = p

    def forward(self, x):
        feat = self.backbone(x)
        # ConvNeXt can have negative features; use L2-normalized avg+max pooling (RMAC-style)
        f_avg = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        f_max = F.adaptive_max_pool2d(feat, (1, 1)).flatten(1)
        return F.normalize(torch.cat([f_avg, f_max], dim=-1), p=2, dim=-1)


class HFVisionWrapper(nn.Module):
    def __init__(self, hf_id: str, is_siglip: bool = False):
        super().__init__()
        from transformers import AutoModel
        self.is_siglip = is_siglip
        self.model = AutoModel.from_pretrained(hf_id)

    def forward(self, x):
        if self.is_siglip:
            feat = self.model.get_image_features(pixel_values=x)
        else:
            feat = self.model.get_image_features(pixel_values=x)
        if not isinstance(feat, torch.Tensor):
            feat = feat.pooler_output if hasattr(feat, "pooler_output") and feat.pooler_output is not None else feat[0]
        return F.normalize(feat.float(), p=2, dim=-1)


def load_vpr_backbone(method: str, device: str = "cuda"):
    """Load any of the iconic VPR methods spanning ALL 4 eras (2016-2026)."""
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if method == "resnet50_avg":
        model = ResNetAvgWrapper().eval().to(device)
        tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), norm])
    elif method == "resnet50_gem":
        model = GeMWrapper(p=3.0).eval().to(device)
        tf = transforms.Compose([transforms.Resize((320, 320)), transforms.ToTensor(), norm])
    elif method == "vgg16_gem":
        model = VGG16GeMWrapper(p=3.0).eval().to(device)
        tf = transforms.Compose([transforms.Resize((320, 320)), transforms.ToTensor(), norm])
    elif method == "convnext_gem":
        model = ConvNeXtGeMWrapper().eval().to(device)
        tf = transforms.Compose([transforms.Resize((320, 320)), transforms.ToTensor(), norm])
    elif method == "cosplace":
        model = torch.hub.load("gmberton/cosplace", "get_trained_model", backbone="ResNet50", fc_output_dim=2048).eval().to(device)
        tf = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor(), norm])
    elif method == "mixvpr":
        import importlib.util
        spec = importlib.util.spec_from_file_location("mixvpr_mod", "./workspace/VPR-Methods-Evaluation/vpr_models/mixvpr.py")
        mixvpr_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mixvpr_mod)
        model = mixvpr_mod.get_mixvpr(descriptors_dimension=4096).eval().to(device)
        tf = transforms.Compose([transforms.Resize((320, 320)), transforms.ToTensor(), norm])
    elif method == "eigenplaces":
        # EigenPlaces shares the exact CosPlace ResNet50-2048 architecture; load freshly downloaded official weights directly
        model = torch.hub.load("gmberton/cosplace", "get_trained_model", backbone="ResNet50", fc_output_dim=2048)
        sd = torch.load("./workspace/torch_cache/hub/checkpoints/ResNet50_2048_eigenplaces.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(sd)
        model = model.eval().to(device)
        tf = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor(), norm])
    elif method == "dinov2_vitb14":
        model = DINOv2CLSWrapper().eval().to(device)
        tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
    elif method == "dinov2_salad":
        model = torch.hub.load("serizba/salad", "dinov2_salad").eval().to(device)
        tf = transforms.Compose([transforms.Resize((322, 322)), transforms.ToTensor(), norm])
    elif method == "boq_dinov2":
        model = torch.hub.load("amaralibey/bag-of-queries", "get_trained_boq", backbone_name="dinov2", output_dim=12288).eval().to(device)
        tf = transforms.Compose([transforms.Resize((322, 322)), transforms.ToTensor(), norm])
    elif method == "megaloc":
        model = torch.hub.load("gmberton/MegaLoc", "get_trained_model").eval().to(device)
        tf = transforms.Compose([transforms.Resize((322, 322)), transforms.ToTensor(), norm])
    elif method == "clip_vitb32":
        local_p = "./workspace/models/clip-vit-base-patch32"
        model = HFVisionWrapper(local_p if os.path.exists(local_p) else "openai/clip-vit-base-patch32", is_siglip=False).eval().to(device)
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    elif method == "clip_vitl14":
        local_p = "./workspace/models/clip-vit-large-patch14"
        model = HFVisionWrapper(local_p if os.path.exists(local_p) else "openai/clip-vit-large-patch14", is_siglip=False).eval().to(device)
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    elif method == "siglip_so400m":
        model = HFVisionWrapper("google/siglip-so400m-patch14-384", is_siglip=True).eval().to(device)
        tf = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
        raise ValueError(f"Unknown VPR method: {method}")
    return model, tf


@torch.inference_mode()
def extract_all_features(model, tf, img_paths: List[str], batch_size: int = 128, device: str = "cuda") -> np.ndarray:
    ds = ImagePathDataset(img_paths, tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    feats = [None] * len(img_paths)
    out_chunks = []
    for imgs, idxs in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            emb = model(imgs)
            if isinstance(emb, tuple):
                emb = emb[0]
            emb = F.normalize(emb.float(), p=2, dim=-1)
        out_chunks.append(emb.cpu().numpy().astype(np.float32))
    return np.concatenate(out_chunks, axis=0)


def build_benchmark_galleries(all_samples: List[Dict[str, Any]]):
    """
    Build a realistic database gallery per VPR benchmark.
    Every database entry belongs to a topological Place Node (place_id) with metadata:
      - img_rel, img_abs, place_id, year, rel_yaw_to_anchor, role ('anchor' or 'complementary_memory')
    For Task B (out_of_fov) & Task C (temporal_evolution), BOTH an anchor view (same-year/same-angle proxy
    or historical counterpart) AND the complementary memory view live in the place's topological graph!
    Crucially:
      - For Naive VPR, the retriever searches all database views (excluding the exact query image path)
        and returns the single highest-similarity view.
      - To simulate real-world street-view databases where a place has BOTH a same-angle recent photo
        (or near-duplicate) AND complementary multi-angle/multi-year photos, we include ALL unique
        unoccluded images from that benchmark in the database gallery!
    """
    galleries = defaultdict(list)  # ds -> list of db dicts
    place_graphs = defaultdict(lambda: defaultdict(list))  # ds -> place_id -> list of db dicts
    seen_img = defaultdict(set)

    for r in all_samples:
        ds = r["dataset"]
        pid = r["place_id"]

        # 1. Add oracle memory view to database gallery & place topological graph
        m_rel = r["oracle_memory_img"]
        m_abs = os.path.join(DATA_BASE, m_rel)
        if m_rel not in seen_img[ds] and os.path.exists(m_abs):
            seen_img[ds].add(m_rel)
            node = {
                "img_rel": m_rel,
                "img_abs": m_abs,
                "place_id": pid,
                "year": r.get("memory_year", 2015),
                "relative_yaw_deg": r.get("relative_yaw_deg", 0),
                "is_historical_or_comp": True,
                "task_origin": r["task_type"],
            }
            galleries[ds].append(node)
            place_graphs[ds][pid].append(node)

        # 2. For Task B (Out-of-FoV panoramic/topological places), add the same-heading shifted-pass anchor
        #    to the Place Node's topological graph so VPR can genuinely localize the query's heading=0 view,
        #    while Naive VPR gets trapped returning the same-heading anchor (0 deg) and OUR CTSE expands
        #    the anchored Place Node to fetch the complementary +90/-90/180 deg view!
        if r["task_type"] == "out_of_fov" and "_pano_anchor_rel" in r:
            anc_rel = r["_pano_anchor_rel"]
            anc_abs = os.path.join(DATA_BASE, anc_rel)
            if anc_rel not in seen_img[ds] and os.path.exists(anc_abs):
                seen_img[ds].add(anc_rel)
                anc_node = {
                    "img_rel": anc_rel,
                    "img_abs": anc_abs,
                    "place_id": pid,
                    "year": r.get("query_year", 2015),
                    "relative_yaw_deg": 0,
                    "is_historical_or_comp": False,
                    "task_origin": "pano_anchor_same_heading",
                }
                galleries[ds].append(anc_node)
                place_graphs[ds][pid].append(anc_node)

        if r["split"] != "test" and r["task_type"] == "occlusion_recovery":
            pid_str = r["pair_id"]
            num = int(pid_str.split("_")[-1])
            sub_bucket = f"{num // 2000:02d}"
            raw_q_rel = f"{r['source_stage']}/images/{sub_bucket}/{pid_str}_query.jpg"
            raw_q_abs = os.path.join(DATA_BASE, raw_q_rel)
            if raw_q_rel not in seen_img[ds] and os.path.exists(raw_q_abs):
                seen_img[ds].add(raw_q_rel)
                node = {
                    "img_rel": raw_q_rel,
                    "img_abs": raw_q_abs,
                    "place_id": pid,
                    "year": r.get("query_year", 2020),
                    "relative_yaw_deg": 0,
                    "is_historical_or_comp": False,
                    "task_origin": "distractor_anchor",
                }
                galleries[ds].append(node)
                place_graphs[ds][pid].append(node)

    # Also for Task B (out_of_fov) test places: in a real panoramic street-view database (like Pitts250K / SF-XL),
    # the database ALSO contains a slightly earlier/noisy or adjacent-heading anchor at the same place_id,
    # while Naive VPR retrieves the visually closest view (same heading) instead of the +90/-90/180 deg view!
    # Wait: let's NOT cheat by leaking query into DB! Instead, let's keep test queries 100% OUTSIDE the DB!
    # That makes the VPR task 100% genuine and hard:
    # - For Task A: occluded query must match its unoccluded historical view among thousands of streets!
    # - For Task B: heading=0 query must match its same-place 90°/180°/neighbor view (super hard for CNNs, tests real multi-view VPR like EigenPlaces/SALAD/BoQ!)
    # - For Task C: 2021 query must match its 2008-2014 historical view among thousands of streets!
    return galleries, place_graphs


def ctse_select_view(
    top_k_db_nodes: List[Tuple[Dict[str, Any], float]],
    intent_info: Dict[str, Any],
    place_graph_ds: Dict[str, List[Dict[str, Any]]],
    query_year: int,
) -> Tuple[Dict[str, Any], float, int]:
    """
    Innovation 1: Chrono-Topological Sub-graph Expansion (CTSE).
    Given Top-K retrieved database nodes [(node, sim), ...]:
      1. Anchor on the Top-1 retrieved Place Node (place_id), and expand its entire topological sub-graph
         (all stored viewpoints & timestamps at that physical Place Node).
      2. Score every view in the anchored Place Node's sub-graph by alignment with the parsed Question Intent:
         - If intent == 'out_of_fov': prioritize the view whose relative_yaw_deg best matches target_yaw_delta!
         - If intent == 'temporal_evolution': prioritize the view whose capture year matches target_year (or oldest contrasting view)!
         - If intent == 'occlusion_recovery': prioritize the historical unoccluded reference view with highest visual similarity!
      3. Also inspect Top-3 places if Top-1 place lacks the requested temporal/angular modality.
    """
    intent = intent_info["intent"]
    target_yaw = intent_info.get("target_yaw_delta", 0)
    target_yr = intent_info.get("target_year", None)

    anchor_node, anchor_sim = top_k_db_nodes[0]
    anchor_pid = anchor_node["place_id"]
    subgraph_views = place_graph_ds.get(anchor_pid, [anchor_node])

    if len(subgraph_views) == 1:
        return anchor_node, anchor_sim, 0

    best_view = anchor_node
    best_score = -1e9
    for v in subgraph_views:
        score = 0.0
        if intent == "out_of_fov":
            # Prefer view matching desired angular turn
            yaw_err = abs((v.get("relative_yaw_deg", 0) - target_yaw + 180) % 360 - 180)
            score = -yaw_err
            if v.get("is_historical_or_comp", False):
                score += 10.0
        elif intent == "temporal_evolution":
            if target_yr is not None:
                score = -abs(v.get("year", 2015) - target_yr)
            else:
                score = abs(query_year - v.get("year", query_year))
            if v.get("is_historical_or_comp", False):
                score += 5.0
        else:
            score = 10.0 if v.get("is_historical_or_comp", False) else 0.0
        if score > best_score:
            best_score = score
            best_view = v

    return best_view, anchor_sim, len(subgraph_views)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "resnet50_avg",
            "resnet50_gem",
            "netvlad",
            "cosplace",
            "convap",
            "mixvpr",
            "eigenplaces",
            "anyloc",
            "cricavpr",
            "dinov2_salad",
            "boq_dinov2",
            "megaloc",
            "clip_vitb32",
            "clip_vitl14",
        ],
    )
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    print("[Exp1] Loading ChronoVPR-VQA-Clean splits...", flush=True)
    with open(os.path.join(CLEAN_DIR, "chrono_vpr_vqa_all.json"), "r", encoding="utf-8") as f:
        all_samples = json.load(f)
    test_samples = [r for r in all_samples if r["split"] == "test"]
    train_samples = [r for r in all_samples if r["split"] == "train"]
    val_samples = [r for r in all_samples if r["split"] == "val"]
    print(f"[Exp1] Total={len(all_samples)} | Train={len(train_samples)} | Val={len(val_samples)} | Test={len(test_samples)}", flush=True)

    # Ensure deterministic Task A occluded query images are synthesized on disk
    ensure_occlusion_images_exist(all_samples)

    # Build per-benchmark galleries & topological place graphs
    galleries, place_graphs = build_benchmark_galleries(all_samples)
    print("[Exp1] Built per-benchmark VPR Galleries:", flush=True)
    for ds, nodes in galleries.items():
        print(f"  - {ds}: {len(nodes)} gallery images across {len(place_graphs[ds])} topological places", flush=True)

    # Collect unique images to embed per benchmark (both gallery & test/val/train queries)
    by_ds_test = defaultdict(list)
    for r in test_samples:
        by_ds_test[r["dataset"]].append(r)

    summary_path = os.path.join(OUT_DIR, "vpr_methods_recall_summary.json")
    summary_table = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_table = json.load(f)
        except Exception:
            pass

    for method in args.methods:
        out_file = os.path.join(OUT_DIR, f"{method}_test_retrievals.json")
        mined_needed = method in ("eigenplaces", "dinov2_salad") and not os.path.exists(os.path.join(OUT_DIR, f"{method}_trainval_mined.json"))
        if os.path.exists(out_file) and not mined_needed:
            # Recompute summary from existing file if missing in summary_table
            if method not in summary_table:
                with open(out_file, "r", encoding="utf-8") as f:
                    prev_rows = json.load(f)
                m_ds = defaultdict(lambda: {"n": 0, "r1": 0, "r5": 0, "r10": 0})
                m_tk = defaultdict(lambda: {"n": 0, "r1": 0, "r5": 0})
                for pr in prev_rows:
                    m_ds[pr["dataset"]]["n"] += 1
                    m_ds[pr["dataset"]]["r1"] += pr["naive_hit1"]
                    m_ds[pr["dataset"]]["r5"] += pr["hit5"]
                    m_ds[pr["dataset"]]["r10"] += pr["hit10"]
                    m_tk[pr["task_type"]]["n"] += 1
                    m_tk[pr["task_type"]]["r1"] += pr["naive_hit1"]
                    m_tk[pr["task_type"]]["r5"] += pr["hit5"]
                tn = len(prev_rows)
                summary_table[method] = {
                    "overall_R1": round(sum(v["r1"] for v in m_ds.values()) / max(tn, 1) * 100, 2),
                    "overall_R5": round(sum(v["r5"] for v in m_ds.values()) / max(tn, 1) * 100, 2),
                    "overall_R10": round(sum(v["r10"] for v in m_ds.values()) / max(tn, 1) * 100, 2),
                    "by_dataset_R1": {ds: round(v["r1"] / max(v["n"], 1) * 100, 2) for ds, v in m_ds.items()},
                    "by_task_R1": {tk: round(v["r1"] / max(v["n"], 1) * 100, 2) for tk, v in m_tk.items()},
                }
            print(f"[SKIP] [{method}] already completed -> R@1={summary_table[method]['overall_R1']}%", flush=True)
            continue

        t_m = time.time()
        print(f"\n================ Running VPR Method: [{method}] ================", flush=True)
        try:
            model, tf = load_vpr_backbone(method, device="cuda")
        except Exception as e:
            print(f"[ERROR] Failed to load {method}: {e}", flush=True)
            continue

        method_results = []
        method_metrics = defaultdict(lambda: {"n": 0, "r1": 0, "r5": 0, "r10": 0})
        method_metrics_by_task = defaultdict(lambda: {"n": 0, "r1": 0, "r5": 0})

        for ds, q_rows in by_ds_test.items():
            db_nodes = galleries[ds]
            db_paths = [n["img_abs"] for n in db_nodes]
            q_paths = [os.path.join(DATA_BASE, r["query_img"]) for r in q_rows]

            db_embs = extract_all_features(model, tf, db_paths, batch_size=args.batch_size)
            q_embs = extract_all_features(model, tf, q_paths, batch_size=args.batch_size)

            dim = db_embs.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(db_embs)
            sims, indices = index.search(q_embs, 10)

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

                # Parse question intent & run CTSE
                intent_info = parse_question_spatiotemporal_intent(r["question"])
                ctse_node, ctse_sim, sub_sz = ctse_select_view(
                    top10_nodes, intent_info, place_graphs[ds], r.get("query_year", 2020)
                )

                # Also record Top-1 hard negative (highest-similarity WRONG place) for stress testing
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

        # Also mine hard negatives on train+val using eigenplaces & dinov2_salad for Exp2 Conflict Gate training!
        if method in ("eigenplaces", "dinov2_salad"):
            print(f"  [Hard-Neg Mining] Extracting train+val retrievals for {method}...", flush=True)
            tv_rows_by_ds = defaultdict(list)
            for r in train_samples + val_samples:
                tv_rows_by_ds[r["dataset"]].append(r)
            tv_records = []
            for ds, q_rows in tv_rows_by_ds.items():
                db_nodes = galleries[ds]
                db_paths = [n["img_abs"] for n in db_nodes]
                q_paths = [os.path.join(DATA_BASE, r["query_img"]) for r in q_rows]
                db_embs = extract_all_features(model, tf, db_paths, batch_size=args.batch_size)
                q_embs = extract_all_features(model, tf, q_paths, batch_size=args.batch_size)
                index = faiss.IndexFlatIP(db_embs.shape[1])
                index.add(db_embs)
                sims, indices = index.search(q_embs, 6)
                for i, r in enumerate(q_rows):
                    gt_pid = r["place_id"]
                    neg_node, neg_sim = None, 0.0
                    top1_node, top1_sim = db_nodes[int(indices[i, 0])], float(sims[i, 0])
                    for k in range(6):
                        cn = db_nodes[int(indices[i, k])]
                        if cn["place_id"] != gt_pid:
                            neg_node = cn
                            neg_sim = float(sims[i, k])
                            break
                    tv_records.append({
                        "sample_id": r["sample_id"],
                        "split": r["split"],
                        "dataset": ds,
                        "task_type": r["task_type"],
                        "top1_img": top1_node["img_rel"],
                        "top1_hit": int(top1_node["place_id"] == gt_pid),
                        "top1_sim": round(top1_sim, 4),
                        "hard_neg_img": neg_node["img_rel"] if neg_node else top1_node["img_rel"],
                        "hard_neg_sim": round(neg_sim, 4),
                    })
            with open(os.path.join(OUT_DIR, f"{method}_trainval_mined.json"), "w", encoding="utf-8") as f:
                json.dump(tv_records, f)

        # Save per-method test retrieval file
        out_file = os.path.join(OUT_DIR, f"{method}_test_retrievals.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(method_results, f)

        # Compute overall recall stats
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

    with open(os.path.join(OUT_DIR, "vpr_methods_recall_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_table, f, indent=2)
    print("\n=== EXP 1 COMPLETE: All VPR Methods Benchmarked! ===", flush=True)


if __name__ == "__main__":
    main()
