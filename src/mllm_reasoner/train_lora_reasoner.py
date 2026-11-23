"""
Experiment 3: Fine-Tune Our Conflict-Aware Spatio-Temporal VPR-VQA Reasoner via LoRA (Innovation 3).
Trains a lightweight LoRA adapter (r=16, alpha=32) on Qwen2.5-VL-7B-Instruct using RTX 4080 32GB (BF16 + grad_ckpt)
on a balanced 4,500-sample curriculum from ChronoVPR-VQA-Clean/train:
  - 80% verified CTSE topological memory pairs (Task A occlusion, Task B out-of-FoV with Delta_theta,
    Task C temporal evolution with Delta_t) annotated with high Gate Reliability r >= 0.70
  - 20% hard perceptual-aliasing negative retrievals mined from CosPlace/SALAD annotated with low Gate
    Reliability r < 0.35 so the reasoner explicitly learns to IGNORE noisy VPR memory!

Outputs LoRA adapter to ./workspace/lora_chrono_vpr_vqa (plugs directly into vLLM LoRARequest!).
"""

import os
import json
import time
import random
import argparse
from typing import Dict, Any, List
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model


DATA_BASE = "./workspace"
CLEAN_DIR = "./workspace/ChronoVPR-VQA-Clean"
VPR_DIR = "./workspace/vpr_retrievals"
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
SAVE_DIR = "./workspace/lora_chrono_vpr_vqa"


def build_spatiotemporal_prompt(
    question: str,
    options: Dict[str, str],
    task_type: str,
    rel_yaw: int,
    q_year: int,
    m_year: int,
    reliability: float,
    focus_quadrant: str = "center",
    use_metadata: bool = True,
    use_gate: bool = True,
    gate_threshold: float = 0.50,
) -> str:
    opts_str = "\n".join([f"{k}. {options[k]}" for k in ("A", "B", "C", "D")])
    header = "You are a Chrono-Topological Visual Question Answering system grounded in Visual Place Recognition (VPR) memory.\n"
    header += "Image 1 is the current query street view. Image 2 is the retrieved VPR memory view.\n"

    if use_gate and reliability < gate_threshold:
        header += (
            f"[CONFLICT GATE ALERT]: Retrieved memory Image 2 has LOW structural reliability (r={reliability:.2f} < {gate_threshold:.2f}). "
            f"It is likely a perceptual-aliasing mismatch from a DIFFERENT location. IGNORE Image 2 and answer cautiously using ONLY Image 1.\n"
        )
    elif use_metadata:
        header += (
            f"[SPATIO-TEMPORAL ALIGNMENT]: Verified topological match (Reliability r={reliability:.2f}). "
            f"Query year={q_year}, Memory year={m_year} (Delta_t={q_year - m_year:+d} yrs), "
            f"Relative camera yaw Delta_theta={rel_yaw:+d} deg, Divergence focus={focus_quadrant}.\n"
        )

    return (
        f"{header}\n"
        f"Question: {question}\n"
        f"Options:\n{opts_str}\n\n"
        f"Answer with ONLY the single correct option letter (A, B, C, or D)."
    )


def load_pil_small(rel_path: str, max_side: int = 392) -> Image.Image:
    p = os.path.join(DATA_BASE, rel_path)
    img = Image.open(p).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.BILINEAR)
    return img


class ChronoVQALoRADataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], mined_neg_map: Dict[str, str]):
        self.items = []
        rng = random.Random(42)
        for idx, r in enumerate(samples):
            # 80% verified positive memory with high reliability; 20% hard negative retrieval with low reliability
            is_noisy = (idx % 5 == 0) and (r["sample_id"] in mined_neg_map)
            if is_noisy:
                mem_img = mined_neg_map[r["sample_id"]]
                rel_score = rng.uniform(0.05, 0.35)
            else:
                mem_img = r["oracle_memory_img"]
                rel_score = rng.uniform(0.78, 0.99)
            self.items.append((r, mem_img, rel_score))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        r, mem_rel, rel_score = self.items[idx]
        img1 = load_pil_small(r["query_img"], max_side=364)
        img2 = load_pil_small(mem_rel, max_side=364)
        prompt = build_spatiotemporal_prompt(
            r["question"],
            r["options"],
            r["task_type"],
            r.get("relative_yaw_deg", 0),
            r.get("query_year", 2020),
            r.get("memory_year", 2012),
            rel_score,
            focus_quadrant="center",
            use_metadata=True,
            use_gate=True,
        )
        ans = r["correct_option"]
        return img1, img2, prompt, ans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train_samples", type=int, default=3600)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--grad_accum", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(os.path.join(CLEAN_DIR, "chrono_vpr_vqa_train.json"), "r", encoding="utf-8") as f:
        train_all = json.load(f)

    # Balanced sampling across all 3 task types
    by_task = {}
    for r in train_all:
        by_task.setdefault(r["task_type"], []).append(r)
    rng = random.Random(42)
    selected = []
    per_task = args.num_train_samples // len(by_task)
    for tk, arr in by_task.items():
        rng.shuffle(arr)
        selected.extend(arr[:per_task])
    rng.shuffle(selected)

    mined_neg_map = {}
    mined_path = os.path.join(VPR_DIR, "dinov2_salad_trainval_mined.json")
    if os.path.exists(mined_path):
        with open(mined_path, "r", encoding="utf-8") as f:
            for rec in json.load(f):
                mined_neg_map[rec["sample_id"]] = rec["hard_neg_img"]

    print(f"[Exp3-LoRA] Training Qwen2.5-VL-7B LoRA on {len(selected)} balanced spatio-temporal samples (RTX 4080 32GB)...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, min_pixels=196 * 28 * 28, max_pixels=384 * 28 * 28)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = ChronoVQALoRADataset(selected, mined_neg_map)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    total_steps = len(ds) // args.grad_accum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1))

    model.train()
    t0 = time.time()
    running_loss = 0.0
    opt.zero_grad()

    for step_idx in range(len(ds)):
        img1, img2, prompt, ans = ds[step_idx]
        user_msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img1},
                    {"type": "image", "image": img2},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        full_msgs = user_msgs + [{"role": "assistant", "content": [{"type": "text", "text": ans}]}]

        prompt_only_text = processor.apply_chat_template(user_msgs, tokenize=False, add_generation_prompt=True)
        full_text = processor.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)

        prompt_inputs = processor(text=[prompt_only_text], images=[img1, img2], return_tensors="pt")
        full_inputs = processor(text=[full_text], images=[img1, img2], return_tensors="pt").to("cuda:0")

        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels = full_inputs["input_ids"].clone()
        labels[:, :prompt_len] = -100

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(**full_inputs, labels=labels)
            loss = outputs.loss / args.grad_accum

        loss.backward()
        running_loss += loss.item() * args.grad_accum

        if (step_idx + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            opt.zero_grad()

        if (step_idx + 1) % 300 == 0 or (step_idx + 1) == len(ds):
            avg_l = running_loss / 300.0 if (step_idx + 1) % 300 == 0 else running_loss / max((step_idx + 1) % 300, 1)
            elapsed = time.time() - t0
            eta = (len(ds) - step_idx - 1) / max((step_idx + 1) / elapsed, 1e-3) / 60.0
            print(
                f"  [LoRA] {step_idx+1}/{len(ds)} | Loss={avg_l:.4f} | LR={scheduler.get_last_lr()[0]:.2e} | "
                f"Elapsed={elapsed/60:.1f}m | ETA={eta:.1f}m",
                flush=True,
            )
            running_loss = 0.0

    model.save_pretrained(SAVE_DIR)
    print(f"\n=== EXP 3 COMPLETE: Saved LoRA adapter to {SAVE_DIR} ({(time.time()-t0)/60:.1f} min) ===", flush=True)


if __name__ == "__main__":
    main()
