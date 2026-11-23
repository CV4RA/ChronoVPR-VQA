# Trained Checkpoints & Model Zoo

## 1. Scale-Invariant Neural Conflict Resolution Gate *(Included in Repo)*
- **File:** `conflict_gate_best.pt` (2.12 MB, 0.53M parameters)
- **Backbone Input:** Frozen `DINOv2-ViT-B/14` $4\times 4$ spatial patch grids + CLS tokens + 7-D scale-invariant metadata (zero raw VPR cosine similarity input).
- **Performance:** Validation AUROC **0.8737** (F1 = 0.7937); transfers **zero-shot** across all 17 VPR methods (up to **0.8642** test AUROC on ResNet50-Avg and **0.8605** on NetVLAD).

## 2. Conflict-Aware LoRA MLLM Adapter *(Host on GitHub Releases / Hugging Face)*
- **Config Included:** `lora_adapter_config.json` (Qwen2.5-VL-7B-Instruct, `r=16`, `alpha=32`, `dropout=0.05`, 7 projections: `q/k/v/o/gate/up/down_proj`, 47.59M trainable parameters = 0.5706%).
- **Weights File:** `adapter_model.safetensors` (~190 MB). Because GitHub enforces a strict 100 MB per-file limit on git commits, upload `adapter_model.safetensors` as a **GitHub Release Asset** or to **Hugging Face Hub**, and place it at:
  ```text
  checkpoints/lora_chrono_vpr_vqa/
  ├── adapter_config.json
  └── adapter_model.safetensors
  ```
- Alternatively, reproduce the exact adapter from scratch in ~74 minutes on a single 32 GB GPU:
  ```bash
  python src/mllm_reasoner/train_lora_reasoner.py --num_train_samples 3600 --lr 1.5e-4 --grad_accum 8
  ```
