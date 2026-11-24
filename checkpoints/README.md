# Trained Checkpoints & Model Zoo

## 1. Scale-Invariant Neural Conflict Resolution Gate *(Included in Repo)*
- **File:** `conflict_gate_best.pt` (2.12 MB, 0.53M parameters)
- **Backbone Input:** Frozen `DINOv2-ViT-B/14` $4\times 4$ spatial patch grids + CLS tokens + 7-D scale-invariant metadata (zero raw VPR cosine similarity input).
- **Performance:** Validation AUROC **0.8737** (F1 = 0.7937); transfers **zero-shot** across all 17 VPR methods (up to **0.8642** test AUROC on ResNet50-Avg and **0.8605** on NetVLAD).

## 2. Conflict-Aware LoRA MLLM Adapter
- **Config Included:** `lora_adapter_config.json` (Qwen2.5-VL-7B-Instruct, `r=16`, `alpha=32`, `dropout=0.05`, 7 projections: `q/k/v/o/gate/up/down_proj`, 47.59M trainable parameters = 0.5706%).
- **Weights file:** `adapter_model.safetensors` (~190 MB). This checkout contains the configuration. If you have the trained weights, use the layout expected by the evaluator:
  ```text
  workspace/lora_chrono_vpr_vqa/
  ├── adapter_config.json
  └── adapter_model.safetensors
  ```
- To train an adapter, prepare the dataset and retrieval outputs using the [reproduction guide](../README.md#reproducing-the-experiments), then run:
  ```bash
  python src/mllm_reasoner/train_lora_reasoner.py --num_train_samples 3600 --lr 1.5e-4 --grad_accum 8
  ```

## Upstream VPR assets

The primary runner fetches several encoders through PyTorch Hub or Hugging Face. It reads the EigenPlaces checkpoint from `workspace/torch_cache/hub/checkpoints/ResNet50_2048_eigenplaces.pth`.

The incremental runner expects these assets from the original experiment workspace. See [its model loaders](../src/vpr_retrieval/run_remaining_vpr_backbones.py) for checkpoint formats and input transforms.

| Retriever | Workspace-relative asset |
| --- | --- |
| NetVLAD | `vpr_repos/Patch-NetVLAD/` and `vpr_weights/pitts_WPCA4096.pth.tar` |
| Patch-NetVLAD | `vpr_repos/Patch-NetVLAD/` and `vpr_weights/mapillary_WPCA4096.pth.tar` |
| Conv-AP | `vpr_weights/resnet50_ConvAP_1024_2x2.ckpt` |
| MixVPR | `vpr_weights/resnet50_MixVPR_4096.ckpt` |
| AnyLoc | `vpr_repos/DINO/model_wrapper.py` with `AnyLocVladNoCacheDinov2` |
| SelaVPR | `vpr_repos/SelaVPR/` and `vpr_weights/SelaVPR_msls.pth` |
| CricaVPR | `vpr_repos/CricaVPR/` and `vpr_weights/CricaVPR.pth` |
