#!/usr/bin/env bash
# GPU experiment runner. Prepare the workspace and upstream assets listed in README.md first.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ ! -f workspace/ChronoVPR-VQA-Clean/chrono_vpr_vqa_all.json ]]; then
    echo "Prepare the experiment workspace as described in README.md." >&2
    exit 1
fi

echo "[1/5] Verifying ChronoVPR-VQA annotations..."
python scripts/verify_dataset.py

echo "[2/5] Running 17-method (4-Era) VPR extraction & CTSE sub-graph expansion..."
python src/vpr_retrieval/run_vpr_gallery_and_ctse.py \
    --methods resnet50_avg resnet50_gem cosplace eigenplaces dinov2_vitb14 \
    dinov2_salad boq_dinov2 megaloc clip_vitb32 clip_vitl14
python -m src.vpr_retrieval.run_remaining_vpr_backbones \
    --methods netvlad patch_netvlad convap mixvpr anyloc selavpr cricavpr

echo "[3/5] Training & zero-shot scoring Scale-Invariant Conflict Resolution Gate..."
python src/conflict_gate/train_conflict_gate.py
python -m src.conflict_gate.score_vpr_retrievals

echo "[4/5] Training Conflict-Aware LoRA MLLM Reasoner (Qwen2.5-VL-7B, r=16)..."
python src/mllm_reasoner/train_lora_reasoner.py --num_train_samples 3600 --lr 1.5e-4 --grad_accum 8

echo "[5/5] Running full 8,047-query deterministic evaluation..."
python src/mllm_reasoner/evaluate_full_benchmark.py
echo "Evaluation results written to workspace/final_paper_results.json."
