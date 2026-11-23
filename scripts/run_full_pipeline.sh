#!/usr/bin/env bash
# End-to-End 4-Stage Reproduction Pipeline for ChronoVPR-VQA
set -euo pipefail

export CHRONOVPR_ROOT="${CHRONOVPR_ROOT:-$(pwd)}"
echo "[1/5] Verifying ChronoVPR-VQA annotations..."
python scripts/verify_dataset.py

echo "[2/5] Running 17-method (4-Era) VPR extraction & CTSE sub-graph expansion..."
python src/vpr_retrieval/run_vpr_gallery_and_ctse.py
python src/vpr_retrieval/run_remaining_vpr_backbones.py

echo "[3/5] Training & zero-shot scoring Scale-Invariant Conflict Resolution Gate..."
python src/conflict_gate/train_conflict_gate.py
python src/conflict_gate/score_vpr_retrievals.py

echo "[4/5] Training Conflict-Aware LoRA MLLM Reasoner (Qwen2.5-VL-7B, r=16)..."
python src/mllm_reasoner/train_lora_reasoner.py --num_train_samples 3600 --lr 1.5e-4 --grad_accum 8

echo "[5/5] Running full 8,047-query deterministic evaluation & generating paper tables/figures..."
python src/mllm_reasoner/evaluate_full_benchmark.py
python src/analysis/compute_paper_tables.py
python src/analysis/plot_main_figures.py
echo "ALL STAGES COMPLETED SUCCESSFULLY."
