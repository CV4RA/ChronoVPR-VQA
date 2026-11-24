"""Build the dependency-free GitHub Pages site from the release artifacts."""

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

METHODS = (
    ("resnet50_avg", "ResNet50-Avg", "early"),
    ("resnet50_gem", "ResNet50-GeM", "early"),
    ("netvlad", "NetVLAD", "early"),
    ("patch_netvlad", "Patch-NetVLAD", "early"),
    ("cosplace", "CosPlace", "cnn"),
    ("eigenplaces", "EigenPlaces", "cnn"),
    ("convap", "Conv-AP", "cnn"),
    ("mixvpr", "MixVPR", "cnn"),
    ("dinov2_vitb14", "DINOv2-ViT-B/14", "vfm"),
    ("anyloc", "AnyLoc", "vfm"),
    ("selavpr", "SelaVPR", "vfm"),
    ("cricavpr", "CricaVPR", "vfm"),
    ("dinov2_salad", "DINOv2-SALAD", "vfm"),
    ("boq_dinov2", "BoQ-DINOv2", "vfm"),
    ("megaloc", "MegaLoc", "vfm"),
    ("clip_vitb32", "CLIP-ViT-B/32", "clip"),
    ("clip_vitl14", "CLIP-ViT-L/14", "clip"),
)

MAIN = {
    "Single view": "Single-View MLLM (No VPR)",
    "Naive memory": "Naive VPR+MLLM (boq_dinov2)",
    "CTSE + Gate": "Ours Zero-Shot (CTSE+Gate, boq_dinov2)",
    "Full system": "Ours Full (CTSE+Gate+LoRA, boq_dinov2)",
    "Oracle + LoRA": "Oracle Upper Bound (+LoRA)",
}


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def release_data():
    results = read_json("results/final_paper_results.json")
    gate = read_json("results/conflict_gate_eval_summary.json")["by_vpr_method"]
    stats = read_json("data/dataset_stats_clean.json")
    example_fields = (
        "sample_id", "split", "dataset", "task_type", "relative_yaw_deg",
        "year_gap", "question", "options", "correct_option", "evidence_summary",
    )
    samples = read_json("data/sample_preview.json")
    return {
        "methods": [
            {
                "id": method, "label": label, "family": family,
                "auroc": gate[method]["test_aliasing_detection_AUROC"],
                **results["table2_all_vpr"][method],
            }
            for method, label, family in METHODS
        ],
        "main": {label: results["table1_main"][key] for label, key in MAIN.items()},
        "noise": [
            {"rate": rate, **results["table4_noise_robustness"][f"noise_{rate}pct"]}
            for rate in (0, 20, 40, 60, 80)
        ],
        "ablation": results["table3_ablation"],
        "splits": stats["by_split"],
        "datasets": stats["by_dataset_and_split"],
        "examples": [
            {field: sample[field] for field in example_fields}
            for entries in samples.values()
            for sample in entries
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="dist", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT or output == ROOT / "site":
        parser.error("Choose a separate output directory.")

    output.mkdir(parents=True, exist_ok=True)
    for filename in ("index.html", "style.css", "app.js"):
        shutil.copyfile(ROOT / "site" / filename, output / filename)
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    for filename in ("teaser.png", "framework.png", "release_results.png"):
        shutil.copyfile(ROOT / "assets" / filename, assets / filename)

    # Local previews can open index.html directly, including on file://.
    payload = json.dumps(release_data(), ensure_ascii=True, separators=(",", ":"))
    (output / "data.js").write_text(f"window.CHRONOVPR_DATA = {payload};\n", encoding="utf-8")
    (output / ".nojekyll").touch()
    print(f"Built {output / 'index.html'}")


if __name__ == "__main__":
    main()
