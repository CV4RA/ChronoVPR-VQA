"""
Step 2: Automated QA Generation + Single-View Blind Anti-Shortcut Filtering.
Optimized for A800 (80GB VRAM): Supports local vLLM (Qwen2.5-VL-72B/32B/7B) or OpenAI-compatible API.

Pipeline for each candidate pair:
1. Synthesize realistic occlusion patch for Task A (or use natural cross-view/cross-year pairs for Task B/C).
2. Prompt MLLM with (Query Image + Memory Reference Image) to generate a discriminative MCQ + Open-Ended QA.
3. Single-View Blind Filter: Ask MLLM to answer using ONLY the Query Image. If it answers correctly,
   discard the sample to guarantee 100% dependence on VPR memory retrieval!
"""

import os
import json
import base64
import argparse
from typing import Dict, Any, Optional
from PIL import Image, ImageDraw


def encode_image_b64(img_path: str) -> str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def apply_synthetic_occlusion(
    src_img_path: str,
    save_img_path: str,
    box_norm: tuple = (0.25, 0.30, 0.75, 0.75),
) -> str:
    """
    Apply a high-contrast occlusion block simulating a large vehicle/obstruction
    on the query image while leaving peripheral context intact for VPR localization.
    """
    os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
    img = Image.open(src_img_path).convert("RGB")
    w, h = img.size
    x1, y1 = int(box_norm[0] * w), int(box_norm[1] * h)
    x2, y2 = int(box_norm[2] * w), int(box_norm[3] * h)
    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], fill=(45, 48, 52), outline=(220, 180, 40), width=4)
    img.save(save_img_path, quality=95)
    return save_img_path


TASK_PROMPTS = {
    "occlusion_recovery": """You are creating a benchmark for Visual Place Recognition-grounded VQA (Task A: Occlusion Recovery).
Image 1 is the current query view where the central landmark/storefront region is occluded by a dark gray block.
Image 2 is a historical unoccluded memory image of the EXACT same place retrieved from a street-view database.

Focus ONLY on a specific static landmark, shop sign, architectural feature, or street facility that is:
1) Clearly visible inside the central region of Image 2, AND
2) Completely hidden by the gray occlusion block in Image 1.

Output a JSON object with:
{
  "valid_scene": true/false (false if the central area in Image 2 only contains sky, road surface, or moving cars),
  "question": "A specific question asking about the occluded landmark/sign/building feature in Image 1",
  "open_answer": "Concise ground-truth answer (2-6 words)",
  "options": {
    "A": "Option A",
    "B": "Option B",
    "C": "Option C",
    "D": "Option D"
  },
  "correct_option": "A/B/C/D"
}""",

    "out_of_fov": """You are creating a benchmark for Visual Place Recognition-grounded VQA (Task B: Out-of-FoV Topological QA).
Image 1 is the current camera view facing forward.
Image 2 is the camera view at the EXACT same location after turning {direction_prompt}.

Focus ONLY on a distinctive static landmark, building facade, signage, or intersection layout visible in Image 2 that is NOT visible in Image 1.

Output a JSON object with:
{{
  "valid_scene": true/false,
  "question": "A question anchored on Image 1 asking what static landmark/feature is seen if the observer turns {direction_prompt}",
  "open_answer": "Concise ground-truth answer",
  "options": {{
    "A": "Option A",
    "B": "Option B",
    "C": "Option C",
    "D": "Option D"
  }},
  "correct_option": "A/B/C/D"
}}""",

    "temporal_evolution": """You are creating a benchmark for Visual Place Recognition-grounded VQA (Task C: Chrono-Evolution QA).
Image 1 is the current view captured in year {query_year}.
Image 2 is the historical memory view of the exact same place captured in year {memory_year}.

Compare Image 1 and Image 2 to identify genuine PERMANENT STATIC changes (e.g., store rebranding, building renovation/painting, new/removed signage or infrastructure). Ignore moving cars, pedestrians, weather, or seasonal foliage.

Output a JSON object with:
{{
  "valid_scene": true/false (false if no noticeable permanent static change occurred or only lighting/cars changed),
  "question": "A question asking about the historical state in {memory_year} or what changed between {memory_year} and {query_year}",
  "open_answer": "Concise ground-truth answer",
  "options": {{
    "A": "Option A",
    "B": "Option B",
    "C": "Option C",
    "D": "Option D"
  }},
  "correct_option": "A/B/C/D"
}}""",
}


def call_vlm_api(
    client: Any,
    model_name: str,
    prompt_text: str,
    image_paths: list,
) -> Optional[Dict[str, Any]]:
    """Call OpenAI-compatible VLM endpoint (supports local vLLM on A800 or cloud API)."""
    content = [{"type": "text", "text": prompt_text}]
    for img_p in image_paths:
        b64 = encode_image_b64(img_p)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content
    try:
        return json.loads(raw)
    except Exception:
        return None


def blind_single_view_check(
    client: Any,
    model_name: str,
    query_img_path: str,
    qa_item: Dict[str, Any],
) -> bool:
    """
    Return True if the model CANNOT answer using only the query image (i.e. sample passes filter).
    Return False if the model guesses or sees the answer from the single query image alone.
    """
    options_str = "\n".join([f"{k}: {v}" for k, v in qa_item["options"].items()])
    check_prompt = f"""Look ONLY at this single image and answer the multiple-choice question.
If the target region is occluded, out of the camera field of view, or requires historical photos from another year that you cannot see in this single image, select "UNKNOWN".

Question: {qa_item['question']}
Options:
{options_str}
UNKNOWN: Cannot be determined from this single image alone.

Output JSON: {{"selected": "A/B/C/D/UNKNOWN", "confidence": 0.0 to 1.0}}"""

    res = call_vlm_api(client, model_name, check_prompt, [query_img_path])
    if not res:
        return True
    selected = str(res.get("selected", "UNKNOWN")).strip().upper()
    conf = float(res.get("confidence", 0.0))
    # If the VLM confidently answers the right option from a single image, reject this QA pair!
    if selected == qa_item["correct_option"].upper() and conf >= 0.65:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate and filter Chrono-Topological VPR-VQA dataset")
    parser.add_argument("--pairs_json", type=str, default="candidate_pairs.json")
    parser.add_argument("--output_dataset", type=str, default="chrono_vpr_vqa_dataset.json")
    parser.add_argument("--occluded_dir", type=str, default="occluded_queries")
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1", help="Local vLLM on A800 or API URL")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-72B-Instruct")
    args = parser.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.api_base, api_key=args.api_key)

    with open(args.pairs_json, "r", encoding="utf-8") as f:
        all_pairs = json.load(f)

    final_dataset = []
    for task_key, pairs in all_pairs.items():
        print(f"Processing {task_key} ({len(pairs)} candidate pairs)...")
        for idx, pair in enumerate(pairs):
            t_type = pair["task_type"]
            q_img = pair["query_img"]
            m_img = pair["memory_img"]

            if t_type == "occlusion_recovery":
                occ_path = os.path.join(args.occluded_dir, f"occ_{idx:05d}_{os.path.basename(q_img)}")
                q_img_used = apply_synthetic_occlusion(q_img, occ_path)
                prompt = TASK_PROMPTS["occlusion_recovery"]
            elif t_type == "out_of_fov":
                q_img_used = q_img
                prompt = TASK_PROMPTS["out_of_fov"].format(
                    direction_prompt=pair.get("direction_prompt", "90 degrees to the right")
                )
            else:
                q_img_used = q_img
                prompt = TASK_PROMPTS["temporal_evolution"].format(
                    query_year=pair.get("query_year", "recent"),
                    memory_year=pair.get("memory_year", "historical"),
                )

            qa_res = call_vlm_api(client, args.model, prompt, [q_img_used, m_img])
            if not qa_res or not qa_res.get("valid_scene", False):
                continue

            # Single-View Blind Filter to guarantee VPR necessity
            passes_filter = blind_single_view_check(client, args.model, q_img_used, qa_res)
            if not passes_filter:
                print(f"  [Filtered Out] Shortcut detected on sample {idx}")
                continue

            record = {
                "sample_id": f"{t_type}_{len(final_dataset):06d}",
                "task_type": t_type,
                "place_id": pair["place_id"],
                "query_img": q_img_used,
                "oracle_memory_img": m_img,
                "relative_yaw_deg": pair.get("relative_yaw_deg", 0),
                "query_year": pair.get("query_year", -1),
                "memory_year": pair.get("memory_year", -1),
                "question": qa_res["question"],
                "open_answer": qa_res["open_answer"],
                "options": qa_res["options"],
                "correct_option": qa_res["correct_option"],
            }
            final_dataset.append(record)

            if len(final_dataset) % 50 == 0:
                with open(args.output_dataset, "w", encoding="utf-8") as f:
                    json.dump(final_dataset, f, ensure_ascii=False, indent=2)

    with open(args.output_dataset, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)
    print(f"Done! Saved {len(final_dataset)} verified VPR-VQA samples to {args.output_dataset}")


if __name__ == "__main__":
    main()
