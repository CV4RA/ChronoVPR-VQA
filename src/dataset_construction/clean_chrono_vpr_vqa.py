"""
Strict post-processing quality sanitizer for ChronoVPR-VQA-43K.
Removes any samples with:
1. Schema placeholder echoes in open_answer (e.g., "2-8 words", "etc.", comma-listed prompt hints).
2. Meta-reference leakage in question or options (e.g., "memory view", "unobstructed view", "image 1/2").
3. Mismatched open_answer vs. options[correct_option] (e.g., open_answer says 'brown' while options have no 'brown').
4. Non-factual / overly meta options (e.g., "Style of the window (arched/rectangular/bay)").
"""

import os
import re
import json
from collections import defaultdict

IN_DIR = r".\ChronoVPR-VQA-43K-annotations"
OUT_DIR = r".\ChronoVPR-VQA-Clean"
os.makedirs(OUT_DIR, exist_ok=True)

BAD_OPEN_PATTERNS = [
    "2-8 words", "etc.", "or obscured", "brick, stone", "arched/rectangular",
    "short factual", "plausible option", "concise ground-truth",
]

BAD_QUESTION_PATTERNS = [
    "image 1", "image 2", "memory view", "unobstructed view", "unoccluded",
    "query image", "second image", "first image", "database image",
]

BAD_OPTION_PATTERNS = [
    "memory view", "image 1", "image 2", "arched/rectangular", "style of the window (",
    "plausible option", "number of stories of the building",
]

COLOR_WORDS = {"red", "blue", "green", "yellow", "white", "black", "brown", "gray", "grey", "orange", "beige", "tan", "pink", "purple"}


def is_clean_sample(r: dict) -> bool:
    q = str(r.get("question", "")).strip()
    q_low = q.lower()
    oa = str(r.get("open_answer", "")).strip()
    oa_low = oa.lower()
    opts = r.get("options", {})
    corr_k = str(r.get("correct_option", "")).strip().upper()

    if corr_k not in ("A", "B", "C", "D") or len(opts) != 4:
        return False
    corr_text = str(opts[corr_k]).strip()
    corr_low = corr_text.lower()

    if len(q) < 18 or len(oa) < 2 or len(corr_text) < 2:
        return False
    if any(p in oa_low for p in BAD_OPEN_PATTERNS):
        return False
    if any(p in q_low for p in BAD_QUESTION_PATTERNS):
        return False
    for k, v in opts.items():
        v_low = str(v).lower()
        if any(p in v_low for p in BAD_OPTION_PATTERNS):
            return False

    # Check color consistency between open_answer and correct option
    oa_colors = set(re.findall(r"\b[a-z]+\b", oa_low)) & COLOR_WORDS
    corr_colors = set(re.findall(r"\b[a-z]+\b", corr_low)) & COLOR_WORDS
    if oa_colors and corr_colors and not (oa_colors & corr_colors):
        return False

    # Ensure meaningful lexical overlap between open_answer and options[correct_option]
    stop = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "to", "at", "it", "be", "has", "have", "with", "and", "or", "by", "for"}
    oa_toks = set(re.findall(r"[a-z0-9]+", oa_low)) - stop
    corr_toks = set(re.findall(r"[a-z0-9]+", corr_low)) - stop
    if oa_toks and corr_toks and len(oa_toks & corr_toks) == 0:
        return False

    return True


def main():
    with open(os.path.join(IN_DIR, "chrono_vpr_vqa_all.json"), "r", encoding="utf-8") as f:
        raw_rows = json.load(f)

    clean_rows = [r for r in raw_rows if is_clean_sample(r)]
    for i, r in enumerate(clean_rows):
        r["sample_id"] = f"chrono_vpr_vqa_{i:06d}"

    splits = {"train": [], "val": [], "test": []}
    by_dataset = defaultdict(int)
    by_task = defaultdict(int)
    by_dataset_task = defaultdict(lambda: defaultdict(int))
    by_dataset_split = defaultdict(lambda: defaultdict(int))
    by_option = {"A": 0, "B": 0, "C": 0, "D": 0}
    sv_hits = 0
    sv_hits_ds = defaultdict(int)
    previews = defaultdict(list)

    for r in clean_rows:
        sp = r["split"]
        ds = r["dataset"]
        tk = r["task_type"]
        splits[sp].append(r)
        by_dataset[ds] += 1
        by_task[tk] += 1
        by_dataset_task[ds][tk] += 1
        by_dataset_split[ds][sp] += 1
        by_option[r["correct_option"]] += 1
        if r.get("single_view_forced_guess") == r["correct_option"]:
            sv_hits += 1
            sv_hits_ds[ds] += 1
        if len(previews[ds]) < 2:
            if tk not in {x["task_type"] for x in previews[ds]}:
                previews[ds].append(r)

    stats = {
        "raw_verified_count": len(raw_rows),
        "strict_clean_count": len(clean_rows),
        "clean_retention_rate": round(len(clean_rows) / max(len(raw_rows), 1), 4),
        "by_split": {k: len(v) for k, v in splits.items()},
        "by_task": dict(by_task),
        "by_dataset": dict(by_dataset),
        "by_dataset_and_task": {ds: dict(v) for ds, v in by_dataset_task.items()},
        "by_dataset_and_split": {ds: dict(v) for ds, v in by_dataset_split.items()},
        "by_correct_option": by_option,
        "overall_single_view_blind_acc": round(sv_hits / max(len(clean_rows), 1), 4),
        "single_view_blind_acc_by_dataset": {
            ds: round(sv_hits_ds[ds] / max(cnt, 1), 4) for ds, cnt in by_dataset.items()
        },
    }

    for sp, rows in splits.items():
        with open(os.path.join(OUT_DIR, f"chrono_vpr_vqa_{sp}.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, "chrono_vpr_vqa_all.json"), "w", encoding="utf-8") as f:
        json.dump(clean_rows, f, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, "dataset_stats_clean.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_DIR, "sample_preview_clean.json"), "w", encoding="utf-8") as f:
        json.dump(dict(previews), f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
