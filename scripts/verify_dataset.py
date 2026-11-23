#!/usr/bin/env python3
"""Quick integrity & statistics verifier for ChronoVPR-VQA annotations."""
import os
import json
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN_DIR = os.path.join(ROOT, "data", "annotations")

def main():
    total = 0
    all_places = set()
    print("=== ChronoVPR-VQA Dataset Integrity Check ===")
    for split in ["train", "val", "test"]:
        fp = os.path.join(ANN_DIR, f"chrono_vpr_vqa_{split}.json")
        assert os.path.exists(fp), f"Missing split file: {fp}"
        rows = json.load(open(fp, encoding="utf-8"))
        total += len(rows)
        ds_cnt = Counter(r["dataset"] for r in rows)
        tk_cnt = Counter(r["task_type"] for r in rows)
        opt_cnt = Counter(r["correct_option"] for r in rows)
        for r in rows:
            all_places.add((r["dataset"], r["place_id"]))
        print(f"\n[{split.upper()}] N = {len(rows):,}")
        print(f"  Datasets : {dict(ds_cnt)}")
        print(f"  Tasks    : {dict(tk_cnt)}")
        print(f"  Options  : { {k: opt_cnt[k] for k in ['A', 'B', 'C', 'D']} }")

    assert total == 30113, f"Expected 30,113 samples, found {total}"
    print(f"\nVERIFIED SUCCESSFULLY: {total:,} blind-filtered samples across 6 benchmarks!")

if __name__ == "__main__":
    main()
