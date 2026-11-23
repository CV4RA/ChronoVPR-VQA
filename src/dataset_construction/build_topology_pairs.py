"""
Step 1: Parse SF-XL and GSV-Cities metadata into unified Chrono-Topological Place Nodes.

Supported raw formats:
1. SF-XL filenames:
   @UTM_east@UTM_north@UTM_zone_num@UTM_zone_let@lat@lon@pano_id@tile_num@heading@pitch@roll@height@timestamp@note@.jpg
2. GSV-Cities:
   Images/{City}/{city}_{place_id}_{year}_{month}_{northdeg}_{lat}_{lon}_{panoid}.jpg
"""

import os
import glob
import json
import math
from collections import defaultdict
from typing import Dict, List, Any, Optional


def parse_sfxl_filename(filepath: str) -> Optional[Dict[str, Any]]:
    """Parse an SF-XL @-delimited filename into structured metadata."""
    fname = os.path.basename(filepath)
    parts = fname.split("@")
    if len(parts) < 14:
        return None
    try:
        utm_east = float(parts[1])
        utm_north = float(parts[2])
        lat = float(parts[5])
        lon = float(parts[6])
        pano_id = parts[7]
        heading = float(parts[9]) if parts[9] else 0.0
        timestamp = parts[13]  # e.g. '201708' or '2014'
        year = int(timestamp[:4]) if len(timestamp) >= 4 and timestamp[:4].isdigit() else -1
        return {
            "dataset": "sf_xl",
            "img_path": filepath,
            "utm_east": utm_east,
            "utm_north": utm_north,
            "lat": lat,
            "lon": lon,
            "pano_id": pano_id,
            "heading": int(round(heading)) % 360,
            "timestamp": timestamp,
            "year": year,
        }
    except (ValueError, IndexError):
        return None


def parse_gsv_cities_filename(filepath: str, city_name: str = "") -> Optional[Dict[str, Any]]:
    """
    Parse a GSV-Cities image filename:
    Typical pattern: {city}_{place_id}_{year}_{month}_{northdeg}_{lat}_{lon}_{panoid}.jpg
    """
    fname = os.path.splitext(os.path.basename(filepath))[0]
    parts = fname.split("_")
    if len(parts) < 8:
        return None
    try:
        panoid = parts[-1]
        lon = float(parts[-2])
        lat = float(parts[-3])
        heading = float(parts[-4])
        month = int(parts[-5])
        year = int(parts[-6])
        place_id = parts[-7]
        city = city_name or "_".join(parts[:-7])
        return {
            "dataset": "gsv_cities",
            "city": city,
            "place_id": f"{city}_{place_id}",
            "img_path": filepath,
            "lat": lat,
            "lon": lon,
            "pano_id": panoid,
            "heading": int(round(heading)) % 360,
            "timestamp": f"{year:04d}{month:02d}",
            "year": year,
        }
    except (ValueError, IndexError):
        return None


def angular_diff(h1: float, h2: float) -> float:
    """Compute signed relative angle (h2 - h1) in [-180, 180]. Positive means turn right."""
    return (h2 - h1 + 180) % 360 - 180


def build_sfxl_task_pairs(
    sfxl_root: str,
    grid_cell_meters: float = 10.0,
    max_pairs_per_task: int = 2000,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Scan an SF-XL directory and construct candidate image pairs for:
      - Task A: Occlusion-Recovery QA (same place & heading, different timestamps or synthetic mask target)
      - Task B: Out-of-FoV Topological QA (same pano_id, heading delta in {+90, -90, 180})
      - Task C: Chrono-Evolution QA (same UTM cell & similar heading, year gap >= 3 years)
    """
    image_files = glob.glob(os.path.join(sfxl_root, "**", "*.jpg"), recursive=True)
    print(f"[SF-XL] Found {len(image_files)} jpg files under {sfxl_root}")

    by_pano: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_spatial_cell: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)

    for fpath in image_files:
        meta = parse_sfxl_filename(fpath)
        if not meta:
            continue
        by_pano[meta["pano_id"]].append(meta)
        cell_key = (
            int(meta["utm_east"] // grid_cell_meters),
            int(meta["utm_north"] // grid_cell_meters),
        )
        by_spatial_cell[cell_key].append(meta)

    task_pairs = {
        "task_a_occlusion": [],
        "task_b_out_of_fov": [],
        "task_c_temporal_evolution": [],
    }

    direction_names = {
        90: "向右转90度 (90 degrees to the right)",
        -90: "向左转90度 (90 degrees to the left)",
        180: "向后转180度 (180 degrees behind)",
    }
    for pano_id, views in by_pano.items():
        if len(task_pairs["task_b_out_of_fov"]) >= max_pairs_per_task:
            break
        heading_map = {v["heading"]: v for v in views}
        for h_query, q_view in heading_map.items():
            for delta_deg, dir_desc in direction_names.items():
                target_h = (h_query + delta_deg) % 360
                if target_h in heading_map:
                    task_pairs["task_b_out_of_fov"].append({
                        "task_type": "out_of_fov",
                        "place_id": f"sfxl_pano_{pano_id}",
                        "query_img": q_view["img_path"],
                        "memory_img": heading_map[target_h]["img_path"],
                        "query_heading": h_query,
                        "memory_heading": target_h,
                        "relative_yaw_deg": delta_deg,
                        "direction_prompt": dir_desc,
                        "query_year": q_view["year"],
                        "memory_year": heading_map[target_h]["year"],
                    })
                    break

    for cell_key, views in by_spatial_cell.items():
        if (
            len(task_pairs["task_c_temporal_evolution"]) >= max_pairs_per_task
            and len(task_pairs["task_a_occlusion"]) >= max_pairs_per_task
        ):
            break
        by_heading: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for v in views:
            by_heading[v["heading"]].append(v)

        for h, h_views in by_heading.items():
            valid_views = [v for v in h_views if v["year"] > 0]
            valid_views.sort(key=lambda x: x["year"])
            if len(valid_views) < 2:
                continue
            oldest, newest = valid_views[0], valid_views[-1]
            year_gap = newest["year"] - oldest["year"]
            dist = math.hypot(
                newest["utm_east"] - oldest["utm_east"],
                newest["utm_north"] - oldest["utm_north"],
            )
            if dist > 15.0:
                continue

            if year_gap >= 3 and len(task_pairs["task_c_temporal_evolution"]) < max_pairs_per_task:
                task_pairs["task_c_temporal_evolution"].append({
                    "task_type": "temporal_evolution",
                    "place_id": f"sfxl_cell_{cell_key[0]}_{cell_key[1]}_h{h}",
                    "query_img": newest["img_path"],
                    "memory_img": oldest["img_path"],
                    "query_year": newest["year"],
                    "memory_year": oldest["year"],
                    "year_gap": year_gap,
                    "relative_yaw_deg": 0,
                })

            if year_gap >= 1 and len(task_pairs["task_a_occlusion"]) < max_pairs_per_task:
                task_pairs["task_a_occlusion"].append({
                    "task_type": "occlusion_recovery",
                    "place_id": f"sfxl_cell_{cell_key[0]}_{cell_key[1]}_h{h}",
                    "query_img": newest["img_path"],
                    "memory_img": oldest["img_path"],
                    "query_year": newest["year"],
                    "memory_year": oldest["year"],
                    "relative_yaw_deg": 0,
                })

    return task_pairs


def build_gsv_cities_task_pairs(
    gsv_root: str,
    max_pairs_per_task: int = 2000,
) -> Dict[str, List[Dict[str, Any]]]:
    """Scan GSV-Cities Images directory and group by place_id."""
    image_files = glob.glob(os.path.join(gsv_root, "**", "*.jpg"), recursive=True)
    print(f"[GSV-Cities] Found {len(image_files)} jpg files under {gsv_root}")

    by_place: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for fpath in image_files:
        city_folder = os.path.basename(os.path.dirname(fpath))
        meta = parse_gsv_cities_filename(fpath, city_name=city_folder)
        if meta:
            by_place[meta["place_id"]].append(meta)

    task_pairs = {
        "task_a_occlusion": [],
        "task_c_temporal_evolution": [],
    }

    for place_id, views in by_place.items():
        if len(views) < 2:
            continue
        views.sort(key=lambda x: (x["year"], x["timestamp"]))
        oldest, newest = views[0], views[-1]
        yaw_diff = abs(angular_diff(oldest["heading"], newest["heading"]))
        if yaw_diff > 25:
            continue
        year_gap = newest["year"] - oldest["year"]

        if year_gap >= 3 and len(task_pairs["task_c_temporal_evolution"]) < max_pairs_per_task:
            task_pairs["task_c_temporal_evolution"].append({
                "task_type": "temporal_evolution",
                "place_id": place_id,
                "query_img": newest["img_path"],
                "memory_img": oldest["img_path"],
                "query_year": newest["year"],
                "memory_year": oldest["year"],
                "year_gap": year_gap,
                "relative_yaw_deg": int(round(angular_diff(newest["heading"], oldest["heading"]))),
            })

        if len(task_pairs["task_a_occlusion"]) < max_pairs_per_task:
            task_pairs["task_a_occlusion"].append({
                "task_type": "occlusion_recovery",
                "place_id": place_id,
                "query_img": newest["img_path"],
                "memory_img": oldest["img_path"],
                "query_year": newest["year"],
                "memory_year": oldest["year"],
                "relative_yaw_deg": int(round(angular_diff(newest["heading"], oldest["heading"]))),
            })

    return task_pairs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Chrono-Topological VPR-VQA candidate pairs")
    parser.add_argument("--sfxl_root", type=str, default="", help="Path to SF-XL images directory")
    parser.add_argument("--gsv_root", type=str, default="", help="Path to GSV-Cities Images directory")
    parser.add_argument("--output_json", type=str, default="candidate_pairs.json", help="Output JSON file")
    parser.add_argument("--max_per_task", type=int, default=1000, help="Max pairs per task category")
    args = parser.parse_args()

    all_pairs = {
        "task_a_occlusion": [],
        "task_b_out_of_fov": [],
        "task_c_temporal_evolution": [],
    }

    if args.sfxl_root and os.path.exists(args.sfxl_root):
        sfxl_pairs = build_sfxl_task_pairs(args.sfxl_root, max_pairs_per_task=args.max_per_task)
        for k in all_pairs:
            all_pairs[k].extend(sfxl_pairs.get(k, []))

    if args.gsv_root and os.path.exists(args.gsv_root):
        gsv_pairs = build_gsv_cities_task_pairs(args.gsv_root, max_pairs_per_task=args.max_per_task)
        for k in gsv_pairs:
            all_pairs[k].extend(gsv_pairs[k])

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)

    print("Saved candidate pairs summary:")
    for k, v in all_pairs.items():
        print(f"  - {k}: {len(v)} pairs")
