# Dataset Card for `ChronoVPR-VQA`

## Dataset Summary

**`ChronoVPR-VQA`** is a strictly blind-filtered benchmark of **30,113 unique physical places** designed to evaluate physical-world multimodal reasoning beyond the single-view observational horizon. Spanning **six canonical Visual Place Recognition (VPR) benchmarks**—**GSV-Cities** (23 global cities), **Pittsburgh250K**, **SF-XL**, **MSLS**, **Nordland** (zero-shot OOD), and **SPED** (zero-shot OOD)—every sample pairs an online street-view query observation $I_q$ with a geo-referenced chrono-topological complementary memory view $I_m$ along three reasoning axes:

1. **Task A — Occlusion Recovery (9,294 places / 3,329 test):** A deterministic foreground barrier obscures the queried storefront/object in $I_q$; resolving the question requires localizing an unoccluded historical pass at the same physical place.
2. **Task B — Out-of-FoV Topological Reasoning (10,163 places / 2,656 test):** The question probes architectural landmarks visible only after an ego-centric camera rotation ($\Delta\theta \in \{\pm 90^\circ, 180^\circ\}$), exposing the *Similarity–Complementarity Paradox*.
3. **Task C — Longitudinal Temporal Evolution (10,656 places / 2,062 test):** The question examines multi-year urban, seasonal, or weather-induced changes ($\Delta t$ up to 15 years, mean 6.7 years) across distinct capture passes.

---

## Dataset Splits & Statistics

| Split | File | Total Places | Task A (Occ.) | Task B (FoV) | Task C (Temp.) | Option A / B / C / D |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | `annotations/chrono_vpr_vqa_train.json` | 19,573 | 5,314 | 6,614 | 7,645 | 4,957 / 4,886 / 4,894 / 4,836 |
| **Val** | `annotations/chrono_vpr_vqa_val.json` | 2,493 | 651 | 893 | 949 | 610 / 651 / 614 / 618 |
| **Test** | `annotations/chrono_vpr_vqa_test.json` | 8,047 | 3,329 | 2,656 | 2,062 | 1,996 / 1,994 / 2,017 / 2,040 |
| **Total** | — | **30,113** | **9,294** | **10,163** | **10,656** | **7,563 / 7,531 / 7,525 / 7,494** |

- **Strict Blind Filtering:** Every candidate question was verified against single-view guessing; any sample answerable from $I_q$ alone was eliminated, yielding a rigorous **28.77%** single-view blind baseline on the verified pool.
- **Zero-Shot Out-of-Domain Holdouts:** All 1,347 **Nordland** (4-season rail) and 229 **SPED** (extreme weather/night) places reside exclusively in `test` with **0 samples** in `train`/`val`.

---

## JSON Schema

Each entry in `chrono_vpr_vqa_{train,val,test}.json` follows this schema (see [`sample_preview.json`](sample_preview.json) for full examples):

```json
{
  "sample_id": "chrono_vpr_vqa_016047",
  "pair_id": "pair_16047",
  "split": "test",
  "dataset": "GSV-Cities",
  "city": "London",
  "task_type": "occlusion_recovery",
  "place_id": "London_0001234",
  "query_img": "chrono_vpr_vqa_30k/images/08/pair_16047_query.jpg",
  "oracle_memory_img": "chrono_vpr_vqa_30k/images/08/pair_16047_memory.jpg",
  "query_orig_name": "<original upstream filename for exact image linking>",
  "memory_orig_name": "<original upstream filename for exact image linking>",
  "query_year": 2019,
  "memory_year": 2009,
  "year_gap": 10,
  "relative_yaw_deg": 19,
  "direction_prompt": "same street location",
  "question": "What type of awning is visible on the building behind the blocked area?",
  "options": {
    "A": "Brown awning with green logo",
    "B": "Blue striped awning",
    "C": "Red awning with white logo",
    "D": "No awning is visible"
  },
  "correct_option": "C",
  "passed_blind_filter": true
}
```

---

## Obtaining Upstream Street-View Imagery

To respect upstream dataset licenses and geographic privacy policies, `ChronoVPR-VQA` distributes **only derived spatio-temporal annotations, original upstream filenames (`query_orig_name`, `memory_orig_name`), and indexing scripts**. Download the six canonical academic VPR datasets directly from their official sources:

1. **GSV-Cities:** [amaralibey/gsv-cities](https://github.com/amaralibey/gsv-cities)
2. **Pittsburgh250K:** [Visual Place Recognition Benchmarks (Torii et al.)](http://www.ok.ctrl.titech.ac.jp/~torii/project/repttile/)
3. **SF-XL:** [gmberton/CosPlace](https://github.com/gmberton/CosPlace)
4. **MSLS:** [Mapillary Street-Level Sequences](https://www.mapillary.com/dataset/places)
5. **Nordland:** [VPR-Bench / Nordland Dataset](https://github.com/MubarizZaffar/VPR-Bench)
6. **SPED:** [SPEDTest / VPR-Bench](https://github.com/MubarizZaffar/VPR-Bench)

---

## Ethics, Privacy & License

- All upstream street-view datasets apply face and license-plate blurring prior to academic release.
- All derived annotations in `ChronoVPR-VQA` are released under **CC-BY-4.0** for non-commercial academic research.
