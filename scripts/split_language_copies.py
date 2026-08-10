#!/usr/bin/env python3
"""Split the combined v2 build into the two released language copies.

    python scripts/split_language_copies.py path/to/dental_clinical_dataset_v2 \
        --outdir path/to/release

Produces `dental_clinical_dataset_v2_zh/` (structured fields + the eight
Chinese narrative fields) and `dental_clinical_dataset_v2_en/` (structured
fields + the eight `_en` fields), each with the shared `metadata.csv`,
`panoramic_radiographs/` and release documentation. This is the final step of
the release-assembly chain: `build_v2.py` writes the combined bilingual build,
and this script derives the two shipped archives from it. The translation
screening log ships with the English copy only.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

STRUCTURED = [
    "patient_id", "image_file", "has_image", "age", "sex",
    "attending_physician", "chief_complaint_category", "treatment_category",
    "primary_diagnosis_icd",
]
NARRATIVE = [
    "chief_complaint", "history_of_present_illness", "oral_examination",
    "imaging_examination", "primary_diagnosis", "treatment_plan",
    "procedure", "physician_advice",
]
SEX_EN = {"男": "male", "女": "female", "male": "male", "female": "female"}
SHARED_FILES = ["metadata.csv", "CHANGELOG.md", "build_report.json"]


def write_copy(records: list[dict], src: Path, out: Path, lang: str) -> None:
    recdir = out / "clinical_records"
    recdir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        keep = {k: rec[k] for k in STRUCTURED if k in rec}
        if lang == "en":
            keep["sex"] = SEX_EN.get(keep.get("sex", ""), keep.get("sex", ""))
            for f in NARRATIVE:
                keep[f + "_en"] = rec.get(f + "_en", "")
        else:
            for f in NARRATIVE:
                keep[f] = rec.get(f, "")
        path = recdir / f"{rec['patient_id']}.json"
        path.write_text(json.dumps(keep, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    imgdir = out / "panoramic_radiographs"
    if not imgdir.exists():
        shutil.copytree(src / "panoramic_radiographs", imgdir)
    for name in SHARED_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, out / name)
    if lang == "en" and (src / "translation_screening_log.json").exists():
        shutil.copy2(src / "translation_screening_log.json",
                     out / "translation_screening_log.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("combined", type=Path,
                    help="combined bilingual v2 build from build_v2.py")
    ap.add_argument("--outdir", type=Path, default=Path("release"))
    args = ap.parse_args()

    records = [json.loads(p.read_text(encoding="utf-8"))
               for p in sorted((args.combined / "clinical_records").glob("*.json"))]
    print(f"loaded {len(records)} combined records")

    write_copy(records, args.combined,
               args.outdir / "dental_clinical_dataset_v2_zh", "zh")
    write_copy(records, args.combined,
               args.outdir / "dental_clinical_dataset_v2_en", "en")
    print(f"wrote zh and en copies under {args.outdir}")
    print("note: per-copy README.md files are authored by hand, not generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
