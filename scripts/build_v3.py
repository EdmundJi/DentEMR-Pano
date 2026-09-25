#!/usr/bin/env python3
"""Derive the v3 release from the v2 language copies.

    PYTHONPATH=code/src python code/scripts/build_v3.py \
        dataset/dental_clinical_dataset_v2 --outdir dataset/dental_clinical_dataset_v3 \
        --id-map private/id_map_v2_to_v3.csv \
        --quality-scores quality_scores_v2_ids.csv

v3 changes three things in the data and nothing else:

1. **Identifiers re-assigned.** Every ``patient_id`` (and therefore every
   image file name) is replaced by a freshly generated random study identifier
   of the form ``DP`` followed by six digits. The v1/v2 identifiers had been
   assigned during collection at the hospital workstation; v3 replaces them so
   that the released identifier demonstrably has no relation to any hospital
   numbering. The mapping is read from ``--id-map`` (created with fresh random
   numbers if the file does not exist), is held by the principal investigator,
   and is not published or shipped.
2. **Radiograph linkage repaired.** A byte-level duplicate check over the 457
   released image files (now part of ``dentemr_pano.validate``) found seven
   image files whose content is identical to another file linked to a
   *different* record. A panoramic radiograph belongs to one patient, so at
   least one record in each pair was paired with the wrong image. The correct
   owner could not be re-established from the source system, so both records
   in each pair are unlinked: ``has_image`` becomes ``no``, ``image_file``
   becomes empty, and the fourteen image files are removed.
3. **Two partial dates reduced.** Two month-day mentions found by the new
   partial-date probe are reduced to the month (``NARRATIVE_FIXES``, matched
   by content so that no v2 identifier appears in this public script).

The script verifies at the end that, record by record, the only fields that
differ from v2 are ``patient_id``, ``image_file``/``has_image`` (for the 14
unlinked records) and the two fixed narrative fields. It also regenerates
``metadata.csv``, ``build_report.json``, ``CHANGELOG.md`` (v3 section
prepended; identifiers in the inherited v2 notes are redacted), the per-copy
``README.md``, ships the rater score table as ``quality_scores.csv`` with its
``patient_id`` column re-keyed, runs the release validator on both copies
(writing ``residual_identifier_scan.json`` into each), and writes the two
release archives ``dental_clinical_dataset_v3_zh.zip`` and
``dental_clinical_dataset_v3_en.zip``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dentemr_pano.validate import audit  # noqa: E402

COPIES = ("zh", "en")
METADATA_COLS = [
    "patient_id", "image_file", "has_image", "age", "sex",
    "treatment_category", "primary_diagnosis_icd",
    "chief_complaint_category", "attending_physician",
]
NEW_ID = re.compile(r"^DP\d{6}$")
OLD_ID = re.compile(r"\b\d{10}\b")

#: Partial-date reductions applied in v3, matched by content: (field, old, new).
#: Both are dates a physician wrote into the narrative; the release policy
#: reduces dates to year-month granularity. Each pattern must match exactly one
#: record of the copy in which the field exists.
NARRATIVE_FIXES: list[tuple[str, str, str]] = [
    ("treatment_plan", "嘱患者10月6日复诊", "嘱患者10月复诊"),
    ("treatment_plan_en", "return for follow-up on October 6", "return for follow-up in October"),
    ("history_of_present_illness", "于8月28日16时30分左石在高速公路", "于8月在高速公路"),
    ("history_of_present_illness_en", "around 16:30 on August 28, a car accident",
     "in August, a car accident"),
]


def duplicate_groups(image_dir: Path) -> list[list[str]]:
    by_hash: dict[str, list[str]] = defaultdict(list)
    for p in sorted(image_dir.glob("*.png")):
        by_hash[hashlib.md5(p.read_bytes()).hexdigest()].append(p.name)
    return [sorted(v) for v in by_hash.values() if len(v) > 1]


def load_records(copy_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((copy_dir / "clinical_records").glob("*.json"))]


def load_or_create_id_map(path: Path, old_ids: list[str]) -> dict[str, str]:
    """Read the old->new identifier map, or create one with fresh random ids.

    The map is the only link between v3 identifiers and earlier ones. It must
    be kept out of the release and of any public repository.
    """
    if path.exists():
        with path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        idmap = {r["old_id"]: r["new_id"] for r in rows}
        missing = sorted(set(old_ids) - set(idmap))
        assert not missing, f"id map lacks {len(missing)} ids, e.g. {missing[:3]}"
        print(f"id map: read {len(idmap)} entries from {path}")
    else:
        rng = random.SystemRandom()
        new_ids: set[str] = set()
        while len(new_ids) < len(old_ids):
            new_ids.add(f"DP{rng.randrange(100000, 1000000):06d}")
        shuffled = list(new_ids)
        rng.shuffle(shuffled)
        idmap = dict(zip(old_ids, shuffled))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["old_id", "new_id"])
            for old in old_ids:
                w.writerow([old, idmap[old]])
        print(f"id map: created {len(idmap)} fresh random ids -> {path} (KEEP PRIVATE)")
    assert all(NEW_ID.match(v) for v in idmap.values()), "new ids must be DP + 6 digits"
    assert len(set(idmap.values())) == len(idmap), "new ids must be unique"
    return idmap


def changelog_v3(unlinked_new: list[tuple[str, str]], fixed_new: list[str], lang: str) -> str:
    pairs = "\n".join(f"  - {a} and {b}" for a, b in unlinked_new)
    fixed = ", ".join(fixed_new)
    return f"""# Changelog - dental_clinical_dataset v3 ({'Chinese source' if lang == 'zh' else 'English'} copy)

## v3 (this release)

- **Identifiers re-assigned.** Every `patient_id`, and with it every image
  file name (`{{patient_id}}_panorama.png`), was replaced by a freshly
  generated random study identifier of the form `DP` followed by six digits.
  The identifiers used in v1 and v2 had been assigned during collection at
  the hospital workstation; v3 replaces them so that the released identifier
  demonstrably has no relation to any hospital numbering. No mapping to the
  v1/v2 identifiers is published or shipped; the identifiers below are v3
  identifiers. Records are otherwise the same records as in v2.
- **Radiograph linkage repaired.** A byte-level duplicate check over the 457
  image files released in v2 found seven image files whose content is
  identical to a file linked to a different record (the same screen capture
  had been saved under two identifiers during collection). Because the
  correct owner of each image could not be re-established from the source
  system, both records of each pair were unlinked from the image:
  `has_image` is now `no` and `image_file` is empty for the 14 records listed
  below, and the 14 image files were removed from the archive. All narrative
  and structured fields of these records are unchanged, and no record was
  dropped. The release now pairs 443 of 463 records with a radiograph (6
  records had no retrievable radiograph in v1/v2; 14 were unlinked in v3).
{pairs}
- **Two partial dates reduced.** A partial-date probe (month-day without a
  year, or a time of day) added to the released validator found two mentions
  written by physicians into narratives: a follow-up date in `treatment_plan`
  and the date and time of an accident in `history_of_present_illness`
  (records {fixed}). Both were reduced to the month in both language copies,
  in line with the release policy of year-month granularity for dates.
- **`quality_scores.csv` restored.** The rater score table of the quality
  evaluation (160 cases x 3 raters x 23 items, long layout; `patient_id`
  given, in v3 identifiers, for the 158 evaluated cases in the release)
  shipped with v1 but was omitted from v2; it is included again so the
  reliability statistics can be recomputed from the archive with the
  released code.
- `residual_identifier_scan.json` added: the full-corpus scan of this copy
  by the released validator (`dentemr_pano.validate`), which now also runs
  the duplicate-image check above and the partial-date probe.
- Provenance note corrected (see the end of the inherited v2 notes below).

"""


def inherit_v2_changelog(old_log: str) -> str:
    old_log = old_log.replace("# Changelog - dental_clinical_dataset v2", "## v2")
    old_log = re.sub(
        r"## Provenance Note.*\Z",
        "## Provenance Note (as corrected in v3)\n\n"
        "The v2 changelog said that the clinical text fields followed an HIS export "
        "and the radiographs a PACS/DICOM export. That was inaccurate: the records "
        "were transcribed by hand from photographs of the HIS record screen (double "
        "entry, reconciled against the source) and the radiographs were saved from "
        "the PACS viewer with its window-capture function as 8-bit bitmaps; no DICOM "
        "object or header was exported. See the v3 section above and the data "
        "descriptor's Methods.\n",
        old_log, flags=re.S)
    old_log = old_log.replace("../dental_clinical_dataset_v2_en/",
                              "the English copy (dental_clinical_dataset_v3_en/)")
    old_log = old_log.replace("../dental_clinical_dataset_v2_zh/",
                              "the Chinese copy (dental_clinical_dataset_v3_zh/)")
    # v1/v2 identifiers are not carried into v3 documentation.
    old_log = OLD_ID.sub("[identifier re-assigned in v3]", old_log)
    return old_log


def readme_v3(text: str) -> str:
    text = text.replace("v2 release", "v3 release")
    text = text.replace("dental_clinical_dataset_v2_en", "dental_clinical_dataset_v3_en")
    text = text.replace("dental_clinical_dataset_v2_zh", "dental_clinical_dataset_v3_zh")
    text = text.replace(
        "- **Panoramic radiographs (PNG)**: 457 (1722 x 922 pixels, 8-bit single-channel grayscale)",
        "- **Panoramic radiographs (PNG)**: 443 (1722 x 922 pixels, 8-bit single-channel "
        "grayscale); 20 records have no linked radiograph (6 not retrievable at "
        "collection, 14 unlinked in v3 after a duplicate-image check, see CHANGELOG.md)")
    text = text.replace(
        "together with a paired panoramic radiograph.",
        "together with a paired panoramic radiograph for 443 of the 463 cases.")
    text = text.replace(
        "97.25%", "97.25% before correction (all identified errors corrected in the release)")
    text = text.replace(
        "- **Inter-rater reliability**: ICC(2,1) = 0.666; Gwet's AC = 0.984",
        "- **Inter-rater reliability**: ICC(2,1) = 0.666 (95% CI 0.584-0.737); Gwet's AC = 0.984; "
        "rater scores in `quality_scores.csv`")
    text = text.replace(
        "- Protected identifiers and date reductions follow the release policy described in the source publication.",
        "- Protected identifiers and date reductions follow the release policy described "
        "in the source publication. `residual_identifier_scan.json` holds the "
        "full-corpus scan of this copy by the released validator.\n"
        "- `patient_id` values are random study identifiers (`DP` + six digits) "
        "assigned for v3; they replace the identifiers of v1/v2 and have no relation "
        "to hospital numbering. No mapping to earlier identifiers is published.")
    text = OLD_ID.sub("DP######", text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("v2", type=Path, help="directory holding the v2 zh and en copies")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--id-map", type=Path, required=True,
                    help="CSV old_id,new_id; created with fresh random ids if absent. "
                         "Keep private; never ship or commit.")
    ap.add_argument("--quality-scores", type=Path, default=None,
                    help="rater score table (long layout, v2 patient_id column) to "
                         "re-key and ship in both copies")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    src = {lang: args.v2 / f"dental_clinical_dataset_v2_{lang}" for lang in COPIES}
    dst = {lang: args.outdir / f"dental_clinical_dataset_v3_{lang}" for lang in COPIES}
    for lang in COPIES:
        if not src[lang].is_dir():
            raise SystemExit(f"missing v2 copy: {src[lang]}")

    old_ids = [r["patient_id"] for r in load_records(src["zh"])]
    idmap = load_or_create_id_map(args.id_map, old_ids)

    groups = duplicate_groups(src["zh"] / "panoramic_radiographs")
    assert groups == duplicate_groups(src["en"] / "panoramic_radiographs"), \
        "zh and en copies do not share identical image files"
    assert all(len(g) == 2 for g in groups), groups
    unlinked_old = sorted(n[:10] for g in groups for n in g)
    removed_files = sorted(n for g in groups for n in g)
    unlinked_new_pairs = sorted(tuple(sorted((idmap[g[0][:10]], idmap[g[1][:10]])))
                                for g in groups)
    print(f"duplicate groups: {len(groups)}  records to unlink: {len(unlinked_old)}")

    stats: Counter = Counter()
    for lang in COPIES:
        out = dst[lang]
        if out.exists():
            shutil.rmtree(out)
        (out / "clinical_records").mkdir(parents=True)
        (out / "panoramic_radiographs").mkdir()

        v2_records = load_records(src[lang])
        v2_by_old = {r["patient_id"]: json.loads(json.dumps(r)) for r in v2_records}
        records = []
        for rec in v2_records:
            old = rec["patient_id"]
            new = idmap[old]
            rec["patient_id"] = new
            if old in unlinked_old:
                assert rec["has_image"] == "yes" and rec["image_file"] in removed_files
                rec["has_image"] = "no"
                rec["image_file"] = ""
                stats[f"{lang} records unlinked"] += 1
            elif rec["has_image"] == "yes":
                assert rec["image_file"] == f"{old}_panorama.png", rec["image_file"]
                rec["image_file"] = f"{new}_panorama.png"
            else:
                assert rec["image_file"] == ""
            records.append(rec)

        fixed_new: list[str] = []
        for field, old_text, new_text in NARRATIVE_FIXES:
            if field not in records[0]:
                continue
            hits = [r for r in records if old_text in str(r.get(field) or "")]
            assert len(hits) == 1, (field, old_text, len(hits))
            hits[0][field] = hits[0][field].replace(old_text, new_text)
            fixed_new.append(hits[0]["patient_id"])
            stats[f"{lang} partial dates reduced"] += 1
        fixed_new = sorted(set(fixed_new))

        records.sort(key=lambda r: r["patient_id"])
        for rec in records:
            # v2 files use CRLF line endings; keep the convention.
            text = json.dumps(rec, ensure_ascii=False, indent=2)
            (out / "clinical_records" / f"{rec['patient_id']}.json").write_bytes(
                (text.replace("\n", "\r\n") + "\r\n").encode("utf-8"))

        for p in sorted((src[lang] / "panoramic_radiographs").glob("*.png")):
            if p.name in removed_files:
                stats[f"{lang} images removed"] += 1
                continue
            shutil.copy2(p, out / "panoramic_radiographs" / f"{idmap[p.name[:10]]}_panorama.png")
            stats[f"{lang} images kept"] += 1

        with (out / "metadata.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=METADATA_COLS)
            w.writeheader()
            for rec in records:
                w.writerow({c: rec.get(c, "") for c in METADATA_COLS})

        (out / "CHANGELOG.md").write_text(
            changelog_v3(unlinked_new_pairs, fixed_new, lang)
            + "# Earlier changes (v2)\n\n"
            + inherit_v2_changelog((src[lang] / "CHANGELOG.md").read_text(encoding="utf-8")),
            encoding="utf-8")
        (out / "README.md").write_text(
            readme_v3((src[lang] / "README.md").read_text(encoding="utf-8")),
            encoding="utf-8")
        extra = src[lang] / "translation_screening_log.json"
        if extra.exists():
            shutil.copy2(extra, out / extra.name)

        if args.quality_scores:
            with args.quality_scores.open(encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
                cols = list(rows[0].keys())
            for r in rows:
                if r.get("patient_id"):
                    r["patient_id"] = idmap[r["patient_id"]]
            with (out / "quality_scores.csv").open("w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
                w.writeheader()
                w.writerows(rows)

        # Verify: only the intended fields differ from v2, record by record.
        new_to_old = {n: o for o, n in idmap.items()}
        fixed_fields = {f for f, _, _ in NARRATIVE_FIXES}
        for rec in records:
            old = new_to_old[rec["patient_id"]]
            v2 = v2_by_old[old]
            assert set(rec) == set(v2), rec["patient_id"]
            diff = {k for k in rec if rec[k] != v2[k]}
            allowed = {"patient_id"} | ({"image_file"} if v2["has_image"] == "yes" else set())
            if old in unlinked_old:
                allowed |= {"image_file", "has_image"}
            if rec["patient_id"] in fixed_new:
                allowed |= fixed_fields
            assert diff <= allowed, (rec["patient_id"], diff - allowed)

        res = audit(out)
        (out / "residual_identifier_scan.json").write_text(
            json.dumps({
                "copy": lang,
                "validator": "dentemr_pano.validate (release v3)",
                "records_scanned": res.n_records,
                "probes": sorted(res.stats["deid_hits"]),
                "hits_per_probe": res.stats["deid_hits"],
                "records_flagged": 0 if not any(res.stats["deid_hits"].values()) else None,
                "duplicate_image_groups": res.stats.get("duplicate_image_groups"),
                "errors": res.errors,
                "warnings": res.warnings,
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not res.ok:
            raise SystemExit(f"{lang}: validator reports errors: {res.errors}")

        has = Counter(r["has_image"] for r in records)
        report = {
            "release": "dental_clinical_dataset_v3",
            "language_copy": "Chinese source copy" if lang == "zh" else "English copy",
            "record_count": len(records),
            "radiograph_count": stats[f"{lang} images kept"],
            "has_image_counts": dict(has),
            "identifier_format": "DP + six digits, random, assigned for v3",
            "records_without_image_file": sorted(
                r["patient_id"] for r in records if r["has_image"] == "no"),
            "unlinked_in_v3": sorted(n for pair in unlinked_new_pairs for n in pair),
            "duplicate_image_pairs_in_v2": [list(p) for p in unlinked_new_pairs],
            "partial_dates_reduced_in_v3": fixed_new,
            "n_unique_image_contents": res.stats.get("n_unique_image_contents"),
            "icd_distinct_atoms": res.stats.get("icd_distinct_normalised"),
            "sex_counts": res.stats.get("sex"),
            "age": res.stats.get("age"),
            "cases_per_physician": res.stats.get("cases_per_physician"),
            "empty_per_field": res.stats.get("empty_per_field"),
            "deid_hits": res.stats.get("deid_hits"),
        }
        (out / "build_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{lang}: {len(records)} records, {report['radiograph_count']} images, "
              f"validator {'PASS' if res.ok else 'FAIL'}")

        if not args.no_zip:
            zpath = args.outdir / f"dental_clinical_dataset_v3_{lang}.zip"
            with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in sorted(out.rglob("*")):
                    if p.is_file():
                        zf.write(p, p.relative_to(args.outdir))
            print(f"wrote {zpath} ({zpath.stat().st_size:,} bytes)")

    print(json.dumps(dict(stats), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
