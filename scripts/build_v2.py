#!/usr/bin/env python3
"""Build the v2 source archive from v1 + archived translation artefacts.

    PYTHONPATH=code/src .venv/bin/python code/scripts/build_v2.py \
        dataset/dental_clinical_dataset_v1 \
        --translation analysis/out/translation \
        --outdir dataset/dental_clinical_dataset_v2

Every change v2 makes to a published field is deterministic, driven by the
inputs named on the command line, and enumerated in the CHANGELOG.md the
script writes. Nothing is silently corrected: the counts printed at the end
are the counts that go into the changelog.

Changes applied (provenance documented in the release CHANGELOG):
  1. ICD-10 normalisation: case, letter-O-for-zero, full-width separators.
  2. treatment_category: 124 combination strings -> controlled vocabulary.
  3. Three records with residual exact dates / pasted haematology values.
  4. Drop `notes` (empty throughout) and `routine_checkup` (quasi-identifier).
  5. Rename `chief_complaint_en` -> `chief_complaint_category` (it holds a
     category, not a translation); unify `others` -> `Others`.
  6. English alignment backfill: eight `_en` fields reconstructed from the
     archived unique-segment translations.
  7. Clinician corrections: the union of both reviewers' findings (11
     segments), read from the archived review files.
  8. Single-digit "tooth N" repairs: FDI numbers are two-digit, so
     "tooth 6" is never a valid reading of a bare "6" in the source.
  9. Radiographs: masked burned-in modality label, single-channel grayscale (verified
     lossless outside the masked region).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dentemr_pano.segments import build_segment_table, restore_tokens  # noqa: E402
from dentemr_pano.translate import TRANSLATABLE_FIELDS  # noqa: E402

# ---------------------------------------------------------------------------
# 1-3. Field-level repairs to the Chinese source
# ---------------------------------------------------------------------------

#: Per-record replacements in `imaging_examination`. Residual exact dates are
#: reduced to YYYY-MM; haematology results pasted into an imaging field are
#: removed (they are lab results, not imaging findings, and two carry exact
#: dates). The pre-clean source strings are stored as SHA-256 digests, not
#: verbatim: publishing the removed identifiers would undo the removal.
IMAGING_FIXES: dict[str, tuple[str, str]] = {
    "0000448185": (
        "8edc557615cb5f6321e3ff6e641bd62b58744dee056dbc35804f4745ace6cf18",
        "曲面断层片示拔牙创末见其他异常。",
    ),
    "0000866026": (
        "ce2f580cac2babc34e20c6b223f3282aee48a3d1588cde0536e34030018bbbdd",
        "2022-03牙片示：27根管内有阻射影。根尖周暗影",
    ),
    "0001232553": (
        "da3819953f33136a548bb359caf4f265b45d260602145a512a9c32543262897b",
        "x线全景片示:牙周病，遗留多数牙根，残冠、38、48理伏阳生齿未萌",
    ),
}

#: treatment_category atoms -> controlled vocabulary. Everything not listed
#: maps to itself; the assertion below guarantees no unmapped variant ships.
CATEGORY_MAP: dict[str, str] = {
    "Endodontic (root canal)": "Endodontic (root canal)",
    "Restorative (fillings, crowns)": "Restorative (fillings, crowns)",
    "Operative Dentistry": "Restorative (fillings, crowns)",
    "Extraction": "Extraction",
    "Oral surgery": "Oral surgery",
    "Surgery": "Oral surgery",
    "Periodontal therapy": "Periodontal therapy",
    "Periodontal": "Periodontal therapy",
    "Periodontics": "Periodontal therapy",
    "Prosthodontic": "Prosthodontics",
    "Prosthodontics": "Prosthodontics",
    "rosthodontic": "Prosthodontics",           # typo in v1
    "Orthodontic": "Orthodontics",
    "Orthodontics": "Orthodontics",
    "Preventive care": "Preventive care",
    "Observation": "Observation",
    "Others": "Others",
    "Desensitization": "Desensitization",
    "Laser Therapy": "Laser therapy",
    "Oral Medicine": "Oral medicine",
    "Diagnostic imaging": "Diagnostic imaging",
    "Imaging": "Diagnostic imaging",
    "Trauma": "Trauma management",
}


def normalise_icd(raw: str) -> str:
    """Case, letter-O and separator repairs; atom order is preserved."""
    atoms = [a.strip() for a in re.split(r"[;；]", raw or "") if a.strip()]
    out = []
    for a in atoms:
        a = a[0].upper() + a[1:]
        a = re.sub(r"^([A-Z])O(\d)", r"\g<1>0\g<2>", a)  # KO3.x -> K03.x
        out.append(a)
    return ";".join(out)


def normalise_category(raw: str) -> str:
    atoms = [a.strip() for a in re.split(r"[;；]", raw or "") if a.strip()]
    mapped = []
    for a in atoms:
        if a not in CATEGORY_MAP:
            raise SystemExit(f"unmapped treatment_category atom: {a!r}")
        m = CATEGORY_MAP[a]
        if m not in mapped:
            mapped.append(m)
    return "; ".join(mapped)


# ---------------------------------------------------------------------------
# 6-8. English overrides applied at the (masked) unique-segment level
# ---------------------------------------------------------------------------

#: Repairs for translations that read a leading HIS order-line number as a
#: tooth number. FDI tooth numbers are two-digit (quadrant 1-4, position 1-8),
#: so "tooth 6" is never a valid reading of a bare single digit. Where the
#:  source is a billing order line, the digit is the line number and the
#: English is rewritten as an enumerator. Where the source is clinical text
#: whose bare digit cannot be resolved to a two-digit FDI tooth, the English
#: mirrors the source's bare numeral instead of asserting a tooth that the
#: source does not name.
TOOTH_FIXES: dict[str, str] = {
    # HIS billing order lines: leading digit is the order-line number
    "6调牙合 每天1次 1": "6. Occlusal adjustment, once daily, 1",
    "7牙髓活力检查 每天1次 1": "7. Pulp vitality test, once daily, 1",
    "8根管消毒术 每天1次 1": "8. Root canal disinfection, once daily, 1",
    "9简单充填术(国产充填材料) 每天1次 1":
        "9. Simple restoration (domestic filling material), once daily, 1",
    "3复杂充填术(进口充填材料) 每天1次 2":
        "3. Complex restoration (imported restorative material), once daily, 2",
    "6调牙合 每天1次 3": "6. Occlusal adjustment, once daily, 3",
    "5牙髓活力检查 每天1次 1": "5. Pulp vitality test, once daily, 1",
    "2 冠周炎局部治疗 每天 1 次 1":
        "2. Local treatment of pericoronitis, once daily, 1",
    # clinical text with an unresolvable bare digit: mirror the source
    "6牙龈愈合，无红肿，触诊无疼痛":
        "6: gingiva healed, no redness or swelling, no pain on palpation.",
    "建议择期拔除4，不适随诊":
        "Recommend elective extraction of 4; return for follow-up if "
        "discomfort occurs.",
}

#: The one Layer-1 empty output: a source-garbled HIS billing line. Translated
#: by the research team; the illegible remainder is marked as such rather
#: than guessed at.
MANUAL_TRANSLATIONS: dict[str, str] = {
    '9，盐酸利多卡因注射液(集兴襄生好5n:".1名0.工:':
        "9. Lidocaine hydrochloride injection [remainder of billing line "
        "illegible in source]",
}

#: Adjudication outcome (third bilingual clinician, blinded A/B pairs,
#: reviews/裁决_zr_8.8.xlsx + adjudication_answer_key.json). All 11 flagged
#: segments were upheld as genuine errors and 10 reviewer corrections were
#: upheld; for this one segment BOTH the machine translation and the
#: reviewers' correction were rejected for fabricating a definite tooth
#: number ("tooth 18") from the illegible source token 她8. The
#: adjudicator's rendering, which marks what is illegible instead of
#: guessing, replaces the reviewer correction.
ADJUDICATED_TRANSLATIONS: dict[str, str] = {
    "周用下拨除她8，嘱患者收纱止血30出，细后进食，4勿刷牙嫩口，不适随诊，8月后复诊行修夏治疗开别缺损":
        'Under local anesthesia, tooth ?8 was extracted (quadrant illegible '
        'in the source, "她8"). The patient was instructed to bite on gauze '
        "for 30 minutes for hemostasis, to eat afterwards, not to brush or "
        "rinse (duration illegible), to return if any discomfort occurs, and "
        "to return in 8 months for prosthodontic treatment of the dentition "
        "defect.",
}

#: English-side repairs inside longer segments (substring surgery), for the
#: two clinical-text cases where the bare digit sits mid-segment.
EN_SUBSTRING_FIXES: list[tuple[str, str, str]] = [
    # (zh segment prefix to locate, english-old, english-new)
    ("全口牙龈乳头增生红肿",
     "tooth 4 residual crown", "4: residual crown"),
    ("曲面断层片示，3近中倾斜",
     "mesial inclination of tooth 3", "mesial inclination of 3"),
]

#: Translations for the segments whose Chinese was repaired by IMAGING_FIXES:
#: the archived translation of the original segment, trimmed or re-dated to
#: match the repaired source exactly.
CLEANED_SEGMENT_TRANSLATIONS: dict[str, str] = {
    "2022-03牙片示：27根管内有阻射影":
        "2022-03 periapical radiograph shows: radiopacity in the root canal "
        "of tooth 27.",
    "x线全景片示:牙周病，遗留多数牙根，残冠、38、48理伏阳生齿未萌":
        "Panoramic radiograph shows: periodontal disease, multiple retained "
        "roots, residual crowns, impacted teeth 38 and 48 unerupted.",
}


def load_reviewer_corrections(tdir: Path) -> dict[str, str]:
    """Union of both reviewers' corrections, keyed by masked segment.

    Reviewer 1's correction is preferred where both exist (the choices are
    semantically equivalent for all three overlapping findings); Reviewer 2's
    is used where Reviewer 1 scored the segment 2 or left it blank.
    """
    import pandas as pd

    ranked = {}
    with (tdir / "review_unique.csv").open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            ranked[int(row["rank"])] = row["chinese"]  # masked segment

    sheets = []
    for name in ("reviews/reviewer1_refill_0953.xlsx", "reviews/reviewer2_0940.xlsx"):
        df = pd.read_excel(tdir / name, sheet_name="待复核 400条")
        sheets.append(df.set_index("rank"))

    corrections: dict[str, str] = {}
    for rank in sheets[0].index:
        for df in sheets:  # R1 first: preferred where both corrected
            v = pd.to_numeric(df.loc[rank, "verdict_0_1_2"], errors="coerce")
            corr = df.loc[rank, "correction"]
            if pd.notna(v) and v < 2 and isinstance(corr, str) and corr.strip():
                corrections.setdefault(ranked[rank], corr.strip())
    return corrections


# ---------------------------------------------------------------------------
# reconstruction
# ---------------------------------------------------------------------------

def join_segments(pieces: list[str]) -> str:
    pieces = [p.strip() for p in pieces if p and p.strip()]
    if not pieces:
        return ""
    out = []
    for p in pieces[:-1]:
        out.append(p if p[-1] in ".;:!?" else p + ";")
    out.append(pieces[-1])
    return " ".join(out)


def build_records(v1: Path, tdir: Path, stats: Counter,
                  extra_corrections: dict[str, str] | None = None) -> list[dict]:
    records = [json.loads(f.read_text(encoding="utf-8"))
               for f in sorted((v1 / "clinical_records").glob("*.json"))]

    # --- Chinese-side repairs -------------------------------------------
    for rec in records:
        pid = rec["patient_id"]
        if pid in IMAGING_FIXES:
            old_sha, new = IMAGING_FIXES[pid]
            got = hashlib.sha256(rec["imaging_examination"].encode()).hexdigest()
            assert got == old_sha, pid
            rec["imaging_examination"] = new
            stats["imaging_examination repaired"] += 1

        icd = rec.get("primary_diagnosis_icd") or ""
        fixed = normalise_icd(icd)
        if fixed != icd:
            rec["primary_diagnosis_icd"] = fixed
            stats["primary_diagnosis_icd normalised"] += 1

        cat = rec.get("treatment_category") or ""
        fixed = normalise_category(cat)
        if fixed != cat:
            rec["treatment_category"] = fixed
            stats["treatment_category normalised"] += 1

    # --- English layer ---------------------------------------------------
    seg_en: dict[str, str] = {}
    for row in json.loads((tdir / "translations.json").read_text(encoding="utf-8")):
        seg_en[row["segment"]] = row["en_primary"]

    final_overrides = dict(TOOTH_FIXES)
    final_overrides.update(MANUAL_TRANSLATIONS)
    final_overrides.update(CLEANED_SEGMENT_TRANSLATIONS)
    corrections = load_reviewer_corrections(tdir)
    corrections.update(ADJUDICATED_TRANSLATIONS)  # adjudication supersedes
    if extra_corrections:
        overlap = set(extra_corrections) & (set(corrections) | set(final_overrides))
        assert not overlap, f"extra corrections collide: {sorted(overlap)[:3]}"
        corrections.update(extra_corrections)
        stats["extra corrections applied"] = len(extra_corrections)
    for seg, corr in corrections.items():
        final_overrides.setdefault(seg, corr)
    stats["reviewer corrections applied"] = len(corrections)
    stats["adjudication replacements"] = len(ADJUDICATED_TRANSLATIONS)
    stats["single-digit tooth repairs"] = len(TOOTH_FIXES) + \
        sum(1 for _, old, _ in EN_SUBSTRING_FIXES if "tooth" in old)
    stats["manual translations"] = len(MANUAL_TRANSLATIONS)

    st = build_segment_table(records, TRANSLATABLE_FIELDS)
    per_record: dict[tuple[str, str], list[tuple[int, str]]] = {}
    missing: list[str] = []
    for occ in st.occurrences:
        seg = occ["segment"]
        if seg in final_overrides:
            en = final_overrides[seg]        # already fully restored text
        elif seg in seg_en and seg_en[seg].strip():
            en = restore_tokens(seg_en[seg], occ["tokens"])
        else:
            missing.append(seg)
            en = ""
        for prefix, old, new in EN_SUBSTRING_FIXES:
            if seg.startswith(prefix):
                assert old in en, (prefix, en)
                en = en.replace(old, new)
        per_record.setdefault((occ["patient_id"], occ["field"]), []).append(
            (occ["order"], en))
    if missing:
        raise SystemExit(f"{len(missing)} segments lack a translation: "
                         f"{missing[:5]!r}")

    # --- assemble v2 records --------------------------------------------
    out_records = []
    for rec in records:
        pid = rec["patient_id"]
        new: dict = {}
        for key, val in rec.items():
            if key in ("notes", "routine_checkup"):
                continue
            if key == "chief_complaint_en":
                v = (val or "").strip()
                new["chief_complaint_category"] = "Others" if v == "others" else v
                if v == "others":
                    stats["chief_complaint category case unified"] += 1
                continue
            new[key] = val
            if key in TRANSLATABLE_FIELDS:
                pieces = sorted(per_record.get((pid, key), []))
                new[f"{key}_en"] = join_segments([p for _, p in pieces])
        out_records.append(new)
    stats["records"] = len(out_records)
    return out_records


# ---------------------------------------------------------------------------
# 9. radiographs
# ---------------------------------------------------------------------------

def build_images(v1: Path, outdir: Path, stats: Counter) -> None:
    from PIL import Image
    import numpy as np

    src = v1 / "panoramic_radiographs"
    dst = outdir / "panoramic_radiographs"
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.png")):
        a = np.asarray(Image.open(f)).copy()
        assert a.shape == (922, 1722, 3), f.name
        a[10:29, 0:46] = 0                    # burned-in modality label
        diff = np.abs(a[..., 0].astype(int) - a[..., 1].astype(int)).max()
        diff = max(diff, np.abs(a[..., 1].astype(int) - a[..., 2].astype(int)).max())
        assert diff == 0, f"{f.name}: channels differ by {diff} outside mask"
        Image.fromarray(a[..., 0], mode="L").save(dst / f.name, optimize=True)
        stats["images converted"] += 1


# ---------------------------------------------------------------------------

def qa(records: list[dict], stats: Counter) -> dict:
    icd_atoms = Counter()
    for r in records:
        for a in (r["primary_diagnosis_icd"] or "").split(";"):
            if a:
                icd_atoms[a] += 1
    cats = sorted({a for r in records
                   for a in (r["treatment_category"] or "").split("; ") if a})
    cc = sorted({(r["chief_complaint_category"] or "") for r in records})
    single_tooth = []
    dur = re.compile(r"\b(?:tooth|teeth)\s+\d\s?(?:month|week|day|year|hour)", re.I)
    pat = re.compile(r"\b(?:tooth|teeth)\s+\d(?!\d|\.\d)", re.I)
    zh_res = []
    for r in records:
        for f in TRANSLATABLE_FIELDS:
            en = r.get(f + "_en", "")
            for m in pat.finditer(en):
                if not dur.match(en[m.start():m.start() + 40]):
                    single_tooth.append((r["patient_id"], f, en[m.start():m.start() + 40]))
            resid = [ch for ch in en if "一" <= ch <= "鿿"]
            if resid:
                zh_res.append((r["patient_id"], f, "".join(resid[:10])))
    return {
        "icd_distinct_atoms": len(icd_atoms),
        "icd_atoms": sorted(icd_atoms),
        "treatment_category_vocabulary": cats,
        "chief_complaint_categories": [c for c in cc if c],
        "residual_single_digit_tooth": single_tooth,
        "english_fields_with_chinese": len(zh_res),
        "chinese_residual_examples": zh_res[:10],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("v1", type=Path)
    ap.add_argument("--translation", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--skip-images", action="store_true")
    ap.add_argument("--extra-corrections", type=Path, default=None,
                    help="JSON of {masked_segment: final_english} applied on "
                         "top of the reviewer corrections (screening pass)")
    args = ap.parse_args()

    stats: Counter = Counter()
    extra = (json.loads(args.extra_corrections.read_text(encoding="utf-8"))
             if args.extra_corrections else None)
    records = build_records(args.v1, args.translation, stats, extra)

    recdir = args.outdir / "clinical_records"
    recdir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        (recdir / f"{rec['patient_id']}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")

    with (args.outdir / "metadata.csv").open("w", newline="", encoding="utf-8") as fh:
        cols = ["patient_id", "image_file", "has_image", "age", "sex",
                "treatment_category", "primary_diagnosis_icd",
                "chief_complaint_category", "attending_physician"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for rec in records:
            w.writerow({c: rec.get(c, "") for c in cols})

    if not args.skip_images:
        build_images(args.v1, args.outdir, stats)

    report = qa(records, stats)
    report["stats"] = dict(stats)
    (args.outdir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
