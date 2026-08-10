"""Structural, completeness and de-identification checks for the release.

Every number the manuscript reports about dataset composition should come from
:func:`audit` rather than from a spreadsheet, so that a regenerated table can
never disagree with the shipped archive.

Usage::

    python -m dentemr_pano.validate path/to/dental_clinical_dataset_v1 \
        --json analysis/out/audit.json
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Iterable

from .schema import BY_NAME, DROPPED_IN_V2, FIELDS, PHYSICIANS

# ── Patterns ─────────────────────────────────────────────────────────────────

#: A well-formed ICD-10 atom: one letter, two digits, optional sub-classifier.
#: ``K04.x`` is accepted: the source records use ``.x`` where the EMR did not
#: record a sub-category.
ICD_ATOM = re.compile(r"[A-Z]\d{2}(\.[0-9xX]+)?$")

#: Splits a multi-code diagnosis string. Full-width separators occur in the
#: source data and are treated as valid delimiters, then reported separately.
ICD_SPLIT = re.compile(r"[;；,，]")

#: Residual-identifier probes for the de-identification audit. These are
#: deliberately generic; institution-specific de-identification rules are not
#: part of this repository (see docs/EXCLUSIONS.md).
DEID_PROBES: dict[str, re.Pattern[str]] = {
    "exact_date": re.compile(r"\d{4}\s*[-/年]\s*\d{1,2}\s*[-/月]\s*\d{1,2}"),
    "cn_id_number": re.compile(r"\b\d{15}(\d{2}[0-9Xx])?\b"),
    "phone_number": re.compile(r"\b1[3-9]\d{9}\b"),
    "unreplaced_placeholder": re.compile(r"\[(EMPLOYER|RELATIVE|LOCATION)\]"),
}

#: Markers showing that a ``procedure`` value is a raw HIS billing order list
#: rather than a clinical procedure narrative.
ORDER_LIST_MARKERS = re.compile(r"每[天日]\d+次|\(集\)|\[\s*\d+\s*ml[::]")


# ── Result container ─────────────────────────────────────────────────────────


@dataclass
class Audit:
    """Outcome of a dataset audit."""

    n_records: int = 0
    n_images_on_disk: int = 0
    errors: list[str] = dc_field(default_factory=list)
    warnings: list[str] = dc_field(default_factory=list)
    stats: dict[str, Any] = dc_field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_json(self) -> str:
        return json.dumps(
            {
                "n_records": self.n_records,
                "n_images_on_disk": self.n_images_on_disk,
                "ok": self.ok,
                "errors": self.errors,
                "warnings": self.warnings,
                "stats": self.stats,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


# ── Loading ──────────────────────────────────────────────────────────────────


def load_records(root: Path) -> list[dict[str, Any]]:
    """Read every ``clinical_records/*.json`` under *root*, sorted by filename."""
    paths = sorted((root / "clinical_records").glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no clinical_records/*.json under {root}")
    records = []
    for p in paths:
        with p.open(encoding="utf-8") as fh:
            records.append(json.load(fh))
    return records


def _blank(value: Any) -> bool:
    return str(value if value is not None else "").strip() == ""


def normalise_icd(atom: str) -> str:
    """Canonicalise one ICD atom.

    Fixes the two clerical error classes present in v1: lower-case section
    letters (``k02.x``) and the letter ``O`` typed for the digit ``0``
    (``KO3.x``).
    """
    atom = atom.strip()
    if not atom:
        return atom
    # Only the section letter is case-normalised; the ``.x`` placeholder that
    # marks an unrecorded sub-category is lower-case by convention.
    head, sep, tail = atom.partition(".")
    head = head.upper()
    if len(head) >= 3 and head[0].isalpha() and head[1] == "O":
        head = head[0] + "0" + head[2:]
    return head + sep + tail


def icd_atoms(value: str) -> list[str]:
    """Split a diagnosis code string into normalised atoms."""
    return [normalise_icd(a) for a in ICD_SPLIT.split(value or "") if a.strip()]


# ── The audit ────────────────────────────────────────────────────────────────


def audit(root: Path, records: Iterable[dict[str, Any]] | None = None) -> Audit:
    """Run all checks against the dataset rooted at *root*."""
    recs = list(records) if records is not None else load_records(root)
    res = Audit(n_records=len(recs))

    image_dir = root / "panoramic_radiographs"
    on_disk = {p.name for p in image_dir.glob("*.png")} if image_dir.is_dir() else set()
    res.n_images_on_disk = len(on_disk)

    _check_keys(recs, res)
    _check_pairing(recs, on_disk, image_dir.is_dir(), res)
    _check_completeness(recs, res)
    _check_icd(recs, res)
    _check_demographics(recs, res)
    _check_physicians(recs, res)
    _check_deidentification(recs, res)
    _check_order_list_leakage(recs, res)
    return res


def _check_keys(recs, res: Audit) -> None:
    expected = {f.name for f in FIELDS}
    seen: Counter[str] = Counter()
    for r in recs:
        seen.update(r.keys())

    missing = sorted(expected - set(seen))
    if missing:
        res.errors.append(f"schema: fields absent from every record: {missing}")

    for name in sorted(set(seen) - expected):
        reason = DROPPED_IN_V2.get(name)
        msg = f"schema: undeclared field {name!r} present in {seen[name]} records"
        (res.warnings if reason else res.errors).append(
            f"{msg} ({reason})" if reason else msg
        )

    ragged = [r.get("patient_id") for r in recs if set(r) != set(seen)]
    if ragged:
        res.errors.append(f"schema: {len(ragged)} records have a non-uniform key set")

    res.stats["fields_present"] = sorted(seen)


def _check_pairing(recs, on_disk: set[str], have_dir: bool, res: Audit) -> None:
    declared_yes = {r["image_file"] for r in recs if r.get("has_image") == "yes"}
    n_no = sum(1 for r in recs if r.get("has_image") == "no")

    for r in recs:
        has, fname = r.get("has_image"), str(r.get("image_file", "")).strip()
        if has == "yes" and not fname:
            res.errors.append(f"pairing: {r['patient_id']} has_image=yes but no filename")
        if has == "no" and fname:
            res.errors.append(f"pairing: {r['patient_id']} has_image=no but filename set")
        if fname and not fname.startswith(str(r["patient_id"])):
            res.errors.append(
                f"pairing: {r['patient_id']} filename {fname!r} does not match id"
            )

    if have_dir:
        for name in sorted(declared_yes - on_disk):
            res.errors.append(f"pairing: declared image {name} missing from disk")
        for name in sorted(on_disk - declared_yes):
            res.errors.append(f"pairing: orphan image {name} not referenced")
    else:
        res.warnings.append(
            "pairing: panoramic_radiographs/ not present, image checks skipped"
        )

    res.stats["n_paired"] = len(declared_yes)
    res.stats["n_unpaired"] = n_no
    res.stats["unpaired_ids"] = sorted(
        r["patient_id"] for r in recs if r.get("has_image") == "no"
    )


def _check_completeness(recs, res: Audit) -> None:
    """Report emptiness per field, split by declared requirement level."""
    per_field: dict[str, int] = {}
    incomplete: set[str] = set()

    for f in FIELDS:
        blanks = [r["patient_id"] for r in recs if _blank(r.get(f.name))]
        per_field[f.name] = len(blanks)
        if not blanks:
            continue
        if f.requirement == "required":
            res.errors.append(
                f"completeness: required field {f.name} empty in {len(blanks)} records"
            )
        elif f.requirement == "expected":
            incomplete.update(blanks)
            res.warnings.append(
                f"completeness: expected field {f.name} empty in {len(blanks)} records"
            )

    res.stats["empty_per_field"] = per_field
    res.stats["n_incomplete_expected"] = len(incomplete)
    res.stats["incomplete_expected_ids"] = sorted(incomplete)


def _check_icd(recs, res: Audit) -> None:
    raw_atoms: Counter[str] = Counter()
    norm_atoms: Counter[str] = Counter()
    malformed: Counter[str] = Counter()
    fullwidth: list[str] = []

    for r in recs:
        value = r.get("primary_diagnosis_icd", "") or ""
        if "；" in value or "，" in value:
            fullwidth.append(r["patient_id"])
        for raw in ICD_SPLIT.split(value):
            raw = raw.strip()
            if not raw:
                continue
            raw_atoms[raw] += 1
            atom = normalise_icd(raw)
            norm_atoms[atom] += 1
            if not ICD_ATOM.match(atom):
                malformed[raw] += 1
            elif atom != raw:
                malformed[raw] += 1

    if malformed:
        res.warnings.append(
            f"icd: {sum(malformed.values())} code atoms need normalisation: "
            f"{dict(malformed)}"
        )
    if fullwidth:
        res.warnings.append(
            f"icd: full-width separator in {len(fullwidth)} records: {fullwidth}"
        )

    res.stats["icd_distinct_raw"] = len(raw_atoms)
    res.stats["icd_distinct_normalised"] = len(norm_atoms)
    res.stats["icd_counts"] = dict(norm_atoms.most_common())
    res.stats["icd_malformed"] = dict(malformed)
    res.stats["icd_multi_code_records"] = sum(
        1 for r in recs if len(icd_atoms(r.get("primary_diagnosis_icd", ""))) > 1
    )


def _check_demographics(recs, res: Audit) -> None:
    ages: list[int] = []
    for r in recs:
        try:
            ages.append(int(str(r["age"]).strip()))
        except (ValueError, KeyError):
            res.errors.append(f"demographics: {r.get('patient_id')} has unparsable age")
    if not ages:
        return

    below = [a for a in ages if a < 18]
    if below:
        res.errors.append(
            f"demographics: {len(below)} records below the stated age>=18 criterion"
        )

    sexes = Counter(r.get("sex") for r in recs)
    unknown = set(sexes) - {"男", "女"}
    if unknown:
        res.errors.append(f"demographics: unexpected sex values {sorted(unknown)}")

    res.stats["age"] = {
        "min": min(ages),
        "max": max(ages),
        "mean": round(statistics.mean(ages), 2),
        "median": statistics.median(ages),
        "sd": round(statistics.stdev(ages), 2) if len(ages) > 1 else 0.0,
        "q1": round(statistics.quantiles(ages, n=4)[0], 1),
        "q3": round(statistics.quantiles(ages, n=4)[2], 1),
        "n_ge_50": sum(1 for a in ages if a >= 50),
        "pct_ge_50": round(100 * sum(1 for a in ages if a >= 50) / len(ages), 1),
    }
    res.stats["sex"] = {
        "male": sexes.get("男", 0),
        "female": sexes.get("女", 0),
        "pct_female": round(100 * sexes.get("女", 0) / len(recs), 1),
    }


def _check_physicians(recs, res: Audit) -> None:
    counts = Counter(r.get("attending_physician") for r in recs)
    unknown = set(counts) - set(PHYSICIANS)
    if unknown:
        res.errors.append(f"physicians: unexpected codes {sorted(unknown)}")
    res.stats["cases_per_physician"] = {p: counts.get(p, 0) for p in PHYSICIANS}


def _check_deidentification(recs, res: Audit) -> None:
    """Scan every free-text field for residual identifiers."""
    hits: dict[str, list[str]] = {k: [] for k in DEID_PROBES}
    text_fields = [f.name for f in FIELDS if f.kind == "string"]

    for r in recs:
        for fname in text_fields:
            value = str(r.get(fname) or "")
            if not value:
                continue
            for probe, rx in DEID_PROBES.items():
                if rx.search(value):
                    hits[probe].append(f"{r['patient_id']}:{fname}")

    for probe, found in hits.items():
        if not found:
            continue
        # An intact placeholder is the de-identification working, not a leak.
        bucket = res.warnings if probe == "unreplaced_placeholder" else res.errors
        bucket.append(f"deid: {probe} matched {len(found)} field(s): {found[:10]}")

    res.stats["deid_hits"] = {k: len(v) for k, v in hits.items()}


def _check_order_list_leakage(recs, res: Audit) -> None:
    """Quantify raw HIS order-list content inside ``procedure``.

    The submitted Methods claimed billing order items were excluded from the
    procedure field. They were not, so the revision must either strip them or
    correct the claim; this check supplies the number either way.
    """
    contaminated = [
        r["patient_id"]
        for r in recs
        if ORDER_LIST_MARKERS.search(str(r.get("procedure") or ""))
    ]
    if contaminated:
        res.warnings.append(
            f"provenance: {len(contaminated)} procedure values contain raw HIS "
            "order-list markers (billing frequency, drug packaging)"
        )
    res.stats["n_order_list_in_procedure"] = len(contaminated)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=Path, help="dataset root containing clinical_records/")
    ap.add_argument("--json", type=Path, help="write the full report here")
    ap.add_argument("--quiet", action="store_true", help="suppress the text summary")
    args = ap.parse_args(argv)

    res = audit(args.root)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(res.to_json(), encoding="utf-8")

    if not args.quiet:
        print(f"records {res.n_records}   images {res.n_images_on_disk}")
        for label, items in (("ERROR", res.errors), ("WARN", res.warnings)):
            for item in items:
                print(f"  [{label}] {item}")
        print("PASS" if res.ok else f"FAIL ({len(res.errors)} error(s))")

    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
