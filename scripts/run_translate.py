#!/usr/bin/env python3
"""Produce the bilingual v2 clinical records.

    python scripts/run_translate.py path/to/dental_clinical_dataset_v1 \
        --outdir out/bilingual --workers 8

Writes one JSON per record with the original Chinese fields untouched and an
``_en`` sibling for each of the eight free-text fields, plus:

  translation_log.json   per-record model, prompt hash, attempts, any error
  translation_audit.json automatic checks (tooth-number preservation, residual
                         Chinese, empty/non-empty mismatches)
  review_sample.csv      stratified sample for clinician review

Results are cached per record under ``--outdir/.cache``; re-running only calls
the API for records that are missing or previously errored, so an interrupted
run resumes without paying twice.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dentemr_pano.translate import (  # noqa: E402
    TRANSLATABLE_FIELDS,
    Translator,
    audit,
)

# The published v1 archive already has a `chief_complaint_en` holding a
# 13-value category, not a translation. It is renamed here so that `_en` means
# "English translation of the adjacent field" consistently across the schema.
CATEGORY_RENAME = ("chief_complaint_en", "chief_complaint_category")


def load_records(root: Path) -> list[dict]:
    files = sorted((root / "clinical_records").glob("*.json"))
    if not files:
        sys.exit(f"no clinical_records/*.json under {root}")
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--outdir", type=Path, default=Path("out/bilingual"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="translate only the first N records (smoke test)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--review-sample", type=int, default=60,
                    help="records drawn for clinician review")
    ap.add_argument("--seed", type=int, default=20260806)
    args = ap.parse_args()

    records = load_records(args.dataset)
    if args.limit:
        records = records[: args.limit]

    outdir = args.outdir
    recdir = outdir / "clinical_records"
    cache = outdir / ".cache"
    for d in (recdir, cache):
        d.mkdir(parents=True, exist_ok=True)

    kw = {"model": args.model} if args.model else {}
    tr = Translator.from_env(**kw)

    todo, results = [], {}
    for rec in records:
        pid = rec["patient_id"]
        cp = cache / f"{pid}.json"
        if cp.exists():
            cached = json.loads(cp.read_text(encoding="utf-8"))
            if not cached.get("error"):
                results[pid] = cached
                continue
        todo.append(rec)

    print(f"{len(records)} records · {len(results)} cached · {len(todo)} to translate",
          flush=True)

    done = 0
    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(tr.translate_record, r): r for r in todo}
            for fut in as_completed(futs):
                rec = futs[fut]
                pid = rec["patient_id"]
                try:
                    res = fut.result()
                    payload = {
                        "patient_id": res.patient_id,
                        "translations": res.translations,
                        "model": res.model,
                        "prompt_sha256": res.prompt_sha256,
                        "attempts": res.attempts,
                        "error": res.error,
                    }
                except Exception as exc:  # noqa: BLE001
                    payload = {"patient_id": pid, "translations": {},
                               "model": tr.model, "prompt_sha256": "",
                               "attempts": 0, "error": repr(exc)}
                (cache / f"{pid}.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")
                results[pid] = payload
                done += 1
                if done % 25 == 0 or done == len(todo):
                    print(f"  {done}/{len(todo)}", flush=True)

    # -- assemble bilingual records --------------------------------------
    bilingual, log = [], []
    for rec in records:
        pid = rec["patient_id"]
        res = results.get(pid, {})
        out = dict(rec)
        old, new = CATEGORY_RENAME
        if old in out:
            out[new] = out.pop(old)
        for f in TRANSLATABLE_FIELDS:
            out[f"{f}_en"] = res.get("translations", {}).get(f"{f}_en", "")
        # keep field order readable: zh field immediately followed by its _en
        ordered: dict[str, str] = {}
        for k, v in out.items():
            if k.endswith("_en") and k[:-3] in TRANSLATABLE_FIELDS:
                continue
            ordered[k] = v
            if k in TRANSLATABLE_FIELDS:
                ordered[f"{k}_en"] = out[f"{k}_en"]
        bilingual.append(ordered)
        (recdir / f"{pid}.json").write_text(
            json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
        log.append({k: res.get(k) for k in
                    ("patient_id", "model", "prompt_sha256", "attempts", "error")})

    (outdir / "translation_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    report = audit(bilingual)
    (outdir / "translation_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # -- clinician review sample -----------------------------------------
    rng = random.Random(args.seed)
    sample = rng.sample(bilingual, min(args.review_sample, len(bilingual)))
    with (outdir / "review_sample.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["patient_id", "field", "chinese", "english",
                    "accurate_0_1_2", "reviewer_correction", "reviewer_note"])
        for rec in sample:
            for f in TRANSLATABLE_FIELDS:
                if (rec.get(f) or "").strip():
                    w.writerow([rec["patient_id"], f, rec[f], rec.get(f"{f}_en", ""),
                                "", "", ""])

    errs = [r for r in log if r.get("error")]
    print(f"\nwrote {len(bilingual)} bilingual records to {recdir}")
    print(f"errors: {len(errs)}")
    print(f"audit: {report['strings_flagged']}/{report['strings_checked']} strings "
          f"flagged ({report['flag_rate']:.1%})")
    print(f"review sample: {len(sample)} records -> {outdir/'review_sample.csv'}")
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
