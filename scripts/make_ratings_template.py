#!/usr/bin/env python3
"""Emit a blank rating template, and a synthetic rating file for pipeline tests.

The template fixes the exact column layout :mod:`dentemr_pano.ratings` expects,
so the raters' scores can be transcribed into it without guesswork.

The synthetic file is *not* data. It is generated to sit near the dimension
means the manuscript reports, purely so the analysis pipeline can be exercised
end to end before the real scores are transcribed. Anything it produces is
labelled as synthetic and must never reach the paper.

    python code/scripts/make_ratings_template.py --dataset dataset/dental_clinical_dataset_v2_zh
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from dentemr_pano.checklist import ITEMS

RATERS = ("R1", "R2", "R3")

#: Evaluated-case counts per physician, from the manuscript's sampling table.
SAMPLE_PER_PHYSICIAN = {"P1": 64, "P2": 36, "P3": 20, "P4": 20, "P5": 20}

#: Per-item probability of the maximum score, tuned so that the synthetic
#: dimension means land close to the reported 98.4 / 93.3 / 98.0 / 83.4 / 100.
_SYNTHETIC_P_MAX = {
    "Dim1": 0.95, "Dim2": 0.90, "Dim3": 0.96, "Dim4": 0.55, "Dim5": 1.00,
}


def stratified_case_ids(dataset: Path, seed: int = 20250801) -> list[str]:
    """Draw the evaluated sample, stratified by contributing physician."""
    by_physician: dict[str, list[str]] = {}
    for path in sorted((dataset / "clinical_records").glob("*.json")):
        with path.open(encoding="utf-8") as fh:
            rec = json.load(fh)
        by_physician.setdefault(rec["attending_physician"], []).append(rec["patient_id"])

    rng = random.Random(seed)
    chosen: list[str] = []
    for physician, wanted in SAMPLE_PER_PHYSICIAN.items():
        pool = sorted(by_physician.get(physician, []))
        if len(pool) < wanted:
            raise ValueError(f"{physician}: need {wanted} cases, pool has {len(pool)}")
        chosen.extend(rng.sample(pool, wanted))
    return sorted(chosen)


def write_long(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["case_id", "rater", "item", "score"])
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("analysis/out"))
    ap.add_argument("--seed", type=int, default=20250801)
    args = ap.parse_args(argv)

    cases = stratified_case_ids(args.dataset, args.seed)

    blank = [(c, r, i.code, "") for c in cases for r in RATERS for i in ITEMS]
    write_long(args.outdir / "ratings_template.csv", blank)

    rng = random.Random(args.seed)
    synthetic: list[tuple[str, str, str, str]] = []
    for case in cases:
        # A per-case quality offset makes some cases genuinely worse than
        # others, which is what gives ICC any between-case variance at all.
        offset = rng.gauss(0.0, 0.06)
        for rater in RATERS:
            for item in ITEMS:
                p_max = min(0.999, max(0.05, _SYNTHETIC_P_MAX[item.dimension] + offset))
                if rng.random() < p_max:
                    score = item.max_score
                elif item.binary:
                    score = 0
                else:
                    score = 1 if rng.random() < 0.93 else 0
                synthetic.append((case, rater, item.code, str(score)))
    write_long(args.outdir / "ratings_synthetic.csv", synthetic)

    print(f"{len(cases)} cases x {len(RATERS)} raters x {len(ITEMS)} items")
    print(f"  template   -> {args.outdir / 'ratings_template.csv'}  ({len(blank)} rows)")
    print(f"  SYNTHETIC  -> {args.outdir / 'ratings_synthetic.csv'} ({len(synthetic)} rows)")
    print("\nThe synthetic file is for pipeline testing only; it is not data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
