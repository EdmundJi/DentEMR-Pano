#!/usr/bin/env python3
"""Regenerate every cohort statistic, table and figure from the release.

    python code/scripts/run_cohort.py dataset/dental_clinical_dataset_v1 \
        --outdir analysis/out

Writes the three dataset-characteristics figures as PDF and PNG, the counts
behind them as CSV, and a ``cohort.json`` holding every number the manuscript
quotes about cohort composition.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from dentemr_pano.cohort import (
    AGE_BANDS,
    band_label,
    disease_by_age_band,
    load_cohort,
    sex_difference_tests,
    summarise,
)
from dentemr_pano.figures import age_distribution, burden_by_age, spectrum


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--outdir", type=Path, default=Path("analysis/out"))
    args = ap.parse_args(argv)

    records = load_cohort(args.dataset)
    args.outdir.mkdir(parents=True, exist_ok=True)

    stats = summarise(records)
    tests = sex_difference_tests(records)

    for ext in ("pdf", "png"):
        age_distribution(records, args.outdir / f"fig_age_distribution.{ext}")
        spectrum(records, args.outdir / f"fig_disease_spectrum.{ext}")
        burden_by_age(records, args.outdir / f"fig_burden_by_age.{ext}")

    _write_csv(
        args.outdir / "cohort_disease_spectrum.csv",
        ["category", "label", "n_cases", "pct_of_cohort"],
        [[r["key"], r["label"], r["n_cases"], r["pct"]] for r in stats["disease_spectrum"]],
    )

    bands = [band_label(lo, hi, i == len(AGE_BANDS) - 1)
             for i, (lo, hi) in enumerate(AGE_BANDS)]
    table = disease_by_age_band(records)
    _write_csv(
        args.outdir / "cohort_burden_by_age.csv",
        ["category", *bands],
        [[key, *[table[key][b] for b in bands]] for key in table],
    )

    _write_csv(
        args.outdir / "cohort_sex_tests.csv",
        ["category", "label", "male_n", "male_pct", "female_n", "female_pct",
         "test", "statistic", "p_raw", "p_holm", "significant"],
        [[t["category"], t["label"], t["male_n"], t["male_pct"], t["female_n"],
          t["female_pct"], t["test"], round(t["statistic"], 3) if t["statistic"] == t["statistic"] else "",
          round(t["p_raw"], 5), round(t["p_holm"], 5), t["significant"]] for t in tests],
    )

    payload = {**stats, "age_bands": dict(stats["age_bands"]), "sex_tests": tests}
    (args.outdir / "cohort.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"n = {stats['n_cases']}   age {stats['age']}   sex {stats['sex']}")
    print("\ndisease spectrum:")
    for r in stats["disease_spectrum"]:
        print(f"  {r['n_cases']:4d}  {r['pct']:5.1f}%  {r['label']}")

    significant = [t for t in tests if t["significant"]]
    print(f"\nsex differences: {len(significant)} of {len(tests)} significant "
          f"after Holm correction")
    for t in significant:
        print(f"  {t['label']}: M {t['male_pct']}% vs F {t['female_pct']}%  "
              f"p={t['p_raw']:.4f} (Holm {t['p_holm']:.4f})")

    print(f"\nwrote figures + 4 data files to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
