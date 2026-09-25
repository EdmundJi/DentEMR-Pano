#!/usr/bin/env python3
"""Compute the full reliability report and emit the manuscript's Table 6.

    python code/scripts/run_reliability.py ratings.csv --outdir analysis/out

Writes ``reliability_items.csv``, ``reliability_dimensions.csv``,
``reliability.json`` and a LaTeX fragment ``table_reliability.tex`` that can be
``\\input`` directly by the manuscript, so the printed table can never drift
from the computed numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dentemr_pano.checklist import (
    BY_CODE,
    BY_DIMENSION,
    COMPOSITE_MAX,
    DIMENSION_MAX,
    DIMENSION_NAMES,
    ITEMS,
)
from dentemr_pano.ratings import (
    composite_matrix,
    coverage,
    dimension_matrix,
    item_matrix,
    load_ratings,
    pooled_item_matrix,
)
from dentemr_pano.reliability import exact_agreement, gwet_ac, icc_two_way, pabak


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value:.{digits}f}"


def item_report(tidy: pd.DataFrame) -> pd.DataFrame:
    """Per-item exact agreement, AC1/AC2 and PABAK."""
    rows = []
    for item in ITEMS:
        matrix, cases, raters = item_matrix(tidy, item.code)
        weights = "identity" if item.binary else "quadratic"
        ac = gwet_ac(matrix, item.scale, weights=weights)
        pk = pabak(matrix, item.scale)
        complete = matrix[~np.isnan(matrix).any(axis=1)]
        rows.append(
            {
                "item": item.code,
                "name": item.name,
                "dimension": item.dimension,
                "scale": "binary" if item.binary else "ordinal",
                "n_cases": len(cases),
                "mean_score": float(np.nanmean(matrix)),
                "exact_agreement": exact_agreement(complete) if len(complete) else np.nan,
                "coefficient": ac.name,
                "ac": ac.value,
                "ac_se": ac.se,
                "ac_ci_low": ac.ci_low,
                "ac_ci_high": ac.ci_high,
                "pabak": pk.value,
                "pabak_ci_low": pk.ci_low,
                "pabak_ci_high": pk.ci_high,
            }
        )
    return pd.DataFrame(rows)


def dimension_report(tidy: pd.DataFrame) -> pd.DataFrame:
    """Per-dimension ICC alongside AC1/AC2 and PABAK, plus the composite row.

    ICC is computed on the dimension *total*, matching how the manuscript
    reports case-level quality. The chance-corrected coefficients are computed
    on the pooled item scores instead, because a weighted coefficient on a wide
    total would be dominated by the weighting rather than by the ratings.
    """
    rows = []

    def add(label: str, totals: np.ndarray, maximum: int, codes: list[str]) -> None:
        complete = totals[~np.isnan(totals).any(axis=1)]
        icc = icc_two_way(complete) if len(complete) >= 2 else None

        pooled, _, _ = pooled_item_matrix(tidy, codes)
        binary_only = all(BY_CODE[c].binary for c in codes)
        scale = (0, 2) if binary_only else (0, 1, 2)
        weights = "identity" if binary_only else "quadratic"
        ac = gwet_ac(pooled, scale, weights=weights)
        pk = pabak(pooled, scale)

        per_rater = np.nanmean(totals, axis=0)
        rows.append(
            {
                "dimension": label,
                "max": maximum,
                "n_cases": int(len(complete)),
                **{f"rater{i + 1}_mean": float(m) for i, m in enumerate(per_rater)},
                "case_mean_pct": 100.0 * float(np.nanmean(totals)) / maximum,
                "icc_2_1": None if icc is None or icc.degenerate else icc.icc_2_1,
                "icc_2_1_ci_low": None if icc is None or icc.degenerate else icc.icc_2_1_ci_low,
                "icc_2_1_ci_high": None if icc is None or icc.degenerate else icc.icc_2_1_ci_high,
                "icc_2_k": None if icc is None or icc.degenerate else icc.icc_2_k,
                "icc_2_k_ci_low": None if icc is None or icc.degenerate else icc.icc_2_k_ci_low,
                "icc_2_k_ci_high": None if icc is None or icc.degenerate else icc.icc_2_k_ci_high,
                "icc_degenerate": bool(icc.degenerate) if icc else True,
                "ac_name": ac.name,
                "ac": ac.value,
                "ac_ci_low": ac.ci_low,
                "ac_ci_high": ac.ci_high,
                "pabak": pk.value,
                "pabak_ci_low": pk.ci_low,
                "pabak_ci_high": pk.ci_high,
                "exact_agreement": exact_agreement(complete) if len(complete) else np.nan,
                "item_exact_agreement": exact_agreement(
                    pooled[~np.isnan(pooled).any(axis=1)]
                ),
            }
        )

    for dim, name in DIMENSION_NAMES.items():
        totals, _, _ = dimension_matrix(tidy, dim)
        codes = [i.code for i in BY_DIMENSION[dim]]
        add(f"{dim}: {name}", totals, DIMENSION_MAX[dim], codes)

    totals, _, _ = composite_matrix(tidy)
    add("Composite", totals, COMPOSITE_MAX, [i.code for i in ITEMS])
    return pd.DataFrame(rows)


def _ci(value, low, high, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "undefined"
    return f"{value:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def latex_table(dims: pd.DataFrame) -> str:
    """Render the dimension report as the manuscript's reliability table."""
    head = r"""% Generated by code/scripts/run_reliability.py -- do not edit by hand.
\begin{tabular}{lrrlll}
    \toprule
    Dimension & Mean (\%) & Exact agr. (\%) & ICC(2,1) [95\% CI] & Gwet's AC [95\% CI] & PABAK [95\% CI] \\
    \midrule
"""
    body = []
    for _, r in dims.iterrows():
        icc = ("undefined" if r["icc_degenerate"]
               else _ci(r["icc_2_1"], r["icc_2_1_ci_low"], r["icc_2_1_ci_high"]))
        ac = _ci(r["ac"], r["ac_ci_low"], r["ac_ci_high"])
        pk = _ci(r["pabak"], r["pabak_ci_low"], r["pabak_ci_high"])
        label = str(r["dimension"]).replace("&", r"\&")
        bold = label == "Composite"
        cells = [label, f"{r['case_mean_pct']:.1f}",
                 f"{100 * r['item_exact_agreement']:.1f}", icc, ac, pk]
        if bold:
            body.append(r"    \midrule")
            cells = [rf"\textbf{{{c}}}" for c in cells]
        body.append("    " + " & ".join(cells) + r" \\")
    tail = r"""
    \bottomrule
\end{tabular}
"""
    return head + "\n".join(body) + tail


def latex_item_table(items: pd.DataFrame) -> str:
    """Render the per-item report (Supplementary Table S3)."""
    head = r"""% Generated by code/scripts/run_reliability.py -- do not edit by hand.
\begin{tabular}{llrrlll}
    \toprule
    Item & Name & Mean score & Exact agr. (\%) & Coefficient & Gwet's AC [95\% CI] & PABAK [95\% CI] \\
    \midrule
"""
    body = []
    for _, r in items.iterrows():
        body.append("    " + " & ".join([
            r["item"], str(r["name"]).replace("&", r"\&"), f"{r['mean_score']:.2f}",
            f"{100 * r['exact_agreement']:.1f}", r["coefficient"],
            _ci(r["ac"], r["ac_ci_low"], r["ac_ci_high"]),
            _ci(r["pabak"], r["pabak_ci_low"], r["pabak_ci_high"]),
        ]) + r" \\")
    tail = r"""
    \bottomrule
\end{tabular}
"""
    return head + "\n".join(body) + tail


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ratings", type=Path, help="rater score file (long or wide)")
    ap.add_argument("--outdir", type=Path, default=Path("analysis/out"))
    args = ap.parse_args(argv)

    tidy = load_ratings(args.ratings)
    cov = coverage(tidy)
    items = item_report(tidy)
    dims = dimension_report(tidy)

    args.outdir.mkdir(parents=True, exist_ok=True)
    items.to_csv(args.outdir / "reliability_items.csv", index=False)
    dims.to_csv(args.outdir / "reliability_dimensions.csv", index=False)
    (args.outdir / "table_reliability.tex").write_text(latex_table(dims), encoding="utf-8")
    (args.outdir / "table_reliability_items.tex").write_text(
        latex_item_table(items), encoding="utf-8")
    s3 = pd.DataFrame({
        "Item": items["item"],
        "Name": items["name"],
        "Dimension": items["dimension"],
        "Scale": items["scale"],
        "Cases": items["n_cases"],
        "Mean score": items["mean_score"].round(2),
        "Exact agreement (%)": (100 * items["exact_agreement"]).round(1),
        "Coefficient": items["coefficient"],
        "Gwet's AC": items["ac"].round(3),
        "AC 95% CI low": items["ac_ci_low"].round(3),
        "AC 95% CI high": items["ac_ci_high"].round(3),
        "PABAK": items["pabak"].round(3),
        "PABAK 95% CI low": items["pabak_ci_low"].round(3),
        "PABAK 95% CI high": items["pabak_ci_high"].round(3),
    })
    with pd.ExcelWriter(args.outdir / "Supplementary_Table_S3_item_reliability.xlsx") as xw:
        s3.to_excel(xw, index=False, sheet_name="S3 item reliability", startrow=2)
        ws = xw.sheets["S3 item reliability"]
        ws.cell(row=1, column=1, value=(
            "Supplementary Table S3. Per-item inter-rater agreement for the 23 checklist "
            "items over the 160 evaluated cases: Gwet's AC2 (quadratic weights) for "
            "ordinal 0/1/2 items and AC1 for binary items, and PABAK, each with 95% "
            "confidence intervals. Generated by scripts/run_reliability.py."))
    (args.outdir / "reliability.json").write_text(
        json.dumps(
            {
                "coverage": cov,
                "items": items.to_dict(orient="records"),
                "dimensions": dims.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
            default=lambda o: None if isinstance(o, float) and np.isnan(o) else o,
        ),
        encoding="utf-8",
    )

    print(f"cases {cov['n_cases']}  raters {cov['n_raters']}  "
          f"scores {cov['n_scores']}/{cov['expected_scores']}")
    print()
    print(f"{'Dimension':<34}{'Mean%':>8}{'ItemAgr':>8}{'ICC(2,1)':>10}{'AC':>8}{'PABAK':>8}")
    for _, r in dims.iterrows():
        icc = "N/A" if r["icc_degenerate"] else f"{r['icc_2_1']:.3f}"
        print(f"{r['dimension']:<34}{r['case_mean_pct']:>8.1f}"
              f"{100 * r['item_exact_agreement']:>8.1f}{icc:>10}"
              f"{r['ac']:>8.3f}{r['pabak']:>8.3f}")
    print(f"\nwrote 6 files to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
