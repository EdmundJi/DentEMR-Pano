"""Cohort description: demographics, disease spectrum, age-stratified burden.

Supplies every number and figure in the manuscript's dataset-characteristics
material. Two conventions here are deliberate and must stay documented in the
paper, because both change what the figures say:

*Age bands are half-open.* The submitted Figure 6 used the labels 18-24,
24-32, 32-50 and 50-90, which leaves 24, 32 and 50 belonging to two bands at
once. Bands are defined here as ``[low, high)`` with an inclusive final upper
bound, and the labels are rendered to match.

*Cases are multi-label.* 127 of 463 records carry more than one ICD code, so
disease-category counts sum to more than the cohort size. Percentages are
reported over cases, not over category assignments.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .validate import icd_atoms


@dataclass(frozen=True)
class DiseaseCategory:
    """One reporting bucket of the disease spectrum."""

    key: str
    label: str
    chapters: tuple[str, ...]


#: Reporting categories, defined over ICD-10 three-character chapters.
#:
#: The eight buckets follow the chapter structure of ICD-10 Chapter XI rather
#: than clinical convenience, so that the assignment of any code is checkable
#: against the standard. Everything outside the dental chapters -- trauma,
#: systemic comorbidity, encounter codes -- collects in ``other``.
DISEASE_CATEGORIES: tuple[DiseaseCategory, ...] = (
    DiseaseCategory("tooth_loss", "Tooth loss & other disorders of teeth", ("K08",)),
    DiseaseCategory("pulp_periapical", "Pulp & periapical diseases", ("K04",)),
    DiseaseCategory("caries", "Dental caries", ("K02",)),
    DiseaseCategory("periodontal", "Periodontal & gingival diseases", ("K05", "K06")),
    DiseaseCategory("impacted", "Embedded & impacted teeth", ("K01",)),
    DiseaseCategory("hard_tissue", "Developmental & other hard-tissue disorders",
                    ("K00", "K03")),
    DiseaseCategory("dentofacial", "Dentofacial anomalies", ("K07", "M26")),
    DiseaseCategory("other", "Other oral, trauma & systemic conditions", ()),
)

CATEGORY_BY_CHAPTER: dict[str, str] = {
    chapter: cat.key for cat in DISEASE_CATEGORIES for chapter in cat.chapters
}

#: Half-open age bands ``[low, high)``; the last band includes its upper bound.
AGE_BANDS: tuple[tuple[int, int], ...] = ((18, 25), (25, 35), (35, 50), (50, 65), (65, 91))


def band_label(low: int, high: int, last: bool) -> str:
    """Render a half-open band as an unambiguous printed label."""
    return f"{low}-{high - 1}" if not last else f"{low}+"


def load_cohort(dataset: Path) -> list[dict]:
    """Read all clinical records from a dataset root."""
    out = []
    for path in sorted((dataset / "clinical_records").glob("*.json")):
        with path.open(encoding="utf-8") as fh:
            out.append(json.load(fh))
    return out


def categories_of(record: dict) -> set[str]:
    """Disease categories a record belongs to, from its ICD codes."""
    chapters = {a[:3] for a in icd_atoms(record.get("primary_diagnosis_icd", ""))}
    if not chapters:
        return set()
    found = {CATEGORY_BY_CHAPTER[c] for c in chapters if c in CATEGORY_BY_CHAPTER}
    if any(c not in CATEGORY_BY_CHAPTER for c in chapters):
        found.add("other")
    return found


def age_band_of(age: int) -> str:
    """Label of the half-open band containing *age*."""
    for i, (low, high) in enumerate(AGE_BANDS):
        last = i == len(AGE_BANDS) - 1
        if low <= age < high or (last and age == high - 1):
            return band_label(low, high, last)
    raise ValueError(f"age {age} falls outside the defined bands")


def disease_spectrum(records: Iterable[dict]) -> list[dict]:
    """Case counts per disease category, descending.

    Returns:
        One row per category with ``key``, ``label``, ``n_cases`` and ``pct``,
        where ``pct`` is over the cohort, not over category assignments.
    """
    recs = list(records)
    counts: Counter[str] = Counter()
    for r in recs:
        counts.update(categories_of(r))
    labels = {c.key: c.label for c in DISEASE_CATEGORIES}
    rows = [
        {
            "key": key,
            "label": labels[key],
            "n_cases": counts.get(key, 0),
            "pct": round(100.0 * counts.get(key, 0) / len(recs), 1),
        }
        for key in labels
    ]
    return sorted(rows, key=lambda r: -r["n_cases"])


def age_by_sex(records: Iterable[dict]) -> dict[str, list[int]]:
    """Ages split by recorded sex, for the violin plot."""
    out: dict[str, list[int]] = {"male": [], "female": []}
    for r in records:
        key = "male" if r.get("sex") == "男" else "female"
        out[key].append(int(r["age"]))
    return out


def disease_by_age_band(records: Iterable[dict]) -> dict[str, dict[str, int]]:
    """Cross-tabulate disease category against age band."""
    recs = list(records)
    bands = [band_label(lo, hi, i == len(AGE_BANDS) - 1)
             for i, (lo, hi) in enumerate(AGE_BANDS)]
    table = {c.key: {b: 0 for b in bands} for c in DISEASE_CATEGORIES}
    for r in recs:
        band = age_band_of(int(r["age"]))
        for key in categories_of(r):
            table[key][band] += 1
    return table


# ── Hypothesis test for the sex-difference claim ─────────────────────────────


def sex_difference_tests(records: Iterable[dict]) -> list[dict]:
    """Chi-square test of independence between sex and each disease category.

    The submitted manuscript asserted "no significant sex-dependent bias in
    disease prevalence" without running a test. This supplies one, with
    Holm-Bonferroni correction across the eight categories.

    Returns:
        One row per category with the 2x2 counts, chi-square statistic, raw and
        adjusted p-values, and whether the adjusted p clears 0.05.
    """
    from scipy import stats  # imported lazily; only this function needs SciPy

    recs = list(records)
    male = [r for r in recs if r.get("sex") == "男"]
    female = [r for r in recs if r.get("sex") == "女"]

    rows = []
    for cat in DISEASE_CATEGORIES:
        m_yes = sum(1 for r in male if cat.key in categories_of(r))
        f_yes = sum(1 for r in female if cat.key in categories_of(r))
        table = [[m_yes, len(male) - m_yes], [f_yes, len(female) - f_yes]]
        if min(min(row) for row in table) < 5:
            # Expected counts too small for chi-square; use the exact test.
            stat, p = float("nan"), float(stats.fisher_exact(table)[1])
            test = "Fisher exact"
        else:
            chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
            stat, test = float(chi2), "chi-square"
        rows.append(
            {
                "category": cat.key,
                "label": cat.label,
                "male_n": m_yes,
                "male_pct": round(100.0 * m_yes / len(male), 1),
                "female_n": f_yes,
                "female_pct": round(100.0 * f_yes / len(female), 1),
                "test": test,
                "statistic": stat,
                "p_raw": float(p),
            }
        )

    # Holm-Bonferroni: sort by p, scale by the number of remaining hypotheses.
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p_raw"])
    running = 0.0
    for rank, idx in enumerate(order):
        m = len(rows) - rank
        running = max(running, min(1.0, rows[idx]["p_raw"] * m))
        rows[idx]["p_holm"] = running
        rows[idx]["significant"] = running < 0.05
    return rows


def summarise(records: Sequence[dict]) -> dict:
    """One dictionary with every cohort number the manuscript quotes."""
    ages = [int(r["age"]) for r in records]
    sexes = Counter(r.get("sex") for r in records)
    return {
        "n_cases": len(records),
        "age": {
            "min": min(ages),
            "max": max(ages),
            "mean": round(sum(ages) / len(ages), 1),
            "median": sorted(ages)[len(ages) // 2],
        },
        "sex": {
            "male": sexes.get("男", 0),
            "female": sexes.get("女", 0),
            "pct_female": round(100.0 * sexes.get("女", 0) / len(records), 1),
        },
        "age_bands": Counter(age_band_of(a) for a in ages),
        "disease_spectrum": disease_spectrum(records),
    }
