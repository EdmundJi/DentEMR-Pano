"""Loading and reshaping the three-rater quality-evaluation scores.

Two on-disk layouts are accepted so the raters' own spreadsheets can be used
with minimal reformatting:

*long*
    one row per (case, rater, item)::

        case_id,rater,item,score
        E011,R1,1.1,2
        E011,R1,1.2,1

*wide*
    one row per (case, rater), one column per item code::

        case_id,rater,1.1,1.2,...,5.4
        E011,R1,2,1,...,2

Both are normalised to a single tidy frame; :func:`item_matrix` then produces
the ``(n_cases, n_raters)`` arrays the coefficients in
:mod:`dentemr_pano.reliability` consume.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .checklist import BY_CODE, BY_DIMENSION, DIMENSION_MAX, ITEMS

REQUIRED_LONG = {"case_id", "rater", "item", "score"}


def load_ratings(path: Path) -> pd.DataFrame:
    """Read a rating file in either layout into tidy long form.

    Args:
        path: A ``.csv``, ``.tsv`` or ``.xlsx`` rating file.

    Returns:
        Columns ``case_id`` (str), ``rater`` (str), ``item`` (str),
        ``score`` (float, NaN where unrated).

    Raises:
        ValueError: If the layout cannot be recognised, if an item code is not
            part of the 23-item checklist, or if a score is outside the scale
            declared for its item.
    """
    frame = _read_any(path)
    frame.columns = [str(c).strip() for c in frame.columns]
    lowered = {c.lower(): c for c in frame.columns}

    if REQUIRED_LONG.issubset(lowered):
        tidy = frame.rename(columns={lowered[k]: k for k in REQUIRED_LONG})
        tidy = tidy[list(REQUIRED_LONG)].copy()
    elif {"case_id", "rater"}.issubset(lowered):
        id_cols = [lowered["case_id"], lowered["rater"]]
        tidy = frame.melt(id_vars=id_cols, var_name="item", value_name="score")
        tidy = tidy.rename(
            columns={lowered["case_id"]: "case_id", lowered["rater"]: "rater"}
        )
    else:
        raise ValueError(
            f"{path}: need columns case_id+rater (+item,score for long layout); "
            f"found {list(frame.columns)}"
        )

    tidy["case_id"] = tidy["case_id"].astype(str).str.strip()
    tidy["rater"] = tidy["rater"].astype(str).str.strip()
    tidy["item"] = tidy["item"].astype(str).str.strip()
    tidy["score"] = pd.to_numeric(tidy["score"], errors="coerce")

    _validate(tidy, path)
    return tidy.sort_values(["case_id", "rater", "item"]).reset_index(drop=True)


def _read_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _validate(tidy: pd.DataFrame, path: Path) -> None:
    unknown = sorted(set(tidy["item"]) - set(BY_CODE))
    if unknown:
        raise ValueError(f"{path}: item codes not in the checklist: {unknown}")

    missing = sorted(set(BY_CODE) - set(tidy["item"]))
    if missing:
        raise ValueError(f"{path}: no scores for checklist items {missing}")

    for code, group in tidy.groupby("item"):
        allowed = set(BY_CODE[code].scale)
        seen = {int(s) for s in group["score"].dropna().unique()}
        if not seen <= allowed:
            raise ValueError(
                f"{path}: item {code} allows {sorted(allowed)} but found "
                f"{sorted(seen - allowed)}"
            )


def item_matrix(tidy: pd.DataFrame, item: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Pivot one checklist item into a ``(n_cases, n_raters)`` score matrix.

    Returns:
        The matrix, the case ids in row order, and the rater ids in column
        order. Unrated cells are ``np.nan``.
    """
    sub = tidy[tidy["item"] == item]
    wide = sub.pivot_table(
        index="case_id", columns="rater", values="score", aggfunc="first"
    ).sort_index()
    return wide.to_numpy(dtype=float), list(wide.index), list(wide.columns)


def dimension_matrix(tidy: pd.DataFrame, dimension: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Sum a dimension's items per (case, rater) into a score matrix.

    A case is left as ``NaN`` for a rater who did not score every item in the
    dimension, so partial dimension totals never enter the ICC.
    """
    codes = [i.code for i in BY_DIMENSION[dimension]]
    sub = tidy[tidy["item"].isin(codes)]
    totals = sub.pivot_table(
        index="case_id", columns="rater", values="score", aggfunc="sum"
    ).sort_index()
    complete = sub.pivot_table(
        index="case_id", columns="rater", values="score", aggfunc="count"
    ).sort_index()
    totals = totals.where(complete == len(codes))
    return totals.to_numpy(dtype=float), list(totals.index), list(totals.columns)


def pooled_item_matrix(
    tidy: pd.DataFrame, codes: list[str]
) -> tuple[np.ndarray, list[tuple[str, str]], list[str]]:
    """Stack several items into one ``(n_case_items, n_raters)`` matrix.

    Each ``(case, item)`` pair becomes a row, so a chance-corrected coefficient
    for a dimension is computed on the native 0/1/2 item scale rather than on
    the dimension total. Applying quadratic weights to the 46-point composite
    total would treat almost every pair of scores as near-agreement and drive
    the coefficient to 1 regardless of the data; pooling avoids that.

    Returns:
        The matrix, the ``(case_id, item)`` keys in row order, and the rater
        ids in column order.
    """
    sub = tidy[tidy["item"].isin(codes)]
    wide = sub.pivot_table(
        index=["case_id", "item"], columns="rater", values="score", aggfunc="first"
    ).sort_index()
    return wide.to_numpy(dtype=float), list(wide.index), list(wide.columns)


def composite_matrix(tidy: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    """Sum all 23 items per (case, rater); ``NaN`` unless all items are scored."""
    totals = tidy.pivot_table(
        index="case_id", columns="rater", values="score", aggfunc="sum"
    ).sort_index()
    complete = tidy.pivot_table(
        index="case_id", columns="rater", values="score", aggfunc="count"
    ).sort_index()
    totals = totals.where(complete == len(ITEMS))
    return totals.to_numpy(dtype=float), list(totals.index), list(totals.columns)


def coverage(tidy: pd.DataFrame) -> dict[str, object]:
    """Summarise what the rating file actually contains."""
    return {
        "n_cases": int(tidy["case_id"].nunique()),
        "n_raters": int(tidy["rater"].nunique()),
        "raters": sorted(tidy["rater"].unique()),
        "n_items": int(tidy["item"].nunique()),
        "n_scores": int(tidy["score"].notna().sum()),
        "n_missing": int(tidy["score"].isna().sum()),
        "expected_scores": int(
            tidy["case_id"].nunique() * tidy["rater"].nunique() * len(ITEMS)
        ),
        "dimension_max": dict(DIMENSION_MAX),
    }
