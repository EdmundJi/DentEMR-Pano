"""Inter-rater reliability: ICC(2,1)/ICC(2,k), Gwet's AC1/AC2 and PABAK.

Why more than one coefficient
-----------------------------
ICC is a variance-ratio statistic. When almost every case receives the same
score, between-case variance approaches zero and the ratio collapses even
though the raters agree on nearly every item. Three of the five checklist
dimensions sit in exactly that regime, and Dim5 is degenerate outright -- every
rater gave every case full marks, so ICC is undefined rather than perfect.

Gwet's AC1 (and its weighted form AC2) replaces the chance-agreement term with
one that does not inflate as the marginal distribution becomes concentrated, so
it stays interpretable under those conditions. PABAK is reported as a third,
simpler reference point.

References:
    Gwet, K.L. (2008). Computing inter-rater reliability and its variance in
        the presence of high agreement. Br J Math Stat Psychol 61:29-48.
    Shrout, P.E. & Fleiss, J.L. (1979). Intraclass correlations: uses in
        assessing rater reliability. Psychol Bull 86:420-428.
    McGraw, K.O. & Wong, S.P. (1996). Forming inferences about some intraclass
        correlation coefficients. Psychol Methods 1:30-46.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import numpy as np

WeightKind = Literal["identity", "linear", "quadratic", "ordinal"]


# ── Weight matrices ──────────────────────────────────────────────────────────


def weight_matrix(categories: Sequence[int], kind: WeightKind) -> np.ndarray:
    """Agreement weights ``w[k, l]`` for scoring category *k* against *l*.

    ``identity`` yields unweighted agreement (AC1). The other kinds give
    partial credit to near-misses, which is the appropriate treatment for an
    ordinal 0/1/2 scale where scoring 1 against 2 is a smaller disagreement
    than scoring 0 against 2 (AC2).

    Args:
        categories: The admissible scores, ascending.
        kind: Weighting scheme.

    Returns:
        A ``q x q`` matrix with 1.0 on the diagonal and values in ``[0, 1]``.
    """
    cats = np.asarray(categories, dtype=float)
    q = len(cats)
    if q < 2:
        raise ValueError("need at least two categories")

    if kind == "identity":
        return np.eye(q)

    if kind == "linear":
        spread = cats.max() - cats.min()
        return 1.0 - np.abs(cats[:, None] - cats[None, :]) / spread

    if kind == "quadratic":
        spread = cats.max() - cats.min()
        return 1.0 - ((cats[:, None] - cats[None, :]) / spread) ** 2

    if kind == "ordinal":
        # Gwet's ordinal weights: penalise by the number of categories spanned.
        idx = np.arange(q)
        span = np.abs(idx[:, None] - idx[None, :]) + 1.0
        return 1.0 - (span * (span - 1.0) / 2.0) / (q * (q - 1.0) / 2.0)

    raise ValueError(f"unknown weight kind {kind!r}")


# ── Result containers ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Coefficient:
    """A chance-corrected agreement coefficient with its uncertainty."""

    name: str
    value: float
    se: float
    ci_low: float
    ci_high: float
    percent_agreement: float
    percent_chance: float
    n_subjects: int
    n_raters_max: int

    def as_dict(self) -> dict[str, float | str | int]:
        return asdict(self)


@dataclass(frozen=True)
class IccResult:
    """Two-way random-effects ICC estimates."""

    icc_2_1: float | None
    icc_2_k: float | None
    icc_2_k_spearman_brown: float | None
    n_subjects: int
    n_raters: int
    degenerate: bool
    note: str = ""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


# ── Gwet's AC1 / AC2 ─────────────────────────────────────────────────────────


def gwet_ac(
    ratings: np.ndarray,
    categories: Sequence[int],
    weights: WeightKind = "identity",
    conf_level: float = 0.95,
) -> Coefficient:
    """Gwet's AC1 (unweighted) or AC2 (weighted) for two or more raters.

    Args:
        ratings: ``(n_subjects, n_raters)`` score matrix. ``np.nan`` marks a
            missing rating; subjects rated fewer than twice are excluded from
            the observed-agreement term but still inform the marginals.
        categories: Admissible scores, ascending.
        weights: ``"identity"`` gives AC1; any other scheme gives AC2.
        conf_level: Two-sided confidence level for the interval.

    Returns:
        The coefficient, its linearised standard error and confidence interval.
    """
    r = np.asarray(ratings, dtype=float)
    if r.ndim != 2:
        raise ValueError("ratings must be a 2-D (subjects x raters) array")

    cats = list(categories)
    q = len(cats)
    w = weight_matrix(cats, weights)
    t_w = float(w.sum())

    # r_ik: how many raters put subject i in category k.
    counts = np.stack([(r == c).sum(axis=1) for c in cats], axis=1).astype(float)
    r_i = counts.sum(axis=1)  # raters per subject

    scored = r_i >= 2
    if not scored.any():
        raise ValueError("no subject was rated by at least two raters")

    # Weighted counts: r*_ik = sum_l w[k,l] r_il
    counts_w = counts @ w.T

    # Observed agreement, per subject then averaged over scorable subjects.
    with np.errstate(invalid="ignore", divide="ignore"):
        pa_i = (counts * (counts_w - 1.0)).sum(axis=1) / (r_i * (r_i - 1.0))
    pa_i = np.where(scored, pa_i, 0.0)
    n_eff = int(scored.sum())
    p_a = float(pa_i[scored].mean())

    # Chance agreement from the mean marginal distribution.
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(r_i[:, None] > 0, counts / np.maximum(r_i, 1)[:, None], 0.0)
    pi = share[r_i > 0].mean(axis=0)
    p_e = t_w * float((pi * (1.0 - pi)).sum()) / (q * (q - 1.0))

    denom = 1.0 - p_e
    gamma = (p_a - p_e) / denom if denom != 0 else float("nan")

    # Gwet's linearisation variance.
    pe_i = (t_w / (q * (q - 1.0))) * (share * (1.0 - pi)).sum(axis=1)
    gamma_i = (pa_i - p_e) / denom - 2.0 * (1.0 - gamma) * (pe_i - p_e) / denom
    sub = gamma_i[scored]
    var = float(((sub - gamma) ** 2).sum() / (n_eff * (n_eff - 1))) if n_eff > 1 else 0.0
    se = math.sqrt(max(var, 0.0))

    z = _z_for(conf_level)
    return Coefficient(
        name="AC1" if weights == "identity" else f"AC2({weights})",
        value=gamma,
        se=se,
        ci_low=max(-1.0, gamma - z * se),
        ci_high=min(1.0, gamma + z * se),
        percent_agreement=p_a,
        percent_chance=p_e,
        n_subjects=n_eff,
        n_raters_max=int(r_i.max()),
    )


def pabak(ratings: np.ndarray, categories: Sequence[int]) -> Coefficient:
    """Prevalence-adjusted bias-adjusted kappa.

    Chance agreement is fixed at ``1/q`` regardless of the observed marginals,
    which is what makes the statistic immune to prevalence distortion -- and
    also what makes it cruder than AC1.
    """
    r = np.asarray(ratings, dtype=float)
    cats = list(categories)
    q = len(cats)

    counts = np.stack([(r == c).sum(axis=1) for c in cats], axis=1).astype(float)
    r_i = counts.sum(axis=1)
    scored = r_i >= 2
    if not scored.any():
        raise ValueError("no subject was rated by at least two raters")

    with np.errstate(invalid="ignore", divide="ignore"):
        pa_i = (counts * (counts - 1.0)).sum(axis=1) / (r_i * (r_i - 1.0))
    p_a = float(pa_i[scored].mean())
    p_e = 1.0 / q
    value = (p_a - p_e) / (1.0 - p_e)

    n_eff = int(scored.sum())
    se = float(np.std(pa_i[scored], ddof=1) / math.sqrt(n_eff) / (1.0 - p_e)) if n_eff > 1 else 0.0
    z = _z_for(0.95)
    return Coefficient(
        name="PABAK",
        value=value,
        se=se,
        ci_low=max(-1.0, value - z * se),
        ci_high=min(1.0, value + z * se),
        percent_agreement=p_a,
        percent_chance=p_e,
        n_subjects=n_eff,
        n_raters_max=int(r_i.max()),
    )


# ── ICC ──────────────────────────────────────────────────────────────────────


def icc_two_way(ratings: np.ndarray) -> IccResult:
    """ICC(2,1) and ICC(2,k): two-way random effects, absolute agreement.

    Args:
        ratings: ``(n_subjects, n_raters)`` complete score matrix.

    Returns:
        Both single- and average-measure estimates. ``ICC(2,k)`` is reported
        twice: from the McGraw & Wong mean-squares form, and by Spearman-Brown
        extrapolation from ICC(2,1). The two are algebraically identical for
        this model -- see ``tests/test_reliability.py`` for the derivation --
        so the pair doubles as a self-check on the mean-squares arithmetic.
    """
    x = np.asarray(ratings, dtype=float)
    if x.ndim != 2:
        raise ValueError("ratings must be a 2-D (subjects x raters) array")
    if np.isnan(x).any():
        raise ValueError("ICC requires a complete matrix; drop or impute missing")

    n, k = x.shape
    if n < 2 or k < 2:
        raise ValueError("need at least two subjects and two raters")

    if np.allclose(x, x.flat[0]):
        return IccResult(
            None, None, None, n, k, degenerate=True,
            note="all scores identical: between-subject variance is zero and "
                 "ICC is undefined. Agreement is total; use AC1 instead.",
        )

    grand = x.mean()
    ss_total = ((x - grand) ** 2).sum()
    ss_rows = k * ((x.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((x.mean(axis=0) - grand) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))

    denom_1 = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    denom_k = ms_rows + (ms_cols - ms_err) / n

    icc1 = (ms_rows - ms_err) / denom_1 if denom_1 != 0 else float("nan")
    icck = (ms_rows - ms_err) / denom_k if denom_k != 0 else float("nan")
    sb = (k * icc1) / (1.0 + (k - 1) * icc1) if icc1 > -1 / (k - 1) else float("nan")

    return IccResult(
        icc_2_1=float(icc1),
        icc_2_k=float(icck),
        icc_2_k_spearman_brown=float(sb),
        n_subjects=n,
        n_raters=k,
        degenerate=False,
    )


# ── Exact agreement ──────────────────────────────────────────────────────────


def exact_agreement(ratings: np.ndarray) -> float:
    """Fraction of subjects on which every rater gave an identical score."""
    x = np.asarray(ratings, dtype=float)
    if x.size == 0:
        return float("nan")
    same = (x == x[:, [0]]).all(axis=1)
    return float(same.mean())


def _z_for(conf_level: float) -> float:
    """Normal quantile for a two-sided interval, without a SciPy dependency."""
    p = 1.0 - (1.0 - conf_level) / 2.0
    # Acklam's inverse-normal approximation; accurate to ~1e-9 in this range.
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        z = math.sqrt(-2 * math.log(p))
        return (((((c[0] * z + c[1]) * z + c[2]) * z + c[3]) * z + c[4]) * z + c[5]) / \
               ((((d[0] * z + d[1]) * z + d[2]) * z + d[3]) * z + 1)
    if p > p_high:
        z = math.sqrt(-2 * math.log(1 - p))
        return -((((((c[0] * z + c[1]) * z + c[2]) * z + c[3]) * z + c[4]) * z + c[5]) /
                 ((((d[0] * z + d[1]) * z + d[2]) * z + d[3]) * z + 1))
    z = p - 0.5
    r = z * z
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * z / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
