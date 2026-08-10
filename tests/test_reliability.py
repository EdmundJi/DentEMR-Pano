"""Correctness tests for the agreement coefficients.

The ICC test reproduces the published values for the Shrout & Fleiss (1979)
worked example. The AC1 tests use a four-subject table small enough to verify
by hand, plus the degenerate all-maximum case that the checklist's Dim5
actually produces.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dentemr_pano.reliability import (
    exact_agreement,
    gwet_ac,
    icc_two_way,
    pabak,
    weight_matrix,
)

# Shrout & Fleiss (1979), Table 1: 6 targets rated by 4 judges.
SHROUT_FLEISS = np.array(
    [
        [9, 2, 5, 8],
        [6, 1, 3, 2],
        [8, 4, 6, 8],
        [7, 1, 2, 6],
        [10, 5, 6, 9],
        [6, 2, 4, 7],
    ],
    dtype=float,
)


class TestIcc:
    def test_matches_published_shrout_fleiss_values(self):
        res = icc_two_way(SHROUT_FLEISS)
        # Published: ICC(2,1) = 0.29, ICC(2,k) = 0.62.
        assert res.icc_2_1 == pytest.approx(0.290, abs=0.005)
        assert res.icc_2_k == pytest.approx(0.620, abs=0.005)
        assert not res.degenerate

    def test_spearman_brown_is_algebraically_identical_to_icc_2k(self):
        """The manuscript derived ICC(2,k) by Spearman-Brown. That is exact.

        Writing A = MSR - MSE, the single-measure denominator satisfies
        D1 + (k-1)A = k*(MSR + (MSC - MSE)/n) = k*Dk, so

            k*I1 / (1 + (k-1)*I1) = kA / (D1 + (k-1)A) = A / Dk = Ik

        identically, not approximately. The identity holds whatever the rater
        main effect is, so it is checked here on data with a large judge
        effect and on the degenerate-free random case.
        """
        res = icc_two_way(SHROUT_FLEISS)
        assert res.icc_2_k_spearman_brown == pytest.approx(res.icc_2_k, abs=1e-12)

        rng = np.random.default_rng(3)
        for _ in range(5):
            other = icc_two_way(rng.normal(size=(25, 3)) + rng.normal(size=(25, 1)))
            assert other.icc_2_k_spearman_brown == pytest.approx(
                other.icc_2_k, abs=1e-12
            )

    def test_perfect_agreement_gives_unity(self):
        x = np.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=float)
        assert icc_two_way(x).icc_2_1 == pytest.approx(1.0)

    def test_zero_variance_is_reported_as_degenerate_not_zero(self):
        """Dim5's real shape: every rater gives every case full marks."""
        x = np.full((160, 3), 8.0)
        res = icc_two_way(x)
        assert res.degenerate
        assert res.icc_2_1 is None
        assert "undefined" in res.note

    def test_rejects_incomplete_matrix(self):
        x = np.array([[1.0, np.nan], [2.0, 2.0]])
        with pytest.raises(ValueError, match="complete matrix"):
            icc_two_way(x)


class TestWeights:
    def test_identity_is_the_unweighted_case(self):
        assert np.array_equal(weight_matrix((0, 1, 2), "identity"), np.eye(3))

    def test_quadratic_weights_on_a_three_point_scale(self):
        w = weight_matrix((0, 1, 2), "quadratic")
        assert w[0, 0] == pytest.approx(1.0)
        assert w[0, 1] == pytest.approx(0.75)
        assert w[0, 2] == pytest.approx(0.0)
        assert w.sum() == pytest.approx(6.0)

    def test_all_schemes_are_symmetric_with_unit_diagonal(self):
        for kind in ("identity", "linear", "quadratic", "ordinal"):
            w = weight_matrix((0, 1, 2), kind)
            assert np.allclose(w, w.T), kind
            assert np.allclose(np.diag(w), 1.0), kind
            assert w.min() >= 0.0 and w.max() <= 1.0, kind

    def test_partial_credit_orders_near_misses_above_far_misses(self):
        for kind in ("linear", "quadratic", "ordinal"):
            w = weight_matrix((0, 1, 2), kind)
            assert w[0, 1] > w[0, 2], kind


class TestGwetAc1:
    def test_hand_computed_four_subject_table(self):
        """Two raters, binary scale, agreement on 3 of 4 subjects.

        p_a = 0.75, marginals pi = (0.375, 0.625), so
        p_e = 2 * 2 * 0.375 * 0.625 / 2 = 0.46875 and
        AC1 = (0.75 - 0.46875) / (1 - 0.46875) = 0.5294...
        """
        x = np.array([[1, 1], [1, 1], [1, 0], [0, 0]], dtype=float)
        res = gwet_ac(x, categories=(0, 1))
        assert res.percent_agreement == pytest.approx(0.75)
        assert res.percent_chance == pytest.approx(0.46875)
        assert res.value == pytest.approx(0.28125 / 0.53125)
        assert res.name == "AC1"

    def test_unanimous_maximum_scores_give_ac1_of_one(self):
        """The Dim5 result: ICC is undefined here, AC1 is exactly 1.00.

        This is the substantive gain from adding AC1 -- an entry the manuscript
        currently reports as 'N/A' becomes 'perfect agreement'.
        """
        x = np.full((160, 3), 2.0)
        res = gwet_ac(x, categories=(0, 2))
        assert res.value == pytest.approx(1.0)
        assert res.percent_agreement == pytest.approx(1.0)
        assert res.percent_chance == pytest.approx(0.0)

    def test_high_agreement_with_skewed_marginals_beats_chance(self):
        """The prevalence regime where kappa collapses but AC1 does not."""
        x = np.full((100, 3), 2.0)
        x[:4] = [2, 2, 1]  # a few near-misses
        res = gwet_ac(x, categories=(0, 1, 2))
        assert res.value > 0.9

    def test_weighted_ac2_exceeds_unweighted_when_misses_are_adjacent(self):
        rng = np.random.default_rng(0)
        base = rng.integers(0, 3, size=(80, 1))
        x = np.hstack([base, np.clip(base + rng.integers(-1, 2, (80, 1)), 0, 2),
                       np.clip(base + rng.integers(-1, 2, (80, 1)), 0, 2)]).astype(float)
        ac1 = gwet_ac(x, (0, 1, 2), weights="identity").value
        ac2 = gwet_ac(x, (0, 1, 2), weights="quadratic").value
        assert ac2 > ac1

    def test_confidence_interval_brackets_the_estimate(self):
        rng = np.random.default_rng(7)
        x = rng.integers(0, 3, size=(60, 3)).astype(float)
        res = gwet_ac(x, (0, 1, 2))
        assert res.ci_low <= res.value <= res.ci_high
        assert res.se > 0

    def test_missing_ratings_are_tolerated(self):
        x = np.array([[2, 2, 2], [2, 2, np.nan], [1, 1, 1], [0, 0, np.nan]])
        res = gwet_ac(x, (0, 1, 2))
        assert res.n_subjects == 4
        assert res.value == pytest.approx(1.0)


class TestPabak:
    def test_binary_pabak_is_twice_observed_agreement_minus_one(self):
        x = np.array([[1, 1], [1, 1], [1, 0], [0, 0]], dtype=float)
        res = pabak(x, categories=(0, 1))
        assert res.value == pytest.approx(2 * 0.75 - 1)

    def test_three_category_pabak_uses_one_third_chance(self):
        x = np.full((50, 3), 2.0)
        res = pabak(x, categories=(0, 1, 2))
        assert res.percent_chance == pytest.approx(1 / 3)
        assert res.value == pytest.approx(1.0)


class TestExactAgreement:
    def test_counts_only_unanimous_rows(self):
        x = np.array([[2, 2, 2], [2, 2, 1], [0, 0, 0], [1, 2, 0]], dtype=float)
        assert exact_agreement(x) == pytest.approx(0.5)

    def test_unanimous_matrix_is_one(self):
        assert exact_agreement(np.full((10, 3), 2.0)) == pytest.approx(1.0)


def test_ac1_and_icc_diverge_exactly_where_the_reviewer_predicted():
    """Reviewer 2's premise, made reproducible.

    A near-ceiling dimension: raters agree on 95% of cases, but almost all
    cases sit at the maximum. ICC should be poor while AC1 stays high.
    """
    rng = np.random.default_rng(42)
    x = np.full((160, 3), 2.0)
    disagree = rng.choice(160, size=8, replace=False)
    x[disagree, rng.integers(0, 3, size=8)] = 1.0

    icc = icc_two_way(x).icc_2_1
    ac1 = gwet_ac(x, (0, 1, 2)).value

    assert icc < 0.35, f"ICC should be deflated, got {icc}"
    assert ac1 > 0.95, f"AC1 should stay high, got {ac1}"
    assert not math.isnan(ac1)
