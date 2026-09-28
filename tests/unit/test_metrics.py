import math

import pytest

from lexeu.eval.metrics import hit_at_k, matches, mrr_at_k, ndcg_at_k, recall_at_k

GDPR33 = "32016R0679:art_33:p1"
NIS23 = "32022L2555:art_23:p4"


def test_matching_is_exact_or_nested() -> None:
    assert matches("32016R0679:art_4:u1", "32016R0679:art_4")
    assert matches(GDPR33, GDPR33)
    assert not matches("32016R0679:art_40:p1", "32016R0679:art_4")  # not a prefix match
    assert not matches("32016R0679:art_33:p2", GDPR33)


def test_hit_and_mrr() -> None:
    retrieved = ["a:art_1:p1", GDPR33, "b:art_2:p1"]
    assert hit_at_k(retrieved, [GDPR33], 1) == 0.0
    assert hit_at_k(retrieved, [GDPR33], 3) == 1.0
    assert mrr_at_k(retrieved, [GDPR33], 10) == pytest.approx(0.5)
    assert mrr_at_k(retrieved, ["x:art_9:p9"], 10) == 0.0


def test_recall_counts_each_expected_provision() -> None:
    retrieved = [NIS23, "a:art_1:p1"]
    assert recall_at_k(retrieved, [GDPR33, NIS23], 10) == 0.5
    assert recall_at_k([NIS23, GDPR33], [GDPR33, NIS23], 10) == 1.0
    assert recall_at_k([NIS23, GDPR33], [GDPR33, NIS23], 1) == 0.5


def test_ndcg_perfect_and_partial() -> None:
    assert ndcg_at_k([GDPR33, NIS23], [GDPR33, NIS23], 10) == pytest.approx(1.0)
    partial = ndcg_at_k(["x:art_1:p1", GDPR33], [GDPR33], 10)
    assert partial == pytest.approx(1 / math.log2(3))
    # a second chunk of an already-credited provision earns nothing
    assert ndcg_at_k([GDPR33, GDPR33], [GDPR33, NIS23], 10) == pytest.approx(
        1.0 / (1.0 + 1 / math.log2(3))
    )
