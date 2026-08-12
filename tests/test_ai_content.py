import math

import daft
import pytest

from llm_data.factories.ai_content import (
    AIContentScorer,
    normalize_label_scores,
)
from llm_data.filters.ai_content import ReviewedAIDomainFilter
from llm_data.reports.ai_content import AIDomainReviewReport


class FakePredictor:
    @property
    def id2label(self) -> dict[int, str]:
        return {0: "human", 1: "generated"}

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, float(text.count("AI"))] for text in texts]


def test_normalize_label_scores_uses_configured_label_mapping():
    scores = normalize_label_scores(
        [[3.0, 1.0], [0.0, 2.0]],
        ai_label="generated",
        id2label={"0": "generated", "1": "human"},
    )

    assert scores == pytest.approx(
        [1.0 / (1.0 + math.exp(-2.0)), 1.0 / (1.0 + math.exp(2.0))]
    )


def test_ai_content_scorer_handles_null_and_blank_without_model_inference():
    df = daft.from_pydict({"text": [None, "", "   ", "one AI marker", "AI twice AI"]})

    result = (
        AIContentScorer(
            model_name_or_path="unused-in-test",
            ai_label="generated",
            batch_size=5,
            predictor_factory=FakePredictor,
        )(df)
        .collect()
        .to_pydict()
    )

    assert result["text"] == [None, "", "   ", "one AI marker", "AI twice AI"]
    assert result["ai_content_score"] == pytest.approx(
        [0.0, 0.0, 0.0, 1.0 / (1.0 + math.exp(-1.0)), 1.0 / (1.0 + math.exp(-2.0))]
    )


def test_domain_review_report_applies_page_and_fraction_thresholds():
    df = daft.from_pydict(
        {
            "url": [
                "https://candidate.example/a",
                "https://candidate.example/b",
                "https://candidate.example/c",
                "https://small.example/a",
                "https://small.example/b",
                None,
            ],
            "ai_content_score": [0.9, 0.8, 0.1, 0.9, 0.8, 1.0],
        }
    )

    result = (
        AIDomainReviewReport(
            page_score_threshold=0.8,
            min_pages=3,
            min_candidate_fraction=0.6,
        )(df)
        .collect()
        .to_pydict()
    )

    assert result == {
        "domain": ["candidate.example"],
        "page_count": [3],
        "pages_above_threshold": [2],
        "fraction_above_threshold": pytest.approx([2 / 3]),
        "mean_score": pytest.approx([0.6]),
    }


def test_reviewed_domain_filter_reuses_parent_domain_matching():
    df = daft.from_pydict(
        {
            "url": [
                "https://reviewed.example/page",
                "https://deep.sub.reviewed.example/page",
                "https://allowed.example/page",
                None,
            ],
            "value": [1, 2, 3, 4],
        }
    )

    result = ReviewedAIDomainFilter({"reviewed.example"})(df).collect().to_pydict()

    assert result == {
        "url": ["https://allowed.example/page", None],
        "value": [3, 4],
    }
