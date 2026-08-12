```bash
ruff check --select I --fix . && ruff format .
```

## AI-content domain review

AI-content detection is a review workflow rather than an automatic source-of-truth
filter. Score pages with a sequence-classification model, aggregate likely-positive
pages into candidate domains for manual inspection, and apply only the domain set
that reviewers approve:

```python
from llm_data.factories.ai_content import AIContentScorer
from llm_data.filters.ai_content import ReviewedAIDomainFilter
from llm_data.reports.ai_content import AIDomainReviewReport

scored = AIContentScorer(
    model_name_or_path="organization/model-name",
    ai_label="generated",
    normalization="softmax",
    batch_size=32,
    gpus=1,
)(pages)

candidates = AIDomainReviewReport(
    page_score_threshold=0.8,
    min_pages=20,
    min_candidate_fraction=0.6,
)(scored)

# Populate this set only after reviewing and validating the candidate report.
reviewed_domains = {"reviewed-generated-content.example"}
filtered = ReviewedAIDomainFilter(reviewed_domains)(scored)
```

Detector scores are probabilistic and model-specific. The configured output label
and normalization must match the selected model, and thresholds require validation
on representative data. A candidate domain is a review artifact, not a claim that
the domain is AI-generated; only manually reviewed domains should enter the final
blocklist.

```
@misc{paster2023openwebmath,
    title={OpenWebMath: An Open Dataset of High-Quality Mathematical Web Text},
    author={Keiran Paster and Marco Dos Santos and Zhangir Azerbayev and Jimmy Ba},
    year={2023},
    eprint={2310.06786},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```