import daft
from daft import DataFrame, col

from llm_data.filters.url_blacklist import extract_domain


class AIDomainReviewReport:
    """Aggregate model scores into candidate domains for manual review."""

    def __init__(
        self,
        url_column: str = "url",
        score_column: str = "ai_content_score",
        page_score_threshold: float = 0.8,
        min_pages: int = 10,
        min_candidate_fraction: float = 0.5,
        domain_column: str = "domain",
        name: str = "AIDomainReviewReport",
    ):
        if not 0.0 <= page_score_threshold <= 1.0:
            raise ValueError("page_score_threshold must be between 0 and 1")
        if min_pages < 1:
            raise ValueError("min_pages must be at least 1")
        if not 0.0 <= min_candidate_fraction <= 1.0:
            raise ValueError("min_candidate_fraction must be between 0 and 1")

        self.url_column = url_column
        self.score_column = score_column
        self.page_score_threshold = page_score_threshold
        self.min_pages = min_pages
        self.min_candidate_fraction = min_candidate_fraction
        self.domain_column = domain_column
        self.name = name

    def __call__(self, df: DataFrame) -> DataFrame:
        with_domains = df.with_column(
            self.domain_column, extract_domain(col(self.url_column))
        ).where(col(self.domain_column).not_null() & (col(self.domain_column) != ""))
        above_threshold = (col(self.score_column) >= self.page_score_threshold).cast(
            daft.DataType.int64()
        )
        report = with_domains.groupby(self.domain_column).agg(
            col(self.domain_column).count().alias("page_count"),
            above_threshold.sum().alias("pages_above_threshold"),
            col(self.score_column).mean().alias("mean_score"),
        )
        report = report.with_column(
            "fraction_above_threshold",
            col("pages_above_threshold") / col("page_count"),
        )
        return report.where(
            (col("page_count") >= self.min_pages)
            & (col("fraction_above_threshold") >= self.min_candidate_fraction)
        ).select(
            self.domain_column,
            "page_count",
            "pages_above_threshold",
            "fraction_above_threshold",
            "mean_score",
        )
