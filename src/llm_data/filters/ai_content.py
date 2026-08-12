from collections.abc import Iterable

from daft import DataFrame

from llm_data.filters.url_blacklist import URLBlacklistFilter


class ReviewedAIDomainFilter:
    """Filter only domains that have been manually reviewed and approved for removal."""

    def __init__(
        self,
        reviewed_domains: Iterable[str],
        input_column: str = "url",
        name: str = "ReviewedAIDomainFilter",
    ):
        self.name = name
        self._filter = URLBlacklistFilter(
            input_column=input_column,
            blacklist=set(reviewed_domains),
            name=name,
        )

    def __call__(self, df: DataFrame) -> DataFrame:
        return self._filter(df)
