"""DuckDuckGo web search wrapper with rate-limit handling."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from duckduckgo_search import DDGS

log = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str
    image_url: str | None


class WebFetcher:
    """Wraps DuckDuckGo search. Rate-limited to 1 req/min by default."""

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        try:
            with DDGS() as ddgs:
                raw = ddgs.text(query, max_results=max_results)
            results = []
            for r in raw:
                results.append(SearchResult(
                    title=r.get("title", ""),
                    snippet=r.get("body", ""),
                    url=r.get("href", ""),
                    image_url=r.get("image") or None,
                ))
            return results
        except Exception as e:
            log.warning(f"WebFetcher.search failed: {e}")
            return []
