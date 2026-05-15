"""Web fetch — DuckDuckGo search, no API key needed."""
from __future__ import annotations
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str


def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search DuckDuckGo. Returns empty list on failure (never crashes)."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
        results = []
        for r in raw:
            results.append(SearchResult(
                title=r.get("title", ""),
                snippet=r.get("body", ""),
                url=r.get("href", ""),
            ))
        return results
    except Exception as e:
        log.warning(f"Web search failed for '{query}': {e}")
        return []


def results_to_texts(results: list[SearchResult]) -> list[str]:
    """Convert search results to list of text chunks for embedding."""
    texts = []
    for r in results:
        if r.title:
            texts.append(r.title)
        if r.snippet:
            # Split long snippets into ~50 word chunks
            words = r.snippet.split()
            for i in range(0, len(words), 50):
                chunk = " ".join(words[i:i+50])
                if chunk:
                    texts.append(chunk)
    return texts
