"""Web fetch — DuckDuckGo, Wikipedia, and arXiv search, no API key needed."""
from __future__ import annotations
import json
import logging
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass

log = logging.getLogger(__name__)

_SCIENCE_WORDS = {
    "quantum", "neural", "protein", "theorem", "entropy", "photon",
    "genome", "algorithm", "topology", "manifold", "plasma", "crystal",
    "enzyme",
}

_TIMEOUT = 8      # seconds for every HTTP call
_DDG_TIMEOUT = 12  # hard cap on DuckDuckGo (no built-in timeout in ddgs library)
_FETCH_TIMEOUT = 10  # trafilatura full-page fetch timeout


def _fetch_full_content(url: str) -> str | None:
    """Fetch full article text via trafilatura. Returns None on any failure."""
    if not url:
        return None
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url, timeout=_FETCH_TIMEOUT)
        if not downloaded:
            return None
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        return text if text and len(text) > 100 else None
    except Exception as e:
        log.debug(f"trafilatura fetch failed for '{url}': {e}")
        return None


@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str


def _ddg_search(query: str, max_results: int) -> list[SearchResult]:
    def _do_search():
        from ddgs import DDGS
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
        return [SearchResult(title=r.get("title", ""), snippet=r.get("body", ""), url=r.get("href", "")) for r in raw]

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_do_search)
            return future.result(timeout=_DDG_TIMEOUT)
    except FuturesTimeoutError:
        log.warning(f"DuckDuckGo timed out after {_DDG_TIMEOUT}s for '{query}'")
        return []
    except Exception as e:
        log.warning(f"DuckDuckGo search failed for '{query}': {e}")
        return []


def wikipedia_search(query: str, max_results: int = 3) -> list[SearchResult]:
    """Search Wikipedia. Returns empty list on failure (never crashes)."""
    results: list[SearchResult] = []
    try:
        search_url = (
            "https://en.wikipedia.org/w/api.php?action=query&list=search"
            f"&srsearch={urllib.parse.quote(query)}&srlimit={max_results}&format=json"
        )
        req = urllib.request.Request(search_url, headers={"User-Agent": "HaloFEP/3"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for hit in data.get("query", {}).get("search", []):
            title = hit.get("title", "")
            if not title:
                continue
            snippet = hit.get("snippet", "")
            page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
            try:
                sum_req = urllib.request.Request(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}",
                    headers={"User-Agent": "HaloFEP/3"},
                )
                with urllib.request.urlopen(sum_req, timeout=_TIMEOUT) as r:
                    sd = json.loads(r.read().decode("utf-8"))
                if sd.get("extract"):
                    snippet = sd["extract"]
                page_url = sd.get("content_urls", {}).get("desktop", {}).get("page", page_url)
            except Exception:
                pass
            results.append(SearchResult(title=title, snippet=snippet, url=page_url))
    except Exception as e:
        log.warning(f"Wikipedia search failed for '{query}': {e}")
    return results


def arxiv_search(query: str, max_results: int = 3) -> list[SearchResult]:
    """Search arXiv preprints. Returns empty list on failure (never crashes)."""
    results: list[SearchResult] = []
    try:
        url = (
            f"http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query)}"
            f"&start=0&max_results={max_results}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "HaloFEP/3"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            root = ET.fromstring(resp.read())
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("atom:entry", ns):
            title = (entry.findtext("atom:title", "", ns) or "").strip().replace("\n", " ")
            snippet = (entry.findtext("atom:summary", "", ns) or "").strip().replace("\n", " ")
            url_str = (entry.findtext("atom:id", "", ns) or "").strip()
            if title:
                results.append(SearchResult(title=title, snippet=snippet, url=url_str))
    except Exception as e:
        log.warning(f"arXiv search failed for '{query}': {e}")
    return results


def _is_scientific(query: str) -> bool:
    """Check if query is scientific enough to warrant arXiv search."""
    return bool(set(query.lower().split()) & _SCIENCE_WORDS)


def _title_overlap(a: str, b: str) -> float:
    """Jaccard overlap of title words — used for deduplication."""
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def multi_source_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search DuckDuckGo + Wikipedia (+ arXiv for science queries) in parallel.

    All sources run concurrently — total time = slowest source, not sum.
    DuckDuckGo results come first. Wikipedia and arXiv fill remaining slots,
    deduplicated by title overlap > 60%.
    """
    is_sci = _is_scientific(query)
    with ThreadPoolExecutor(max_workers=3) as ex:
        ddg_future = ex.submit(_ddg_search, query, max_results)
        wiki_future = ex.submit(wikipedia_search, query, 3)
        arxiv_future = ex.submit(arxiv_search, query, 3) if is_sci else None

        ddg_results = ddg_future.result()
        wiki_results = wiki_future.result()
        arxiv_results = arxiv_future.result() if arxiv_future else []

    # Enrich top-2 DDG results with full article text (parallel, non-blocking on failure)
    if ddg_results:
        top_urls = [r.url for r in ddg_results[:2] if r.url]
        with ThreadPoolExecutor(max_workers=2) as fetch_ex:
            full_texts = list(fetch_ex.map(_fetch_full_content, top_urls, timeout=_FETCH_TIMEOUT + 2))
        for i, text in enumerate(full_texts):
            if text:
                ddg_results[i] = SearchResult(
                    title=ddg_results[i].title,
                    snippet=text[:2000],  # cap at 2000 chars
                    url=ddg_results[i].url,
                )

    n_ddg = len(ddg_results)
    combined = list(ddg_results)
    seen_titles = [r.title for r in combined]

    def _is_dup(title: str) -> bool:
        return any(_title_overlap(title, s) > 0.60 for s in seen_titles)

    n_wiki = 0
    for r in wiki_results:
        if len(combined) >= max_results:
            break
        if not _is_dup(r.title):
            combined.append(r)
            seen_titles.append(r.title)
            n_wiki += 1

    n_arxiv = 0
    for r in arxiv_results:
        if len(combined) >= max_results:
            break
        if not _is_dup(r.title):
            combined.append(r)
            seen_titles.append(r.title)
            n_arxiv += 1

    log.info(f"Perception: ddg={n_ddg} wiki={n_wiki} arxiv={n_arxiv}")
    return combined[:max_results]


def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search the web using multiple sources. Returns empty list on failure (never crashes)."""
    return multi_source_search(query, max_results)


def results_to_texts(results: list[SearchResult]) -> list[str]:
    """Convert search results to list of text chunks for embedding."""
    texts = []
    for r in results:
        if r.title:
            texts.append(r.title)
        if r.snippet:
            words = r.snippet.split()
            for i in range(0, len(words), 50):
                chunk = " ".join(words[i:i+50])
                if chunk:
                    texts.append(chunk)
    return texts
