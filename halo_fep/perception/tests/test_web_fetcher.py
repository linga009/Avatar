from unittest.mock import patch, MagicMock
from halo_fep.perception.web_fetcher import WebFetcher, SearchResult


def test_search_result_fields():
    r = SearchResult(title="Test", snippet="A snippet", url="http://x.com", image_url=None)
    assert r.title == "Test"
    assert r.image_url is None


def test_web_fetcher_returns_list():
    mock_results = [
        {"title": "T1", "body": "B1", "href": "http://a.com", "image": None},
        {"title": "T2", "body": "B2", "href": "http://b.com", "image": "http://img.com/x.jpg"},
    ]
    with patch("halo_fep.perception.web_fetcher.DDGS") as MockDDGS:
        mock_ddgs = MagicMock()
        MockDDGS.return_value.__enter__.return_value = mock_ddgs
        mock_ddgs.text.return_value = mock_results
        fetcher = WebFetcher()
        results = fetcher.search("test query", max_results=2)
    assert len(results) == 2
    assert results[0].title == "T1"
    assert results[1].image_url == "http://img.com/x.jpg"


def test_web_fetcher_handles_empty():
    with patch("halo_fep.perception.web_fetcher.DDGS") as MockDDGS:
        mock_ddgs = MagicMock()
        MockDDGS.return_value.__enter__.return_value = mock_ddgs
        mock_ddgs.text.return_value = []
        fetcher = WebFetcher()
        results = fetcher.search("nonexistent xyz", max_results=5)
    assert results == []


def test_web_fetcher_handles_exception():
    with patch("halo_fep.perception.web_fetcher.DDGS") as MockDDGS:
        MockDDGS.return_value.__enter__.side_effect = Exception("rate limit")
        fetcher = WebFetcher()
        results = fetcher.search("query", max_results=5)
    assert results == []
