"""Optional: Web fetcher for augmentation candidates from external sources.

Supports fetching from arXiv abstracts, Wikipedia, and custom web sources.
This module is only active when `modules.web_fetcher: true` in config.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class WebFetcher:
    """Fetches text from web sources for data augmentation mixing."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.sources = config.get("pipeline", {}).get("data_mixing", {}).get("sources", [])

    def fetch(self, query: str, max_results: int = 50) -> list[str]:
        """Fetch text samples from configured sources.

        Args:
            query: Search query relevant to the training domain.
            max_results: Maximum number of text samples to return.

        Returns:
            List of text strings.
        """
        results = []
        for source in self.sources:
            try:
                if source == "arxiv":
                    results.extend(self._fetch_arxiv(query, max_results))
                elif source == "wikipedia":
                    results.extend(self._fetch_wikipedia(query, max_results))
                else:
                    logger.warning(f"Unknown source: {source}")
            except Exception as e:
                logger.warning(f"Failed to fetch from {source}: {e}")

        return results[:max_results]

    def _fetch_arxiv(self, query: str, max_results: int) -> list[str]:
        """Fetch abstracts from arXiv."""
        import urllib.parse
        import urllib.request
        import xml.etree.ElementTree as ET

        base_url = "http://export.arxiv.org/api/query"
        params = urllib.parse.urlencode({
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
        })

        url = f"{base_url}?{params}"
        response = urllib.request.urlopen(url, timeout=30)
        data = response.read().decode("utf-8")

        root = ET.fromstring(data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        abstracts = []
        for entry in root.findall("atom:entry", ns):
            summary = entry.find("atom:summary", ns)
            if summary is not None and summary.text:
                abstracts.append(summary.text.strip())

        return abstracts

    def _fetch_wikipedia(self, query: str, max_results: int) -> list[str]:
        """Fetch Wikipedia article extracts."""
        import json
        import urllib.parse
        import urllib.request

        base_url = "https://en.wikipedia.org/w/api.php"
        params = urllib.parse.urlencode({
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max_results,
            "format": "json",
        })

        url = f"{base_url}?{params}"
        response = urllib.request.urlopen(url, timeout=30)
        data = json.loads(response.read().decode("utf-8"))

        texts = []
        for result in data.get("query", {}).get("search", []):
            # Get full extract for each page
            page_title = result["title"]
            extract = self._get_wikipedia_extract(page_title)
            if extract:
                texts.append(extract)

        return texts

    def _get_wikipedia_extract(self, title: str) -> str | None:
        """Get the full text extract for a Wikipedia page."""
        import json
        import urllib.parse
        import urllib.request

        params = urllib.parse.urlencode({
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "format": "json",
        })

        url = f"https://en.wikipedia.org/w/api.php?{params}"
        try:
            response = urllib.request.urlopen(url, timeout=15)
            data = json.loads(response.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                extract = page.get("extract", "")
                if extract and len(extract) > 50:
                    return extract
        except Exception:
            pass
        return None
