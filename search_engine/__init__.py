"""Custom Search Engine: an LLM-powered web search CLI built on the Claude API."""

__version__ = "0.1.0"

from .engine import SearchEngine, SearchResult, Source  # noqa: E402

__all__ = ["SearchEngine", "SearchResult", "Source", "__version__"]
