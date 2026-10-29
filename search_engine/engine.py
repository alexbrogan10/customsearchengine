"""Core search engine: asks Claude to research a query on the live web and
returns a cited answer.

Claude runs the searches itself through Anthropic's server-side ``web_search``
and ``web_fetch`` tools, so there is no tool loop to drive here beyond
resuming a turn the server paused.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import anthropic

DEFAULT_MODEL = "claude-opus-5-5"
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# A paused server-tool turn is resumed by sending it back; cap the resumes so
# a runaway research loop can't spin forever.
MAX_CONTINUATIONS = 5

SYSTEM_PROMPT = """\
You are a search engine. For every query, search the web for current, \
authoritative information, then answer from what you found.

- Open with the direct answer in one or two sentences, then add the detail \
the query calls for. Keep it skimmable: short paragraphs, and bullets only \
for genuinely list-shaped content.
- Ground every factual claim in a search result so it carries a citation. \
Prefer primary and recent sources; say so when sources disagree or when \
information may be out of date.
- If the query is ambiguous, answer the most likely reading and note the \
alternative in one line.
- If the web doesn't settle the question, say what is known and what isn't \
rather than guessing.
"""

# Event callback: kind is one of "search", "fetch", "tool_error", "text",
# "citations", "notice"; payload is a str (or list[int] for "citations").
EventHandler = Callable[[str, Any], None]


@dataclass
class Source:
    url: str
    title: str
    page_age: Optional[str] = None


@dataclass
class SearchResult:
    query: str
    answer: str
    sources: list[Source]  # cited sources; answer markers [n] index this list (1-based)
    searches: list[str] = field(default_factory=list)
    fetched: list[str] = field(default_factory=list)
    results_seen: list[Source] = field(default_factory=list)  # every search hit, cited or not
    stop_reason: Optional[str] = None
    model: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": [vars(s) for s in self.sources],
            "searches": self.searches,
            "fetched": self.fetched,
            "results_seen": [vars(s) for s in self.results_seen],
            "stop_reason": self.stop_reason,
            "model": self.model,
            "usage": {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens},
        }


class CitationIndex:
    """Assigns stable 1-based numbers to cited URLs across a whole answer."""

    def __init__(self) -> None:
        self._numbers: dict[str, int] = {}
        self.sources: list[Source] = []

    def number_for(self, url: str, title: Optional[str]) -> int:
        if url not in self._numbers:
            self.sources.append(Source(url=url, title=title or url))
            self._numbers[url] = len(self.sources)
        return self._numbers[url]

    def numbers_for_block(self, block: Any) -> list[int]:
        """Citation numbers for one text block, de-duplicated, in order."""
        numbers: list[int] = []
        for citation in getattr(block, "citations", None) or []:
            url = getattr(citation, "url", None)
            if not url:
                continue
            n = self.number_for(url, getattr(citation, "title", None))
            if n not in numbers:
                numbers.append(n)
        return numbers


def format_markers(numbers: list[int]) -> str:
    return "".join(f"[{n}]" for n in numbers)


class SearchEngine:
    """An LLM search engine with conversational follow-ups.

    ``search`` answers a query in the context of earlier queries in this
    session; call ``reset`` to start fresh.
    """

    def __init__(
        self,
        client: Optional[anthropic.Anthropic] = None,
        *,
        model: str = DEFAULT_MODEL,
        effort: str = "medium",
        max_searches: int = 5,
        allow_fetch: bool = True,
        allowed_domains: Optional[list[str]] = None,
        blocked_domains: Optional[list[str]] = None,
        country: Optional[str] = None,
    ) -> None:
        if effort not in EFFORT_LEVELS:
            raise ValueError(f"effort must be one of {', '.join(EFFORT_LEVELS)}")
        if allowed_domains and blocked_domains:
            raise ValueError("use allowed_domains or blocked_domains, not both")
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.tools = self._build_tools(max_searches, allow_fetch, allowed_domains, blocked_domains, country)
        self.messages: list[dict[str, Any]] = []

    @staticmethod
    def _build_tools(
        max_searches: int,
        allow_fetch: bool,
        allowed_domains: Optional[list[str]],
        blocked_domains: Optional[list[str]],
        country: Optional[str],
    ) -> list[dict[str, Any]]:
        domain_filter: dict[str, Any] = {}
        if allowed_domains:
            domain_filter["allowed_domains"] = allowed_domains
        if blocked_domains:
            domain_filter["blocked_domains"] = blocked_domains

        search: dict[str, Any] = {
            "type": "web_search_20260209",
            "name": "web_search",
            "max_uses": max_searches,
            **domain_filter,
        }
        if country:
            search["user_location"] = {"type": "approximate", "country": country.upper()}
        tools = [search]
        if allow_fetch:
            tools.append({
                "type": "web_fetch_20260209",
                "name": "web_fetch",
                "max_uses": max_searches,
                "citations": {"enabled": True},
                **domain_filter,
            })
        return tools

    def reset(self) -> None:
        self.messages = []

    def search(self, query: str, on_event: Optional[EventHandler] = None) -> SearchResult:
        emit = on_event or (lambda kind, payload: None)
        self.messages.append({"role": "user", "content": query})
        # Index of this query's user turn, so a failed request can be rolled back.
        turn_start = len(self.messages) - 1

        citations = CitationIndex()
        result = SearchResult(query=query, answer="", sources=citations.sources)
        answer_parts: list[str] = []

        try:
            for _ in range(MAX_CONTINUATIONS + 1):
                response = self._stream_turn(citations, result, answer_parts, emit)
                self.messages.append({"role": "assistant", "content": response.content})
                result.model = response.model
                result.stop_reason = response.stop_reason
                result.input_tokens += response.usage.input_tokens
                result.output_tokens += response.usage.output_tokens
                if response.stop_reason != "pause_turn":
                    break
            else:
                emit("notice", "Research was cut short after too many continuations.")
        except BaseException:
            del self.messages[turn_start:]
            raise

        if result.stop_reason == "refusal":
            # A declined turn can't be continued from; drop it so follow-ups still work.
            del self.messages[turn_start:]
        elif result.stop_reason == "max_tokens":
            emit("notice", "The answer hit the output limit and may be incomplete.")

        result.answer = "".join(answer_parts).strip()
        return result

    def _stream_turn(
        self,
        citations: CitationIndex,
        result: SearchResult,
        answer_parts: list[str],
        emit: EventHandler,
    ) -> Any:
        today = dt.date.today().isoformat()
        with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=64000,
            system=f"{SYSTEM_PROMPT}\nToday's date is {today}.",
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            tools=self.tools,
            messages=self.messages,
            betas=[FALLBACK_BETA],
            fallbacks="default",
        ) as stream:
            for event in stream:
                if event.type == "text":
                    answer_parts.append(event.text)
                    emit("text", event.text)
                elif event.type == "content_block_stop":
                    self._on_block_done(event.content_block, citations, result, answer_parts, emit)
            return stream.get_final_message()

    @staticmethod
    def _on_block_done(
        block: Any,
        citations: CitationIndex,
        result: SearchResult,
        answer_parts: list[str],
        emit: EventHandler,
    ) -> None:
        if block.type == "server_tool_use":
            if block.name == "web_search":
                query = str(block.input.get("query", ""))
                result.searches.append(query)
                emit("search", query)
            elif block.name == "web_fetch":
                url = str(block.input.get("url", ""))
                result.fetched.append(url)
                emit("fetch", url)
        elif block.type == "web_search_tool_result":
            # Success is a list of results; an error is a single object.
            if isinstance(block.content, list):
                result.results_seen.extend(
                    Source(url=r.url, title=r.title, page_age=getattr(r, "page_age", None))
                    for r in block.content
                )
            else:
                emit("tool_error", f"search failed: {getattr(block.content, 'error_code', 'unknown error')}")
        elif block.type == "web_fetch_tool_result":
            if getattr(block.content, "type", None) != "web_fetch_result":
                emit("tool_error", f"fetch failed: {getattr(block.content, 'error_code', 'unknown error')}")
        elif block.type == "text":
            numbers = citations.numbers_for_block(block)
            if numbers:
                answer_parts.append(format_markers(numbers))
                emit("citations", numbers)
        elif block.type == "fallback":
            emit("notice", f"{block.from_.model} declined; answered by {block.to.model}.")
