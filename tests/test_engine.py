from types import SimpleNamespace as NS

import pytest

from search_engine.cli import main
from search_engine.engine import SearchEngine


def block(type_, **fields):
    return NS(type=type_, **fields)


def text_block(text, citations=None):
    return block("text", text=text, citations=citations)


def cite(url, title):
    return NS(type="web_search_result_location", url=url, title=title, cited_text="...")


def message(content, stop_reason="end_turn"):
    return NS(
        content=content,
        stop_reason=stop_reason,
        model="claude-opus-5-5",
        usage=NS(input_tokens=100, output_tokens=20),
    )


class FakeStream:
    """Replays a final message as the event sequence the SDK stream helper yields."""

    def __init__(self, final):
        self.final = final

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        for b in self.final.content:
            if b.type == "text":
                yield NS(type="text", text=b.text)
            yield NS(type="content_block_stop", content_block=b)

    def get_final_message(self):
        return self.final


class FakeClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.beta = NS(messages=NS(stream=self._stream))

    def _stream(self, **kwargs):
        # Snapshot messages: the engine mutates the same list after the call.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return FakeStream(response)


SEARCH_TURN = [
    block("server_tool_use", name="web_search", input={"query": "tokyo population"}),
    block(
        "web_search_tool_result",
        content=[
            NS(url="https://a.example", title="A", page_age="2 days ago"),
            NS(url="https://b.example", title="B", page_age=None),
        ],
    ),
    text_block("Tokyo has about 14 million people", [cite("https://a.example", "A")]),
    text_block(" in the city proper", [cite("https://b.example", "B"), cite("https://a.example", "A")]),
    text_block("."),
]


def test_search_builds_cited_answer_and_sources():
    client = FakeClient(message(SEARCH_TURN))
    events = []
    result = SearchEngine(client).search("population of tokyo", on_event=lambda k, p: events.append((k, p)))

    assert result.answer == "Tokyo has about 14 million people[1] in the city proper[2][1]."
    assert [s.url for s in result.sources] == ["https://a.example", "https://b.example"]
    assert result.searches == ["tokyo population"]
    assert len(result.results_seen) == 2
    assert ("search", "tokyo population") in events
    assert ("citations", [2, 1]) in events


def test_request_shape():
    client = FakeClient(message([text_block("ok")]))
    SearchEngine(client, effort="high", max_searches=3, allowed_domains=["python.org"], country="gb").search("q")

    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"] == {"effort": "high"}
    assert call["fallbacks"] == "default"
    search_tool, fetch_tool = call["tools"]
    assert search_tool["type"] == "web_search_20260209"
    assert search_tool["max_uses"] == 3
    assert search_tool["allowed_domains"] == ["python.org"]
    assert search_tool["user_location"]["country"] == "GB"
    assert fetch_tool["type"] == "web_fetch_20260209"


def test_no_fetch_omits_fetch_tool():
    engine = SearchEngine(FakeClient(), allow_fetch=False)
    assert [t["name"] for t in engine.tools] == ["web_search"]


def test_pause_turn_is_resumed_without_extra_user_message():
    paused = message([block("server_tool_use", name="web_search", input={"query": "x"})], stop_reason="pause_turn")
    client = FakeClient(paused, message([text_block("done")]))
    result = SearchEngine(client).search("q")

    assert result.answer == "done"
    assert len(client.calls) == 2
    resumed = client.calls[1]["messages"]
    assert [m["role"] for m in resumed] == ["user", "assistant"]
    assert result.input_tokens == 200


def test_follow_up_keeps_history_and_reset_clears_it():
    client = FakeClient(message([text_block("one")]), message([text_block("two")]), message([text_block("three")]))
    engine = SearchEngine(client)
    engine.search("first")
    engine.search("and second?")
    assert [m["role"] for m in client.calls[1]["messages"]] == ["user", "assistant", "user"]

    engine.reset()
    engine.search("fresh")
    assert len(client.calls[2]["messages"]) == 1


def test_refusal_is_reported_and_dropped_from_history():
    client = FakeClient(message([], stop_reason="refusal"), message([text_block("fine")]))
    engine = SearchEngine(client)
    assert engine.search("bad").refused
    assert engine.messages == []
    engine.search("ok")
    assert len(client.calls[1]["messages"]) == 1


def test_failed_request_rolls_back_history():
    client = FakeClient(RuntimeError("boom"), message([text_block("ok")]))
    engine = SearchEngine(client)
    with pytest.raises(RuntimeError):
        engine.search("q")
    assert engine.messages == []


def test_search_error_is_reported_not_raised():
    content = [
        block("server_tool_use", name="web_search", input={"query": "x"}),
        block("web_search_tool_result", content=NS(type="web_search_tool_result_error", error_code="max_uses_exceeded")),
        text_block("partial"),
    ]
    events = []
    result = SearchEngine(FakeClient(message(content))).search("q", on_event=lambda k, p: events.append((k, p)))
    assert result.answer == "partial"
    assert ("tool_error", "search failed: max_uses_exceeded") in events


def test_invalid_options():
    with pytest.raises(ValueError):
        SearchEngine(FakeClient(), effort="extreme")
    with pytest.raises(ValueError):
        SearchEngine(FakeClient(), allowed_domains=["a.com"], blocked_domains=["b.com"])


def test_cli_json_output(monkeypatch, capsys):
    import json

    import search_engine.engine as engine_mod

    monkeypatch.setattr(engine_mod.anthropic, "Anthropic", lambda: FakeClient(message(SEARCH_TURN)))
    assert main(["--json", "population", "of", "tokyo"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["query"] == "population of tokyo"
    assert out["sources"][1]["url"] == "https://b.example"


def test_cli_streams_answer_and_sources(monkeypatch, capsys):
    import search_engine.engine as engine_mod

    monkeypatch.setattr(engine_mod.anthropic, "Anthropic", lambda: FakeClient(message(SEARCH_TURN)))
    assert main(["--no-color", "population of tokyo"]) == 0
    out = capsys.readouterr().out
    assert "searching: tokyo population" in out
    assert "14 million people[1] in the city proper[2][1]." in out
    assert "[2] B" in out and "https://b.example" in out


def test_interactive_follow_ups_and_commands(monkeypatch, capsys):
    import builtins

    import search_engine.cli as cli
    import search_engine.engine as engine_mod

    client = FakeClient(message(SEARCH_TURN), message([text_block("Yokohama is second.")]))
    monkeypatch.setattr(engine_mod.anthropic, "Anthropic", lambda: client)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    lines = iter(["population of tokyo", "/sources", "what about the second city?", "/new", "/quit"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(lines))

    assert main(["--no-color"]) == 0
    out = capsys.readouterr().out
    assert "All results seen" in out and "https://a.example" in out
    assert "Yokohama is second." in out
    assert "Started a new conversation." in out
    assert [m["role"] for m in client.calls[1]["messages"]] == ["user", "assistant", "user"]
