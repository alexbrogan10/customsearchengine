# customsearchengine

An LLM-powered search engine for the command line. Ask a question and Claude
searches the live web, reads the pages it needs, and streams back a direct
answer with numbered citations and a source list.

Example session (illustrative):

```
$ cse "what's new in python 3.14"
  searching: Python 3.14 release new features
  reading:   https://docs.python.org/3.14/whatsnew/3.14.html
Python 3.14 brings free-threaded builds to officially supported status,
template strings (t-strings) via PEP 750, and deferred evaluation of
annotations[1] ...

Sources
  [1] What's New In Python 3.14 (3 weeks ago)
      https://docs.python.org/3.14/whatsnew/3.14.html
```

## How it works

The engine sends your query to the [Claude API](https://platform.claude.com)
(`claude-opus-5-5` by default) with Anthropic's server-side **web search** and
**web fetch** tools enabled. Claude decides what to search for, runs the
searches, opens full pages when snippets aren't enough, and writes an answer
grounded in what it found. Citations come back from the API attached to each
passage, and the CLI turns them into `[n]` markers and a numbered source list.

- **Streaming**: search activity and the answer appear as they happen.
- **Follow-ups**: interactive mode keeps the conversation, so "what about in
  Europe?" works after a first question.
- **Long research**: if the server pauses a long research turn, the engine
  resumes it automatically.
- **Refusal fallback**: requests opt into the API's server-side `fallbacks:
  "default"`, so if the model declines, a fallback model answers instead.

## Setup

Requires Python 3.10+ and an Anthropic API key.

```bash
git clone https://github.com/alexbrogan10/customsearchengine.git
cd customsearchengine
python -m venv .venv && source .venv/bin/activate
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...   # https://platform.claude.com
```

Web search must be enabled for your organization in the Claude Console. Web
search is billed per search in addition to tokens.

## Usage

```bash
cse "how do heat pumps work in cold climates"        # one-shot search
cse                                                  # interactive mode
echo "latest rust release" | cse                     # query from stdin
python -m search_engine "..."                        # without installing the script
```

Interactive commands: `/new` (fresh conversation), `/sources` (every result
seen for the last query, cited or not), `/help`, `/quit`.

### Options

| Option | Description |
|---|---|
| `--site DOMAIN` | Only search this domain (repeatable) |
| `--exclude DOMAIN` | Never search this domain (repeatable) |
| `--country CC` | Localize results, e.g. `--country GB` |
| `--effort LEVEL` | `low`, `medium` (default), `high`, `xhigh`, `max`: higher digs deeper but costs more |
| `--max-searches N` | Cap web searches (and page reads) per query (default 5) |
| `--no-fetch` | Answer from search snippets only, without opening pages |
| `--model MODEL` | Claude model (default `claude-opus-5-5`, or set `CSE_MODEL`) |
| `--json` | Print a JSON result (answer, sources, searches, usage) for scripting |
| `-q`, `--quiet` | Hide search activity |
| `--stats` | Show search count and token usage after each answer |
| `--no-color` | Plain output (also respects `NO_COLOR`) |

```bash
cse --site docs.python.org "asyncio TaskGroup vs gather"
cse --effort high --stats "compare the 2026 EV tax credits by state"
cse --json "population of tokyo" | jq -r '.sources[].url'
```

Exit codes: `0` success, `1` API or credentials error, `2` bad arguments,
`3` the model declined the query.

## Using it as a library

```python
from search_engine import SearchEngine

engine = SearchEngine(effort="high", allowed_domains=["arxiv.org"])
result = engine.search("recent results on sparse attention")
print(result.answer)
for n, source in enumerate(result.sources, 1):
    print(n, source.title, source.url)

follow_up = engine.search("which of those has code available?")
```

Pass `on_event=callback` to `search()` to receive streaming events:
`("search", query)`, `("fetch", url)`, `("text", chunk)`,
`("citations", [n, ...])`, `("tool_error", msg)` and `("notice", msg)`.

## Project layout

```
search_engine/
  engine.py   SearchEngine: API calls, streaming, citations, pause/resume
  cli.py      the `cse` command: argument parsing, rendering, interactive mode
tests/        offline tests using a fake streaming client
```

## Tests

```bash
pip install -e '.[dev]'
pytest
```

The tests replay canned API responses through a fake client, so they need no
API key or network access.
