"""Command-line interface for the search engine.

    cse "how do solar panels work"     # one-shot search
    cse                                # interactive session with follow-ups
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from typing import Any, Optional

import anthropic

from . import __version__
from .engine import DEFAULT_MODEL, EFFORT_LEVELS, SearchEngine, SearchResult, format_markers

HELP_TEXT = """\
Type a query and press Enter. Follow-up queries remember the conversation.
  /new       start a fresh conversation
  /sources   list every search result seen for the last query
  /help      show this help
  /quit      exit (or Ctrl-D)"""


class Style:
    """ANSI styling, disabled when output isn't a terminal or NO_COLOR is set."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)

    def red(self, text: str) -> str:
        return self._wrap("31", text)


class LiveRenderer:
    """Prints search activity and the answer as they stream in."""

    def __init__(self, style: Style, show_activity: bool = True) -> None:
        self.style = style
        self.show_activity = show_activity
        self._mid_line = False

    def _status(self, text: str) -> None:
        if self._mid_line:
            print()
            self._mid_line = False
        print(text, file=sys.stdout, flush=True)

    def __call__(self, kind: str, payload: Any) -> None:
        if kind == "text":
            print(payload, end="", flush=True)
            self._mid_line = not payload.endswith("\n")
        elif kind == "citations":
            print(self.style.cyan(format_markers(payload)), end="", flush=True)
            self._mid_line = True
        elif kind == "search" and self.show_activity:
            self._status(self.style.dim(f"  searching: {payload}"))
        elif kind == "fetch" and self.show_activity:
            self._status(self.style.dim(f"  reading:   {payload}"))
        elif kind == "tool_error" and self.show_activity:
            self._status(self.style.yellow(f"  {payload}"))
        elif kind == "notice":
            self._status(self.style.yellow(f"note: {payload}"))

    def finish(self) -> None:
        if self._mid_line:
            print()
            self._mid_line = False


def print_sources(result: SearchResult, style: Style, all_results: bool = False) -> None:
    sources = result.results_seen if all_results else result.sources
    if not sources:
        return
    heading = "All results seen" if all_results else "Sources"
    print("\n" + style.bold(heading))
    seen: set[str] = set()
    n = 0
    for source in sources:
        if source.url in seen:
            continue
        seen.add(source.url)
        n += 1
        age = f" ({source.page_age})" if source.page_age else ""
        print(f"  {style.cyan(f'[{n}]')} {source.title}{style.dim(age)}")
        print(f"      {style.dim(source.url)}")


def print_footer(result: SearchResult, style: Style) -> None:
    parts = [f"{len(result.searches)} search{'es' if len(result.searches) != 1 else ''}"]
    if result.fetched:
        parts.append(f"{len(result.fetched)} page{'s' if len(result.fetched) != 1 else ''} read")
    parts.append(f"{result.input_tokens:,} in / {result.output_tokens:,} out tokens")
    if result.model:
        parts.append(result.model)
    print(style.dim("\n" + " · ".join(parts)))


class MissingCredentials(Exception):
    def __init__(self) -> None:
        super().__init__(
            "No Anthropic credentials found. Set ANTHROPIC_API_KEY "
            "(get a key at https://platform.claude.com) or run `ant auth login`."
        )


def run_query(engine: SearchEngine, query: str, args: argparse.Namespace, style: Style) -> Optional[SearchResult]:
    try:
        return _run_query(engine, query, args, style)
    except TypeError as e:
        # The SDK raises TypeError, not an APIError, when it finds no credentials at all.
        if "authentication method" in str(e):
            raise MissingCredentials() from None
        raise


def _run_query(engine: SearchEngine, query: str, args: argparse.Namespace, style: Style) -> Optional[SearchResult]:
    if args.json:
        result = engine.search(query)
        print(json.dumps(result.to_dict(), indent=2))
        return result

    renderer = LiveRenderer(style, show_activity=not args.quiet)
    try:
        result = engine.search(query, on_event=renderer)
    finally:
        renderer.finish()

    if result.refused:
        print(style.red("The model declined to answer this query."))
        return result
    print_sources(result, style)
    if args.stats:
        print_footer(result, style)
    return result


def interactive(engine: SearchEngine, args: argparse.Namespace, style: Style) -> int:
    print(style.bold(f"Custom Search Engine {__version__}") + style.dim(f"  ({engine.model})"))
    print(style.dim(HELP_TEXT) + "\n")
    last: Optional[SearchResult] = None
    while True:
        try:
            query = input(style.bold("search> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not query:
            continue
        if query in ("/quit", "/exit", "/q"):
            return 0
        if query == "/help":
            print(style.dim(HELP_TEXT))
            continue
        if query == "/new":
            engine.reset()
            last = None
            print(style.dim("Started a new conversation."))
            continue
        if query == "/sources":
            if last:
                print_sources(last, style, all_results=True)
            else:
                print(style.dim("No search yet."))
            continue
        if query.startswith("/"):
            print(style.yellow(f"Unknown command {query}. Type /help."))
            continue

        try:
            last = run_query(engine, query, args, style) or last
        except KeyboardInterrupt:
            print(style.yellow("\nSearch cancelled."))
        except anthropic.APIError as e:
            print(style.red(describe_api_error(e)))
        except MissingCredentials as e:
            print(style.red(str(e)))
            return 1
        print()


def describe_api_error(e: anthropic.APIError) -> str:
    if isinstance(e, anthropic.AuthenticationError):
        return "Authentication failed: set ANTHROPIC_API_KEY to a valid key (or run `ant auth login`)."
    if isinstance(e, anthropic.PermissionDeniedError):
        return "This API key isn't allowed to use this model or tool (web search may need enabling in the Console)."
    if isinstance(e, anthropic.NotFoundError):
        return f"Not found: {e.message} (check the --model name)."
    if isinstance(e, anthropic.RateLimitError):
        retry_after = e.response.headers.get("retry-after")
        return "Rate limited" + (f"; retry in {retry_after}s." if retry_after else "; try again shortly.")
    if isinstance(e, anthropic.BadRequestError):
        return f"Bad request: {e.message}"
    if isinstance(e, anthropic.APIStatusError):
        return f"API error {e.status_code}: {e.message}"
    if isinstance(e, anthropic.APIConnectionError):
        return "Couldn't reach the Anthropic API. Check your network connection."
    return f"API error: {e}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cse",
        description="An LLM-powered search engine: Claude searches the web and answers with citations.",
        epilog=textwrap.dedent("""\
            examples:
              cse "what's new in python 3.14"
              cse --site docs.python.org "how do I use asyncio.TaskGroup"
              cse --effort high "compare heat pumps and gas furnaces for cold climates"
              cse --json "population of tokyo" | jq .sources
              cse                       # interactive mode with follow-up questions"""),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", nargs="*", help="what to search for (omit for interactive mode)")
    parser.add_argument("--model", default=os.environ.get("CSE_MODEL", DEFAULT_MODEL),
                        help=f"Claude model to use (default: {DEFAULT_MODEL}, or $CSE_MODEL)")
    parser.add_argument("--effort", choices=EFFORT_LEVELS, default="medium",
                        help="how hard Claude works on the answer (default: medium)")
    parser.add_argument("--max-searches", type=int, default=5, metavar="N",
                        help="cap on web searches (and page reads) per query (default: 5)")
    domains = parser.add_mutually_exclusive_group()
    domains.add_argument("--site", action="append", metavar="DOMAIN", dest="allowed_domains",
                         help="only search this domain (repeatable)")
    domains.add_argument("--exclude", action="append", metavar="DOMAIN", dest="blocked_domains",
                         help="never search this domain (repeatable)")
    parser.add_argument("--country", metavar="CC", help="localize results to a two-letter country code, e.g. US, GB")
    parser.add_argument("--no-fetch", action="store_true", help="don't let Claude open full pages, only search snippets")
    parser.add_argument("--json", action="store_true", help="print the result as JSON instead of streaming text")
    parser.add_argument("-q", "--quiet", action="store_true", help="hide search activity, show only the answer")
    parser.add_argument("--stats", action="store_true", help="show search count and token usage after each answer")
    parser.add_argument("--no-color", action="store_true", help="disable colored output")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    style = Style(sys.stdout.isatty() and not args.no_color and "NO_COLOR" not in os.environ)

    if args.max_searches < 1:
        print("--max-searches must be at least 1", file=sys.stderr)
        return 2

    engine = SearchEngine(
        model=args.model,
        effort=args.effort,
        max_searches=args.max_searches,
        allow_fetch=not args.no_fetch,
        allowed_domains=args.allowed_domains,
        blocked_domains=args.blocked_domains,
        country=args.country,
    )

    query = " ".join(args.query).strip()
    if not query:
        if not sys.stdin.isatty():
            query = sys.stdin.read().strip()
        if not query:
            return interactive(engine, args, style)

    try:
        result = run_query(engine, query, args, style)
    except KeyboardInterrupt:
        return 130
    except anthropic.APIError as e:
        print(describe_api_error(e), file=sys.stderr)
        return 1
    except MissingCredentials as e:
        print(e, file=sys.stderr)
        return 1
    return 3 if result and result.refused else 0


if __name__ == "__main__":
    sys.exit(main())
