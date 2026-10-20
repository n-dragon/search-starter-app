"""The MCP navigation tools keep the shape the agent loop depends on.

The round-trip test covers storing and finding a document; this one covers the seam the
MCP server owns on top of that: that `open` and `read` expose the *distinct* contracts the
agentic search loop is built around, and that `open` actually resolves a chunk id to a
position before it windows around it.

- `open(chunk_id, window)`   -- "I have a chunk from search, show me context around it."
  The caller passes only the opaque chunk `id`; the server resolves its position via
  `NavigableIndex.get_chunk` and pulls in neighbours. If this silently reverts to taking
  raw offsets, the tool becomes indistinguishable from `read` and the distinction the docs
  teach stops being true.
- `read(source_id, start_offset, end_offset)` -- "I know the exact range, give me those
  chunks." Offset-addressed, no expansion.

No backend and no API key are needed: `open`'s logic is exercised against a fake store, and
the schema is read off the registered tools. `mcp_server` fails fast at import without
`MISTRAL_API_KEY` and loads `.env` with `override=True`, so the fixture neutralises the
dotenv load and sets a placeholder key -- it is never used to make a call.
"""

import asyncio
import importlib
from unittest.mock import AsyncMock

import pytest
from fastmcp import Client
from mistralai.search.toolkit.document import ChunkType
from mistralai.search.toolkit.search import NavigationDirection
from mistralai.search.toolkit.search.models import SearchResult, SearchResultChunk


@pytest.fixture
def mcp_server(monkeypatch: pytest.MonkeyPatch):
    """Import `entrypoints.mcp_server` offline, with a placeholder key and dotenv disabled.

    The module reads `.env` with `override=True` at import; the generated `.env` ships an
    empty `MISTRAL_API_KEY`, which would clobber a value set here. Neutralising `load_dotenv`
    before (re)import lets the placeholder stand so the import-time fail-fast passes.
    """
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key-not-used")

    import entrypoints.mcp_server as module

    return importlib.reload(module)


def _result(chunk_id: str, start: int, end: int, content: str) -> SearchResult:
    chunk = SearchResultChunk(
        id=chunk_id,
        source_id="doc-1",
        locator=f"chunk_idx:{start}",
        content=content,
        start_offset=start,
        end_offset=end,
        chunk_type=ChunkType.CONTENT,
    )
    return SearchResult(score=0.0, chunk=chunk)


class _FakeStore:
    """Minimal `NavigableIndex` stand-in: knows one chunk and one neighbour in each direction."""

    async def get_chunk(self, chunk_id: str, **_: object) -> SearchResult | None:
        return _result("c2", 10, 20, "anchor") if chunk_id == "c2" else None

    async def navigate(
        self, source_id: str, start: int, end: int, direction: NavigationDirection, *, top_k: int = 1, **_: object
    ) -> list[SearchResult]:
        if direction == NavigationDirection.PREVIOUS:
            return [_result("c1", 0, 10, "before")]
        return [_result("c3", 20, 30, "after")]


def _open_fn(mcp_server):
    # `open` shadows the builtin; @mcp.tool() wraps it, so reach the coroutine through `.fn`.
    return getattr(mcp_server.open, "fn", mcp_server.open)


def _search_fn(mcp_server):
    return getattr(mcp_server.search, "fn", mcp_server.search)


class _EngineResult:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results


class _RecordingEngine:
    """Captures the kwargs `search` forwards, returns one canned hit."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def search(self, **kwargs: object) -> _EngineResult:
        self.calls.append(kwargs)
        return _EngineResult([_result("c1", 0, 10, "hit")])


def test_search_exposes_exclude_ids(mcp_server) -> None:
    search_tool = asyncio.run(mcp_server.mcp.get_tool("search"))
    props = search_tool.parameters["properties"]

    assert "exclude_ids" in props
    # Only `query` is mandatory; exclude_ids stays optional.
    assert search_tool.parameters.get("required") == ["query"]


def test_search_forwards_exclude_ids_as_a_set(mcp_server, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _RecordingEngine()
    monkeypatch.setattr(mcp_server, "_query_engine", engine)

    asyncio.run(_search_fn(mcp_server)("q", exclude_ids=["a", "b", "a"]))

    assert engine.calls[0]["exclude_ids"] == {"a", "b"}


def test_search_defaults_exclude_ids_to_none(mcp_server, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _RecordingEngine()
    monkeypatch.setattr(mcp_server, "_query_engine", engine)

    asyncio.run(_search_fn(mcp_server)("q"))

    # Empty/absent stays None so the backend skips the filter entirely.
    assert engine.calls[0]["exclude_ids"] is None


def test_open_takes_a_chunk_id_and_read_stays_offset_addressed(mcp_server) -> None:
    open_tool = asyncio.run(mcp_server.mcp.get_tool("open"))
    read_tool = asyncio.run(mcp_server.mcp.get_tool("read"))

    assert set(open_tool.parameters["properties"]) == {"chunk_id", "window"}
    assert open_tool.parameters.get("required") == ["chunk_id"]

    read_props = read_tool.parameters["properties"]
    assert "start_offset" in read_props and "end_offset" in read_props
    assert "chunk_id" not in read_props


def test_open_resolves_the_chunk_and_windows_around_it(mcp_server, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_server, "_navigable_store", _FakeStore())

    out = asyncio.run(_open_fn(mcp_server)("c2", window=1))

    # Previous neighbour, the resolved anchor, then the next neighbour -- in reading order.
    assert [r["id"] for r in out] == ["c1", "c2", "c3"]
    assert [r["content"] for r in out] == ["before", "anchor", "after"]


def test_open_raises_when_the_chunk_id_is_unknown(mcp_server, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastmcp.exceptions import ToolError

    monkeypatch.setattr(mcp_server, "_navigable_store", _FakeStore())

    with pytest.raises(ToolError):
        asyncio.run(_open_fn(mcp_server)("does-not-exist"))


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("search", {"query": "q", "top_k": 0}),
        ("search", {"query": "q", "top_k": -1}),
        ("open", {"chunk_id": "c2", "window": -1}),
        ("navigate", {"source_id": "doc-1", "start_offset": -1, "end_offset": 20, "direction": "next"}),
        ("navigate", {"source_id": "doc-1", "start_offset": 0, "end_offset": -1, "direction": "next"}),
        ("navigate", {"source_id": "doc-1", "start_offset": 0, "end_offset": 20, "direction": "sideways"}),
        ("navigate", {"source_id": "doc-1", "start_offset": 0, "end_offset": 20, "direction": "next", "top_k": 0}),
        ("read", {"source_id": "doc-1", "start_offset": -1}),
        ("read", {"source_id": "doc-1", "end_offset": -1}),
        ("read", {"source_id": "doc-1", "top_k": 0}),
        ("grep", {"source_id": "doc-1", "pattern": "word", "mode": "regex"}),
        ("grep", {"source_id": "doc-1", "pattern": "word", "top_k": 0}),
    ],
)
def test_invalid_tool_arguments_do_not_reach_the_backend(
    mcp_server, monkeypatch: pytest.MonkeyPatch, tool: str, arguments: dict
) -> None:
    engine = AsyncMock()
    store = AsyncMock()
    monkeypatch.setattr(mcp_server, "_query_engine", engine)
    monkeypatch.setattr(mcp_server, "_navigable_store", store)

    async def call_tool():
        async with Client(mcp_server.mcp) as client:
            return await client.call_tool(tool, arguments, raise_on_error=False)

    result = asyncio.run(call_tool())

    assert result.is_error
    engine.search.assert_not_awaited()
    for method in ("get_chunk", "navigate", "read", "grep"):
        getattr(store, method).assert_not_awaited()


def test_open_with_zero_window_returns_only_the_anchor(mcp_server, monkeypatch: pytest.MonkeyPatch) -> None:
    store = AsyncMock()
    store.get_chunk.return_value = _result("c2", 10, 20, "anchor")
    store.navigate.side_effect = AssertionError("a zero window must not request neighbours")
    monkeypatch.setattr(mcp_server, "_navigable_store", store)

    async def call_tool():
        async with Client(mcp_server.mcp) as client:
            return await client.call_tool("open", {"chunk_id": "c2", "window": 0})

    result = asyncio.run(call_tool())

    assert [chunk["id"] for chunk in result.data] == ["c2"]
    store.get_chunk.assert_awaited_once_with("c2")
    store.navigate.assert_not_awaited()


def test_tool_schemas_advertise_limits_and_supported_options(mcp_server) -> None:
    navigate_tool = asyncio.run(mcp_server.mcp.get_tool("navigate"))
    grep_tool = asyncio.run(mcp_server.mcp.get_tool("grep"))
    open_tool = asyncio.run(mcp_server.mcp.get_tool("open"))

    assert navigate_tool.parameters["$defs"]["NavigationDirection"]["enum"] == ["next", "previous"]
    assert grep_tool.parameters["$defs"]["GrepMode"]["enum"] == ["phrase", "term"]
    assert navigate_tool.parameters["properties"]["top_k"]["minimum"] == 1
    assert open_tool.parameters["properties"]["window"]["minimum"] == 0


def test_read_accepts_unbounded_offsets(mcp_server, monkeypatch: pytest.MonkeyPatch) -> None:
    store = AsyncMock()
    store.read.return_value = [_result("c1", 0, 10, "hit")]
    monkeypatch.setattr(mcp_server, "_navigable_store", store)

    async def call_tool():
        async with Client(mcp_server.mcp) as client:
            return await client.call_tool("read", {"source_id": "doc-1", "start_offset": None, "end_offset": None})

    result = asyncio.run(call_tool())

    assert [chunk["id"] for chunk in result.data] == ["c1"]
    store.read.assert_awaited_once_with("doc-1", None, None, top_k=20)
