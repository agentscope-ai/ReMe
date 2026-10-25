"""Tests for ``reme.utils.link_expansion``.

Two pure helpers:

* ``expand_links(file_store, paths, max_per_direction)`` — fetch
  outlinks / inlinks for each path with neighbor meta attached.
* ``render_expansion_lines(expansion)`` — turn one path's expansion
  dict into the indented ``→`` / ``←`` block used by SearchStep
  answers.
"""

# pylint: disable=protected-access

from contextlib import chdir as temp_chdir
import tempfile
import warnings
from pathlib import Path

import pytest

from reme.components.file_store import LocalFileStore
from reme.schema import FileFrontMatter, FileNode
from reme.utils.link_expansion import expand_links, render_expansion_lines
from reme.utils.wikilink_handler import WikilinkHandler

warnings.filterwarnings("ignore", category=DeprecationWarning, module="jieba")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")


async def _store_with(files: dict[str, dict]) -> LocalFileStore:
    """LocalFileStore seeded with files + parsed wikilinks + optional frontmatter meta.

    Each value: ``{"body": str, "name": str?, "description": str?}``. The
    body is written to disk and wikilinks are extracted; ``name`` /
    ``description`` populate FileFrontMatter so neighbor meta lookups
    have something to surface.
    """
    store = LocalFileStore(name="t", embedding_store="")
    await store.start()
    nodes: list[FileNode] = []
    root = Path.cwd()
    for rel, spec in files.items():
        body = spec["body"]
        abs_path = root / rel
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(body, encoding="utf-8")
        fm = FileFrontMatter(
            name=spec.get("name", ""),
            description=spec.get("description", ""),
        )
        nodes.append(
            FileNode(
                path=rel,
                st_mtime=abs_path.stat().st_mtime,
                links=WikilinkHandler.extract_links(body, rel),
                front_matter=fm,
            ),
        )
    if nodes:
        await store.file_graph.upsert_nodes(nodes)
    return store


# -- expand_links -------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_links_empty_paths_short_circuits():
    """No paths ⇒ empty dict, no file_store calls needed."""

    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        store = LocalFileStore(name="t", embedding_store="")
        await store.start()
        result = await expand_links(store, [])
        assert result == {}
        await store.close()


@pytest.mark.asyncio
async def test_expand_links_returns_outlinks_and_inlinks_with_meta():
    """A.md links to B.md ⇒ A has B as outlink, B has A as inlink, meta surfaced."""

    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        store = await _store_with(
            {
                "A.md": {"body": "See [[B.md]] for details.", "name": "A Doc", "description": "alpha"},
                "B.md": {"body": "End node.", "name": "B Doc", "description": "beta"},
            },
        )
        result = await expand_links(store, ["A.md", "B.md"])

        assert set(result.keys()) == {"A.md", "B.md"}

        a_out = result["A.md"]["outlinks"]
        assert len(a_out) == 1
        assert a_out[0]["path"] == "B.md"
        assert a_out[0]["meta"] == {"name": "B Doc", "description": "beta"}
        assert a_out[0]["anchors"] == []
        assert result["A.md"]["inlinks"] == []

        b_in = result["B.md"]["inlinks"]
        assert len(b_in) == 1
        assert b_in[0]["path"] == "A.md"
        assert b_in[0]["meta"] == {"name": "A Doc", "description": "alpha"}
        assert result["B.md"]["outlinks"] == []

        await store.close()


@pytest.mark.asyncio
async def test_expand_links_max_per_direction_caps_neighbors():
    """max_per_direction=2 ⇒ only first two distinct neighbors per direction kept."""

    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        store = await _store_with(
            {
                "hub.md": {
                    "body": "[[a.md]] [[b.md]] [[c.md]] [[d.md]]",
                },
                "a.md": {"body": "a"},
                "b.md": {"body": "b"},
                "c.md": {"body": "c"},
                "d.md": {"body": "d"},
            },
        )
        result = await expand_links(store, ["hub.md"], max_per_direction=2)
        out = result["hub.md"]["outlinks"]
        assert len(out) == 2
        assert [n["path"] for n in out] == ["a.md", "b.md"]
        await store.close()


@pytest.mark.asyncio
async def test_expand_links_node_without_meta_returns_empty_meta_dict():
    """Neighbor with no frontmatter name/description ⇒ meta = {}."""

    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        store = await _store_with(
            {
                "src.md": {"body": "[[dst.md]]"},
                "dst.md": {"body": "no meta"},
            },
        )
        result = await expand_links(store, ["src.md"])
        assert result["src.md"]["outlinks"][0]["meta"] == {}
        await store.close()


# -- render_expansion_lines ---------------------------------------------------


def test_render_expansion_lines_empty_input_yields_empty_list():
    """Both directions empty ⇒ no lines."""
    assert not render_expansion_lines({})
    assert not render_expansion_lines({"outlinks": [], "inlinks": []})


def test_render_expansion_lines_outlinks_only():
    """Single outlink with metadata renders without a redundant edge line."""
    expansion = {
        "outlinks": [
            {
                "path": "B.md",
                "meta": {"name": "B", "description": "beta"},
                "anchors": [],
            },
        ],
        "inlinks": [],
    }
    lines = render_expansion_lines(expansion)
    assert lines == [
        "  outlinks (1):",
        '    → B.md  name="B"  description="beta"',
    ]


def test_render_expansion_lines_inlinks_only_with_anchor():
    """Inlink anchor renders via descriptor."""
    expansion = {
        "outlinks": [],
        "inlinks": [
            {
                "path": "src.md",
                "meta": {},
                "anchors": ["intro"],
            },
        ],
    }
    lines = render_expansion_lines(expansion)
    assert lines == [
        "  inlinks (1):",
        "    ← src.md  (no meta)",
        "        via anchor=#intro",
    ]


def test_render_expansion_lines_both_directions_in_order():
    """outlinks block precedes inlinks block."""
    expansion = {
        "outlinks": [
            {"path": "out.md", "meta": {"name": "Out"}, "anchors": []},
        ],
        "inlinks": [
            {"path": "in.md", "meta": {"description": "incoming"}, "anchors": []},
        ],
    }
    lines = render_expansion_lines(expansion)
    assert lines[0] == "  outlinks (1):"
    assert lines[2] == "  inlinks (1):"
    assert lines[1].lstrip().startswith("→")
    assert lines[3].lstrip().startswith("←")
