"""Tests for ``reme4.utils.link_expansion``.

Two pure helpers:

* ``expand_links(file_store, paths, max_per_direction)`` — fetch
  outlinks / inlinks for each path with neighbor meta attached.
* ``render_expansion_lines(expansion)`` — turn one path's expansion
  dict into the indented ``→`` / ``←`` block used by SearchStep
  answers.
"""

# pylint: disable=protected-access

import asyncio
import os
import tempfile
import warnings
from pathlib import Path

from reme4.components.file_store import LocalFileStore
from reme4.schema import FileFrontMatter, FileNode
from reme4.utils.link_expansion import expand_links, render_expansion_lines
from reme4.utils.wikilink_handler import WikilinkHandler

warnings.filterwarnings("ignore", category=DeprecationWarning, module="jieba")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")


class temp_chdir:
    """Test helper: chdir to ``path`` on enter, restore previous cwd on exit."""

    def __init__(self, path):
        self.path = path
        self.old = None

    def __enter__(self):
        self.old = os.getcwd()
        os.chdir(self.path)
        return self

    def __exit__(self, *exc):
        os.chdir(self.old)


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


def test_expand_links_empty_paths_short_circuits():
    """No paths ⇒ empty dict, no file_store calls needed."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(name="t", embedding_store="")
            await store.start()
            result = await expand_links(store, [])
            assert result == {}
            await store.close()
        print("✓ test_expand_links_empty_paths_short_circuits passed")

    asyncio.run(run())


def test_expand_links_returns_outlinks_and_inlinks_with_meta():
    """A.md links to B.md ⇒ A has B as outlink, B has A as inlink, meta surfaced."""

    async def run():
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
            assert a_out[0]["edges"] == [{"predicate": None, "anchor": None}]
            assert result["A.md"]["inlinks"] == []

            b_in = result["B.md"]["inlinks"]
            assert len(b_in) == 1
            assert b_in[0]["path"] == "A.md"
            assert b_in[0]["meta"] == {"name": "A Doc", "description": "alpha"}
            assert result["B.md"]["outlinks"] == []

            await store.close()
        print("✓ test_expand_links_returns_outlinks_and_inlinks_with_meta passed")

    asyncio.run(run())


def test_expand_links_max_per_direction_caps_neighbors():
    """max_per_direction=2 ⇒ only first two distinct neighbors per direction kept."""

    async def run():
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
        print("✓ test_expand_links_max_per_direction_caps_neighbors passed")

    asyncio.run(run())


def test_expand_links_node_without_meta_returns_empty_meta_dict():
    """Neighbor with no frontmatter name/description ⇒ meta = {}."""

    async def run():
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
        print("✓ test_expand_links_node_without_meta_returns_empty_meta_dict passed")

    asyncio.run(run())


# -- render_expansion_lines ---------------------------------------------------


def test_render_expansion_lines_empty_input_yields_empty_list():
    """Both directions empty ⇒ no lines."""
    assert not render_expansion_lines({})
    assert not render_expansion_lines({"outlinks": [], "inlinks": []})
    print("✓ test_render_expansion_lines_empty_input_yields_empty_list passed")


def test_render_expansion_lines_outlinks_only():
    """Single outlink with meta + plain edge renders as 3 lines."""
    expansion = {
        "outlinks": [
            {
                "path": "B.md",
                "meta": {"name": "B", "description": "beta"},
                "edges": [{"predicate": None, "anchor": None}],
            },
        ],
        "inlinks": [],
    }
    lines = render_expansion_lines(expansion)
    assert lines == [
        "  outlinks (1):",
        '    → B.md  name="B"  description="beta"',
        "        via plain",
    ]
    print("✓ test_render_expansion_lines_outlinks_only passed")


def test_render_expansion_lines_inlinks_only_with_predicate_and_anchor():
    """Inlink edge with predicate + anchor renders via descriptor."""
    expansion = {
        "outlinks": [],
        "inlinks": [
            {
                "path": "src.md",
                "meta": {},
                "edges": [{"predicate": "references", "anchor": "intro"}],
            },
        ],
    }
    lines = render_expansion_lines(expansion)
    assert lines == [
        "  inlinks (1):",
        "    ← src.md  (no meta)",
        "        via predicate=references, anchor=#intro",
    ]
    print("✓ test_render_expansion_lines_inlinks_only_with_predicate_and_anchor passed")


def test_render_expansion_lines_both_directions_in_order():
    """outlinks block precedes inlinks block."""
    expansion = {
        "outlinks": [
            {"path": "out.md", "meta": {"name": "Out"}, "edges": [{"predicate": None, "anchor": None}]},
        ],
        "inlinks": [
            {"path": "in.md", "meta": {"description": "incoming"}, "edges": [{"predicate": None, "anchor": None}]},
        ],
    }
    lines = render_expansion_lines(expansion)
    assert lines[0] == "  outlinks (1):"
    assert lines[3] == "  inlinks (1):"
    assert lines[1].lstrip().startswith("→")
    assert lines[4].lstrip().startswith("←")
    print("✓ test_render_expansion_lines_both_directions_in_order passed")


if __name__ == "__main__":
    test_expand_links_empty_paths_short_circuits()
    test_expand_links_returns_outlinks_and_inlinks_with_meta()
    test_expand_links_max_per_direction_caps_neighbors()
    test_expand_links_node_without_meta_returns_empty_meta_dict()
    test_render_expansion_lines_empty_input_yields_empty_list()
    test_render_expansion_lines_outlinks_only()
    test_render_expansion_lines_inlinks_only_with_predicate_and_anchor()
    test_render_expansion_lines_both_directions_in_order()
