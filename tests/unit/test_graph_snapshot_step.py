"""Tests for the frontend-ready category-rooted digest graph snapshot."""

import asyncio

from reme.components.file_store import LocalFileStore
from reme.schema import FileFrontMatter, FileLink, FileNode, GraphSnapshot
from reme.steps.index.graph_snapshot import GraphSnapshotStep


def _node(
    path: str,
    links: list[tuple[str, str | None]] | None = None,
    *,
    name: str = "",
    description: str = "",
) -> FileNode:
    return FileNode(
        path=path,
        st_mtime=1.0,
        links=[
            FileLink(source_path=path, target_path=target, target_anchor=anchor) for target, anchor in (links or [])
        ],
        front_matter=FileFrontMatter(name=name, description=description),
    )


def test_graph_snapshot_returns_category_roots_digest_links_and_daily_leaves(tmp_path, monkeypatch):
    """The snapshot is rooted at digest categories and stops at indexed daily notes."""
    monkeypatch.chdir(tmp_path)

    async def run():
        store = LocalFileStore(name="snapshot", embedding_store="")
        await store.start()
        try:
            await store.file_graph.upsert_nodes(
                [
                    _node(
                        "digest/wiki/alpha.md",
                        [
                            ("digest/personal/beta.md", "intro"),
                            ("daily/2026-08-05/event.md", None),
                            ("daily/2026-08-05/missing.md", None),
                            ("resource/ignored.md", None),
                        ],
                        name="Alpha",
                        description="Root note",
                    ),
                    _node("digest/personal/beta.md", [("digest/wiki/alpha.md", None)], name="Beta"),
                    _node("digest/procedure/how-to.md", name="How to"),
                    _node(
                        "daily/2026-08-05/event.md",
                        [("digest/procedure/how-to.md", None), ("daily/2026-08-05/hidden.md", None)],
                        name="Event",
                    ),
                    _node("daily/2026-08-05/isolated.md", name="Isolated daily"),
                    _node("resource/ignored.md", name="Ignored resource"),
                ],
            )

            response = await GraphSnapshotStep(file_store=store)()
            graph = GraphSnapshot.model_validate(response.answer)

            assert response.success is True
            assert response.metadata == {}
            assert [node.id for node in graph.nodes[:3]] == [
                "virtual:wiki",
                "virtual:personal",
                "virtual:procedure",
            ]
            assert [node.path for node in graph.nodes[3:]] == [
                "daily/2026-08-05/event.md",
                "digest/personal/beta.md",
                "digest/procedure/how-to.md",
                "digest/wiki/alpha.md",
            ]
            nodes = {node.id: node for node in graph.nodes}
            assert nodes["virtual:wiki"].path == "digest/wiki"
            assert nodes["virtual:procedure"].path == "digest/procedure"
            assert nodes["virtual:wiki"].virtual is True
            assert nodes["digest/wiki/alpha.md"].name == "Alpha"
            assert nodes["digest/wiki/alpha.md"].description == "Root note"
            assert nodes["digest/wiki/alpha.md"].virtual is False
            assert {(edge.source, edge.target, edge.target_anchor) for edge in graph.edges} == {
                ("virtual:wiki", "digest/wiki/alpha.md", None),
                ("virtual:personal", "digest/personal/beta.md", None),
                ("virtual:procedure", "digest/procedure/how-to.md", None),
                ("digest/wiki/alpha.md", "digest/personal/beta.md", "intro"),
                ("digest/wiki/alpha.md", "daily/2026-08-05/event.md", None),
                ("digest/personal/beta.md", "digest/wiki/alpha.md", None),
            }
        finally:
            await store.close()

    asyncio.run(run())
