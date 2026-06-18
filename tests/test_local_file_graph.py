import pytest

from reme4.components.file_graph.local_file_graph import LocalFileGraph
from reme4.enumeration import LinkScopeEnum
from reme4.schema import FileLink, FileNode


def _node(path: str, *targets: str) -> FileNode:
    return FileNode(
        path=path,
        st_mtime=1.0,
        links=[FileLink(source_path=path, target_path=target) for target in targets],
    )


@pytest.mark.asyncio
async def test_local_file_graph_accepts_string_scope_for_outlinks():
    graph = LocalFileGraph(name="test_scope_outlinks")
    await graph.upsert_nodes([_node("A.md", "B.md", "Missing.md"), _node("B.md")])

    assert [link.target_path for link in await graph.get_outlinks("A.md", "real")] == ["B.md"]
    assert [link.target_path for link in await graph.get_outlinks("A.md", "virtual")] == ["Missing.md"]
    assert [link.target_path for link in await graph.get_outlinks("A.md", "all")] == ["B.md", "Missing.md"]


@pytest.mark.asyncio
async def test_local_file_graph_accepts_string_scope_and_orders_inlinks():
    graph = LocalFileGraph(name="test_scope_inlinks")
    await graph.upsert_nodes([
        _node("B.md", "Target.md"),
        _node("A.md", "Target.md"),
        _node("C.md", "Missing.md"),
        _node("Target.md"),
    ])

    assert [link.source_path for link in await graph.get_inlinks("Target.md", "real")] == ["A.md", "B.md"]
    assert [link.source_path for link in await graph.get_inlinks("Missing.md", "virtual")] == ["C.md"]


@pytest.mark.asyncio
async def test_local_file_graph_scope_enum_still_filters_virtual_inlinks():
    graph = LocalFileGraph(name="test_scope_enum_inlinks")
    await graph.upsert_nodes([_node("A.md", "Missing.md")])

    assert await graph.get_inlinks("Missing.md", LinkScopeEnum.REAL) == []
    assert [link.source_path for link in await graph.get_inlinks("Missing.md", LinkScopeEnum.VIRTUAL)] == ["A.md"]
