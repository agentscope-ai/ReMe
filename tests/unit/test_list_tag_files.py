"""Tag browsing stays bounded, scoped, and derived from user-owned files."""

import pytest

from reme.components.file_store import LocalFileStore
from reme.components.application_context import ApplicationContext
from reme.components.tag_index import LocalTagIndex
from reme.config import resolve_app_config
from reme.schema import FileFrontMatter, FileNode
from reme.steps.index.list_tag_files import ListTagFilesStep


@pytest.fixture(name="store")
def file_store(tmp_path):
    """Keep indexed nodes and source files in one isolated workspace."""
    result = LocalFileStore(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        embedding_store="",
        tag_index="",
    )
    result.tag_index = LocalTagIndex()
    return result


@pytest.mark.asyncio
async def test_tag_browse_scopes_before_pagination_and_ignores_deleted_files(tmp_path, store):
    """Filter directories and permissions before counting or slicing results."""
    nodes = []
    for path, tags in [
        ("daily/a.md", ["Python"]),
        ("daily/b.md", ["ReMe"]),
        ("daily2/c.md", ["Python"]),
        ("digest/d.md", ["Python"]),
        ("daily/deleted.md", ["Python"]),
    ]:
        nodes.append(FileNode(path=path, st_mtime=1, front_matter=FileFrontMatter(memory_tags=tags)))
        if "deleted" not in path:
            target = tmp_path / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("source", encoding="utf-8")
    await store.tag_index.rebuild(nodes)
    step = ListTagFilesStep(file_store=store)
    response = await step(tags=["PYTHON", "ReMe"], path="daily", page_size=1, page=2)
    assert response.metadata["tag_filter"]["tags"] == ["python", "reme"]
    assert response.answer == {
        "total_files": 2,
        "total_pages": 2,
        "page": 2,
        "range": (2, 2),
        "items": [{"path": "daily/b.md", "tags": ["reme"]}],
    }
    response = await step(tags=["python"], _allowed_paths=["digest"], page_size=1)
    assert response.answer["items"] == [{"path": "digest/d.md", "tags": ["python"]}]
    assert (await step(tags=["python"], page=20)).answer["range"] == (0, 0)
    assert (tmp_path / "daily/a.md").read_text(encoding="utf-8") == "source"


@pytest.mark.asyncio
async def test_tag_browse_fails_closed_for_invalid_paths_or_unhealthy_index(store):
    """Invalid requests and unavailable derived state must never broaden results."""
    step = ListTagFilesStep(file_store=store)
    for kwargs in [{"path": "../escape"}, {"tags": []}, {"page": False}, {"page_size": 1001}]:
        with pytest.raises(ValueError):
            await step(**{"tags": ["python"], **kwargs})
    store.tag_index.set_healthy(False)
    with pytest.raises(RuntimeError, match="unavailable"):
        await step(tags=["python"])


def test_tag_browse_is_registered_and_exposed_by_default():
    """Keep discovery and the public configuration aligned."""
    config = resolve_app_config(config="default", log_config=False)
    assert config["jobs"]["list_tag_files"]["steps"] == [{"backend": "list_tag_files_step"}]
    assert config["jobs"]["list_tag_files"]["parameters"]["required"] == ["tags"]


def test_proactive_workspace_uses_local_indexes_without_model_requests():
    """The desktop's named config must enable the actual tag/search dependency path."""
    config = resolve_app_config(config="proactive", log_config=False)
    assert config["components"]["file_store"]["default"]["tag_index"] == "default"
    assert config["components"]["tag_index"]["default"]["tag_key"] == "memory_tags"
    assert config["jobs"]["list_tag_files"]["steps"] == [{"backend": "list_tag_files_step"}]
    assert config["jobs"]["search"]["steps"][0]["vector_weight"] == 0
    assert config["jobs"]["index_update_loop"]["watch_dirs"] == ["daily_dir", "digest_dir"]
