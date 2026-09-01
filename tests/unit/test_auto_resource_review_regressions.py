"""Regression tests for the safety findings from the Auto Resource PR review."""

import frontmatter
import pytest

from .auto_resource_test_support import (
    FakeVisionModel,
    StructuredVisionModel,
    caption_json,
    image_bytes,
    write_binary,
    write_note,
)

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("routed", [False, True], ids=["image", "unified-router"])
async def test_image_rejects_paths_outside_the_resource_tree(routed, auto_resource_env, tmp_path):
    """Traversal and external paths fail before image reads."""
    env = auto_resource_env
    outside = write_binary(tmp_path / "outside.png", image_bytes())
    nonresource = env.write_binary("private.png", image_bytes())

    model = FakeVisionModel(caption_json("unsafe", "Unsafe", "Must not be read."))
    response = await env.run(
        env.processor(model, routed=routed),
        [
            {"change": "added", "path": "resource/2026-01-01/../../../outside.png"},
            {"change": "added", "path": str(outside)},
            {"change": "added", "path": str(nonresource)},
        ],
    )

    results = response.metadata["results"]
    assert response.success is False
    assert len(results) == 3
    assert all(item["metadata"]["action"] == "failed" for item in results)
    assert all(item["metadata"]["modified"] is False for item in results)
    assert "cannot contain '.' or '..'" in results[0]["metadata"]["error"]
    assert "must stay inside the workspace" in results[1]["metadata"]["error"]
    assert "configured resource directory" in results[2]["metadata"]["error"]
    assert not model.calls
    assert outside.read_bytes() == image_bytes()


@pytest.mark.parametrize("routed", [False, True], ids=["image", "unified-router"])
async def test_image_rejects_resource_symlink_outside_workspace(routed, auto_resource_env, tmp_path):
    """An escaping resource symlink fails without weakening other containment tests."""
    env = auto_resource_env
    outside = write_binary(tmp_path / "outside.png", image_bytes())
    external_link = env.workspace / "resource/2026-01-01/external.png"
    external_link.parent.mkdir(parents=True, exist_ok=True)
    try:
        external_link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    model = FakeVisionModel(caption_json("unsafe", "Unsafe", "Must not be read."))
    response = await env.run(
        env.processor(model, routed=routed),
        [{"change": "added", "path": str(external_link)}],
    )

    result = response.metadata["results"][0]
    assert response.success is False
    assert result["metadata"]["action"] == "failed"
    assert result["metadata"]["modified"] is False
    assert "must stay inside the workspace" in result["metadata"]["error"]
    assert not model.calls
    assert outside.read_bytes() == image_bytes()


@pytest.mark.parametrize("routed", [False, True], ids=["image", "unified-router"])
async def test_image_internal_symlink_keeps_logical_provenance(routed, auto_resource_env):
    """An internal symlink is read safely while ownership follows the watched alias."""
    env = auto_resource_env
    target = env.write_binary("resource/2026-01-01/original.png", image_bytes())
    link = target.with_name("alias.png")
    try:
        link.symlink_to(target.name)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    model = FakeVisionModel(caption_json("linked-image", "Linked", "An internal linked image."))
    step = env.processor(model, routed=routed)
    response = await env.run(step, [{"change": "added", "path": str(link)}])

    note_path = env.workspace / "daily/2026-01-01/linked-image.md"
    post = frontmatter.loads(note_path.read_text(encoding="utf-8"))
    assert response.success is True
    assert post.metadata["source_resource"] == "[[resource/2026-01-01/alias.png]]"
    assert "![[resource/2026-01-01/alias.png]]" in post.content
    assert len(model.calls) == 1

    link.unlink()
    deleted = await env.run(step, [{"change": "deleted", "path": str(link)}])
    assert deleted.success is True
    assert deleted.metadata["results"][0]["metadata"]["action"] == "deleted"
    assert not note_path.exists()
    assert target.is_file()


@pytest.mark.parametrize("routed", [False, True], ids=["image", "unified-router"])
@pytest.mark.parametrize("existing_owner", ["different-source", "no-source"])
@pytest.mark.parametrize("change", ["modified", "deleted"])
async def test_image_preserves_unowned_same_stem_note(routed, existing_owner, change, auto_resource_env):
    """Upsert and delete never claim a same-stem note without exact ownership."""
    env = auto_resource_env
    source = env.workspace / "resource/2026-01-01/img.png"
    if change == "modified":
        write_binary(source, image_bytes())
    same_stem = env.workspace / "daily/2026-01-01/img.md"
    if existing_owner == "different-source":
        write_note(same_stem, "[[resource/2026-01-01/other.png]]", body="unrelated image note")
    else:
        same_stem.parent.mkdir(parents=True, exist_ok=True)
        same_stem.write_text(
            "---\nname: img\ndescription: user-owned note\n---\nuser-owned bytes\n",
            encoding="utf-8",
        )
    before = same_stem.read_bytes()

    model = FakeVisionModel(caption_json("generated-caption", "Generated", "Generated caption."))
    response = await env.run(env.processor(model, routed=routed), [{"change": change, "path": str(source)}])

    assert response.success is True
    assert same_stem.read_bytes() == before
    if change == "deleted":
        result = response.metadata["results"][0]["metadata"]
        assert result["action"] == "skipped"
        assert result["reason"] == "resource_note_not_found"
        assert result["modified"] is False
    else:
        generated = env.workspace / "daily/2026-01-01/generated-caption.md"
        post = frontmatter.loads(generated.read_text(encoding="utf-8"))
        assert post.metadata["source_resource"] == "[[resource/2026-01-01/img.png]]"
        assert "Generated caption." in post.content


@pytest.mark.parametrize("routed", [False, True], ids=["image", "unified-router"])
@pytest.mark.parametrize("plain_text", ["   ", "```json\n\n```"], ids=["whitespace", "empty-json-fence"])
async def test_blank_plain_caption_does_not_create_or_overwrite_note(routed, plain_text, auto_resource_env):
    """An empty structured result plus blank plain fallback leaves notes untouched."""
    env = auto_resource_env
    new_source = env.write_binary("resource/2026-01-01/blank-new.png", image_bytes())
    old_source = env.write_binary("resource/2026-01-01/blank-old.png", image_bytes())
    old_note = env.write_note(
        "daily/2026-01-01/preserved.md",
        "[[resource/2026-01-01/blank-old.png]]",
        body="caption that must survive",
    )
    before = old_note.read_bytes()
    model = StructuredVisionModel(content={}, plain_text=plain_text)
    step = env.processor(model, routed=routed)

    added = await env.run(step, [{"change": "added", "path": str(new_source)}])
    modified = await env.run(step, [{"change": "modified", "path": str(old_source)}])

    for response in (added, modified):
        result = response.metadata["results"][0]["metadata"]
        assert response.success is False
        assert result["action"] == "failed"
        assert result["modified"] is False
        assert "no usable caption" in result["error"]
    assert not (env.workspace / "daily/2026-01-01/blank-new.md").exists()
    assert old_note.read_bytes() == before
    assert len(model.structured_calls) == len(model.plain_calls) == 2
