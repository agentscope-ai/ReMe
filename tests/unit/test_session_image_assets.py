"""Session attachments preserve bytes, identity, file boundaries and user links."""

# pylint: disable=missing-function-docstring

import asyncio
import base64
import importlib
from types import SimpleNamespace

from agentscope.message import Base64Source, DataBlock, Msg, TextBlock, URLSource
import frontmatter
import httpx
import pytest

from reme.components.runtime_context import RuntimeContext
from reme.steps.evolve._session_image_assets import merge_image_sources, save_session_images
from reme.steps.file_io.write import WriteStep


def _prepared(*sources, message_id="message-a"):
    original = Msg(
        id=message_id,
        name="Alice",
        role="user",
        content=[
            TextBlock(text="before"),
            *[
                DataBlock(
                    id="same-block-id",
                    source=(
                        source
                        if isinstance(source, URLSource)
                        else Base64Source(media_type="image/png", data=base64.b64encode(source).decode())
                    ),
                )
                for source in sources
            ],
            TextBlock(text="after"),
        ],
    )
    prepared = original.model_copy(deep=True)
    images = {}
    for index, block in enumerate(prepared.content):
        if isinstance(block, DataBlock):
            marker = f"__image_{index}__"
            images[marker] = block
            prepared.content[index] = TextBlock(text=marker)
    return original, [prepared], images


@pytest.mark.asyncio
async def test_bytes_positions_repeated_ids_and_reuse(tmp_path):
    original, messages, images = _prepared(b"first image", b"second image")
    snapshot = original.model_dump()
    sources = await save_session_images(tmp_path, "sessions", "chat", messages, images)
    assert len(sources) == 2
    paths = [tmp_path / source[2:-2] for source in sources]
    assert [path.read_bytes() for path in paths] == [b"first image", b"second image"]
    assert [path.name.rsplit("-", 1)[1] for path in paths] == ["1.png", "2.png"]
    assert messages[0].content[1].text == f"Image source: {sources[0]}\n__image_1__"
    assert messages[0].content[2].text == f"Image source: {sources[1]}\n__image_2__"
    assert original.model_dump() == snapshot
    assert [block.model_dump() for block in images.values()] == snapshot["content"][1:3]
    mtimes = [path.stat().st_mtime_ns for path in paths]
    _, replay, replay_images = _prepared(b"first image", b"second image")
    assert await save_session_images(tmp_path, "sessions", "chat", replay, replay_images) == sources
    assert [path.stat().st_mtime_ns for path in paths] == mtimes


@pytest.mark.asyncio
@pytest.mark.parametrize("message_id", ["../../outside", "A", "a", "图像/a%"])
async def test_message_id_is_encoded_not_used_as_a_path(tmp_path, message_id):
    _, messages, images = _prepared(b"image", message_id=message_id)
    sources = await save_session_images(tmp_path, "session", "chat", messages, images)
    path = tmp_path / sources[0][2:-2]
    encoded = path.name.removeprefix("msg-").split("-image-", 1)[0]
    assert bytes.fromhex(encoded).decode() == message_id
    assert path.parent == tmp_path / "session/images/chat"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "session_dir,session_id",
    [("session", "chat#1"), ("session", "[chat]"), ("sessions#1", "chat"), ("[sessions]", "chat")],
)
async def test_unlinkable_attachment_paths_fail_before_writing(tmp_path, session_dir, session_id):
    _, messages, images = _prepared(b"image")
    with pytest.raises(ValueError, match="cannot be linked"):
        await save_session_images(tmp_path, session_dir, session_id, messages, images)
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["base64", "denied", "directory", "conflict", "symlink", "traversal"])
async def test_invalid_inputs_are_checked_before_any_image_write(tmp_path, failure):
    _, messages, images = _prepared(b"first", b"second")
    session_dir = "session"
    allowed_paths = None
    second = tmp_path / f"session/images/chat/msg-{'message-a'.encode().hex()}-image-2.png"
    if failure == "base64":
        images["__image_2__"].source.data = "invalid base64!"
    elif failure == "denied":
        allowed_paths = ["unrelated"]
    elif failure in {"directory", "conflict"}:
        second.parent.mkdir(parents=True)
        if failure == "directory":
            second.mkdir()
        else:
            second.write_bytes(b"user-owned")
    elif failure == "symlink":
        (tmp_path / "session").symlink_to(tmp_path.parent, target_is_directory=True)
    else:
        session_dir = "../outside"
    with pytest.raises((ValueError, PermissionError)):
        await save_session_images(tmp_path, session_dir, "chat", messages, images, allowed_paths)
    assert messages[0].content[1].text == "__image_1__"
    assert not list(tmp_path.glob("session/images/chat/*-image-1.png"))
    if failure == "conflict":
        assert second.read_bytes() == b"user-owned"


@pytest.mark.asyncio
async def test_http_sources_remain_urls_without_local_io(tmp_path, monkeypatch):
    def fail_network(*_args, **_kwargs):
        pytest.fail("Auto Memory must not download image URLs")

    monkeypatch.setattr(httpx.Client, "request", fail_network)
    monkeypatch.setattr(httpx.AsyncClient, "request", fail_network)
    url = "https://example.com/image.png?token=original"
    _, messages, images = _prepared(URLSource(url=url, media_type="image/png"))
    assert await save_session_images(tmp_path, "session", "chat", messages, images) == [url]
    assert messages[0].content[1].text == f"Image source: {url}\n__image_1__"
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_source_links_preserve_body_unknown_links_and_noop_mtime(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("---\nsource_images: ['[[manual.md]]']\nuser_owned: keep\n---\n\n# Body\n\nKeep this text.")
    before = frontmatter.loads(path.read_text())
    await merge_image_sources(tmp_path, "note.md", ["[[new.png]]", "[[new.png]]", "[[manual.md]]"])
    after = frontmatter.loads(path.read_text())
    assert after.content == before.content
    assert after.metadata == {"source_images": ["[[manual.md]]", "[[new.png]]"], "user_owned": "keep"}
    mtime = path.stat().st_mtime_ns
    await merge_image_sources(tmp_path, "note.md", ["[[new.png]]"])
    assert path.stat().st_mtime_ns == mtime


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", ["source_images: manual", "source_images: null", "source_images: [3]"])
async def test_invalid_existing_source_field_is_not_replaced(tmp_path, metadata):
    path = tmp_path / "note.md"
    path.write_text(f"---\n{metadata}\n---\n\nUser body.")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="list of strings"):
        await merge_image_sources(tmp_path, "note.md", ["[[new.png]]"])
    assert path.read_bytes() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("sources", ["[[image.png]]", [None]])
async def test_invalid_incoming_sources_do_not_write(tmp_path, sources):
    path = tmp_path / "note.md"
    path.write_text("User body.")
    with pytest.raises(ValueError, match="list of strings"):
        await merge_image_sources(tmp_path, "note.md", sources)
    assert path.read_text() == "User body."


@pytest.mark.asyncio
async def test_merge_rereads_after_concurrent_native_write(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_module = importlib.import_module("reme.steps.file_io.write")
    native_write = write_module.write_file_safe
    entered, release = asyncio.Event(), asyncio.Event()

    async def delayed_write(*args, **kwargs):
        entered.set()
        await release.wait()
        await native_write(*args, **kwargs)

    monkeypatch.setattr(write_module, "write_file_safe", delayed_write)
    path = tmp_path / "note.md"
    path.write_text("Old content")
    writer = WriteStep(file_store=SimpleNamespace(workspace_path=tmp_path))
    context = RuntimeContext(
        path="note.md",
        content="New user content",
        metadata={"source_images": ["[[manual.md]]"], "user_owned": 7},
    )
    writing = asyncio.create_task(writer(context))
    await asyncio.wait_for(entered.wait(), timeout=5)
    merging = asyncio.create_task(merge_image_sources(tmp_path, "note.md", ["[[image.png]]"]))
    await asyncio.sleep(0)
    assert not merging.done()
    release.set()
    await asyncio.wait_for(asyncio.gather(writing, merging), timeout=5)
    assert context.response.success
    post = frontmatter.loads(path.read_text())
    assert post.content == "New user content"
    assert post.metadata == {"source_images": ["[[manual.md]]", "[[image.png]]"], "user_owned": 7}
