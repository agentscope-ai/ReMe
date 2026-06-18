"""Tests for transfer step filesystem safety."""

# pylint: disable=protected-access

import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from reme4.schema import FileFrontMatter, FileNode
from reme4.steps.transfer.download import DownloadStep
from reme4.steps.transfer.ingest import IngestStep, _DuplicateIngest, _assemble_day_md
from reme4.steps.transfer.upload import UploadStep


def _step(step_cls, vault_path: Path):
    """Build a transfer step with a minimal file_store stub."""
    step = step_cls()
    step.file_store = SimpleNamespace(vault_path=vault_path)
    return step


@pytest.mark.asyncio
async def test_upload_rejects_dst_path_outside_vault(tmp_path):
    """Upload must reject destinations that escape the vault."""
    vault = tmp_path / "vault"
    vault.mkdir()
    source = tmp_path / "source.txt"
    source.write_text("payload", encoding="utf-8")

    payload = await _step(UploadStep, vault)._upload(str(source), "dir/../../outside.txt", False)

    assert payload["error"] == "dst_path must stay inside the vault"
    assert not (tmp_path / "outside.txt").exists()


@pytest.mark.asyncio
async def test_download_rejects_src_path_outside_vault(tmp_path):
    """Download must reject sources that escape the vault."""
    vault = tmp_path / "vault"
    vault.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    payload = await _step(DownloadStep, vault)._download("../outside.txt", str(tmp_path / "copy.txt"), False)

    assert payload["error"] == "src_path must stay inside the vault"
    assert not (tmp_path / "copy.txt").exists()


@pytest.mark.asyncio
async def test_download_requires_absolute_explicit_destination(tmp_path):
    """Explicit download destinations must be absolute filesystem paths."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "in.txt").write_text("content", encoding="utf-8")

    payload = await _step(DownloadStep, vault)._download("in.txt", "relative.txt", False)

    assert payload["error"] == "dst_path must be an absolute filesystem path"


def test_ingest_rejects_case_insensitive_duplicate(tmp_path):
    """Ingest duplicate detection should be case-insensitive."""
    vault = tmp_path / "vault"
    step = _step(IngestStep, vault)
    source = tmp_path / "Report.pdf"
    source.write_text("content", encoding="utf-8")
    date = "2026-06-18"
    final_name = "wechat__120000__Report.pdf"
    bucket = vault / "resource" / date
    bucket.mkdir(parents=True)
    (bucket / "wechat__120000__report.pdf").write_text("old", encoding="utf-8")

    with pytest.raises(_DuplicateIngest):
        step._land(
            source,
            date,
            final_name,
            {"channel": "wechat", "received_at": "2026-06-18T12:00:00", "description": "desc"},
        )


def test_assemble_day_md_quotes_asset_names_for_yaml_frontmatter():
    """Asset names in generated markdown frontmatter should be JSON-quoted."""
    entry = FileNode(
        path="resource/2026-06-18/wechat__120000__a, [b].pdf",
        st_mtime=1.0,
        front_matter=FileFrontMatter(
            description="desc",
            channel="wechat",
            received_at=datetime.datetime(2026, 6, 18, 12, 0, 0).isoformat(),
        ),
    )

    rendered = _assemble_day_md([entry], "2026-06-18")
    assets_line = next(line for line in rendered.splitlines() if line.startswith("assets: "))

    assert json.loads(assets_line.removeprefix("assets: ")) == ["wechat__120000__a, [b].pdf"]
