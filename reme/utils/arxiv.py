"""arXiv validation and PDF download helpers."""

import os
import re
from pathlib import Path
from uuid import uuid4

import aiofiles
import httpx

ARXIV_ID_PATTERN = re.compile(r"^\d{4}\.\d{4,5}$")


class ArxivPdfClient:
    """Download validated arXiv PDFs to local files."""

    def __init__(self, *, timeout: float = 90.0, max_bytes: int = 50 * 1024 * 1024) -> None:
        self.timeout, self.max_bytes = timeout, max_bytes

    async def download(self, arxiv_id: str, target: Path) -> Path:
        """Download one PDF atomically, reusing an existing valid target."""
        if not ARXIV_ID_PATTERN.fullmatch(arxiv_id):
            raise ValueError(f"Invalid arXiv id: {arxiv_id!r}")
        if target.is_file() and target.stat().st_size > 5:
            with target.open("rb") as existing:
                if existing.read(5) == b"%PDF-":
                    return target

        target.parent.mkdir(parents=True, exist_ok=True)
        part_path = target.with_name(f".{target.name}.{uuid4().hex}.part")
        size = 0
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": "ReMe arXiv client"},
            ) as client:
                async with client.stream("GET", f"https://arxiv.org/pdf/{arxiv_id}") as response:
                    response.raise_for_status()
                    content_length = int(response.headers.get("content-length") or 0)
                    if content_length and content_length > self.max_bytes:
                        raise ValueError(f"PDF exceeds maximum size: {content_length} > {self.max_bytes}")
                    async with aiofiles.open(part_path, "wb") as stream:
                        async for chunk in response.aiter_bytes():
                            size += len(chunk)
                            if size > self.max_bytes:
                                raise ValueError(f"PDF exceeds maximum size: {size} > {self.max_bytes}")
                            await stream.write(chunk)
            if size <= 5:
                raise ValueError(f"Downloaded PDF is empty for {arxiv_id}")
            async with aiofiles.open(part_path, "rb") as stream:
                header = await stream.read(5)
            if header != b"%PDF-":
                raise ValueError(f"Downloaded content is not a PDF for {arxiv_id}")
            os.replace(part_path, target)
            return target
        finally:
            if part_path.exists():
                part_path.unlink()
