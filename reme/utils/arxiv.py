"""arXiv validation and PDF download helpers."""

import os
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiofiles
import httpx

from .logger_utils import get_logger
from .proxy_utils import get_ssh_proxy_config, ssh_socks_proxy

ARXIV_ID_PATTERN = re.compile(r"^\d{4}\.\d{4,5}$")


class ArxivPdfClient:
    """Download validated arXiv PDFs to local files."""

    def __init__(
        self,
        *,
        timeout: float = 90.0,
        max_bytes: int = 50 * 1024 * 1024,
        logger: Any | None = None,
    ) -> None:
        self.timeout, self.max_bytes = timeout, max_bytes
        self.logger = logger or get_logger()

    async def download(self, arxiv_id: str, target: Path) -> Path:
        """Download one PDF atomically, reusing an existing valid target."""
        if not ARXIV_ID_PATTERN.fullmatch(arxiv_id):
            raise ValueError(f"Invalid arXiv id: {arxiv_id!r}")
        if target.is_file() and target.stat().st_size > 5:
            with target.open("rb") as existing:
                if existing.read(5) == b"%PDF-":
                    self.logger.debug(
                        f"[ArxivPdfClient] cache hit arxiv_id={arxiv_id} path={target} bytes={target.stat().st_size}",
                    )
                    return target

        target.parent.mkdir(parents=True, exist_ok=True)
        part_path = target.with_name(f".{target.name}.{uuid4().hex}.part")
        size = 0
        self.logger.info(
            f"[ArxivPdfClient] download start arxiv_id={arxiv_id} path={target} timeout={self.timeout:g}s",
        )
        try:
            proxy_config = get_ssh_proxy_config()
            proxy_started = False
            try:
                async with ssh_socks_proxy(connect_timeout=self.timeout) as proxy:
                    client_kwargs = {
                        "timeout": self.timeout,
                        "follow_redirects": True,
                        "headers": {"User-Agent": "ReMe arXiv client"},
                    }
                    if proxy is not None:
                        proxy_started = True
                        client_kwargs["proxy"] = proxy
                        assert proxy_config is not None
                        self.logger.info(
                            f"[ArxivPdfClient] network mode=ssh_socks destination={proxy_config.destination} "
                            f"arxiv_id={arxiv_id}",
                        )
                    else:
                        self.logger.debug(f"[ArxivPdfClient] network mode=direct arxiv_id={arxiv_id}")
                    async with httpx.AsyncClient(**client_kwargs) as client:
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
            finally:
                if proxy_started:
                    self.logger.info(f"[ArxivPdfClient] SSH proxy closed arxiv_id={arxiv_id}")
            if size <= 5:
                raise ValueError(f"Downloaded PDF is empty for {arxiv_id}")
            async with aiofiles.open(part_path, "rb") as stream:
                header = await stream.read(5)
            if header != b"%PDF-":
                raise ValueError(f"Downloaded content is not a PDF for {arxiv_id}")
            os.replace(part_path, target)
            self.logger.info(
                f"[ArxivPdfClient] download done arxiv_id={arxiv_id} path={target} bytes={size}",
            )
            return target
        except Exception as exc:
            detail = str(exc) or "-"
            self.logger.warning(
                f"[ArxivPdfClient] download failed arxiv_id={arxiv_id} error={type(exc).__name__} detail={detail}",
            )
            raise
        finally:
            if part_path.exists():
                part_path.unlink()
