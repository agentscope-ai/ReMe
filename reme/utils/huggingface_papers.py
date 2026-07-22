"""Client and response normalization for Hugging Face Papers."""

import asyncio
import re
import sys
from collections.abc import Iterable
from typing import Any

import httpx

from ..schema import PaperInfo
from .arxiv import ARXIV_ID_PATTERN
from .proxy_utils import get_ssh_proxy_config, ssh_socks_proxy

HF_BASE_URL = "https://huggingface.co"
_PAPER_LINK_PATTERN = re.compile(
    r"href=[\"'](?:https://huggingface\.co)?/papers/(\d{4}\.\d{4,5})(?:[^\"']*)?[\"']",
    re.IGNORECASE,
)


def _organization_name(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    name = value.get("fullname") or value.get("name")
    return str(name).strip() if name else None


def paper_info_from_payload(item: dict[str, Any]) -> PaperInfo:
    """Normalize either a daily-list item or a paper-detail response."""
    nested = item.get("paper")
    paper = nested if isinstance(nested, dict) else item
    arxiv_id = str(paper.get("id") or "").strip()
    if not ARXIV_ID_PATTERN.fullmatch(arxiv_id):
        raise ValueError(f"Invalid arXiv id from Hugging Face: {arxiv_id!r}")

    authors = [
        str(author["name"]).strip()
        for author in paper.get("authors") or []
        if isinstance(author, dict) and author.get("name")
    ]

    organization = paper.get("organization") or item.get("organization")
    return PaperInfo(
        arxiv_id=arxiv_id,
        title=str(paper.get("title") or item.get("title") or "").strip(),
        summary=str(paper.get("summary") or item.get("summary") or "").strip(),
        authors=authors,
        published_at=paper.get("publishedAt") or item.get("publishedAt"),
        submitted_on_daily_at=paper.get("submittedOnDailyAt"),
        upvotes=int(paper.get("upvotes") or 0),
        organization=_organization_name(organization),
        github_repo=paper.get("githubRepo"),
        github_stars=paper.get("githubStars"),
        project_page=paper.get("projectPage"),
        thumbnail=item.get("thumbnail"),
    )


def paper_ids_from_html(content: str) -> list[str]:
    """Return unique paper ids in server-rendered display order."""
    return list(dict.fromkeys(_PAPER_LINK_PATTERN.findall(content)))


class HuggingFacePapersClient:
    """Fetch ranked paper pages and their accompanying JSON metadata."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        detail_concurrency: int = 5,
    ) -> None:
        self._owns_client = client is None
        self._timeout = timeout
        self._proxy_context = None
        self.client = client
        if self.client is None and get_ssh_proxy_config() is None:
            self.client = self._new_client()
        self.max_retries = max(1, int(max_retries))
        self.detail_concurrency = max(1, int(detail_concurrency))

    def _new_client(self, proxy: str | None = None) -> httpx.AsyncClient:
        kwargs: dict[str, Any] = {
            "base_url": HF_BASE_URL,
            "timeout": self._timeout,
            "follow_redirects": True,
            "headers": {"User-Agent": "ReMe daily-paper cookbook"},
        }
        if proxy is not None:
            kwargs["proxy"] = proxy
        return httpx.AsyncClient(**kwargs)

    async def __aenter__(self) -> "HuggingFacePapersClient":
        if self.client is None:
            self._proxy_context = ssh_socks_proxy(connect_timeout=self._timeout)
            try:
                proxy = await self._proxy_context.__aenter__()
                self.client = self._new_client(proxy)
            except BaseException:
                await self._proxy_context.__aexit__(*sys.exc_info())
                self._proxy_context = None
                raise
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        try:
            if self._owns_client and self.client is not None:
                await self.client.aclose()
        finally:
            if self._proxy_context is not None:
                await self._proxy_context.__aexit__(exc_type, exc_value, traceback)

    def _require_client(self) -> httpx.AsyncClient:
        if self.client is None:
            raise RuntimeError("HuggingFacePapersClient must be used as an async context manager with SSH proxying")
        return self.client

    async def _get(self, path: str, *, params: dict[str, Any] | None = None) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = await self._require_client().get(path, params=params)
                response.raise_for_status()
                return response
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt + 1 >= self.max_retries:
                    break
                await asyncio.sleep(0.25 * (2**attempt))
        assert last_error is not None
        raise last_error

    async def _json_list(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        response = await self._get("/api/daily_papers", params=params)
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError(f"Unexpected Hugging Face daily_papers response: {type(payload).__name__}")
        return [item for item in payload if isinstance(item, dict)]

    async def fetch_detail(self, arxiv_id: str) -> PaperInfo:
        """Fetch and normalize one paper-detail response."""
        if not ARXIV_ID_PATTERN.fullmatch(arxiv_id):
            raise ValueError(f"Invalid arXiv id: {arxiv_id!r}")
        response = await self._get(f"/api/papers/{arxiv_id}")
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected paper response for {arxiv_id}")
        return paper_info_from_payload(payload)

    async def fetch_daily_ids(self, day: str) -> set[str]:
        """Return paper ids for exactly one Hugging Face Daily Papers date."""
        items = await self._json_list({"date": day, "limit": 100})
        return {paper_info_from_payload(item).arxiv_id for item in items}

    async def fetch_scope(self, scope: str, value: str) -> list[PaperInfo]:
        """Fetch all paper cards shown on one weekly or monthly page."""
        if scope not in {"week", "month"}:
            raise ValueError("scope must be 'week' or 'month'")

        page_response, api_items = await asyncio.gather(
            self._get(f"/papers/{scope}/{value}"),
            self._json_list({scope: value, "limit": 100}),
        )
        page_ids = paper_ids_from_html(page_response.text)
        api_infos = [paper_info_from_payload(item) for item in api_items]
        api_by_id = {paper.arxiv_id: paper for paper in api_infos}
        ranked_ids = page_ids or [paper.arxiv_id for paper in api_infos]

        missing_ids = [arxiv_id for arxiv_id in ranked_ids if arxiv_id not in api_by_id]
        if missing_ids:
            details = await self._fetch_details_limited(missing_ids)
            api_by_id.update({paper.arxiv_id: paper for paper in details})

        return [api_by_id[arxiv_id] for arxiv_id in ranked_ids if arxiv_id in api_by_id]

    async def _fetch_details_limited(self, arxiv_ids: Iterable[str]) -> list[PaperInfo]:
        semaphore = asyncio.Semaphore(self.detail_concurrency)

        async def fetch(arxiv_id: str) -> PaperInfo:
            async with semaphore:
                return await self.fetch_detail(arxiv_id)

        return list(await asyncio.gather(*(fetch(arxiv_id) for arxiv_id in arxiv_ids)))
