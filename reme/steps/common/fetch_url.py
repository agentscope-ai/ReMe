"""Fetch bounded HTTP content without a browser or another model call."""

import asyncio
import codecs
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from uuid import uuid4

import httpx

from ..proactive.data_connectors.base import BaseConnectorStep
from ...components import R
from ...schema import ConnectorResult


class _PageText(HTMLParser):
    """Keep readable page text and its title, excluding non-content elements."""

    _ignored_tags = {"script", "style", "noscript", "template", "svg", "nav", "footer"}
    _block_tags = {
        "p", "div", "li", "br", "hr", "section", "article", "main", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
        "pre", "blockquote", "ul", "ol", "table",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title: list[str] = []
        self.charset = ""
        self._ignored: list[str] = []
        self._in_title = False
        self._in_pre = False

    def _break(self):
        if self.parts and self.parts[-1] == "\t":
            self.parts.pop()
        if self.parts and not self.parts[-1].endswith("\n"):
            self.parts.append("\n")

    def handle_starttag(self, tag, attrs):
        if tag in self._ignored_tags:
            self._ignored.append(tag)
        if self._ignored:
            return
        if tag == "meta" and not self.charset:
            attrs = dict(attrs)
            self.charset = attrs.get("charset") or ""
            if not self.charset and (attrs.get("http-equiv") or "").lower() == "content-type":
                match = re.search(r"charset\s*=\s*[\"']?([\w-]+)", attrs.get("content") or "", re.IGNORECASE)
                self.charset = match.group(1) if match else ""
        if tag == "title":
            self._in_title = True
        elif tag in self._block_tags:
            self._break()
        if tag == "pre":
            self._in_pre = True

    def handle_endtag(self, tag):
        if tag in self._ignored:
            # Closing a parent also closes unbalanced ignored descendants.
            index = len(self._ignored) - 1 - self._ignored[::-1].index(tag)
            del self._ignored[index:]
        if self._ignored:
            return
        if tag == "title":
            self._in_title = False
        elif tag in {"td", "th"}:
            self.parts.append("\t")
        elif tag in self._block_tags:
            self._break()
        if tag == "pre":
            self._in_pre = False

    def handle_data(self, data):
        if self._ignored:
            return
        if self._in_title:
            self.title.append(data)
        elif self._in_pre:
            self.parts.append(data)
        else:
            data = re.sub(r"\s+", " ", data)
            if not self.parts or self.parts[-1].endswith(("\n", "\t", " ")):
                data = data.lstrip(" ")
            if data:
                self.parts.append(data)


def _decode_text(data: bytes, content_type: str, charset: str | None, is_html: bool) -> tuple[str, str]:
    """Honor BOMs and format-specific declarations without inspecting arbitrary body text."""
    encoding = charset or "utf-8"
    for bom, name in (
        (codecs.BOM_UTF32_LE, "utf-32"), (codecs.BOM_UTF32_BE, "utf-32"),
        (codecs.BOM_UTF16_LE, "utf-16"), (codecs.BOM_UTF16_BE, "utf-16"), (codecs.BOM_UTF8, "utf-8-sig"),
    ):
        if data.startswith(bom):
            encoding = name
            break
    else:
        if not charset:
            if content_type == "application/json" or content_type.endswith("+json"):
                encoding = json.detect_encoding(data)
            else:
                match = re.match(rb"\s*<\?xml\b[^?]*\bencoding\s*=\s*['\"]([\w.-]+)['\"]", data[:1024])
                if match:
                    encoding = match.group(1).decode("ascii")
                elif is_html:
                    page = _PageText()
                    page.feed(data[:8192].decode("ascii", errors="ignore"))
                    encoding = page.charset or encoding
    try:
        return data.decode(encoding, errors="replace"), encoding
    except (LookupError, UnicodeError):
        return data.decode("utf-8", errors="replace"), "utf-8"


def _with_preview(answer: dict, content: str, max_chars: int) -> dict:
    """Keep the complete MCP result below its byte limit; full content lives in paths."""
    answer.update(content="", truncated=bool(content))

    def size():
        # ASCII escaping is a conservative bound for either JSON serialization mode.
        return len(json.dumps(answer, ensure_ascii=True).encode("utf-8"))

    if size() > 8000:
        for key in ("url", "final_url", "title"):
            answer.pop(key, None)
        answer["details_in_metadata_file"] = True
    low, high = 0, min(len(content), max_chars)
    while low < high:
        middle = (low + high + 1) // 2
        answer["content"] = content[:middle]
        if size() <= 8000:
            low = middle
        else:
            high = middle - 1
    answer.update(content=content[:low], truncated=low < len(content))
    return answer


@R.register("fetch_url_step")
class FetchUrlStep(BaseConnectorStep):
    """Download text or PDFs, publish complete files and return a bounded text preview."""

    source = "fetch_url"
    _data: bytes
    _metadata: dict
    _charset: str | None
    _content: str

    async def execute(self):
        raw_url = self.context.get("url")
        if not isinstance(raw_url, str) or not 1 <= len(raw_url.strip()) <= 4096:
            raise ValueError("url must be a non-empty HTTP or HTTPS URL of at most 4096 characters")
        url = httpx.URL(raw_url.strip())
        if url.scheme not in {"http", "https"} or not url.host or url.userinfo:
            raise ValueError("url must use HTTP or HTTPS without embedded credentials")
        max_chars = self.context.get("max_chars", 3000)
        if type(max_chars) is not int or not 1 <= max_chars <= 10000:
            raise ValueError("max_chars must be an integer from 1 to 10000")

        self.resource_target()  # Check workspace containment and permissions before any I/O.
        max_bytes = self.max_file_bytes()
        if max_bytes <= 0:
            raise ValueError("max_file_bytes must be positive")
        data = bytearray()
        async with asyncio.timeout(25):
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(20, connect=5),
                follow_redirects=True,
                max_redirects=5,
                headers={"User-Agent": "ReMe fetch_url"},
            ) as client:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                    if content_type and not (
                        content_type.startswith("text/")
                        or content_type in {
                            "application/xhtml+xml", "application/json", "application/xml", "application/pdf",
                            "application/octet-stream",
                        }
                        or content_type.endswith(("+json", "+xml"))
                    ):
                        raise ValueError(f"Unsupported content type: {content_type}; fetch_url supports text and PDF")
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        if self.context.stop_event is not None and self.context.stop_event.is_set():
                            raise RuntimeError("Download stopped before publication")
                        if len(data) + len(chunk) > max_bytes:
                            raise ValueError(f"Download exceeds {max_bytes} bytes; no partial file saved")
                        data.extend(chunk)
                    self._charset = response.charset_encoding
                    self._metadata = {
                        "url": str(url),
                        "final_url": str(response.url),
                        "content_type": content_type,
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                        "download_bytes": len(data),
                    }
        self._data = bytes(data)
        response = await super().execute()
        response.answer = _with_preview({**response.answer, **self._metadata}, self._content, max_chars)
        return response

    def collect(self, directory: Path, created_at: datetime) -> ConnectorResult:
        content_type = self._metadata["content_type"]
        is_pdf = self._data.lstrip().startswith(b"%PDF-")
        if content_type in {"application/pdf", "application/octet-stream"} and not is_pdf:
            raise ValueError("Response is not a PDF; unsupported binary content")
        folder = f"{created_at:%Y-%m-%d_%H-%M-%S-%f}_{uuid4().hex[:8]}"
        destination = directory / folder
        destination.mkdir()
        files = []
        self._content = ""
        if is_pdf:
            (destination / "source.pdf").write_bytes(self._data)
            files.append("source.pdf")
            self._metadata["status"] = "downloaded"
            context = "PDF downloaded. Use a PDF-capable reader on the saved .pdf file."
        else:
            is_html = content_type in {"text/html", "application/xhtml+xml"} or (
                not content_type and bool(re.search(rb"<\s*(?:!doctype\s+html|html)\b", self._data[:1024], re.I))
            )
            text, encoding = _decode_text(self._data, content_type, self._charset, is_html)
            self._metadata.update(encoding=encoding, title="")
            self._content = text
            context = "Read content.txt for the full text."
            if is_html:
                (destination / "source.html").write_text(text, encoding="utf-8")
                files.append("source.html")
                self._content = self._page_text(text)
                context += " source.html preserves HTML and embedded data; JavaScript is not executed."
            self._metadata["status"] = "ok" if self._content.strip() else "no_readable_text"
            if self._content.strip():
                (destination / "content.txt").write_text(self._content, encoding="utf-8")
                files.insert(0, "content.txt")
            else:
                context = "No readable text found. Check the source; loading more data may require a browser."
        (destination / "metadata.json").write_text(
            json.dumps(self._metadata, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        return ConnectorResult(context=context, paths=[f"{folder}/{name}" for name in [*files, "metadata.json"]])

    def _page_text(self, text: str) -> str:
        page = _PageText()
        page.feed(text)
        page.close()
        self._metadata["title"] = " ".join("".join(page.title).split())[:200]
        return "".join(page.parts).strip("\n")
