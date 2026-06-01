"""Read an image file as base64; oversized images degrade to path + metadata."""

import base64

import aiofiles
import aiofiles.os

from ._path import NON_IMAGE_WARNING, gate_image, resolve_path
from ..base_step import BaseStep
from ...components import R
from ...constants import DEFAULT_MAX_IMAGE_BYTES


@R.register("read_image_step")
class ReadImageStep(BaseStep):
    """Read an image file under ``vault_dir`` and return base64 in ``answer``.

    Step-level attributes (``kwargs``, configured in yaml under ``steps:`` —
    not exposed to LLM):
        max_bytes (int, default ``DEFAULT_MAX_IMAGE_BYTES``): cap for the
            base64 path. Above this, ``answer`` carries a notice and
            ``metadata.oversized=True`` (no base64).

    Invariants (callers depend on these):
        - Normal branch: ``answer`` is pure base64 (no ``data:`` prefix, no
          notice suffix). Use ``f"data:{mime};base64,{answer}"`` if you need a
          data URL.
        - Oversized branch (``metadata.oversized=True``): ``answer`` is a
          human-readable notice, **not** base64. Inspect ``metadata.oversized``
          before decoding.
        - Unknown / missing suffix (compatibility mode): ``answer`` stays pure
          base64; ``metadata.non_image_warning=True`` flags it.
    """

    def _fail(self, message: str, **meta) -> None:
        assert self.context is not None
        self.context.response.success = False
        self.context.response.answer = f"Error: {message}"
        if meta:
            self.context.response.metadata.update(meta)

    async def execute(self):  # pylint: disable=too-many-return-statements
        assert self.context is not None
        raw = str(self.context.get("path") or "")
        max_bytes_raw = self.kwargs.get("max_bytes", DEFAULT_MAX_IMAGE_BYTES)

        try:
            max_bytes = int(max_bytes_raw)
            if max_bytes <= 0:
                raise ValueError()
        except (TypeError, ValueError):
            self._fail(f"`max_bytes` must be a positive integer, got {max_bytes_raw!r}")
            return None

        target, err = resolve_path(self.vault_path, raw)
        if err:
            self._fail(err)
            return None

        target, is_image, mime = gate_image(target)
        if not is_image:
            self.logger.info(f"[{self.name}] {NON_IMAGE_WARNING} path={target}")

        if not target.exists():
            self._fail(f"file {target} does not exist", path=str(target))
            return None
        if not target.is_file():
            self._fail(f"path {target} is not a file", path=str(target))
            return None

        try:
            stat = await aiofiles.os.stat(str(target))
        except Exception as e:  # pylint: disable=broad-except
            self._fail(f"read failed: {e}", path=str(target))
            return None
        size_bytes = stat.st_size

        if size_bytes > max_bytes:
            self.context.response.success = True
            self.context.response.answer = (
                f"image exceeds max_bytes ({size_bytes} > {max_bytes}), "
                f"base64 omitted; use path={target} to access directly"
            )
            md = {
                "path": str(target),
                "size_bytes": size_bytes,
                "mime": mime,
                "oversized": True,
                "max_bytes": max_bytes,
            }
            if not is_image:
                md["non_image_warning"] = True
            self.context.response.metadata.update(md)
            self.logger.info(
                f"[{self.name}] read path={target} size={size_bytes} mime={mime} "
                f"oversized=True is_image={is_image}",
            )
            return self.context.response

        try:
            async with aiofiles.open(str(target), "rb") as f:
                data = await f.read()
        except Exception as e:  # pylint: disable=broad-except
            self._fail(f"read failed: {e}", path=str(target))
            return None

        b64 = base64.b64encode(data).decode("ascii")
        self.context.response.success = True
        self.context.response.answer = b64
        md = {"path": str(target), "size_bytes": size_bytes, "mime": mime}
        if not is_image:
            md["non_image_warning"] = True
        self.context.response.metadata.update(md)
        self.logger.info(
            f"[{self.name}] read path={target} size={size_bytes} mime={mime} oversized=False is_image={is_image}",
        )
        return self.context.response
