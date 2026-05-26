"""Read a markdown file from vault_dir, with line-range slicing and byte-truncation."""

import json

from ._file_io import (
    NON_MD_WARNING,
    gate_md,
    resolve_path,
    read_file_safe,
    truncate_text_output,
)
from ..base_step import BaseStep
from ..common.link_expansion import get_first_order_neighbors
from ...components import R


@R.register("read_step")
class ReadStep(BaseStep):
    """Read a markdown file. Optional `start_line`/`end_line` for ranged reads.

    Opt-in kwargs:
        with_neighbors (bool, default False): when True and the file is markdown,
            append a block listing first-order bidirectional neighbors (out/in
            link targets) with their frontmatter, fetched via the file_store.
        max_neighbors_per_direction (int, default 10): cap per direction.
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
        start_line = self.context.get("start_line")
        end_line = self.context.get("end_line")

        target, err = resolve_path(self.vault_path, raw)
        if err:
            self._fail(err)
            return None

        target, is_md = gate_md(target)
        if not is_md:
            self.logger.info(f"[{self.name}] {NON_MD_WARNING} path={target}")

        for label, value in (("start_line", start_line), ("end_line", end_line)):
            if value is None:
                continue
            try:
                int(value)
            except (TypeError, ValueError):
                self._fail(f"{label} must be an integer, got {value!r}")
                return None

        if not target.exists():
            self._fail(f"file {target} does not exist", path=str(target))
            return None
        if not target.is_file():
            self._fail(f"path {target} is not a file", path=str(target))
            return None

        try:
            content = await read_file_safe(target)
        except Exception as e:
            self._fail(f"read failed: {e}", path=str(target))
            return None

        all_lines = content.split("\n")
        total = len(all_lines)
        s = max(1, int(start_line) if start_line is not None else 1)
        e = min(total, int(end_line) if end_line is not None else total)

        if s > total:
            self._fail(
                f"start_line {s} exceeds file length ({total} lines)",
                path=str(target),
                total_lines=total,
            )
            return None
        if s > e:
            self._fail(f"start_line ({s}) > end_line ({e})", path=str(target))
            return None

        selected = "\n".join(all_lines[s - 1 : e])
        text = truncate_text_output(
            selected,
            start_line=s,
            total_lines=total,
            file_path=str(target),
        )

        self.context.response.success = True
        self.context.response.answer = text
        self.logger.info(
            f"[{self.name}] read path={target} lines={s}-{e}/{total} bytes={len(text.encode('utf-8'))}",
        )

        # Runtime context (HTTP / job parameters) takes precedence over step
        # init kwargs so callers can toggle neighbor injection per-request.
        with_neighbors = bool(
            self.context.get("with_neighbors", self.kwargs.get("with_neighbors", False)),
        )
        max_per_direction = int(
            self.context.get(
                "max_neighbors_per_direction",
                self.kwargs.get("max_neighbors_per_direction", 10),
            ),
        )
        if with_neighbors and is_md:
            await self._maybe_inject_neighbors(target, text, max_per_direction)

        return self.context.response

    async def _maybe_inject_neighbors(self, target, text: str, max_per_direction: int) -> None:
        """Append the first-order neighbor frontmatter block to response.answer when available."""
        assert self.context is not None
        try:
            rel_path = str(target.relative_to(self.vault_path))
        except ValueError:
            self.logger.info(f"[{self.name}] skip neighbors: path outside vault_path path={target}")
            return

        try:
            neighbors = await get_first_order_neighbors(
                self.file_store,
                rel_path,
                max_per_direction=max_per_direction,
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                f"[{self.name}] neighbor fetch failed: {type(exc).__name__}: {exc}",
            )
            return

        block = _render_neighbor_block(neighbors)
        if not block:
            return

        self.context.response.answer = f"{text}\n\n{block}"
        self.context.response.metadata["link_expansion"] = _neighbors_to_serializable(neighbors)


def _render_neighbor_block(neighbors: dict) -> str:
    """Render the appended block. Returns '' when both directions are empty."""
    out = neighbors.get("outlinks", [])
    inn = neighbors.get("inlinks", [])
    if not out and not inn:
        return ""

    lines = [
        f"========== Related neighbors (outlinks={len(out)}, inlinks={len(inn)}) ==========",
    ]
    for arrow, items in (("→", out), ("←", inn)):
        for item in items:
            lines.append(f"  {arrow} {item['path']}")
            node = item.get("node")
            if node is None:
                continue
            fm_dict = node.front_matter.model_dump(exclude_defaults=True, exclude_none=True)
            if fm_dict:
                lines.append(f"      frontmatter: {json.dumps(fm_dict, ensure_ascii=False)}")
    return "\n".join(lines)


def _neighbors_to_serializable(neighbors: dict) -> dict:
    """Replace FileNode values with plain dicts so metadata is JSON-friendly."""
    serialized: dict[str, list[dict]] = {}
    for direction in ("outlinks", "inlinks"):
        items = []
        for entry in neighbors.get(direction, []):
            node = entry.get("node")
            items.append(
                {
                    "path": entry["path"],
                    "node": node.model_dump(exclude_none=True) if node is not None else None,
                    "edges": entry.get("edges", []),
                },
            )
        serialized[direction] = items
    return serialized
