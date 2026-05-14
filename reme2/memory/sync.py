"""sync — continuously sync key materials into an event folder.

Layout (one folder per logical thread):

    events/{date}/{name}/
    ├── {name}.md               # index: frontmatter + narrative + Materials footer
    ├── {material_filename}     # raw artifact 1
    ├── ...
    └── {material_filename}     # raw artifact N

Hot-path write entry. The agent picks a stable `name` per logical
thread and calls `sync` repeatedly through the task — each call
extends the same event folder rather than creating a new one. This
turns discrete writes into a coherent stream and lets PreCompact /
SessionEnd hooks treat sync as the last-chance flush before context
loss.

Behavior:
  * If `events/{date}/{name}/` does NOT exist → create new folder,
    write index `{name}.md` (status=active), write materials.
  * If it exists with `status: active` → APPEND:
      - new content → `## Update — {iso}` section appended to the body
      - new materials → siblings (auto-suffix on filename collision)
      - Materials footer regenerated as the trailing section, listing
        every artifact actually present in the folder
      - frontmatter `topics` + `tags` unioned, `updated` set to today
  * If it exists with `status: distilled` / `archived` → REFUSE,
    return suggested_name so the agent can start a new thread.

Zero LLM cost.
"""

import json
import re
from collections.abc import Iterable
from datetime import date as date_type, datetime, timezone
from pathlib import Path

import frontmatter
from pydantic import ValidationError

from reme2.component import R
from reme2.component.base_step import BaseStep
from .memory_io import collisions_after_create, create_file
from .schema import EVENT_PRESET, MemoryFileNode


_SAFE_FILENAME_RE = re.compile(r"^[a-zA-Z0-9._-]+$")
_MATERIALS_HEADER = "## Materials"


def _event_path(
    working_dir: str | Path,
    name: str,
    on_date: date_type | str | None = None,
    events_dir: str = "events",
) -> Path:
    """events/{YYYY-MM-DD}/{name}/{name}.md under the given vault root."""
    if on_date is None:
        on_date = date_type.today()
    if isinstance(on_date, date_type):
        on_date = on_date.isoformat()
    return Path(working_dir) / events_dir / on_date / name / f"{name}.md"


def _next_suffixed_stem(taken: Iterable[str], base: str) -> str:
    """Lowest unused `<base>-N` (N≥2). Returns `base` itself if not taken.

    Why advisory: when a same-name-on-same-day collision is detected, the
    suggested suffix nudges the agent to pick a domain-specific qualifier
    (e.g. `Apple-Inc` vs `Apple-Fruit`) over a numeric one.
    """
    taken_set = set(taken)
    if base not in taken_set:
        return base
    pattern = re.compile(rf"^{re.escape(base)}-(\d+)$")
    used: set[int] = set()
    for s in taken_set:
        m = pattern.match(s)
        if m:
            try:
                used.add(int(m.group(1)))
            except ValueError:
                continue
    n = 2
    while n in used:
        n += 1
    return f"{base}-{n}"


@R.register("sync")
class Sync(BaseStep):
    """Upsert into an event folder under events/{date}/{name}/.

    Inputs (RuntimeContext):
        name (str, required):       kebab-case event identifier; becomes both
                                    the parent dir and the index filename stem.
                                    Reuse the same name across calls in one
                                    thread to keep extending the same folder.
        description (str):          one-line summary for index frontmatter
                                    (only set on initial create).
        content (str):              markdown body. On create it's the initial
                                    body; on append it's added under a new
                                    `## Update — {iso}` section.
        topics (list[str]):         related topic wikilinks; merged (union)
                                    into frontmatter on append.
        tags (list[str]):           free-form tags; merged (union) on append.
        materials (list[dict]):     [{filename, content}, ...] — raw artifacts
                                    written as siblings of the index. Filename
                                    collisions auto-suffix (foo.txt → foo-2.txt).
                                    Index body's Materials footer regenerated
                                    each call from actual folder contents.
        on_date (str | None):       ISO date for the events/{date}/ bucket.
                                    Defaults to today.
        origin_session_id (str):    optional source session identifier (only
                                    set on initial create).

    Output (context.response.answer):
        JSON {path, materials: [paths of NEW materials this call],
              created: bool, action: "created"|"appended"} on success;
        {error, ...} on failure (including refusal when existing event
        has status != "active").
    """

    def __init__(self, events_dir: str = "events", **kwargs):
        super().__init__(**kwargs)
        self.events_dir = events_dir

    def _root(self) -> Path:
        vr = getattr(self.file_store, "working_dir", None)
        if vr is not None:
            return Path(vr)
        raise RuntimeError("sync requires file_store.working_dir to be configured")

    @staticmethod
    def _validate_materials(materials: list, index_filename: str) -> tuple[list[dict], str | None]:
        """Sanity-check the materials list. Returns (cleaned, error_message)."""
        cleaned: list[dict] = []
        seen_in_call: set[str] = set()
        for i, m in enumerate(materials):
            if not isinstance(m, dict):
                return [], f"materials[{i}] must be an object {{filename, content}}"
            fname = m.get("filename")
            if not isinstance(fname, str) or not fname:
                return [], f"materials[{i}].filename is required"
            if not _SAFE_FILENAME_RE.match(fname):
                return [], (
                    f"materials[{i}].filename {fname!r} is unsafe — only "
                    f"letters / digits / dot / underscore / dash allowed"
                )
            if fname == index_filename:
                return [], f"materials[{i}].filename {fname!r} collides with the index file"
            if fname in seen_in_call:
                return [], f"materials[{i}].filename {fname!r} duplicated within the same call"
            seen_in_call.add(fname)
            content = m.get("content", "")
            if not isinstance(content, str):
                return [], f"materials[{i}].content must be a string"
            cleaned.append({"filename": fname, "content": content})
        return cleaned, None

    @staticmethod
    def _strip_materials_footer(body: str) -> str:
        """Drop our trailing `## Materials` footer if present; return narrative."""
        if not body:
            return ""
        # Match the footer at end-of-doc: `## Materials\n\n- [...](./...)\n` repeated.
        # Cheaper rule: find the LAST `## Materials` heading at line start; strip
        # from there to EOF. Whatever the user wrote above stays intact.
        m = re.search(r"(?:\A|\n)##\s+Materials[ \t]*\n", body)
        if m is None:
            return body.rstrip()
        # Find the LAST such heading by scanning all matches.
        last = None
        for hit in re.finditer(r"(?:\A|\n)##\s+Materials[ \t]*\n", body):
            last = hit
        assert last is not None
        cut = last.start()
        # If the heading was at offset 0 (no leading \n), keep nothing before;
        # otherwise keep up to (but not including) the leading \n.
        return body[:cut].rstrip()

    @staticmethod
    def _emit_body(narrative: str, material_filenames: list[str]) -> str:
        """Assemble body = narrative (possibly empty) + Materials footer."""
        narrative = (narrative or "").rstrip()
        if not material_filenames:
            return f"{narrative}\n" if narrative else ""
        listing = "\n".join(f"- [{f}](./{f})" for f in material_filenames)
        if narrative:
            return f"{narrative}\n\n{_MATERIALS_HEADER}\n\n{listing}\n"
        return f"{_MATERIALS_HEADER}\n\n{listing}\n"

    @staticmethod
    def _resolve_filename(existing: set[str], requested: str) -> str:
        """Auto-suffix `foo.txt` → `foo-2.txt` (then -3, -4, …) on collision."""
        if requested not in existing:
            return requested
        if "." in requested:
            stem, _, ext = requested.rpartition(".")
            n = 2
            while f"{stem}-{n}.{ext}" in existing:
                n += 1
            return f"{stem}-{n}.{ext}"
        n = 2
        while f"{requested}-{n}" in existing:
            n += 1
        return f"{requested}-{n}"

    @staticmethod
    def _list_existing_materials(folder: Path, index_filename: str) -> list[str]:
        """Filenames in `folder` excluding the index, sorted for stability."""
        if not folder.is_dir():
            return []
        return sorted(entry.name for entry in folder.iterdir() if entry.is_file() and entry.name != index_filename)

    @staticmethod
    def _union(prior: list, incoming: list) -> list:
        """Order-preserving union: keep prior order, append new items in input order."""
        out = list(prior)
        seen = set(prior)
        for item in incoming:
            if item not in seen:
                out.append(item)
                seen.add(item)
        return out

    def _set_error(self, payload: dict) -> None:
        assert self.context is not None
        self.context.response.success = False
        self.context.response.answer = json.dumps(payload, ensure_ascii=False)

    def _set_ok(self, payload: dict) -> None:
        assert self.context is not None
        self.context.response.success = True
        self.context.response.answer = json.dumps(payload, ensure_ascii=False)

    async def execute(self):
        assert self.context is not None
        name: str = self.context.get("name", "") or ""
        description: str = self.context.get("description", "") or ""
        content: str = self.context.get("content", "") or ""
        topics: list[str] = list(self.context.get("topics") or [])
        tags: list[str] = list(self.context.get("tags") or [])
        materials_in = list(self.context.get("materials") or [])
        on_date = self.context.get("on_date")
        origin_session_id = self.context.get("origin_session_id")

        assert name, "name is required"

        target = _event_path(self._root(), name, on_date, self.events_dir)
        materials, mat_err = self._validate_materials(materials_in, target.name)
        if mat_err is not None:
            self._set_error({"error": mat_err})
            return

        if target.exists():
            await self._append(target, content, materials, topics, tags)
        else:
            await self._create(
                target,
                name,
                description,
                content,
                materials,
                topics,
                tags,
                on_date,
                origin_session_id,
            )

    async def _create(
        self,
        target: Path,
        name: str,
        description: str,
        content: str,
        materials: list[dict],
        topics: list[str],
        tags: list[str],
        on_date,
        origin_session_id,
    ) -> None:
        today = date_type.today().isoformat()
        on_date_str = on_date.isoformat() if isinstance(on_date, date_type) else (on_date or today)
        # Start from EVENT_PRESET (4 axes + status + legacy `category`),
        # layer caller-supplied identity fields on top.
        metadata: dict = {
            **EVENT_PRESET,
            "title": name,
            "description": description,
            "tags": tags,
            "topics": topics,
            "created": on_date_str,
            "updated": today,
        }
        if origin_session_id:
            metadata["originSessionId"] = origin_session_id

        try:
            MemoryFileNode.model_validate(
                {
                    "path": str(target.resolve()),
                    "st_mtime": 0.0,
                    **metadata,
                }
            )
        except ValidationError as e:
            self._set_error(
                {
                    "error": "MemoryFileNode schema validation failed",
                    "details": e.errors(include_context=False, include_url=False),
                }
            )
            return

        graph = self.file_store
        conflicts = collisions_after_create(graph, target)
        if conflicts:
            taken = {Path(p).stem for p in graph.nodes}
            suggested_name = _next_suffixed_stem(taken, name)
            self._set_error(
                {
                    "error": (
                        f"stem `[[{name}]]` would resolve ambiguously "
                        f"to {len(conflicts) + 1} paths after this create"
                    ),
                    "conflicts": conflicts,
                    "suggested_name": suggested_name,
                    "hint": (
                        f"retry with name='{suggested_name}', or pick a "
                        f"semantic qualifier (e.g. '{name}-followup')."
                    ),
                }
            )
            return

        material_filenames = [m["filename"] for m in materials]
        index_body = self._emit_body(content, material_filenames)
        ok, payload = create_file(
            self.file_store,
            target,
            metadata=metadata,
            content=index_body,
        )
        if not ok:
            self._set_error(
                {
                    "path": str(target.resolve()),
                    "error": payload.get("error", "create failed"),
                    "details": payload,
                }
            )
            return

        material_paths: list[str] = []
        for m in materials:
            material_path = target.parent / m["filename"]
            try:
                material_path.write_text(m["content"], encoding="utf-8")
            except Exception as e:
                self.logger.warning(
                    f"sync: failed to write material {m['filename']}: {e}",
                )
                continue
            material_paths.append(str(material_path.resolve()))

        self._set_ok(
            {
                "path": str(target.resolve()),
                "category": "event",
                "status": "active",
                "topics": topics,
                "materials": material_paths,
                "created": True,
                "action": "created",
            }
        )

    async def _append(
        self,
        target: Path,
        content: str,
        materials: list[dict],
        topics: list[str],
        tags: list[str],
    ) -> None:
        # Read current frontmatter + body.
        try:
            raw = target.read_text(encoding="utf-8")
        except Exception as e:
            self._set_error({"path": str(target.resolve()), "error": f"read failed: {e}"})
            return
        post = frontmatter.loads(raw)
        meta = dict(post.metadata)

        status = meta.get("status")
        if status != "active":
            # Don't extend a distilled / archived thread — make the agent pick
            # a new name so the prior cognition isn't silently mutated.
            graph = self.file_store
            taken = {Path(p).stem for p in graph.nodes}
            base = target.stem
            suggested = _next_suffixed_stem(taken, base)
            self._set_error(
                {
                    "path": str(target.resolve()),
                    "error": (
                        f"event `{base}` already exists with status={status!r}; "
                        f"pick a new name to start a fresh thread"
                    ),
                    "status": status,
                    "suggested_name": suggested,
                }
            )
            return

        folder = target.parent
        existing_filenames = self._list_existing_materials(folder, target.name)
        existing_set = set(existing_filenames)

        # Resolve filename collisions for new materials.
        resolved: list[tuple[str, str]] = []  # (filename_on_disk, content)
        all_taken = set(existing_set)
        for m in materials:
            fname = self._resolve_filename(all_taken, m["filename"])
            all_taken.add(fname)
            resolved.append((fname, m["content"]))

        # Write new materials to disk.
        new_material_paths: list[str] = []
        for fname, mcontent in resolved:
            material_path = folder / fname
            try:
                material_path.write_text(mcontent, encoding="utf-8")
            except Exception as e:
                self.logger.warning(f"sync: failed to write material {fname}: {e}")
                continue
            new_material_paths.append(str(material_path.resolve()))
            existing_filenames.append(fname)

        # Rebuild the index body: narrative (existing + optional new Update
        # section) + Materials footer (regenerated from disk).
        narrative = self._strip_materials_footer(post.content)
        if content.strip():
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            update_section = f"## Update — {ts}\n\n{content.rstrip()}\n"
            narrative = f"{narrative.rstrip()}\n\n{update_section}" if narrative else update_section

        all_filenames_sorted = sorted(set(existing_filenames))
        new_body = self._emit_body(narrative, all_filenames_sorted)

        # Update frontmatter: union topics/tags, bump updated.
        meta["topics"] = self._union(list(meta.get("topics") or []), topics)
        meta["tags"] = self._union(list(meta.get("tags") or []), tags)
        meta["updated"] = date_type.today().isoformat()

        try:
            MemoryFileNode.model_validate(
                {
                    "path": str(target.resolve()),
                    "st_mtime": 0.0,
                    **meta,
                }
            )
        except ValidationError as e:
            self._set_error(
                {
                    "path": str(target.resolve()),
                    "error": "MemoryFileNode schema validation failed on append",
                    "details": e.errors(include_context=False, include_url=False),
                }
            )
            return

        new_post = frontmatter.Post(new_body, **meta)
        try:
            target.write_text(frontmatter.dumps(new_post), encoding="utf-8")
        except Exception as e:
            self._set_error({"path": str(target.resolve()), "error": f"write failed: {e}"})
            return

        self._set_ok(
            {
                "path": str(target.resolve()),
                "category": "event",
                "status": "active",
                "topics": meta["topics"],
                "materials": new_material_paths,
                "created": False,
                "action": "appended",
            }
        )
