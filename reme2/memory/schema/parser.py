"""Tolerant parser — never raises, always returns (memory_or_None, errors).

Used by:
    - Maintainer.lint  : surfaces schema violations on existing files
    - any read path    : turn raw frontmatter into a typed view when possible

For *write* paths (`sync`, Ingestor) call `MemoryFileNode.model_validate`
directly so validation errors propagate as exceptions and stop the write.
"""

from __future__ import annotations

from pydantic import ValidationError

from .memory import MemoryFileNode


def parse_frontmatter(raw: dict) -> tuple[MemoryFileNode | None, list[str]]:
    """Tolerant parse. Returns (parsed, errors).

    Success → (MemoryFileNode(...), [])
    Failure → (None, ["loc: msg", ...])

    Migration is automatic via MemoryFileNode's `model_validator(mode='before')`,
    so frontmatter that only carries the legacy `category` field still
    parses successfully.
    """
    if not isinstance(raw, dict):
        return None, [f"frontmatter must be a dict, got {type(raw).__name__}"]
    try:
        return MemoryFileNode.model_validate(raw), []
    except ValidationError as e:
        msgs: list[str] = []
        for err in e.errors(include_context=False, include_url=False):
            loc = ".".join(str(x) for x in err.get("loc", ()))
            msgs.append(f"{loc}: {err.get('msg', '')}".strip(": "))
        return None, msgs
    except Exception as e:  # noqa: BLE001 — defensive: never raise from a tolerant parser
        return None, [f"{type(e).__name__}: {e}"]
