"""``init_step`` — idempotently scaffold a reme vault.

Creates the minimal layout a vault_dir needs:

  daily/          hot, streaming workspace tier (per-day, per-slug folders)
  digest/       cold, curated knowledge tier (canonical entries)
  .gitignore      excludes reme_metadata / .reme_cache / logs / .env
  .env.example    template for LLM credentials (copy to .env, fill keys)

Idempotent — running twice is safe; existing files / directories are
never overwritten. The response payload reports which entries were
``created`` vs ``existed``.

Operates on ``self.vault_path`` (the configured vault root). The root
itself is created if missing.
"""

from __future__ import annotations

from ..base_step import BaseStep

from ...components import R


_GITIGNORE = """\
reme_metadata/
.reme_cache/
logs/
.env
"""

_ENV_EXAMPLE = """\
# LLM credentials (required for vector search + service-tier LLM loops).
# BM25-only fallback works if unset.
LLM_API_KEY=sk-xxxx
LLM_BASE_URL=https://xxxx/v1
LLM_MODEL_NAME=xxxx

# Optional — set only when embeddings use a different endpoint.
#EMBEDDING_API_KEY=sk-xxxx
#EMBEDDING_BASE_URL=https://xxxx/v1
#EMBEDDING_MODEL_NAME=text-embedding-v4
"""

_DIRS = ("daily", "digest")
_FILES = {
    ".gitignore": _GITIGNORE,
    ".env.example": _ENV_EXAMPLE,
}


@R.register("init_step")
class InitStep(BaseStep):
    """Scaffold daily/, digest/, .gitignore, .env.example in vault_path."""

    async def execute(self):
        assert self.context is not None
        root = self.vault_path
        root.mkdir(parents=True, exist_ok=True)

        created: list[str] = []
        existed: list[str] = []

        for name in _DIRS:
            path = root / name
            if path.exists():
                existed.append(f"{name}/")
            else:
                path.mkdir(parents=True)
                created.append(f"{name}/")

        for name, content in _FILES.items():
            path = root / name
            if path.exists():
                existed.append(name)
            else:
                path.write_text(content, encoding="utf-8")
                created.append(name)

        payload = {"root": str(root), "created": created, "existed": existed}
        self.logger.info(
            f"[{self.name}] init at {root}: created={len(created)}, existed={len(existed)}",
        )
        self.context.response.answer = (
            f"✅ Initialized vault at {root} " f"(created {len(created)}, existed {len(existed)})"
        )
        self.context.response.metadata["init"] = payload
        return self.context.response
