"""Build the loopback deployment's allowlisted ReMe configuration."""

from __future__ import annotations

from pathlib import Path

import yaml


REME_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REME_ROOT / "reme" / "config" / "default.yaml"
OUTPUT_CONFIG = Path(__file__).resolve().with_name("safe.yaml")

ALLOWED_JOBS = (
    "index_update_loop",
    "version",
    "health_check",
    "status",
    "help",
    "traverse",
    "reindex",
    "search",
    "node_search",
    "daily_list",
    "daily_reindex",
    "frontmatter_delete",
    "frontmatter_read",
    "frontmatter_update",
    "stat",
    "list",
    "move",
    "delete",
    "read",
    "read_image",
    "write",
    "daily_write",
    "edit",
)

ALLOWED_COMPONENTS = (
    "tokenizer",
    "file_graph",
    "file_catalog",
    "file_chunker",
    "keyword_index",
    "file_store",
)


def select(mapping: dict, names: tuple[str, ...], label: str) -> dict:
    missing = [name for name in names if name not in mapping]
    if missing:
        raise RuntimeError(f"Default config is missing required {label}: {missing}")
    return {name: mapping[name] for name in names}


def build_safe_config() -> dict:
    with DEFAULT_CONFIG.open(encoding="utf-8") as source:
        config = yaml.safe_load(source)

    config["jobs"] = select(config["jobs"], ALLOWED_JOBS, "jobs")
    config["components"] = select(config["components"], ALLOWED_COMPONENTS, "components")
    return config


def validate_safe_config(config: dict) -> None:
    if tuple(config.get("jobs", {})) != ALLOWED_JOBS:
        raise RuntimeError("Generated job set does not match the deployment allowlist")
    if tuple(config.get("components", {})) != ALLOWED_COMPONENTS:
        raise RuntimeError("Generated component set does not match the deployment allowlist")


def main() -> None:
    config = build_safe_config()
    validate_safe_config(config)
    OUTPUT_CONFIG.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    with OUTPUT_CONFIG.open(encoding="utf-8") as generated:
        validate_safe_config(yaml.safe_load(generated))
    print(f"Wrote {OUTPUT_CONFIG}")


if __name__ == "__main__":
    main()
