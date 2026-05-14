"""Shared fixtures for the MCP profile tests.

Each suite uses one app per profile (boot cost ~2s) and a fresh temp
vault. The app is configured with `service.backend = http` so the
stdio MCP listener never starts — jobs are invoked directly via
`app.run_job`.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Make the repo importable when this module is loaded standalone
# (`python -m reme2.mcp.test` already has the path; direct imports
# from a fresh interpreter need this guard).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Eager-import side-effect modules so all @R.register() decorators run
# before Application introspects the registry.
import reme2  # noqa: E402,F401
import reme2.memory  # noqa: E402,F401  -- registers every agent-facing tool

from reme2.application import Application  # noqa: E402
from reme2.config import parse_args  # noqa: E402


CONFIG_DIR = _REPO_ROOT / "reme2" / "config"
PROFILES = {
    "expert": CONFIG_DIR / "expert.yaml",
    "service": CONFIG_DIR / "service.yaml",
}

# How many seed files `seed_vault` writes — checks that need to wait
# for indexing budget against this baseline.
SEED_FILE_COUNT = 3


@dataclass
class AppContext:
    """Bundle of everything a test function needs."""

    app: Application
    vault: Path
    jobs: list[str]

    @property
    def watcher(self):
        return self.app.context.components["file_watcher"]["default"]

    @property
    def file_store(self):
        return self.watcher.file_store

    def abs_path(self, *parts: str) -> str:
        return str((self.vault.joinpath(*parts)).resolve())


def seed_vault(vault: Path) -> None:
    """Drop a small connected topic graph (Alice ↔ Bob ↔ Project X)."""
    (vault / "topics" / "Alice").mkdir(parents=True, exist_ok=True)
    (vault / "topics" / "Alice" / "Alice.md").write_text(
        "---\ntitle: Alice\ncategory: profile\ntags: [person]\n---\n"
        "# Alice\n\nAlice works on [[Project X]] with [[Bob]].\n"
        "[author:: [[Alice]]]\n",
        encoding="utf-8",
    )
    (vault / "topics" / "Bob").mkdir(parents=True, exist_ok=True)
    (vault / "topics" / "Bob" / "Bob.md").write_text(
        "---\ntitle: Bob\ncategory: profile\ntags: [person]\n---\n"
        "# Bob\n\nBob collaborates with [[Alice]] on [[Project X]].\n",
        encoding="utf-8",
    )
    (vault / "topics" / "Project X").mkdir(parents=True, exist_ok=True)
    (vault / "topics" / "Project X" / "Project X.md").write_text(
        "---\ntitle: Project X\ncategory: concept\n---\n"
        "# Project X\n\nA major initiative led by [[Alice]] with [[Bob]].\n",
        encoding="utf-8",
    )


async def wait_for_index(watcher, expected_min: int, timeout_s: float = 15.0) -> None:
    """Poll `len(file_store)` until it reaches `expected_min` and stays
    stable across 4 ticks. Raises if the budget runs out — a watcher
    that never indexed seed files is a hard wiring failure, not a
    soft check."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    last = -1
    stable = 0
    while asyncio.get_event_loop().time() < deadline:
        now = len(watcher.file_store)
        if now == last and now >= expected_min:
            stable += 1
            if stable >= 4:
                return
        else:
            stable = 0
            last = now
        await asyncio.sleep(0.25)
    raise RuntimeError(
        f"watcher did not reach >= {expected_min} files within {timeout_s}s " f"(last seen: {last})",
    )


def decode(resp) -> object:
    """Job answers come back as JSON strings — decode for inspection.
    Pass through dicts / lists if the step already returned native types."""
    if isinstance(resp.answer, (dict, list)):
        return resp.answer
    if isinstance(resp.answer, str):
        try:
            return json.loads(resp.answer)
        except (json.JSONDecodeError, TypeError):
            return resp.answer
    return resp.answer


async def build_app(profile_path: Path, vault: Path) -> Application:
    """Boot an Application against `vault` using the given profile.

    Forces `service.backend = http` so the stdio MCP listener never
    starts (we only call jobs directly), and rewrites the watcher /
    file_store paths to point at the temp vault.
    """
    _, cfg = parse_args(
        "start",
        f"config={profile_path}",
        f"components.file_store.default.working_dir={vault}",
        f"components.file_store.default.db_path={vault}/.reme",
    )
    cfg["service"] = {"backend": "http"}
    app = Application(**cfg)
    await app.start()
    return app


async def make_context(profile: str) -> tuple[AppContext, Path]:
    """Build a fresh temp vault + app for `profile`, wait for indexing.

    The caller owns the temp dir lifecycle — returns its Path so it can
    be `shutil.rmtree`d after `app.close()`.
    """
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix=f"reme2-mcp-{profile}-"))
    vault = tmp / "vault"
    vault.mkdir()
    seed_vault(vault)
    app = await build_app(PROFILES[profile], vault)
    await wait_for_index(
        app.context.components["file_watcher"]["default"],
        expected_min=SEED_FILE_COUNT,
    )
    jobs = sorted(app.context.jobs.keys())
    return AppContext(app=app, vault=vault, jobs=jobs), tmp
