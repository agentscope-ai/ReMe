"""Run the ReMe e2e loop against the real ./.reme workspace (no temp dir).

Same 5 stages as tests/integration/test_reme_e2e.py, but the workspace is
pinned to <repo>/.reme so the produced files survive for inspection.

    python tests/integration/run_reme_e2e_local.py
"""

import asyncio
import sys
from pathlib import Path

INTEGRATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = INTEGRATION_DIR.parents[1]
sys.path.insert(0, str(INTEGRATION_DIR))

# pylint: disable=wrong-import-position
from _workspace_fixture import WorkspaceEnv, temp_chdir  # noqa: E402
import test_reme_e2e as e2e  # noqa: E402

from reme.utils import load_env  # noqa: E402


async def main() -> None:
    load_env()
    workspace = REPO_ROOT / ".reme"
    workspace.mkdir(parents=True, exist_ok=True)
    env = WorkspaceEnv(workspace=workspace, workspace_dir=workspace)
    env.clean()  # wipe daily/digest/resource/metadata from any prior run

    print(f"[local] workspace = {workspace}")
    with temp_chdir(REPO_ROOT):
        reme = await env.make_reme()
        try:
            await e2e._run_loop(env, reme)  # noqa: SLF001
        finally:
            await env.close_all()
    print(f"\n[local] done — inspect files under: {workspace}")


if __name__ == "__main__":
    asyncio.run(main())
