"""Start a local Home application and prepare once: python prepare.py WORKSPACE."""

import asyncio
from pathlib import Path
import sys

from reme.application import Application


async def main(workspace: str):
    """Read WORKSPACE/prepare.yaml; a dictionary can instead be passed as spec=."""
    app = Application(
        workspace_dir=str(Path(workspace).resolve()),
        plugins=["home"],
        service={"backend": "cli"},
    )
    try:
        await app.start()
        response = await app.run_job("home_prepare", spec_path="prepare.yaml")
        if not response.success:
            raise RuntimeError(response.answer)
        print(response.answer)
    finally:
        await app.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
