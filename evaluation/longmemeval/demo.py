"""Demo: drive the three demo.yaml jobs through Application directly (no service).

demo.yaml defines:
  - demo         : query → echo (two-step pipeline)
  - add          : a + b → sum
  - stream_demo  : query → char-by-char streaming

Run:
    python evaluation/longmemeval/demo.py
"""

import asyncio
import tempfile
from pathlib import Path


async def main():
    # ── 0. Import reme (triggers @R.register for all demo steps) ──────────
    from reme import Application
    from reme.config import resolve_app_config

    # ── 1. Build config ──────────────────────────────────────────────────
    # demo.yaml doesn't touch workspace files, but Application still needs
    # a workspace_dir to exist.  Use a throwaway temp directory.
    with tempfile.TemporaryDirectory() as tmp:
        workspace_dir = str(Path(tmp) / ".reme")

        cfg = resolve_app_config(
            config="demo.yaml",              # load reme/config/demo.yaml
            workspace_dir=workspace_dir,
            log_to_console=True,
            log_to_file=False,
            enable_logo=True,
        )

        # ── 2. Create and start Application ──────────────────────────────
        app = Application(**cfg)
        await app.start()

        try:
            # ── 3. Run "demo" job ────────────────────────────────────────
            # demo_echo_step1: normalises query, scales min_score × 0.9
            # demo_echo_step2: echoes the processed result
            print("\n" + "=" * 60)
            print("[Job 1] demo")
            print("=" * 60)
            resp = await app.run_job("demo", query="  Hello ReMe  ", min_score=0.8)
            print(f"  success : {resp.success}")
            print(f"  answer  : {resp.answer}")
            print(f"  metadata: {resp.metadata}")
            # 0.8 * 0.9 = 0.7200000000000001 (IEEE 754)
            assert "hello reme" in resp.answer
            assert "0.72" in resp.answer

            # ── 4. Run "add" job ─────────────────────────────────────────
            print("\n" + "=" * 60)
            print("[Job 2] add")
            print("=" * 60)
            resp = await app.run_job("add", a=3.14, b=2.86)
            print(f"  success : {resp.success}")
            print(f"  answer  : {resp.answer}")
            print(f"  metadata: {resp.metadata}")
            assert float(resp.answer) == 6.0

            # ── 5. Run "stream_demo" job ─────────────────────────────────
            # Uses run_stream_job which yields StreamChunks.
            print("\n" + "=" * 60)
            print("[Job 3] stream_demo")
            print("=" * 60)
            chunks = []
            async for chunk in app.run_stream_job(
                "stream_demo",
                query="Hi!",
                repeat=3,
                interval=0.01,
            ):
                chunks.append(chunk)
            text = "".join(str(c.chunk) for c in chunks if c.chunk)
            print(f"  streamed {len(chunks)} chunk(s), total {len(text)} chars")
            print(f"  text: {text!r}")
            assert text == "Hi!" * 3

            print("\n" + "=" * 60)
            print("All demo jobs passed!")
            print("=" * 60)

        finally:
            await app.close()


if __name__ == "__main__":
    asyncio.run(main())
