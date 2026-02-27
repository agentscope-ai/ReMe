"""Translate operation for translating text."""

from pathlib import Path

from loguru import logger

from ..core.enumeration import Role
from ..core.op import BaseOp
from ..core.schema import Message


class TranslateTs(BaseOp):
    """
    Translate operation for translating text.

    reme2 backend=cmd cmd.flow="TranslateTs()" cmd.params.target_dir=""
    """

    async def execute_single_file(self, ts_file: Path):
        """Translate a single ts file."""
        ts_code = ts_file.read_text(encoding="utf-8")
        logger.info(f"Translating {ts_file}")

        def parse_python(assistant_message: Message):
            python_code = assistant_message.content
            assert "```python" in python_code, "Invalid python code"
            python_code = python_code.split("```python", 1)[1]
            python_code_split = python_code.split("```")
            python_code = "```".join(python_code_split[:-1])
            return python_code.strip(), assistant_message.content.strip()

        output = await self.llm.chat(
            messages=[
                Message(
                    role=Role.USER,
                    content=self.prompt_format(prompt_name="translate_prompt", ts_code=ts_code),
                ),
            ],
            callback_fn=parse_python,
        )

        ts_file.with_suffix(".py").write_text(output[0], encoding="utf-8")
        ts_file.with_suffix(".txt").write_text(output[1], encoding="utf-8")
        logger.info(f"Translate {ts_file} complete.")

    async def execute(self):
        """Execute the operation."""
        target_dir = Path(self.context.target_dir)
        ts_files = [p for p in target_dir.rglob("*.ts") if p.is_file() and not p.name.endswith(".test.ts")]
        logger.info(f"Translating {target_dir}, finding {len(ts_files)} ts files")

        for ts_file in ts_files:
            self.submit_async_task(self.execute_single_file, ts_file)

        await self.join_async_tasks()
