"""ReMe File System - Typer CLI"""

import asyncio
import os
from typing import AsyncGenerator

import typer
from prompt_toolkit import PromptSession

from .core.enumeration import ChunkEnum
from .core.op import BaseTool
from .core.schema import StreamChunk
from .core.tools import (
    BashTool,
    EditTool,
    LsTool,
    ReadTool,
    WriteTool,
    ExecuteCode,
    DashscopeSearch,
    TavilySearch,
)
from .core.utils import execute_stream_task, play_horse_easter_egg
from .memory.file_based import FbCli
from .memory.tools import MemorySearch
from .reme_fb import ReMeFb


class ReMeCli(ReMeFb):
    """ReMe Cli"""

    def __init__(self, *args, config_path: str = "cli", **kwargs):
        """Initialize ReMe with config."""
        super().__init__(*args, config_path=config_path, **kwargs)
        self.commands = {
            "/new": "Create a new conversation.",
            "/compact": "Compact messages into a summary.",
            "/exit": "Exit the application.",
            "/clear": "Clear the history.",
            "/help": "Show help.",
            "/horse": "A surprise.",
        }
        self.working_dir = self.service_config.working_dir

    async def chat_with_remy(self, tool_result_max_size: int = 100, **kwargs):
        """Interactive CLI chat with Remy using simple streaming output."""
        language = self.service_config.language
        typer.echo("")
        typer.secho(f"  🔤 ReMe language: {language or 'default'}", dim=True)
        tools: list[BaseTool] = [
            MemorySearch(
                vector_weight=self.service_config.metadata["vector_weight"],
                candidate_multiplier=self.service_config.metadata["candidate_multiplier"],
            ),
            BashTool(cwd=self.working_dir),
            LsTool(cwd=self.working_dir),
            ReadTool(cwd=self.working_dir),
            EditTool(cwd=self.working_dir),
            WriteTool(cwd=self.working_dir),
            ExecuteCode(),
        ]
        tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
        dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
        if tavily_api_key:
            tools.append(TavilySearch(name="web_search", language=language))
            typer.secho("  🔍 Found tavily_api_key, append Tavily search tool", dim=True)
        elif dashscope_api_key:
            tools.append(DashscopeSearch(name="web_search", language=language))
            typer.secho("  🔍 Found dashscope_api_key, append Dashscope search tool", dim=True)
        else:
            typer.secho("  ⚠️ No Tavily or Dashscope API key found, skip search tool", fg=typer.colors.YELLOW)

        fb_cli = FbCli(
            tools=tools,
            context_window_tokens=self.service_config.metadata["context_window_tokens"],
            reserve_tokens=self.service_config.metadata["reserve_tokens"],
            keep_recent_tokens=self.service_config.metadata["keep_recent_tokens"],
            working_dir=self.working_dir,
            language=language,
            **kwargs,
        )
        session = PromptSession()

        # Print welcome banner
        typer.echo("")
        typer.secho("┌────────────────────────────────────────┐", fg=typer.colors.BRIGHT_BLUE)
        typer.secho("│   🤖 Welcome to Remy Chat!             │", fg=typer.colors.BRIGHT_GREEN, bold=True)
        typer.secho("└────────────────────────────────────────┘", fg=typer.colors.BRIGHT_BLUE)
        typer.secho("💡 Tip: Type /help for commands, /exit to quit")
        typer.echo("")

        async def chat(q: str) -> AsyncGenerator[StreamChunk, None]:
            """Execute chat query and yield streaming chunks."""
            stream_queue = asyncio.Queue()
            task = asyncio.create_task(
                fb_cli.call(
                    query=q,
                    stream_queue=stream_queue,
                    service_context=self.service_context,
                ),
            )
            async for _chunk in execute_stream_task(
                stream_queue=stream_queue,
                task=task,
                task_name="cli",
                output_format="chunk",
            ):
                yield _chunk

        while True:
            try:
                # Get user input (async)
                user_input = await session.prompt_async("You: ")
                user_input = user_input.strip()
                if not user_input:
                    continue

                # Handle commands
                if user_input == "/exit":
                    break

                if user_input == "/new":
                    result = await fb_cli.new()
                    typer.secho(f"  ✅ {result} — Conversation reset.", fg=typer.colors.GREEN)
                    typer.echo("")
                    continue

                if user_input == "/compact":
                    result = await fb_cli.compact(force_compact=True)
                    typer.secho(f"  ✅ {result} — History compacted.", fg=typer.colors.GREEN)
                    typer.echo("")
                    continue

                if user_input == "/history":
                    typer.secho("📋 Formated History:", fg=typer.colors.BRIGHT_CYAN)
                    typer.secho("────────────────────────────────────────")
                    result = fb_cli.format_history()
                    print(result)
                    typer.secho("────────────────────────────────────────")
                    continue

                if user_input == "/clear":
                    fb_cli.messages.clear()
                    typer.secho("  ⚠️  History cleared.", fg=typer.colors.YELLOW)
                    typer.echo("")
                    continue

                if user_input == "/help":
                    typer.secho("  📖 Commands:", fg=typer.colors.BRIGHT_CYAN, bold=True)
                    for command, description in self.commands.items():
                        cmd_styled = typer.style(f"  {command:<12}", fg=typer.colors.MAGENTA, bold=True)
                        desc_styled = typer.style(description, dim=True)
                        typer.echo(f"{cmd_styled} {desc_styled}")
                    continue

                if user_input == "/horse":
                    play_horse_easter_egg()
                    continue

                # Stream processing state
                in_thinking = False
                in_answer = False

                try:
                    async for chunk in chat(user_input):
                        if chunk.chunk_type == ChunkEnum.THINK:
                            if not in_thinking:
                                print("\033[90mThinking: ", end="", flush=True)
                                in_thinking = True
                            print(chunk.chunk, end="", flush=True)

                        elif chunk.chunk_type == ChunkEnum.ANSWER:
                            if in_thinking:
                                print("\033[0m")  # reset color after thinking
                                in_thinking = False
                            if not in_answer:
                                print("\n  🤖 Remy: ", end="", flush=True)
                                in_answer = True
                            print(chunk.chunk, end="", flush=True)

                        elif chunk.chunk_type == ChunkEnum.TOOL:
                            if in_thinking:
                                print("\033[0m")  # reset color after thinking
                                in_thinking = False
                            print(f"\033[36m  -> {chunk.chunk}\033[0m")

                        elif chunk.chunk_type == ChunkEnum.TOOL_RESULT:
                            tool_name = chunk.metadata.get("tool_name", "unknown")
                            result = chunk.chunk
                            if len(result) > tool_result_max_size:
                                result = result[:tool_result_max_size] + f"... ({len(chunk.chunk)} chars total)"
                            print(f"\033[36m  -> Tool result for {tool_name}: {result.strip()}\033[0m")

                        elif chunk.chunk_type == ChunkEnum.ERROR:
                            typer.secho(f"\n  ❌ {chunk.chunk}", fg=typer.colors.RED, bold=True, err=True)
                            # Also log the full error metadata if available
                            detail = chunk.metadata.get("traceback", "")
                            if detail:
                                typer.secho(f"  {detail}", dim=True, err=True)

                        elif chunk.chunk_type == ChunkEnum.DONE:
                            break

                except Exception as e:
                    typer.secho(f"\n  ❌ Stream error: {e}", fg=typer.colors.RED, err=True)

                # End current streaming line
                print("\n")
                typer.secho("----------------------------------------\n", dim=True)

            except EOFError:
                break
            except KeyboardInterrupt:
                print("")
                typer.secho("  ⚠️  Interrupted.", fg=typer.colors.YELLOW)
                break
            except Exception as e:
                typer.secho(f"  ❌ Error: {e}", fg=typer.colors.RED, err=True)
                print("   Full traceback:")
                import traceback

                traceback.print_exc()

        typer.echo("")
        typer.secho("  👋 Goodbye! See you next time.", fg=typer.colors.BRIGHT_GREEN, bold=True)
        typer.echo("")


app = typer.Typer(
    help="🤖 ReMe File System — 基于 MD 文档的 FS Memory 对话 CLI",
    add_completion=False,
    no_args_is_help=True,
)


@app.command()
def cli():
    """Start interactive chat session with Remy.

    Supports commands: /exit, /new, /compact, /clear, /help, /horse
    """

    async def _run():
        async with ReMeCli(log_to_console=False) as reme:
            await reme.chat_with_remy()

    asyncio.run(_run())


def main():
    """Main function for testing the ReMeFs CLI."""
    app()


if __name__ == "__main__":
    main()
    
