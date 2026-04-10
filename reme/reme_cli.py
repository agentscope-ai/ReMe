"""ReMe File System"""

import asyncio
import sys
from pathlib import Path

from prompt_toolkit import PromptSession

from .config import ReMeConfigParser
from .core import Application

from .core.utils import play_horse_easter_egg
from .memory.file_based.components import CliAgent


class ReMeCli(Application):
    """ReMe Cli"""

    def __init__(
        self,
        *args,
        working_dir: str = ".reme",
        config_path: str = "cli",
        enable_logo: bool = True,
        log_to_console: bool = True,
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
        default_as_llm_config: dict | None = None,
        default_embedding_model_config: dict | None = None,
        default_file_store_config: dict | None = None,
        default_token_counter_config: dict | None = None,
        default_file_watcher_config: dict | None = None,
        context_window_tokens: int = 128000,
        reserve_tokens: int = 36000,
        keep_recent_tokens: int = 20000,
        vector_weight: float = 0.7,
        candidate_multiplier: float = 3.0,
        **kwargs,
    ):
        """Initialize ReMe with config."""
        working_path = Path(working_dir)
        working_path.mkdir(parents=True, exist_ok=True)
        memory_path = working_path / "memory"
        memory_path.mkdir(parents=True, exist_ok=True)
        self.working_dir: str = str(working_path.absolute())

        default_file_watcher_config = default_file_watcher_config or {}
        if not default_file_watcher_config.get("watch_paths", None):
            default_file_watcher_config["watch_paths"] = [
                str(working_path / "MEMORY.md"),
                str(working_path / "memory.md"),
                str(memory_path),
            ]
        super().__init__(
            *args,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            embedding_api_key=embedding_api_key,
            embedding_base_url=embedding_base_url,
            working_dir=working_dir,
            config_path=config_path,
            enable_logo=enable_logo,
            log_to_console=log_to_console,
            parser=ReMeConfigParser,
            default_as_llm_config=default_as_llm_config,
            default_embedding_model_config=default_embedding_model_config,
            default_file_store_config=default_file_store_config,
            default_token_counter_config=default_token_counter_config,
            default_file_watcher_config=default_file_watcher_config,
            **kwargs,
        )

        self.service_config.metadata.setdefault("context_window_tokens", context_window_tokens)
        self.service_config.metadata.setdefault("reserve_tokens", reserve_tokens)
        self.service_config.metadata.setdefault("keep_recent_tokens", keep_recent_tokens)
        self.service_config.metadata.setdefault("vector_weight", vector_weight)
        self.service_config.metadata.setdefault("candidate_multiplier", candidate_multiplier)

        self.commands = {
            "/new": "Create a new conversation.",
            "/compact": "Compact messages into a summary.",
            "/exit": "Exit the application.",
            "/clear": "Clear the history.",
            "/help": "Show help.",
            "/horse": "A surprise.",
        }

    async def chat_with_remy(self, **kwargs):
        """Interactive CLI chat with Remy using simple streaming output."""
        language = self.service_config.language
        print(f"ReMe language={language or 'default'}")

        cli_agent = CliAgent(
            vector_weight=self.service_config.metadata["vector_weight"],
            candidate_multiplier=self.service_config.metadata["candidate_multiplier"],
            context_window_tokens=self.service_config.metadata["context_window_tokens"],
            reserve_tokens=self.service_config.metadata["reserve_tokens"],
            keep_recent_tokens=self.service_config.metadata["keep_recent_tokens"],
            working_dir=self.working_dir,
            language=language,
            **kwargs,
        )
        session = PromptSession()

        # Print welcome banner
        print("\n========================================")
        print("  Welcome to Remy Chat!")
        print("========================================\n")

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
                    result = await cli_agent.new()
                    print(f"{result}\nConversation reset\n")
                    continue

                if user_input == "/compact":
                    result = await cli_agent.compact(force_compact=True)
                    print(f"{result}\nHistory compacted.\n")
                    continue

                if user_input == "/history":
                    result = cli_agent.format_history()
                    print(f"Formated History:\n{result}\n")
                    continue

                if user_input == "/clear":
                    cli_agent.messages.clear()
                    print("History cleared.\n")
                    continue

                if user_input == "/help":
                    print("\nCommands:")
                    for command, description in self.commands.items():
                        print(f"  {command}: {description}")
                    continue

                if user_input == "/horse":
                    play_horse_easter_egg()
                    continue

                try:
                    await cli_agent.call(
                        query=user_input,
                        service_context=self.service_context,
                    )
                except Exception as e:
                    print(f"\nStream error: {e}")

                # End current streaming line
                print("\n")
                print("----------------------------------------\n")

            except EOFError:
                break
            except KeyboardInterrupt:
                print("\nInterrupted.")
                break
            except Exception as e:
                print(f"Error: {e}")
                import traceback

                traceback.print_exc()

        print("\nGoodbye!\n")


async def async_main():
    """Main function for testing the ReMeFs CLI."""
    async with ReMeCli(*sys.argv[1:], log_to_console=False) as reme:
        await reme.chat_with_remy()


def main():
    """Main function for testing the ReMeFs CLI."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
