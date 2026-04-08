from agentscope.message import Msg

from reme_cli import Application


class ReMeCli(Application):

    async def init(self, **kwargs) -> None:
        """Initialize the application."""
        ...

    async def read(self, file: str, **kwargs) -> None:
        """Read a note file."""
        # obsidian read file="My Note"
        ...

    async def create(self, name: str, content: str, template: str, silent: bool, **kwargs) -> None:
        """Create a new note."""
        # obsidian create name="New Note" content="# Hello" template="Template" silent
        ...

    async def append(self, file: str, content: str, **kwargs) -> None:
        """Append content to a note."""
        # obsidian append file="My Note" content="New line"
        ...

    async def search(self, query: str, limit: int, **kwargs) -> None:
        """Search for notes."""
        # obsidian search query="search term" limit=10
        ...

    async def daily_read(self, **kwargs) -> None:
        """Read daily note."""
        # obsidian daily:read
        ...

    async def daily_append(self, content: str, **kwargs) -> None:
        """Append content to daily note."""
        # obsidian daily:append content="- [ ] New task"
        ...

    async def property_set(self, name: str, value: str, file: str, **kwargs) -> None:
        """Set a property on a note."""
        # obsidian property:set name="status" value="done" file="My Note"
        ...

    async def tasks(self, daily: bool, todo: bool, **kwargs) -> None:
        """Manage tasks."""
        # obsidian tasks daily todo
        ...

    async def tags(self, sort: str, counts: bool, **kwargs) -> None:
        """Manage tags."""
        # obsidian tags sort=count counts
        ...

    async def backlinks(self, file: str, **kwargs) -> None:
        """Get backlinks for a note."""
        # obsidian backlinks file="My Note"
        ...

    async def summary(self, messages: list[Msg], **kwargs):
        ...

    async def dream(self) -> dict:
        ...

    async def proactive(self, messages: list[Msg], **kwargs) -> dict:
        ...


def main():
    """Main entry point for running ReMe from command line."""
    ReMeCli(*sys.argv[1:], config_path="service").run_service()


if __name__ == "__main__":
    main()
