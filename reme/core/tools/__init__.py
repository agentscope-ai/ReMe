"""tools"""

from .execute_code import ExecuteCode
from .execute_shell import ExecuteShell

# file tools
from .file.base_file_tool import BaseFileTool
from .file.bash_tool import BashTool
from .file.edit_tool import EditTool
from .file.find_tool import FindTool
from .file.grep_tool import GrepTool
from .file.ls_tool import LsTool
from .file.read_tool import ReadTool
from .file.write_tool import WriteTool

# search tools
from .search.dashscope_search import DashscopeSearch
from .search.mock_search import MockSearch
from .search.tavily_search import TavilySearch
from .think_tool import ThinkTool
from ..registry_factory import R

__all__ = [
    # base tools
    "ThinkTool",
    "ExecuteCode",
    "ExecuteShell",
    # file tools
    "BaseFileTool",
    "BashTool",
    "EditTool",
    "FindTool",
    "GrepTool",
    "LsTool",
    "ReadTool",
    "WriteTool",
    # search tools
    "DashscopeSearch",
    "TavilySearch",
    "MockSearch",
]

for name in __all__:
    tool_class = globals()[name]
    R.ops.register(tool_class)
