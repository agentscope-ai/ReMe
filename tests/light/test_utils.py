"""Test utilities for copaw tests."""

import os

from agentscope.message import Msg, ThinkingBlock, TextBlock, ToolUseBlock, ToolResultBlock

from reme.memory.file_based.utils import AsMsgHandler


def get_token_counter():
    """Get HF token counter instance."""
    from reme.core.utils import get_hf_token_counter

    return get_hf_token_counter()


def get_msg_handler() -> AsMsgHandler:
    """Get AsMsgHandler instance."""
    return AsMsgHandler(token_counter=get_token_counter())


def get_dash_chat_model(model_name: str = "qwen3.5-plus"):
    """Get DashScope chat model instance."""
    from agentscope.model import OpenAIChatModel
    from reme.core.utils import load_env

    load_env()
    return OpenAIChatModel(
        api_key=os.environ["LLM_API_KEY"],
        client_kwargs={"base_url": os.environ["LLM_BASE_URL"]},
        model_name=model_name,
    )


def get_formatter():
    """Get formatter instance."""
    from agentscope.formatter import OpenAIChatFormatter

    return OpenAIChatFormatter()


def generate_large_code_content(target_tokens: int = 50000) -> str:
    """生成大量代码内容，用于测试超长 tool_result。

    Args:
        target_tokens: 目标 token 数（约 4 字符/token）

    Returns:
        生成的代码内容字符串
    """
    code_template = '''
# === File: src/module_{idx}/handlers.py ===
"""Handler module {idx} for processing requests."""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class RequestContext_{idx}:
    """Context for request processing in module {idx}."""
    request_id: str
    user_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None
    processed: bool = False
    error_message: Optional[str] = None


class Handler_{idx}:
    """Main handler class for module {idx}."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache: Dict[str, Any] = {{}}
        self.metrics: Dict[str, int] = {{
            "requests_processed": 0,
            "errors": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }}
        self._initialized = False
        logger.info(f"Handler_{idx} initialized with config: {{config}}")

    async def initialize(self) -> None:
        """Initialize the handler with async resources."""
        if self._initialized:
            logger.warning("Handler_{idx} already initialized")
            return

        # Simulate async initialization
        await asyncio.sleep(0.01)
        self._initialized = True
        logger.info("Handler_{idx} initialization complete")

    async def process_request(self, context: RequestContext_{idx}) -> Dict[str, Any]:
        """Process an incoming request.

        Args:
            context: The request context containing all request data

        Returns:
            Dict containing the response data
        """
        if not self._initialized:
            raise RuntimeError("Handler not initialized")

        self.metrics["requests_processed"] += 1

        try:
            # Check cache first
            cache_key = f"{{context.request_id}}_{{context.user_id}}"
            if cache_key in self.cache:
                self.metrics["cache_hits"] += 1
                return self.cache[cache_key]

            self.metrics["cache_misses"] += 1

            # Process the request
            result = await self._do_process(context)

            # Cache the result
            self.cache[cache_key] = result
            context.processed = True

            return result

        except Exception as e:
            self.metrics["errors"] += 1
            context.error_message = str(e)
            logger.exception(f"Error processing request {{context.request_id}}: {{e}}")
            raise

    async def _do_process(self, context: RequestContext_{idx}) -> Dict[str, Any]:
        """Internal processing logic."""
        # Simulate some processing
        await asyncio.sleep(0.001)

        return {{
            "status": "success",
            "request_id": context.request_id,
            "user_id": context.user_id,
            "processed_at": datetime.now().isoformat(),
            "module": "module_{idx}",
            "data": {{
                "result": f"Processed by handler_{idx}",
                "metadata": context.metadata,
            }}
        }}

    def get_metrics(self) -> Dict[str, int]:
        """Return current metrics."""
        return self.metrics.copy()

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self.cache.clear()
        self._initialized = False
        logger.info("Handler_{idx} cleaned up")

'''

    # 每个模块约 2000 字符 ≈ 500 tokens
    # 目标 target_tokens，需要 target_tokens / 500 个模块
    num_modules = max(1, target_tokens // 500)

    parts = [f"# 大型项目代码检索结果\n# 共找到 {num_modules} 个相关模块\n"]
    for i in range(num_modules):
        parts.append(code_template.format(idx=i))

    return "".join(parts)


def build_sample_messages(include_large_tool_result: bool = True) -> list[Msg]:
    """构建一段包含多种消息类型的模拟对话。

    Args:
        include_large_tool_result: 是否包含大型 tool_result，确保超过 128K token

    Returns:
        消息列表
    """
    messages = [
        Msg(
            name="user",
            role="user",
            content="我想设置一个 Python 开发环境，你有什么建议？",
        ),
        Msg(
            name="assistant",
            role="assistant",
            content=[
                ThinkingBlock(type="thinking", thinking="用户想要搭建 Python 开发环境，我需要了解他的需求和偏好..."),
                TextBlock(
                    type="text",
                    text="好的！我建议使用 Python 3.11 或 3.12 版本，它们性能更好且功能丰富。"
                    "你希望用于什么类型的开发？Web、数据科学还是其他？",
                ),
            ],
        ),
        Msg(name="user", role="user", content="主要是做 Web 开发，使用 FastAPI 框架。另外我喜欢用 pyenv 管理版本。"),
        Msg(
            name="assistant",
            role="assistant",
            content=[
                ThinkingBlock(type="thinking", thinking="用户要做 FastAPI Web 开发，我需要搜索相关信息..."),
                ToolUseBlock(
                    type="tool_use",
                    id="call_001",
                    name="search_web",
                    input={"query": "FastAPI Python version compatibility 2024"},
                ),
            ],
        ),
        Msg(
            name="system",
            role="system",
            content=[
                ToolResultBlock(
                    type="tool_result",
                    id="call_001",
                    name="search_web",
                    output=[
                        TextBlock(
                            type="text",
                            text="FastAPI 官方推荐使用 Python 3.8+ 版本，但 3.11/3.12 性能最佳。\n"
                            "主要依赖：\n"
                            "- Starlette: ASGI 框架\n"
                            "- Pydantic v2: 数据验证\n"
                            "- Uvicorn: ASGI 服务器\n"
                            "最新版本 FastAPI 0.109+ 完全支持 Python 3.12。\n"
                            "建议搭配 uv 或 pip-tools 进行依赖管理。",
                        ),
                    ],
                ),
            ],
        ),
        Msg(
            name="assistant",
            role="assistant",
            content=[
                ThinkingBlock(
                    type="thinking",
                    thinking="根据搜索结果，FastAPI 完全支持 Python 3.12，结合用户使用 pyenv 的习惯，我来给出完整的环境配置建议...",
                ),
                TextBlock(
                    type="text",
                    text="根据最新信息，我为你整理了 FastAPI 开发环境配置方案：\n\n"
                    "**1. Python 版本**\n"
                    "推荐 Python 3.12，性能最佳且完全兼容 FastAPI 0.109+。\n\n"
                    "**2. pyenv 安装步骤**\n"
                    "```bash\n"
                    "pyenv install 3.12.2\n"
                    "pyenv local 3.12.2\n"
                    "```\n\n"
                    "**3. 虚拟环境与依赖管理**\n"
                    "建议使用 `uv` 作为包管理器，它比 pip 快 10-100 倍。\n\n"
                    "需要我帮你生成项目初始化脚本吗？",
                ),
            ],
        ),
        Msg(name="user", role="user", content="好的，请帮我生成一个完整的项目初始化脚本，包括 pyproject.toml 配置。"),
        Msg(
            name="assistant",
            role="assistant",
            content=[
                ThinkingBlock(
                    type="thinking",
                    thinking="用户需要完整的项目初始化脚本，我需要创建 pyproject.toml 并设置好 FastAPI 相关依赖...",
                ),
                ToolUseBlock(
                    type="tool_use",
                    id="call_002",
                    name="create_file",
                    input={
                        "path": "pyproject.toml",
                        "content": "[project]\nname = 'my-fastapi-app'\nversion = '0.1.0'\nrequires-python = "
                        "'>=3.11'\ndependencies = ['fastapi>=0.109.0', 'uvicorn[standard]>=0.27.0', "
                        "'pydantic>=2.0']",
                    },
                ),
            ],
        ),
        Msg(
            name="system",
            role="system",
            content=[
                ToolResultBlock(
                    type="tool_result",
                    id="call_002",
                    name="create_file",
                    output=[
                        TextBlock(type="text", text="文件 pyproject.toml 已创建成功。"),
                    ],
                ),
            ],
        ),
        Msg(
            name="assistant",
            role="assistant",
            content=[
                ToolUseBlock(
                    type="tool_use",
                    id="call_003",
                    name="create_file",
                    input={
                        "path": "init_project.sh",
                        "content": "#!/bin/bash\nset -e\n\n# 使用 pyenv 设置 Python 版本\npyenv install 3.12.2 "
                        "--skip-existing\npyenv local 3.12.2\n\n# 创建虚拟环境\npython -m venv "
                        ".venv\nsource .venv/bin/activate\n\n# 安装 uv 并使用它安装依赖\npip install uv"
                        "\nuv pip install -e .\n\necho '环境初始化完成！'",
                    },
                ),
            ],
        ),
        Msg(
            name="system",
            role="system",
            content=[
                ToolResultBlock(
                    type="tool_result",
                    id="call_003",
                    name="create_file",
                    output=[
                        TextBlock(type="text", text="文件 init_project.sh 已创建成功。"),
                    ],
                ),
            ],
        ),
        Msg(
            name="assistant",
            role="assistant",
            content=[
                TextBlock(
                    type="text",
                    text="我已为你创建了两个文件：\n\n"
                    "1. **pyproject.toml** - 项目配置文件，包含 FastAPI 核心依赖\n"
                    "2. **init_project.sh** - 一键初始化脚本\n\n"
                    "运行以下命令即可初始化项目：\n"
                    "```bash\n"
                    "chmod +x init_project.sh && ./init_project.sh\n"
                    "```\n\n"
                    "还有什么需要帮助的吗？",
                ),
            ],
        ),
        Msg(name="user", role="user", content="太棒了！请帮我搜索一下项目中所有的 handler 相关代码。"),
    ]

    # 添加大型代码搜索结果（确保超过 128K token）
    if include_large_tool_result:
        # 生成超大的代码搜索结果，目标 ~140K tokens
        large_code_content = generate_large_code_content(target_tokens=140000)

        messages.extend(
            [
                Msg(
                    name="assistant",
                    role="assistant",
                    content=[
                        ThinkingBlock(
                            type="thinking",
                            thinking="用户要我搜索项目中的 handler 代码，我需要使用代码搜索工具...",
                        ),
                        ToolUseBlock(
                            type="tool_use",
                            id="call_004",
                            name="search_codebase",
                            input={"query": "handler class implementation"},
                        ),
                    ],
                ),
                Msg(
                    name="system",
                    role="system",
                    content=[
                        ToolResultBlock(
                            type="tool_result",
                            id="call_004",
                            name="search_codebase",
                            output=[
                                TextBlock(type="text", text=large_code_content),
                            ],
                        ),
                    ],
                ),
                Msg(
                    name="assistant",
                    role="assistant",
                    content=[
                        ThinkingBlock(type="thinking", thinking="搜索返回了大量 handler 代码，我需要为用户整理一下..."),
                        TextBlock(
                            type="text",
                            text="我已经找到了项目中所有的 handler 相关代码。\n\n"
                            "这些 handler 类包含：\n"
                            "- 请求处理逻辑\n"
                            "- 缓存管理\n"
                            "- 指标统计\n"
                            "- 异步初始化\n\n"
                            "你需要我详细解释某个具体的 handler 吗？",
                        ),
                    ],
                ),
            ],
        )

    # 添加更多对话
    messages.extend(
        [
            Msg(
                name="user",
                role="user",
                content="还有一个问题，我应该如何配置 VS Code 来获得最佳的 FastAPI 开发体验？",
            ),
            Msg(
                name="assistant",
                role="assistant",
                content=[
                    ThinkingBlock(
                        type="thinking",
                        thinking="用户询问 VS Code 配置，我需要推荐适合 FastAPI 开发的扩展和设置...",
                    ),
                    TextBlock(
                        type="text",
                        text="VS Code 的 FastAPI 开发配置建议：\n\n"
                        "**推荐扩展：**\n"
                        "- Python (Microsoft)\n"
                        "- Pylance - 类型检查和智能补全\n"
                        "- Ruff - 快速 linter 和 formatter\n"
                        "- REST Client - API 测试\n\n"
                        "**settings.json 配置：**\n"
                        "```json\n"
                        "{\n"
                        '  "python.defaultInterpreterPath": ".venv/bin/python",\n'
                        '  "[python]": {\n'
                        '    "editor.defaultFormatter": "charliermarsh.ruff",\n'
                        '    "editor.formatOnSave": true\n'
                        "  }\n"
                        "}\n"
                        "```\n\n"
                        "这样配置后，你就能获得完整的类型提示和自动格式化支持了！",
                    ),
                ],
            ),
        ],
    )

    return messages
