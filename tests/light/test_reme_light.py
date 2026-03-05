"""测试 ReMeLight"""

import asyncio

from agentscope.message import Msg
from reme.reme_light import ReMeLight


# ==================== 消息创建辅助函数 ====================
def create_user_msg(content: str) -> Msg:
    """创建用户消息"""
    return Msg(name="user", role="user", content=content)


def create_assistant_msg(content: str) -> Msg:
    """创建助手消息"""
    return Msg(name="assistant", role="assistant", content=content)


def create_tool_use_msg(tool_id: str, tool_name: str, tool_input: dict) -> Msg:
    """创建工具调用消息"""
    return Msg(
        name="assistant",
        role="assistant",
        content=[
            {
                "type": "tool_use",
                "id": tool_id,
                "name": tool_name,
                "input": tool_input,
            },
        ],
    )


def create_tool_result_msg(tool_id: str, tool_name: str, output: str) -> Msg:
    """创建工具结果消息"""
    return Msg(
        name="tool",
        role="user",
        content=[
            {
                "type": "tool_result",
                "id": tool_id,
                "name": tool_name,
                "output": output,
            },
        ],
    )


def create_thinking_msg(thinking_content: str) -> Msg:
    """创建思考消息"""
    return Msg(
        name="assistant",
        role="assistant",
        content=[
            {
                "type": "thinking",
                "text": thinking_content,
            },
        ],
    )


# ==================== 构建模拟对话历史 ====================
def build_sample_messages() -> list[Msg]:
    """构建一段包含多种消息类型的模拟对话"""
    messages = [
        # 用户询问 Python 版本
        create_user_msg("我想设置一个 Python 开发环境，你有什么建议？"),
        # 助手思考
        create_thinking_msg("用户想要搭建 Python 开发环境，我需要了解他的需求和偏好..."),
        # 助手回复
        create_assistant_msg(
            "好的！我建议使用 Python 3.11 或 3.12 版本，它们性能更好且功能丰富。"
            "你希望用于什么类型的开发？Web、数据科学还是其他？",
        ),
        # 用户提供更多信息
        create_user_msg("主要是做 Web 开发，使用 FastAPI 框架。另外我喜欢用 pyenv 管理版本。"),
        # 助手调用工具查询
        create_tool_use_msg(
            tool_id="call_001",
            tool_name="search_web",
            tool_input={"query": "FastAPI Python version compatibility 2024"},
        ),
        # 工具返回结果（模拟较长的输出）
        create_tool_result_msg(
            tool_id="call_001",
            tool_name="search_web",
            output=(
                "FastAPI 官方推荐使用 Python 3.8+ 版本，但 3.11/3.12 性能最佳。\n"
                "主要依赖：\n"
                "- Starlette: ASGI 框架\n"
                "- Pydantic v2: 数据验证\n"
                "- Uvicorn: ASGI 服务器\n"
                "最新版本 FastAPI 0.109+ 完全支持 Python 3.12。\n"
                "建议搭配 uv 或 pip-tools 进行依赖管理。"
            ),
        ),
        # 助手总结建议
        create_assistant_msg(
            "根据查询结果，我的建议是：\n"
            "1. **Python 版本**: 使用 Python 3.11 或 3.12（通过 pyenv 安装）\n"
            "2. **框架**: FastAPI 0.109+ 完全兼容这些版本\n"
            "3. **依赖管理**: 推荐使用 uv（更快）或 pip-tools\n"
            "4. **ASGI 服务器**: Uvicorn 配合 gunicorn 用于生产环境\n\n"
            "需要我帮你生成一个项目模板吗？",
        ),
        # 用户确认偏好
        create_user_msg("好的，我决定用 Python 3.12 + FastAPI + uv。请记住我的这些偏好。"),
        # 助手确认
        create_assistant_msg(
            "已记录你的开发偏好：\n"
            "- Python 版本: 3.12 (通过 pyenv 管理)\n"
            "- Web 框架: FastAPI\n"
            "- 包管理器: uv\n"
            "以后有相关问题我会参考这些偏好给你建议！",
        ),
    ]
    return messages


# ==================== 主测试流程 ====================
async def main():
    """ReMeLight 主测试流程，演示完整的记忆管理功能。"""
    # 初始化 ReMeLight
    reme = ReMeLight(
        working_dir=".reme",  # 记忆文件存储目录
        max_input_length=128000,  # 模型上下文窗口（tokens）
        memory_compact_ratio=0.7,  # 达到 max_input_length * 0.7 时触发压缩
        language="zh",  # 摘要语言（zh / ""）
        tool_result_threshold=1000,  # 超过此字符数的工具输出自动转存
        retention_days=7,  # tool_result/ 文件保留天数
    )
    await reme.start()
    print("=" * 60)
    print("ReMeLight 已启动")
    print("=" * 60)

    # 构建模拟对话历史
    messages = build_sample_messages()
    print(f"\n[原始消息数量]: {len(messages)} 条")

    # 1. 压缩超长工具输出（防止工具结果撑爆上下文）
    print("\n" + "-" * 40)
    print("[步骤 1] 压缩超长工具输出...")
    messages = await reme.compact_tool_result(messages)
    print(f"处理后消息数量: {len(messages)} 条")

    # 2. 将历史对话压缩为结构化摘要（触发时机：上下文接近上限）
    print("\n" + "-" * 40)
    print("[步骤 2] 生成结构化压缩摘要...")
    summary = await reme.compact_memory(
        messages=messages,
        previous_summary="",  # 可传入上轮摘要，实现增量更新
    )
    print(f"压缩摘要:\n{summary[:500]}..." if len(summary) > 500 else f"压缩摘要:\n{summary}")

    # 3. 后台异步提交摘要任务（不阻塞对话，摘要写入 memory/YYYY-MM-DD.md）
    print("\n" + "-" * 40)
    print("[步骤 3] 提交后台异步摘要任务...")
    reme.add_async_summary_task(messages=messages)
    print("异步任务已提交")

    # 4. 语义搜索记忆（向量 + BM25 混合检索）
    print("\n" + "-" * 40)
    print("[步骤 4] 语义搜索记忆...")
    result = await reme.memory_search(query="Python 版本偏好", max_results=5)
    print(f"搜索结果: {result}")

    # 5. 获取会话内存实例（ReMeInMemoryMemory，管理单次对话的上下文）
    print("\n" + "-" * 40)
    print("[步骤 5] 获取会话内存实例并估算 Token 使用...")
    memory = reme.get_in_memory_memory()
    # 将消息添加到内存中以便估算
    for msg in messages:
        await memory.add(msg)
    token_stats = await memory.estimate_tokens()
    print(f"当前上下文使用率: {token_stats['context_usage_ratio']:.1f}%")
    print(f"消息 Token 数: {token_stats['messages_tokens']}")
    print(f"预估总 Token 数: {token_stats['estimated_tokens']}")

    # 6. 关闭前等待后台任务完成
    print("\n" + "-" * 40)
    print("[步骤 6] 等待后台任务完成...")
    summary_result = await reme.await_summary_tasks()
    print(f"后台摘要任务完成，结果长度: {len(summary_result)} 字符")

    # 关闭 ReMeLight
    await reme.close()
    print("\n" + "=" * 60)
    print("ReMeLight 已关闭")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
