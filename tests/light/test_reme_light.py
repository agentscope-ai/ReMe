"""测试 ReMeLight

演示 ReMeLight 的完整功能，并使用 AsMsgHandler 跟踪每步 Token 变化：
1. compact_tool_result - 压缩超长工具输出
2. compact_memory - 生成压缩摘要
3. summary_memory - 生成完整摘要并写入文件
4. pre_reasoning_hook - 推理前预处理钩子
5. memory_search - 语义搜索记忆
6. ReMeInMemoryMemory.estimate_tokens - 估算 Token 使用
7. ReMeInMemoryMemory.get_history_str - 获取格式化历史记录
"""

import asyncio
import logging
from test_utils import build_sample_messages, get_msg_handler
from reme.reme_light import ReMeLight


def print_token_change(_step_name: str, before: int, after: int):
    """打印 Token 变化统计。"""
    change = after - before
    change_pct = (change / before * 100) if before > 0 else 0
    print(f"  📊 Token 统计: {before:,} → {after:,} (变化: {change:+,}, {change_pct:+.1f}%)")


# ==================== 主测试流程 ====================
async def main():
    """测试 ReMeLight 的完整功能，并跟踪每步 Token 变化。"""
    # 初始化 AsMsgHandler 用于 Token 统计
    msg_handler = get_msg_handler()

    # 初始化 ReMeLight
    reme = ReMeLight(
        default_as_llm_config={"model_name": "qwen3.5-35b-a3b"},
        # default_embedding_model_config={"model_name": "text-embedding-v4"},
        default_file_store_config={"fts_enabled": True, "vector_enabled": False},
    )
    logging.getLogger("reme").setLevel(logging.WARNING)
    await reme.start()
    print("=" * 70)
    print("ReMeLight 已启动")
    print("=" * 70)

    # 构建模拟对话历史（包含超长 tool_result，确保超过 128K token）
    original_messages = build_sample_messages(include_large_tool_result=True)
    initial_tokens = msg_handler.count_msgs_token(original_messages)

    print(f"\n[原始消息]: {len(original_messages)} 条, {initial_tokens:,} tokens")
    print(f"  目标阈值: 128K = {128 * 1024:,} tokens")
    print(f"  超出阈值: {initial_tokens > 128 * 1024}")

    # ==================== 1. compact_tool_result ====================
    print("\n" + "=" * 70)
    print("[步骤 1] compact_tool_result - 压缩超长工具输出")
    print("=" * 70)

    # 重新获取原始消息
    messages = build_sample_messages(include_large_tool_result=True)
    tokens_before = msg_handler.count_msgs_token(messages)
    messages_after_step1 = await reme.compact_tool_result(messages)
    tokens_after = msg_handler.count_msgs_token(messages_after_step1)

    print(f"  消息数量: {len(messages)} → {len(messages_after_step1)}")
    print_token_change("compact_tool_result", tokens_before, tokens_after)

    # ==================== 2. compact_memory ====================
    print("\n" + "=" * 70)
    print("[步骤 2] compact_memory - 生成结构化压缩摘要")
    print("=" * 70)

    # 重新获取原始消息
    messages = build_sample_messages(include_large_tool_result=True)
    tokens_before = msg_handler.count_msgs_token(messages)
    compact_summary = await reme.compact_memory(
        messages=messages,
        previous_summary="",
    )
    summary_tokens = msg_handler.count_str_token(compact_summary)

    print(f"  输入消息 tokens: {tokens_before:,}")
    print(f"  压缩摘要长度: {len(compact_summary)} 字符, {summary_tokens:,} tokens")
    print(f"  压缩比: {summary_tokens / tokens_before * 100:.1f}%" if tokens_before > 0 else "  压缩比: N/A")
    print(f"  摘要预览: {compact_summary[:200]}..." if len(compact_summary) > 200 else f"  摘要: {compact_summary}")

    # ==================== 3. summary_memory ====================
    print("\n" + "=" * 70)
    print("[步骤 3] summary_memory - 生成完整摘要并写入文件")
    print("=" * 70)

    # 重新获取原始消息
    messages = build_sample_messages(include_large_tool_result=True)
    tokens_before = msg_handler.count_msgs_token(messages)
    summary_result = await reme.summary_memory(messages=messages)

    print(f"  输入消息 tokens: {tokens_before:,}")
    print(f"  摘要结果长度: {len(summary_result)} 字符")
    print(f"  摘要预览: {summary_result[:200]}..." if len(summary_result) > 200 else f"  摘要: {summary_result}")

    # ==================== 4. pre_reasoning_hook ====================
    print("\n" + "=" * 70)
    print("[步骤 4] pre_reasoning_hook - 推理前预处理")
    print("=" * 70)

    # 重新获取原始消息
    messages = build_sample_messages(include_large_tool_result=True)
    tokens_before = msg_handler.count_msgs_token(messages)
    processed_messages, compressed_summary = await reme.pre_reasoning_hook(
        messages=messages,
        system_prompt="你是一个有帮助的 AI 助手。",
        compressed_summary="",
        max_input_length=128000,
        compact_ratio=0.7,
        memory_compact_reserve=10000,
        enable_tool_result_compact=True,
        tool_result_compact_keep_n=3,
    )
    tokens_after = msg_handler.count_msgs_token(processed_messages)
    compressed_summary_tokens = msg_handler.count_str_token(compressed_summary)

    print(f"  消息数量: {len(messages)} → {len(processed_messages)}")
    print_token_change("pre_reasoning_hook", tokens_before, tokens_after)
    print(f"  压缩摘要: {len(compressed_summary)} 字符, {compressed_summary_tokens:,} tokens")
    print(f"  总上下文: {tokens_after + compressed_summary_tokens:,} tokens")

    # ==================== 5. memory_search ====================
    print("\n" + "=" * 70)
    print("[步骤 5] memory_search - 语义搜索记忆")
    print("=" * 70)

    search_result = await reme.memory_search(query="Python 版本偏好", max_results=5)
    if search_result.content:
        print(f"  搜索结果: {search_result.content}")
    else:
        print("  未找到相关记忆")

    # ==================== 6 & 7. ReMeInMemoryMemory ====================
    print("\n" + "=" * 70)
    print("[步骤 6] ReMeInMemoryMemory - 会话内存管理")
    print("=" * 70)

    # 重新获取原始消息
    messages = build_sample_messages(include_large_tool_result=True)
    memory = ReMeLight.get_in_memory_memory()
    for msg in messages:
        await memory.add(msg)
    print(f"  已添加 {len(messages)} 条原始消息到内存")

    # 6.1 estimate_tokens
    print("\n[6.1] estimate_tokens - 估算 Token 使用:")
    token_stats = await memory.estimate_tokens(max_input_length=128000)
    print(f"  - 总消息数: {token_stats['total_messages']}")
    print(f"  - 消息 Token 数: {token_stats['messages_tokens']:,}")
    print(f"  - 压缩摘要 Token 数: {token_stats['compressed_summary_tokens']:,}")
    print(f"  - 预估总 Token 数: {token_stats['estimated_tokens']:,}")
    print(f"  - 最大输入长度: {token_stats['max_input_length']:,}")
    print(f"  - 上下文使用率: {token_stats['context_usage_ratio']:.2f}%")

    # 6.2 get_history_str
    print("\n[6.2] get_history_str - 格式化历史记录:")
    history_str = await memory.get_history_str(max_input_length=128000)
    print(history_str[:1000] + "..." if len(history_str) > 1000 else history_str)

    # ==================== 等待后台任务完成 ====================
    print("\n" + "=" * 70)
    print("[步骤 7] 等待后台任务完成")
    print("=" * 70)
    await_result = await reme.await_summary_tasks()
    print(f"  后台任务完成，结果长度: {len(await_result)} 字符")

    # ==================== 总结 ====================
    print("\n" + "=" * 70)
    print("📊 Token 变化总结")
    print("=" * 70)
    print(f"  原始消息: {initial_tokens:,} tokens")
    print(f"  Step 1 compact_tool_result 后: {msg_handler.count_msgs_token(messages_after_step1):,} tokens")
    print(f"  Step 2 compact_memory 摘要: {summary_tokens:,} tokens")
    print(
        f"  Step 4 pre_reasoning_hook 后: {tokens_after:,} tokens + 摘要 {compressed_summary_tokens:,} "
        f"tokens = {tokens_after + compressed_summary_tokens:,} tokens",
    )
    print(
        f"  最大节省: {initial_tokens - tokens_after:,} "
        f"tokens ({(initial_tokens - tokens_after) / initial_tokens * 100:.1f}%)",
    )
    print(f"  目标阈值: {128 * 1024:,} tokens")

    # 关闭 ReMeLight
    await reme.close()
    print("\n" + "=" * 70)
    print("ReMeLight 已关闭")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
