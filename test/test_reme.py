"""Test module for ReMe memory system functionality."""

import asyncio

from reme import ReMe
from reme.core.schema import VectorNode, MemoryNode


async def test_reme():
    """Tests ReMe memory system with personal information storage and retrieval."""
    # 构建一段包含个人信息的对话
    reme = ReMe(default_vector_store_config={"collection_name": "reme"})
    await reme.start()
    # reme = await ReMe.create(vector_store={"collection_name": "reme"})

    await reme.default_vector_store.delete_all()

    messages = [
        {
            "role": "user",
            "content": "你好，我是张伟，今年28岁，是一名软件工程师。",
            "time_created": "2026-01-29 10:00:00",
        },
        {
            "role": "assistant",
            "content": "你好张伟！很高兴认识你。作为一名软件工程师，你主要从事什么方向的开发工作呢？",
            "time_created": "2026-01-29 10:00:05",
        },
        {
            "role": "user",
            "content": "我主要做后端开发，擅长Python和Go语言。最近在研究AI Agent相关的技术。",
            "time_created": "2026-01-29 10:00:30",
        },
        {
            "role": "assistant",
            "content": "很棒！Python和Go都是非常实用的语言。AI Agent是当前很热门的方向，你在这方面有什么具体的研究重点吗？",
            "time_created": "2026-01-29 10:00:35",
        },
        {
            "role": "user",
            "content": "我特别关注记忆系统的设计，希望能让AI Agent具有长期记忆能力。我的工作地点在北京，平时喜欢看技术博客和参加技术分享会。",
            "time_created": "2026-01-29 10:01:00",
        },
        {
            "role": "assistant",
            "content": "记忆系统确实是AI Agent的核心能力之一。北京有很多优秀的技术社区和活动，相信你能找到很多志同道合的朋友。",
            "time_created": "2026-01-29 10:01:05",
        },
        {
            "role": "user",
            "content": "是的，我每周末都会去参加一些技术沙龙。对了，我的邮箱是zhangwei@example.com，如果有好的技术资料可以发给我。",
            "time_created": "2026-01-29 10:01:30",
        },
        {
            "role": "assistant",
            "content": "好的，我记下了。保持学习的热情很重要，祝你在AI Agent领域的研究越来越深入！",
            "time_created": "2026-01-29 10:01:35",
        },
    ]

    print("=" * 60)
    print("步骤1: 开始总结对话并生成记忆")
    print("=" * 60)

    # 对对话进行总结，生成记忆
    await reme.summarize_memory(
        messages=messages,
        user_name="zhangwei",
        description="用户自我介绍和技术兴趣分享",
    )

    print("\n✓ 记忆总结完成")

    print("\n" + "=" * 60)
    print("步骤2: 查看已存储的记忆节点")
    print("=" * 60)

    # 列出所有存储的记忆节点
    nodes: list[VectorNode] = await reme.default_vector_store.list()
    for i, node in enumerate(nodes, 1):
        memory_node = MemoryNode.from_vector_node(node)
        print(f"{i} {memory_node.model_dump_json()}")

    print("\n" + "=" * 60)
    print("步骤3: 测试记忆检索 - 验证个人信息")
    print("=" * 60)

    # 测试问题1: 检索用户姓名
    query1 = "用户叫什么名字？"
    print(f"\n问题1: {query1}")
    result1 = await reme.retrieve_memory(query=query1, user_name="zhangwei")
    print(f"检索结果:\n{result1}")

    # 测试问题2: 检索技术背景
    query2 = "用户擅长什么编程语言和技术方向？"
    print(f"\n问题2: {query2}")
    result2 = await reme.retrieve_memory(query=query2, user_name="zhangwei")
    print(f"检索结果:\n{result2}")

    # 测试问题3: 检索个人信息
    query3 = "用户的工作地点和联系方式是什么？"
    print(f"\n问题3: {query3}")
    result3 = await reme.retrieve_memory(query=query3, user_name="zhangwei")
    print(f"检索结果:\n{result3}")

    # 测试问题4: 检索兴趣爱好
    query4 = "用户平时有什么爱好或活动？"
    print(f"\n问题4: {query4}")
    result4 = await reme.retrieve_memory(query=query4, user_name="zhangwei")
    print(f"检索结果:\n{result4}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_reme())
