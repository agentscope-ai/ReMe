"""ReMeFb summary接口测试。

本模块测试ReMeFb类的summary()方法，该方法提供了
将用户个人信息存储到记忆文件的高级接口。

summary函数应该能够让LLM：
1. 从用户消息中提取个人信息（姓名、偏好、需求）
2. 调用文件系统工具（WriteTool、EditTool）存储这些信息
3. 为未来的对话维护个性化记忆
"""

import asyncio
import shutil
from pathlib import Path

from reme import ReMeFb
from reme.core.enumeration import Role
from reme.core.schema import Message


def print_messages(messages: list[Message], title: str = "消息列表", max_content_len: int = 150):
    """打印消息及其角色和内容。

    Args:
        messages: 要打印的消息列表
        title: 消息列表的标题
        max_content_len: 显示的最大内容长度（超过则截断）
    """
    print(f"\n{title}：（共 {len(messages)} 条）")
    print("-" * 80)
    for i, msg in enumerate(messages):
        content = str(msg.content)
        if len(content) > max_content_len:
            content = content[:max_content_len] + "..."
        print(f"  [{i}] {msg.role.value:10s} [{msg.time_created}]: {content}")
    print("-" * 80)


def print_result(result: dict, title: str = "执行结果"):
    """打印summary()调用的结果。

    Args:
        result: summary()返回的结果字典
        title: 结果部分的标题
    """
    print(f"\n{'=' * 80}")
    print(f"{title}：")
    print(f"  成功: {result.get('success')}")
    print(f"  跳过: {result.get('skipped', False)}")

    tools_used = result.get("tools", [])
    print(f"  调用工具数: {len(tools_used)}")

    if tools_used:
        print("\n  工具使用详情：")
        for i, tool in enumerate(tools_used):
            print(f"    [{i}] 工具: {tool.name} 参数: {tool.tool_call.arguments}")

    answer = result.get("answer", "")
    if answer:
        answer_preview = answer[:300] + "..." if len(answer) > 300 else answer
        print(f"\n  回答: {answer_preview}")

    print(f"{'=' * 80}")


def delete_memory_file(working_dir: str, date: str):
    """删除指定日期的记忆文件。

    Args:
        working_dir: 工作目录路径
        date: 日期格式 %Y-%m-%d
    """
    memory_path = Path(working_dir) / "memory" / f"{date}.md"
    if memory_path.exists():
        memory_path.unlink()
        print(f"✓ 已删除记忆文件：{memory_path}")
    else:
        print(f"  记忆文件不存在：{memory_path}")


def check_memory_file(working_dir: str, date: str):
    """检查记忆文件是否存在并打印其内容。

    Args:
        working_dir: 工作目录路径
        date: 日期格式 %Y-%m-%d
    """
    memory_path = Path(working_dir) / "memory" / f"{date}.md"
    if memory_path.exists():
        print("✓ 记忆文件存在：")
        print("内容预览：")
        print("-" * 80)
        content = memory_path.read_text()
        print(content[:500] + ("..." if len(content) > 500 else ""))
        print("-" * 80)
    else:
        print(f"\n✗ 记忆文件不存在：{memory_path}")


async def test_summary_first_write():
    """测试1：删除记忆文件，首次summary应该直接写入。

    测试当指定日期没有记忆文件时，summary会创建新文件并直接写入信息。
    """
    print("\n" + "=" * 80)
    print("测试1：Summary - 首次写入（直接写入）")
    print("=" * 80)

    test_date = "2024-01-15"
    working_dir = ".reme_test1"

    # 测试前清理
    if Path(working_dir).exists():
        shutil.rmtree(working_dir)

    reme_fs = ReMeFb(enable_logo=False, working_dir=working_dir, vector_store=None)
    await reme_fs.start()

    # 确保记忆文件已删除
    delete_memory_file(working_dir, test_date)

    # 创建time_created与日期对齐的消息
    messages = [
        Message(
            role=Role.USER,
            content="你好！我叫小明，是一名软件工程师。",
            time_created=f"{test_date} 09:30:00",
        ),
        Message(
            role=Role.ASSISTANT,
            content="很高兴认识你，小明！有什么我可以帮你的吗？",
            time_created=f"{test_date} 09:30:05",
        ),
        Message(
            role=Role.USER,
            content="我热爱Python编程，喜欢做AI项目。空闲时间我还喜欢看科幻小说。",
            time_created=f"{test_date} 09:31:00",
        ),
    ]

    print_messages(messages, "输入消息", max_content_len=200)

    print("\n预期结果：")
    print("  - summary前记忆文件不存在")
    print("  - LLM应该调用WriteTool创建新的记忆文件")
    print("  - 记忆文件应该包含用户的姓名、职业和兴趣")

    result = await reme_fs.summary(
        messages=messages,
        date=test_date,
    )

    print_result(result, "SUMMARY执行结果")
    check_memory_file(working_dir, test_date)

    await reme_fs.close()

    # 测试后清理
    if Path(working_dir).exists():
        shutil.rmtree(working_dir)


async def test_summary_complementary_info():
    """测试2：删除记忆文件，两次summary补充信息且不冲突。

    测试当进行两次summary时，第二次summary添加补充信息而不产生冲突。
    """
    print("\n" + "=" * 80)
    print("测试2：Summary - 补充信息（无冲突）")
    print("=" * 80)

    test_date = "2024-02-20"
    working_dir = ".reme_test2"

    # 测试前清理
    if Path(working_dir).exists():
        shutil.rmtree(working_dir)

    reme_fs = ReMeFb(enable_logo=False, working_dir=working_dir, vector_store=None)
    await reme_fs.start()

    # 确保记忆文件已删除
    delete_memory_file(working_dir, test_date)

    # 第一批消息
    messages1 = [
        Message(
            role=Role.USER,
            content="你好！我叫小红，是一名数据科学家。",
            time_created=f"{test_date} 10:00:00",
        ),
        Message(
            role=Role.ASSISTANT,
            content="你好小红！很高兴认识你。",
            time_created=f"{test_date} 10:00:05",
        ),
        Message(
            role=Role.USER,
            content="我主要从事机器学习和深度学习模型的研究工作。",
            time_created=f"{test_date} 10:01:00",
        ),
    ]

    print_messages(messages1, "第一批消息", max_content_len=200)

    print("\n【第一次Summary】")
    result1 = await reme_fs.summary(
        messages=messages1,
        date=test_date,
    )

    print_result(result1, "第一次SUMMARY执行结果")
    check_memory_file(working_dir, test_date)

    # 第二批消息，包含补充信息
    messages2 = [
        Message(
            role=Role.USER,
            content="周末的时候我还喜欢爬山和摄影。",
            time_created=f"{test_date} 14:30:00",
        ),
        Message(
            role=Role.ASSISTANT,
            content="听起来很不错！这些爱好可以很好地平衡工作。",
            time_created=f"{test_date} 14:30:05",
        ),
        Message(
            role=Role.USER,
            content="我最喜欢的编程语言是Python，经常使用PyTorch框架。",
            time_created=f"{test_date} 14:31:00",
        ),
    ]

    print_messages(messages2, "第二批消息（补充信息）", max_content_len=200)

    print("\n【第二次Summary】")
    print("预期结果：")
    print("  - LLM应该调用EditTool添加补充信息")
    print("  - 新信息：爱好（爬山、摄影）、喜欢的语言（Python）、工具（PyTorch）")
    print("  - 与已有信息无冲突（姓名：小红、职业：数据科学家、技能：ML/DL）")

    result2 = await reme_fs.summary(
        messages=messages2,
        date=test_date,
    )

    print_result(result2, "第二次SUMMARY执行结果")
    check_memory_file(working_dir, test_date)

    await reme_fs.close()

    # 测试后清理
    if Path(working_dir).exists():
        shutil.rmtree(working_dir)


async def test_summary_conflicting_info():
    """测试3：删除记忆文件，两次summary包含冲突和新增信息。

    测试当进行两次summary时，第二次summary包含部分冲突信息和部分新信息。
    """
    print("\n" + "=" * 80)
    print("测试3：Summary - 冲突信息和新增信息")
    print("=" * 80)

    test_date = "2024-03-10"
    working_dir = ".reme_test3"

    # 测试前清理
    if Path(working_dir).exists():
        shutil.rmtree(working_dir)

    reme_fs = ReMeFb(enable_logo=False, working_dir=working_dir, vector_store=None)
    await reme_fs.start()

    # 确保记忆文件已删除
    delete_memory_file(working_dir, test_date)

    # 第一批消息
    messages1 = [
        Message(
            role=Role.USER,
            content="你好！我叫李华，是一名软件工程师，目前在北京工作。",
            time_created=f"{test_date} 08:00:00",
        ),
        Message(
            role=Role.ASSISTANT,
            content="你好李华！很高兴认识你。",
            time_created=f"{test_date} 08:00:05",
        ),
        Message(
            role=Role.USER,
            content="我喜欢用JavaScript和React开发，主要做Web应用。",
            time_created=f"{test_date} 08:01:00",
        ),
    ]

    print_messages(messages1, "第一批消息", max_content_len=200)

    print("\n【第一次Summary】")
    result1 = await reme_fs.summary(
        messages=messages1,
        date=test_date,
    )

    print_result(result1, "第一次SUMMARY执行结果")
    check_memory_file(working_dir, test_date)

    # 第二批消息，包含冲突和新增信息
    messages2 = [
        Message(
            role=Role.USER,
            content="其实上个月我已经搬到上海了，这边的机会更好。",
            time_created=f"{test_date} 16:00:00",
        ),
        Message(
            role=Role.ASSISTANT,
            content="哦，恭喜你搬到新城市！",
            time_created=f"{test_date} 16:00:05",
        ),
        Message(
            role=Role.USER,
            content="最近我也开始学Python和机器学习了。Web开发还在做，但也在探索AI项目。",
            time_created=f"{test_date} 16:01:00",
        ),
    ]

    print_messages(messages2, "第二批消息（冲突+新增信息）", max_content_len=200)

    print("\n【第二次Summary】")
    print("预期结果：")
    print("  - LLM应该调用EditTool更新冲突信息")
    print("  - 冲突信息：位置从'北京'变更为'上海'")
    print("  - 新增信息：学习Python、探索机器学习和AI项目")
    print("  - 保留信息：姓名（李华）、职业（软件工程师）、技能（JavaScript、React、Web开发）")

    result2 = await reme_fs.summary(
        messages=messages2,
        date=test_date,
    )

    print_result(result2, "第二次SUMMARY执行结果")
    check_memory_file(working_dir, test_date)

    await reme_fs.close()

    # 测试后清理
    if Path(working_dir).exists():
        shutil.rmtree(working_dir)


async def main():
    """运行时间对齐的summary接口测试。"""
    print("\n" + "=" * 80)
    print("ReMeFb Summary接口 - 时间对齐的记忆存储测试")
    print("=" * 80)
    print("\n本测试套件验证summary()函数：")
    print("  1. 正确处理消息中的time_created字段（%Y-%m-%d %H:%M:%S）")
    print("  2. 将消息时间戳与summary的date参数（%Y-%m-%d）对齐")
    print("  3. 处理首次写入、补充更新和冲突更新")
    print("\n测试场景：")
    print("  1. 首次写入 - 直接写入新的记忆文件")
    print("  2. 补充信息 - 添加不冲突的信息")
    print("  3. 冲突信息 - 更新冲突信息并添加新信息")
    print("=" * 80)

    # 测试1：首次写入
    await test_summary_first_write()

    # 测试2：补充信息
    await test_summary_complementary_info()

    # 测试3：冲突信息
    await test_summary_conflicting_info()

    print("\n" + "=" * 80)
    print("所有summary测试已完成！")
    print("=" * 80)
    print("\n注意：这些测试需要调用LLM来：")
    print("  - 分析用户消息中的个人信息")
    print("  - 决定存储或更新哪些信息")
    print("  - 调用合适的工具（WriteTool/EditTool）保存到记忆文件")
    print("\n运行前请确保已正确配置API密钥。")


if __name__ == "__main__":
    asyncio.run(main())
