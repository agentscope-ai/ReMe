"""
测试场景：对比 DeltaFileWatcher 和 FullFileWatcher 的行为差异

场景流程：
1. 测试 FullFileWatcher - 完整更新模式
   - 创建初始文件
   - 修改文件
   - 验证数据库：所有 chunks 被重新创建

2. 测试 DeltaFileWatcher - 增量更新模式
   - 创建初始文件
   - 追加内容到文件
   - 验证数据库：只有新增的 chunks，旧 chunks 保留

3. 直接查询数据库验证更新结果
"""

import asyncio
import os
import tempfile
from datetime import datetime

from reme.core.embedding import OpenAIEmbeddingModel
from reme.core.enumeration import MemorySource
from reme.core.file_watcher.delta_file_watcher import DeltaFileWatcher
from reme.core.file_watcher.full_file_watcher import FullFileWatcher
from reme.core.file_store import SqliteFileStore
from reme.core.utils import load_env

load_env()

# 配置
DB_PATH_FULL = "./demo_memory_search/full_watcher.db"
DB_PATH_DELTA = "./demo_memory_search/delta_watcher.db"
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIMENSIONS = 64


def print_separator(title):
    """打印分隔符"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def create_file(workspace_dir: str, filename: str, content: str) -> str:
    """在工作空间创建文件"""
    file_path = os.path.join(workspace_dir, filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✓ 创建文件: {filename}")
    return file_path


def append_to_file(file_path: str, content: str):
    """追加内容到文件"""
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(content)
    print(f"✓ 追加内容到: {file_path}")


async def verify_database(store: SqliteFileStore, title: str):
    """验证数据库内容"""
    print_separator(title)

    # 查询文件列表
    files = await store.list_files(MemorySource.MEMORY)
    print(f"📁 数据库中的文件数量: {len(files)}\n")

    for file_path in files:
        print(f"文件: {file_path}")

        # 获取文件元数据
        file_meta = await store.get_file_metadata(file_path, MemorySource.MEMORY)
        if file_meta:
            print(f"  - Hash: {file_meta.hash[:16]}...")
            print(f"  - Size: {file_meta.size} bytes")
            print(f"  - Chunk count: {file_meta.chunk_count}")

        # 获取文件的所有 chunks
        chunks = await store.get_file_chunks(file_path, MemorySource.MEMORY)
        print(f"  - Chunks in database: {len(chunks)}")

        for i, chunk in enumerate(chunks, 1):
            print(f"    Chunk #{i}:")
            print(f"      ID: {chunk.id[:16]}...")
            print(f"      Lines: {chunk.start_line}-{chunk.end_line}")
            print(f"      Hash: {chunk.hash[:16]}...")
            print(f"      Text preview: {chunk.text[:100]}...")
            print(f"      Has embedding: {chunk.embedding is not None}")

        print()


async def test_full_file_watcher():
    """测试 FullFileWatcher - 完整更新模式"""
    print_separator("测试 1: FullFileWatcher (完整更新模式)")

    # 创建临时工作目录
    temp_workspace = tempfile.mkdtemp(prefix="full_watcher_")
    print(f"工作目录: {temp_workspace}\n")

    # 准备数据库
    os.makedirs(os.path.dirname(DB_PATH_FULL), exist_ok=True)

    # 初始化组件
    embedding_model = OpenAIEmbeddingModel(
        model_name=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
    )

    store = SqliteFileStore(
        db_path=DB_PATH_FULL,
        vec_ext_path="",
        embedding_model=embedding_model,
        fts_enabled=True,
    )
    await store.start()
    await store.clear_all()  # 清空数据库

    # 创建 FullFileWatcher
    watcher = FullFileWatcher(
        watch_paths=temp_workspace,
        file_store=store,
        chunk_tokens=200,
        chunk_overlap=20,
        recursive=True,
        suffix_filters=["md"],
    )

    try:
        # 阶段 1: 创建初始文件
        print("📝 阶段 1: 创建初始文件\n")
        _ = create_file(
            temp_workspace,
            "MEMORY.md",
            """# Python 基础

## 变量和数据类型
Python 是动态类型语言。

## 控制流
if、for、while 语句。
""",
        )

        # 启动 watcher
        await watcher.start()
        print("✓ FullFileWatcher 启动\n")

        # 等待文件被检测和处理
        await asyncio.sleep(2)

        # 验证数据库 - 初始状态
        await verify_database(store, "数据库验证 1.1: 初始文件索引后")

        # 阶段 2: 修改文件（非追加，而是完全修改）
        print("📝 阶段 2: 修改文件内容\n")
        create_file(
            temp_workspace,
            "MEMORY.md",
            """# Python 进阶

## 变量和数据类型
Python 是动态类型语言，支持多种数据类型。

## 控制流
if、for、while 语句用于控制程序流程。

## 函数
def 关键字定义函数。

## 类和对象
面向对象编程的核心概念。
""",
        )

        # 等待文件变化被检测和处理
        await asyncio.sleep(2)

        # 验证数据库 - 修改后
        await verify_database(store, "数据库验证 1.2: 文件修改后（完整更新）")

        print("📊 观察要点:")
        print("  - FullFileWatcher 每次修改都会删除所有旧 chunks")
        print("  - 然后重新创建所有新 chunks")
        print("  - Chunk IDs 会完全不同")

    finally:
        await watcher.close()
        await store.close()
        print(f"\n💡 工作目录: {temp_workspace}")
        print(f"💡 数据库: {DB_PATH_FULL}")


async def test_delta_file_watcher():
    """测试 DeltaFileWatcher - 增量更新模式"""
    print_separator("测试 2: DeltaFileWatcher (增量更新模式)")

    # 创建临时工作目录
    temp_workspace = tempfile.mkdtemp(prefix="delta_watcher_")
    print(f"工作目录: {temp_workspace}\n")

    # 准备数据库
    os.makedirs(os.path.dirname(DB_PATH_DELTA), exist_ok=True)

    # 初始化组件
    embedding_model = OpenAIEmbeddingModel(
        model_name=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
    )

    store = SqliteFileStore(
        db_path=DB_PATH_DELTA,
        vec_ext_path="",
        embedding_model=embedding_model,
        fts_enabled=True,
    )
    await store.start()
    await store.clear_all()  # 清空数据库

    # 创建 DeltaFileWatcher
    watcher = DeltaFileWatcher(
        watch_paths=temp_workspace,
        file_store=store,
        chunk_tokens=200,
        chunk_overlap=20,
        overlap_lines=2,
        recursive=True,
        suffix_filters=["md"],
    )

    try:
        # 阶段 1: 创建初始文件
        print("📝 阶段 1: 创建初始文件\n")
        test_file = create_file(
            temp_workspace,
            "MEMORY.md",
            """# Python 基础

## 变量和数据类型
Python 是动态类型语言。

## 控制流
if、for、while 语句。
""",
        )

        # 启动 watcher
        await watcher.start()
        print("✓ DeltaFileWatcher 启动\n")

        # 等待文件被检测和处理
        await asyncio.sleep(2)

        # 验证数据库 - 初始状态
        await verify_database(store, "数据库验证 2.1: 初始文件索引后")

        # 保存初始 chunk IDs
        initial_chunks = await store.get_file_chunks(test_file, MemorySource.MEMORY)
        initial_chunk_ids = {chunk.id for chunk in initial_chunks}
        print(f"📌 初始 chunk IDs: {len(initial_chunk_ids)} 个\n")

        # 阶段 2: 追加内容到文件（append-only）
        print("📝 阶段 2: 追加新内容到文件\n")
        append_to_file(
            test_file,
            """

## 函数
def 关键字定义函数。

## 类和对象
面向对象编程的核心概念。

## 模块和包
代码组织和复用。
""",
        )

        # 等待文件变化被检测和处理
        await asyncio.sleep(2)

        # 验证数据库 - 追加后
        await verify_database(store, "数据库验证 2.2: 追加内容后（增量更新）")

        # 检查哪些 chunks 被保留
        updated_chunks = await store.get_file_chunks(test_file, MemorySource.MEMORY)
        updated_chunk_ids = {chunk.id for chunk in updated_chunks}

        preserved_ids = initial_chunk_ids & updated_chunk_ids
        new_ids = updated_chunk_ids - initial_chunk_ids
        deleted_ids = initial_chunk_ids - updated_chunk_ids

        print("\n📊 Chunk 变化统计:")
        print(f"  - 保留的 chunks: {len(preserved_ids)} 个")
        print(f"  - 新增的 chunks: {len(new_ids)} 个")
        print(f"  - 删除的 chunks: {len(deleted_ids)} 个")

        print("\n📊 观察要点:")
        print("  - DeltaFileWatcher 检测到 append-only 模式")
        print("  - 保留了大部分旧 chunks（除了重叠部分）")
        print("  - 只处理和嵌入新增的内容")
        print("  - 节省了 embedding API 调用")

        # 阶段 3: 非追加式修改（触发完整更新）
        print("\n📝 阶段 3: 修改文件中间部分（非追加）\n")
        create_file(
            temp_workspace,
            "MEMORY.md",
            """# Python 完全改版

## 这是全新的内容
完全不同的文档结构。

## 新的章节
之前的内容都不见了。
""",
        )

        # 等待文件变化被检测和处理
        await asyncio.sleep(2)

        # 验证数据库 - 非追加修改后
        await verify_database(store, "数据库验证 2.3: 非追加修改后（回退到完整更新）")

        final_chunks = await store.get_file_chunks(test_file, MemorySource.MEMORY)
        final_chunk_ids = {chunk.id for chunk in final_chunks}

        print("\n📊 第二次修改后的 Chunk 统计:")
        print(f"  - 当前 chunks: {len(final_chunk_ids)} 个")
        print(f"  - 所有 chunk IDs 都是新的: {len(final_chunk_ids & updated_chunk_ids) == 0}")

        print("\n📊 观察要点:")
        print("  - DeltaFileWatcher 检测到非追加式修改")
        print("  - 自动回退到完整更新模式")
        print("  - 所有 chunks 被重新创建")

    finally:
        await watcher.close()
        await store.close()
        print(f"\n💡 工作目录: {temp_workspace}")
        print(f"💡 数据库: {DB_PATH_DELTA}")


async def compare_watchers():
    """对比两种 watcher 的性能和行为"""
    print_separator("对比分析")

    print("🔍 FullFileWatcher vs DeltaFileWatcher\n")

    print("FullFileWatcher 特点:")
    print("  ✓ 实现简单")
    print("  ✓ 适合频繁修改的文档")
    print("  ✓ 每次修改都保证完整性")
    print("  ✗ 每次都重新处理整个文件")
    print("  ✗ 更多的 embedding API 调用")

    print("\nDeltaFileWatcher 特点:")
    print("  ✓ 支持增量更新")
    print("  ✓ 节省 embedding API 调用")
    print("  ✓ 适合日志类追加文件")
    print("  ✓ 检测到非追加时自动回退")
    print("  ✗ 实现复杂")
    print("  ✗ 需要维护 cutoff line 逻辑")

    print("\n推荐使用场景:")
    print("  - 日志文件、聊天记录 → DeltaFileWatcher")
    print("  - 文档、代码文件 → FullFileWatcher")
    print("  - 不确定的场景 → DeltaFileWatcher (会自动回退)")


async def main():
    """主函数"""
    print_separator("FileWatcher 对比测试")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 测试 1: FullFileWatcher
    await test_full_file_watcher()

    # 测试 2: DeltaFileWatcher
    await test_delta_file_watcher()

    # 对比分析
    await compare_watchers()

    print_separator("测试完成")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("📂 生成的文件:")
    print(f"  - Full watcher DB: {DB_PATH_FULL}")
    print(f"  - Delta watcher DB: {DB_PATH_DELTA}")

    print("\n🔧 清理命令:")
    print(f"  rm -rf {os.path.dirname(DB_PATH_FULL)}")


if __name__ == "__main__":
    asyncio.run(main())
