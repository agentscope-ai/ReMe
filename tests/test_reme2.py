import asyncio
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from reme2.config.config_parser import _load_yaml
from reme2.reme import ReMe


async def test_full_watcher():
    """测试full watcher模式 - 处理所有文件类型"""
    print("\n=== 测试 Full Watcher 模式 ===")

    config = _load_yaml("paw")

    # 设置API密钥
    config["components"]["embedding_model"]["default"]["api_key"] = os.environ["EMBEDDING_API_KEY"]
    config["components"]["embedding_model"]["default"]["base_url"] = os.environ["EMBEDDING_BASE_URL"]

    config["components"]["as_llm"]["default"]["api_key"] = os.environ["LLM_API_KEY"]
    config["components"]["as_llm"]["default"]["base_url"] = os.environ["LLM_BASE_URL"]
    config["components"]["as_llm"]["default"]["model_name"] = os.environ["LLM_MODEL_NAME"]

    # 创建临时目录用于测试
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 创建测试文件
        md_file = temp_path / "test.md"
        txt_file = temp_path / "test.txt"
        py_file = temp_path / "test.py"

        md_content = """# Test Markdown File

This is a test markdown file for full watcher.

## Section
Some content here about Python programming.
"""

        txt_content = """Plain text file for testing full watcher.
Contains some interesting content about machine learning AI."""

        py_content = '''"""
Test Python file for full watcher.
Demonstrates Python programming concepts.
"""
def hello_world():
    """A simple function."""
    print("Hello, world!")
    # Machine learning related code
    return True
'''

        md_file.write_text(md_content)
        txt_file.write_text(txt_content)
        py_file.write_text(py_content)

        # 更新配置使用临时目录
        config["components"]["file_watcher"]["default"]["watch_paths"] = [str(temp_path)]
        config["components"]["file_watcher"]["default"]["recursive"] = False

        reme = ReMe(**config)

        # Start components (embedding, file_store, file_parser, file_watcher)
        await reme.start()

        # Wait for watcher to scan existing files
        await asyncio.sleep(3)

        # Test: list indexed files
        from reme2.enumeration import ComponentEnum

        store = reme.context.components[ComponentEnum.FILE_STORE]["default"]

        files = await store.list_files()
        print(f"\n=== Indexed files by Full Watcher ({len(files)}) ===")
        for f in files:
            meta = await store.get_file_metadata(f)
            chunks = await store.get_file_chunks(f)
            print(f"  {f} — {meta.chunk_count} chunks, metadata={meta.metadata}")

        # Test: keyword search
        results = await store.keyword_search("Python programming", limit=5)
        print(f"\n=== Keyword search 'Python programming' ({len(results)} results) ===")
        for r in results:
            print(f"  [{r.score:.2f}] {r.path}:{r.start_line}-{r.end_line} — {r.text[:80]}")

        # Test: hybrid search
        results = await store.hybrid_search("machine learning AI", limit=5)
        print(f"\n=== Hybrid search 'machine learning AI' ({len(results)} results) ===")
        for r in results:
            print(f"  [{r.score:.2f}] {r.path}:{r.start_line}-{r.end_line} — {r.text[:80]}")

        await reme.close()
        print("\n=== Full Watcher 测试完成 ===")


async def test_light_watcher():
    """测试light watcher模式 - 只处理markdown文件"""
    print("\n=== 测试 Light Watcher 模式 ===")

    config = _load_yaml("paw")

    # 设置API密钥
    config["components"]["embedding_model"]["default"]["api_key"] = os.environ["EMBEDDING_API_KEY"]
    config["components"]["embedding_model"]["default"]["base_url"] = os.environ["EMBEDDING_BASE_URL"]

    config["components"]["as_llm"]["default"]["api_key"] = os.environ["LLM_API_KEY"]
    config["components"]["as_llm"]["default"]["base_url"] = os.environ["LLM_BASE_URL"]
    config["components"]["as_llm"]["default"]["model_name"] = os.environ["LLM_MODEL_NAME"]

    # 更改配置使用light watcher
    config["components"]["file_watcher"]["default"]["backend"] = "light"

    # 创建临时目录用于测试
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 创建测试文件 - 包含markdown和其他类型文件
        md_file = temp_path / "light_test.md"
        txt_file = temp_path / "light_test.txt"
        py_file = temp_path / "light_test.py"

        md_content = """# Light Test Markdown File

This is a test markdown file for light watcher.

## Section
Only markdown files should be processed by light watcher.
"""

        txt_content = "This text file should be ignored by light watcher."
        py_content = "# This Python file should be ignored by light watcher"

        md_file.write_text(md_content)
        txt_file.write_text(txt_content)
        py_file.write_text(py_content)

        # 更新配置使用临时目录
        config["components"]["file_watcher"]["default"]["watch_paths"] = [str(temp_path)]
        config["components"]["file_watcher"]["default"]["recursive"] = False

        reme = ReMe(**config)

        # Start components
        await reme.start()

        # Wait for watcher to scan existing files
        await asyncio.sleep(3)

        # Test: list indexed files - should only contain markdown files
        from reme2.enumeration import ComponentEnum

        store = reme.context.components[ComponentEnum.FILE_STORE]["default"]

        files = await store.list_files()
        print(f"\n=== Indexed files by Light Watcher ({len(files)}) ===")
        for f in files:
            # 验证只有markdown文件被索引
            file_path = Path(f)
            if file_path.suffix.lower() in [".md", ".markdown"]:
                meta = await store.get_file_metadata(f)
                chunks = await store.get_file_chunks(f)
                print(f"  {f} — {meta.chunk_count} chunks, metadata={meta.metadata}")
            else:
                print(f"  ERROR: Non-markdown file indexed: {f}")

        print(f"\nExpected: Only markdown files should be indexed. Actually indexed: {len(files)} files")

        # Test: search
        results = await store.keyword_search("light watcher", limit=5)
        print(f"\n=== Keyword search 'light watcher' ({len(results)} results) ===")
        for r in results:
            print(f"  [{r.score:.2f}] {r.path}:{r.start_line}-{r.end_line} — {r.text[:80]}")

        await reme.close()
        print("\n=== Light Watcher 测试完成 ===")


async def main():
    """主函数，运行所有测试"""
    print("开始测试ReMe系统的Full和Light Watcher模式")

    try:
        await test_full_watcher()
        await test_light_watcher()
        print("\n✅ 所有测试完成！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
