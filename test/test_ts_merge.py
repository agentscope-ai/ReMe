"""批量提取 .py 文件的 import/from 之前的注释"""

from pathlib import Path


def extract_leading_comments(file_path: Path) -> str:
    """
    从 .py 文件中提取 import/from 之前的全部内容（包括注释、docstring、空行等）
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except (UnicodeDecodeError, OSError) as e:
        print(f"⚠️ 无法读取文件 {file_path}: {e}")
        return ""

    # 找到第一个非空且以 import 或 from 开头的行（忽略前面的空白和注释）
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("import ", "from ")):
            # 找到了第一个 import/from，返回之前的所有内容
            return "".join(lines[:i]).rstrip() + "\n\n"

    # 如果整个文件都没有 import/from，则返回全部内容（可能是纯脚本或空文件）
    content = "".join(lines).rstrip()
    return content + "\n\n" if content else ""


def main(root_dir: str, output_file: str = "merged_comments.txt"):
    """批量提取 .py 文件的 import/from 之前的注释"""
    root = Path(root_dir)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"目录不存在: {root_dir}")

    all_content = []
    py_files = list(root.rglob("*.py"))

    print(f"🔍 找到 {len(py_files)} 个 .py 文件，开始提取前导注释...")

    for py_file in py_files:
        comment = extract_leading_comments(py_file)
        if comment.strip():  # 只保留非空内容
            header = f"\n{'=' * 60}\n# 文件: {py_file.relative_to(root)}\n{'=' * 60}\n"
            all_content.append(header + comment)

    # 写入合并文件
    with open(output_file, "w", encoding="utf-8") as out:
        out.write("# 合并自所有 .py 文件的前导注释（import 之前的内容）\n")
        out.write("=" * 70 + "\n\n")
        out.writelines(all_content)

    print(f"✅ 合并完成！结果已保存至: {output_file}")


if __name__ == "__main__":
    # 👇 修改为你自己的目标目录
    target_directory = "/Users/yuli/workspace/ali_openclaw/src"  # 例如："/home/user/myproject" 或 r"C:\myproject"
    main(target_directory)
