#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool Memory Demo - 展示工具记忆的完整生命周期"""

import json
import time
from typing import List, Dict, Any, Optional

import requests
from dotenv import load_dotenv

from reme_ai.utils.tool_memory_utils import create_mock_tool_call_results

load_dotenv()

BASE_URL = "http://0.0.0.0:8002/"
WORKSPACE_ID = "test_tool_memory_workspace"


def api_call(endpoint: str, data: dict) -> Optional[Dict[str, Any]]:
    """统一的API调用处理"""
    response = requests.post(f"{BASE_URL}{endpoint}", json=data)
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return None
    return response.json()


def delete_workspace() -> None:
    """删除工作空间

    curl example:
    curl -X POST http://0.0.0.0:8002/vector_store \
      -H "Content-Type: application/json" \
      -d '{
        "workspace_id": "test_tool_memory_workspace",
        "action": "delete"
      }'
    """
    result = api_call("vector_store", {"workspace_id": WORKSPACE_ID, "action": "delete"})
    if result:
        print(f"✓ Workspace '{WORKSPACE_ID}' deleted")


def add_tool_call_results(tool_call_results: List[Dict[str, Any]]) -> bool:
    """添加工具调用结果到记忆库

    curl example:
    curl -X POST http://0.0.0.0:8002/add_tool_call_result \
      -H "Content-Type: application/json" \
      -d '{
        "workspace_id": "test_tool_memory_workspace",
        "tool_call_results": [
          {
            "tool_name": "web_search",
            "tool_input": "Python tutorials",
            "tool_output": "Found 100 results about Python",
            "execution_time": 0.5
          }
        ]
      }'
    """
    # 统计不同的工具
    tool_names = set(r.get("tool_name") for r in tool_call_results)
    print(f"\n[ADD] {len(tool_call_results)} results for {len(tool_names)} tools: {', '.join(sorted(tool_names))}")

    result = api_call(
        "add_tool_call_result",
        {
            "workspace_id": WORKSPACE_ID,
            "tool_call_results": tool_call_results,
        },
    )
    if result:
        memory_list = result.get("metadata", {}).get("memory_list", [])
        print(f"✓ Added successfully, created/updated {len(memory_list)} tool memories")
        return True
    return False


def summarize_tool_memory(tool_names: str) -> Optional[Dict[str, Any]]:
    """总结工具使用模式

    curl example:
    curl -X POST http://0.0.0.0:8002/summary_tool_memory \
      -H "Content-Type: application/json" \
      -d '{
        "workspace_id": "test_tool_memory_workspace",
        "tool_names": "web_search,database_query"
      }'
    """
    print(f"\n[SUMMARIZE] {tool_names}")
    result = api_call(
        "summary_tool_memory",
        {
            "workspace_id": WORKSPACE_ID,
            "tool_names": tool_names,
        },
    )

    if result:
        memory_list = result.get("metadata", {}).get("memory_list", [])
        print(f"✓ Summarized {len(memory_list)} tool memories")
        for memory in memory_list:
            print(f"\n{'=' * 60}")
            print(f"Tool: {memory.get('when_to_use', 'N/A')}")
            print(f"{'=' * 60}")
            print(memory.get("content", "No content"))
    return result


def retrieve_tool_memory(tool_names: str, save_to_file: bool = False) -> str:
    """检索工具记忆

    curl example:
    curl -X POST http://0.0.0.0:8002/retrieve_tool_memory \
      -H "Content-Type: application/json" \
      -d '{
        "workspace_id": "test_tool_memory_workspace",
        "tool_names": "web_search"
      }'
    """
    print(f"\n[RETRIEVE] {tool_names}")
    result = api_call(
        "retrieve_tool_memory",
        {
            "workspace_id": WORKSPACE_ID,
            "tool_names": tool_names,
        },
    )

    if not result:
        return ""

    memory_list = result.get("metadata", {}).get("memory_list", [])
    if not memory_list:
        print("No memories found")
        return ""

    print(f"✓ Retrieved {len(memory_list)} memories")

    formatted_memories = []
    for memory in memory_list:
        content = (
            f"\nTool: {memory.get('when_to_use', 'N/A')}\n"
            f"Calls: {len(memory.get('tool_call_results', []))}\n"
            f"{'-' * 60}\n{memory.get('content', 'No content')}\n"
        )
        formatted_memories.append(content)
        print(content)

    if save_to_file:
        with open("tool_memory.json", "w") as f:
            json.dump(memory_list, f, indent=2, ensure_ascii=False)
        print("✓ Saved to tool_memory.json")

    return "\n".join(formatted_memories)


def dump_memory(path: str = "./") -> None:
    """导出记忆到磁盘

    curl example:
    curl -X POST http://0.0.0.0:8002/vector_store \
      -H "Content-Type: application/json" \
      -d '{
        "workspace_id": "test_tool_memory_workspace",
        "action": "dump",
        "path": "./"
      }'
    """
    result = api_call(
        "vector_store",
        {
            "workspace_id": WORKSPACE_ID,
            "action": "dump",
            "path": path,
        },
    )
    if result:
        print(f"✓ Memory dumped to {path}")


def load_memory(path: str = "./") -> None:
    """从磁盘加载记忆

    curl example:
    curl -X POST http://0.0.0.0:8002/vector_store \
      -H "Content-Type: application/json" \
      -d '{
        "workspace_id": "test_tool_memory_workspace",
        "action": "load",
        "path": "./"
      }'
    """
    result = api_call(
        "vector_store",
        {
            "workspace_id": WORKSPACE_ID,
            "action": "load",
            "path": path,
        },
    )
    if result:
        print(f"✓ Memory loaded from {path}")


def main() -> None:
    # 1. 清理工作空间
    print("\n[1] Cleaning workspace...")
    delete_workspace()
    time.sleep(1)

    # 2. 创建和添加模拟工具调用结果
    print("\n[2] Adding mock tool call results...")
    tools_to_test = [
        ("web_search", 30),
        ("database_query", 22),
        ("file_processor", 18),
    ]

    # 收集所有工具的结果，然后一次性添加
    all_mock_results = []
    for tool_name, count in tools_to_test:
        mock_results = create_mock_tool_call_results(tool_name, count)
        all_mock_results.extend(mock_results)

    if not add_tool_call_results(all_mock_results):
        print("✗ Failed to add results")
    else:
        time.sleep(1)

    # 3. 总结工具记忆
    print("\n[3] Summarizing tool memories...")
    all_tool_names = ",".join([tool[0] for tool in tools_to_test])
    summarize_tool_memory(all_tool_names)
    time.sleep(1)

    # 4. 检索工具记忆
    print("\n[4] Retrieving tool memories...")
    for tool_name, _ in tools_to_test:
        retrieve_tool_memory(tool_name, save_to_file=True)
        time.sleep(0.5)

    # 5. 测试记忆持久化
    print("\n[5] Testing memory persistence...")
    dump_memory()
    load_memory()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
