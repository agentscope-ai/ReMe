"""Utility helpers for retrieving tool memory hints."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List

import httpx

logger = logging.getLogger(__name__)


class ToolMemoryFetcher:
    """Helper class that orchestrates tool memory collection."""

    SUMMARY_ENDPOINT = "http://localhost:8002/summary_tool_memory"
    RETRIEVE_ENDPOINT = "http://localhost:8002/retrieve_tool_memory"

    def __init__(self, workspace_id: str, source_task: str, model_name: str) -> None:
        self.workspace_id = workspace_id
        self.source_task = source_task
        self.model_name = model_name

    def collect_memory(self, tool_names: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Collect tool memory hints for a batch of tools (并行 HTTP 调用，加速总结).

        Returns:
            Dict[str, Dict[str, str]]: 字典，key 为工具名，value 为包含 'content' 和 'add_to' 的字典
        """
        if not tool_names:
            return {}

        try:
            # 在没有事件循环的环境下运行异步版本
            return asyncio.run(self._run_summary_async(tool_names))
        except RuntimeError:
            print("没有事件循环，退回到同步版本")
            return self._run_summary_sync(tool_names)

    async def _run_summary_async(self, tool_names: List[str]) -> Dict[str, Dict[str, str]]:
        """异步并行调用 summary 服务，加速多工具 memory 获取。"""
        ans: Dict[str, Dict[str, str]] = {}

        async def fetch_one(client: httpx.AsyncClient, tool_name: str) -> None:
            # 先检查是否有保存的 non_icl memory 文件
            non_icl_content = self._load_non_icl_memory(tool_name)

            payload = {
                "workspace_id": self.workspace_id,
                "tool_names": tool_name,
                "source_task": self.source_task,
                # 标记为"并行/不持久化"模式，服务端据此只返回结果，不写回向量库文件
                "no_persist": True,
            }
            try:
                resp = await client.post(self.SUMMARY_ENDPOINT, json=payload, timeout=60.0)
                resp.raise_for_status()
                data = resp.json()
                if data.get("success"):
                    content = data.get("metadata", {}).get("memory_list", [{}])[0].get("content", "")
                    formatted_content = (
                        "Besides, this is optional guidance on how to better use the tool. "
                        "You may refer to it selectively:" + content
                    )
                    print(f"成功为工具{tool_name}执行总结，总结内容{content}")

                    # 默认 add_to 为 "icl"
                    add_to = "icl"

                    # 在总结执行完之后，调用 retrieve 方法提取工具 memory
                    try:
                        retrieve_payload = {
                            "workspace_id": self.workspace_id,
                            "tool_names": tool_name,
                        }
                        retrieve_resp = await client.post(self.RETRIEVE_ENDPOINT, json=retrieve_payload, timeout=60.0)
                        retrieve_resp.raise_for_status()
                        retrieve_data = retrieve_resp.json()
                        retrieve_metadata = retrieve_data.get("metadata", {})

                        # 如果返回的是列表，取第一个；如果是 dict，直接使用
                        if isinstance(retrieve_metadata, list) and len(retrieve_metadata) > 0:
                            retrieve_item = retrieve_metadata[0]
                        elif isinstance(retrieve_metadata, dict):
                            retrieve_item = retrieve_metadata
                        else:
                            retrieve_item = {}

                        retrieve_content = retrieve_item.get("content", "")
                        add_to = retrieve_item.get("add_to", "icl")

                        # 如果 add_to 不是 'icl'，单独保存在文件里
                        if add_to != "icl":
                            self._save_non_icl_memory(tool_name, retrieve_content, add_to)
                    except Exception as retrieve_exc:
                        logger.warning("⚠️工具%s retrieve失败%s", tool_name, retrieve_exc)

                    # 组合 content：先添加文件中的 non_icl memory（如果有），再添加从服务获取的 memory
                    final_content = ""
                    if non_icl_content:
                        final_content = non_icl_content + " "
                    final_content += formatted_content

                    # 返回包含 content 和 add_to 的字典
                    ans[tool_name] = {
                        "content": final_content,
                        "add_to": add_to,
                    }
                else:
                    # 即使 summary 失败，如果有 non_icl memory，也要返回
                    final_content = non_icl_content if non_icl_content else ""
                    ans[tool_name] = {"content": final_content, "add_to": "icl"}
                    print(f"工具{tool_name}总结条件未满足，但执行没有错误")
            except Exception as exc:  # noqa: BLE001
                logger.warning("⚠️工具%s总结失败%s", tool_name, exc)
                # 即使总结失败，如果有 non_icl memory，也要返回
                final_content = non_icl_content if non_icl_content else ""
                ans[tool_name] = {"content": final_content, "add_to": "icl"}

        async with httpx.AsyncClient() as client:
            await asyncio.gather(*(fetch_one(client, name) for name in tool_names))

        return ans

    def _run_summary_sync(self, tool_names: List[str]) -> Dict[str, Dict[str, str]]:
        """同步版本作为兜底实现（顺序 HTTP 调用）。"""
        ans: Dict[str, Dict[str, str]] = {}
        for tool_name in tool_names:
            # 先检查是否有保存的 non_icl memory 文件
            non_icl_content = self._load_non_icl_memory(tool_name)

            payload = {
                "workspace_id": self.workspace_id,
                "tool_names": tool_name,
                "source_task": self.source_task,
                "no_persist": False,
            }
            try:
                response = httpx.post(self.SUMMARY_ENDPOINT, json=payload, timeout=60.0)
                response.raise_for_status()
                data = response.json()
                if data.get("success"):
                    content = data.get("metadata", {}).get("memory_list", [{}])[0].get("content", "")
                    formatted_content = (
                        "Besides, this is optional guidance on how to better use the tool. "
                        "You may refer to it selectively:" + content
                    )
                    print(f"成功为工具{tool_name}执行总结，总结内容{content}")

                    # 默认 add_to 为 "icl"
                    add_to = "icl"

                    # 在总结执行完之后，调用 retrieve 方法提取工具 memory
                    try:
                        retrieve_payload = {
                            "workspace_id": self.workspace_id,
                            "tool_names": tool_name,
                        }
                        retrieve_resp = httpx.post(self.RETRIEVE_ENDPOINT, json=retrieve_payload, timeout=60.0)
                        retrieve_resp.raise_for_status()
                        retrieve_data = retrieve_resp.json()
                        retrieve_metadata = retrieve_data.get("metadata", {})

                        # 如果返回的是列表，取第一个；如果是 dict，直接使用
                        if isinstance(retrieve_metadata, list) and len(retrieve_metadata) > 0:
                            retrieve_item = retrieve_metadata[0]
                        elif isinstance(retrieve_metadata, dict):
                            retrieve_item = retrieve_metadata
                        else:
                            retrieve_item = {}

                        retrieve_content = retrieve_item.get("content", "")
                        add_to = retrieve_item.get("add_to", "icl")

                        # 如果 add_to 不是 'icl'，单独保存在文件里
                        if add_to != "icl":
                            self._save_non_icl_memory(tool_name, retrieve_content, add_to)
                    except Exception as retrieve_exc:
                        logger.warning("⚠️工具%s retrieve失败%s", tool_name, retrieve_exc)

                    # 组合 content：先添加文件中的 non_icl memory（如果有），再添加从服务获取的 memory
                    final_content = ""
                    if non_icl_content:
                        final_content = non_icl_content + " "
                    final_content += formatted_content

                    # 返回包含 content 和 add_to 的字典
                    ans[tool_name] = {
                        "content": final_content,
                        "add_to": add_to,
                    }
                else:
                    # 即使 summary 失败，如果有 non_icl memory，也要返回
                    final_content = non_icl_content if non_icl_content else ""
                    ans[tool_name] = {"content": final_content, "add_to": "icl"}
                    print(f"工具{tool_name}总结条件未满足，但执行没有错误")
            except Exception as exc:  # noqa: BLE001
                logger.warning("⚠️工具%s总结失败%s", tool_name, exc)
                # 即使总结失败，如果有 non_icl memory，也要返回
                final_content = non_icl_content if non_icl_content else ""
                ans[tool_name] = {"content": final_content, "add_to": "icl"}
        return ans

    def _get_memory(self, tool_names: List[str]) -> Dict[str, str]:
        """Temporary placeholder for actual tool memory retrieval."""
        return {name: f" Memory placeholder for tool '{name}'." for name in tool_names}

    def _load_non_icl_memory(self, tool_name: str) -> str:
        """
        从 non_icl_memories 目录中加载指定工具的 non_icl memory。
        每个工具只有一个文件（覆盖模式）。

        Args:
            tool_name: 工具名称

        Returns:
            str: memory 内容，如果不存在则返回空字符串
        """
        # 获取 non_icl_memories 目录路径
        save_dir = os.path.join(os.path.dirname(__file__), "non_icl_memories")

        # 直接使用固定的文件名（覆盖模式）
        filename = os.path.join(save_dir, f"non_icl_{tool_name}.json")

        if not os.path.exists(filename):
            return ""

        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
                content = data.get("content", "")
                return content
        except Exception as e:
            logger.warning(f"读取 non_icl memory 文件 {filename} 失败: {e}")
            return ""

    def _save_non_icl_memory(self, tool_name: str, content: str, add_to: str) -> None:
        """
        将 add_to 不是 'icl' 的工具 memory 保存到文件中。
        对于每个工具，会覆盖之前的文件（每个工具只有一个文件）。

        Args:
            tool_name: 工具名称
            content: memory 内容
            add_to: add_to 字段的值
        """
        # 创建保存目录（如果不存在）
        save_dir = os.path.join(os.path.dirname(__file__), "non_icl_memories")
        os.makedirs(save_dir, exist_ok=True)

        # 生成文件名（每个工具固定文件名，覆盖模式）
        filename = os.path.join(save_dir, f"non_icl_{tool_name}.json")

        # 保存数据
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data = {
            "tool_name": tool_name,
            "content": "\nThis is guidance on using the tool learned from previous experience. You must follow it:\n"
            + content,
            "add_to": add_to,
            "timestamp": timestamp,
        }

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"已保存非 ICL memory 到文件: {filename} (覆盖模式)")
        except Exception as e:
            logger.warning(f"保存非 ICL memory 失败: {e}")
