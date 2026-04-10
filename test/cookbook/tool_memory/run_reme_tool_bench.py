#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tool Memory Benchmark Script
评估工具记忆在不同场景下的效果，包括有记忆和无记忆的对比

Dependencies:
    pip install requests python-dotenv loguru tabulate
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
from dotenv import load_dotenv
from loguru import logger
from tabulate import tabulate

from reme_ai.schema.memory import ToolCallResult, ToolMemory

load_dotenv()

BASE_URL = "http://0.0.0.0:8002/"
TRAIN_WORKSPACE = "train_tool_workspace"
TEST_WORKSPACE = "test_tool_workspace"


class BenchmarkStats:
    """统计数据收集器"""

    def __init__(self, name: str):
        self.name = name
        self.total_count = 0
        self.scores = []

    def add_result(self, result: Dict[str, Any]):
        """添加一个工具调用结果

        Note:
        - score: Quality/relevance of the result (0.0 or 1.0)
        """
        self.total_count += 1

        # Collect score
        score = result.get("score", 0)
        self.scores.append(score)

    def get_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        if self.total_count == 0:
            return {
                "name": self.name,
                "total_calls": 0,
                "avg_score": 0.0,
            }

        return {
            "name": self.name,
            "total_calls": self.total_count,
            "avg_score": round(sum(self.scores) / len(self.scores), 3),
        }


def api_call(endpoint: str, data: dict) -> Optional[Dict[str, Any]]:
    """统一的API调用处理"""
    try:
        response = requests.post(f"{BASE_URL}{endpoint}", json=data, timeout=120)
        if response.status_code != 200:
            logger.error(f"API Error: {response.status_code} - {response.text}")
            return None
        return response.json()
    except Exception as e:
        logger.error(f"API call failed: {e}")
        return None


def delete_workspace(workspace_id: str) -> bool:
    """删除工作空间"""
    logger.info(f"Deleting workspace: {workspace_id}")
    result = api_call("vector_store", {"workspace_id": workspace_id, "action": "delete"})
    return result is not None


def load_queries(query_file: str = "query.json") -> Dict[str, Any]:
    """加载查询数据"""
    query_path = Path(__file__).parent / query_file
    with open(query_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_use_mock_search(workspace_id: str, queries: List[str], prompt_template: str = "") -> List[ToolCallResult]:
    """运行use_mock_search并收集结果（支持并发）

    Args:
        workspace_id: 工作空间ID
        queries: 查询列表
        prompt_template: 提示模板

    Returns:
        工具调用结果列表
    """
    logger.info(f"Running use_mock_search on {workspace_id} with {len(queries)} queries (max concurrency: 4)")
    results: List[ToolCallResult] = []

    def process_single_query(idx: int, query: str) -> Optional[ToolCallResult]:
        """处理单个查询"""
        logger.info(f"[{idx + 1}/{len(queries)}] Processing: {query}")

        # 提交之前sleep 1秒
        time.sleep(1)

        result = api_call(
            "use_mock_search",
            {
                "workspace_id": workspace_id,
                "query": prompt_template.format(query=query),
            },
        )

        if result:
            tool_call_result = result.get("answer")
            return ToolCallResult(**json.loads(tool_call_result))
        else:
            logger.warning(f"No result for query: {query}")
            return None

    # 使用线程池并发处理，最大并发数为4
    with ThreadPoolExecutor(max_workers=4) as executor:
        # 提交所有任务
        future_to_query = {
            executor.submit(process_single_query, idx, query): (idx, query) for idx, query in enumerate(queries)
        }

        # 收集结果（按完成顺序）
        completed_results = []
        for future in as_completed(future_to_query):
            idx, query = future_to_query[future]
            try:
                result = future.result()
                if result:
                    completed_results.append((idx, result))
            except Exception as e:
                logger.error(f"Error processing query [{idx + 1}]: {query}, error: {e}")

        # 按原始顺序排序结果
        completed_results.sort(key=lambda x: x[0])
        results = [result for _, result in completed_results]

    logger.info(f"Collected {len(results)} results out of {len(queries)} queries")
    return results


def add_tool_call_results(workspace_id: str, results: List[ToolCallResult]) -> List[ToolCallResult]:
    """批量添加工具调用结果到记忆库，并返回带评分的结果

    Args:
        workspace_id: 工作空间ID
        results: 工具调用结果列表

    Returns:
        从API返回的memory_list中提取的带评分的ToolCallResult列表
    """
    if not results:
        logger.warning("No results to add")
        return []

    # 转换为字典用于API调用
    tool_call_results = [result.model_dump() for result in results]

    logger.info(f"Adding tool call results to {workspace_id}: {len(tool_call_results)} results")

    # 统一调用API，让后端自动按tool_name分组处理
    api_result = api_call(
        "add_tool_call_result",
        {
            "workspace_id": workspace_id,
            "tool_call_results": tool_call_results,
        },
    )

    if not api_result:
        logger.error("Failed to add results")
        return []

    # 收集所有带评分的结果
    all_scored_results: List[ToolCallResult] = []

    # 解析返回的memory_list（可能包含多个工具的记忆）
    memory_list = api_result.get("metadata", {}).get("memory_list", [])
    logger.info(f"Received {len(memory_list)} tool memories from API")

    for memory_dict in memory_list:
        tool_memory = ToolMemory(**memory_dict)
        tool_name = tool_memory.when_to_use
        scored_results = tool_memory.tool_call_results
        all_scored_results.extend(scored_results)

        logger.info(f"Extracted {len(scored_results)} scored results from {tool_name}")

        # 打印一些评分示例
        for idx, result in enumerate(scored_results[:3]):
            logger.info(f"  Result #{idx + 1}: score={result.score}, success={result.success}")

    logger.info(f"Total scored results collected: {len(all_scored_results)}")
    return all_scored_results


def summarize_tool_memory(workspace_id: str, tool_names: str) -> bool:
    """总结工具记忆"""
    logger.info(f"Summarizing tool memory for {workspace_id}: {tool_names}")
    result = api_call(
        "summary_tool_memory",
        {
            "workspace_id": workspace_id,
            "tool_names": tool_names,
        },
    )

    if result:
        memory_list = result.get("metadata", {}).get("memory_list", [])
        logger.info(f"Summarized {len(memory_list)} tool memories")
        return True

    return False


def retrieve_tool_memory(workspace_id: str, tool_names: str) -> str:
    """检索工具记忆并返回格式化的内容

    Args:
        workspace_id: 工作空间ID
        tool_names: 逗号分隔的工具名称

    Returns:
        格式化的工具记忆内容，每个工具名称作为一级markdown标题
    """
    logger.info(f"Retrieving tool memory for {workspace_id}: {tool_names}")
    result = api_call(
        "retrieve_tool_memory",
        {
            "workspace_id": workspace_id,
            "tool_names": tool_names,
        },
    )

    if not result:
        logger.error("Failed to retrieve tool memory")
        return ""

    memory_list = result.get("metadata", {}).get("memory_list", [])
    logger.info(f"Retrieved {len(memory_list)} tool memories")

    # 提取每个工具记忆的content字段，并格式化为markdown
    formatted_contents = []
    for memory_dict in memory_list:
        tool_memory = ToolMemory(**memory_dict)
        if tool_memory.content:
            # 使用工具名称作为一级markdown标题
            tool_name = tool_memory.when_to_use or "Unknown Tool"
            formatted_section = f"# {tool_name}\n\n{tool_memory.content}"
            formatted_contents.append(formatted_section)
            logger.info(
                f"Retrieved content for tool: {tool_name}, " f"content_length={len(tool_memory.content)}",
            )

    # 用两个换行符分隔不同工具的记忆
    joined_content = "\n\n".join(formatted_contents)
    logger.info(f"Total content length: {len(joined_content)}")

    return joined_content


def collect_statistics(results: List[ToolCallResult], stats: BenchmarkStats) -> None:
    """从结果列表中收集统计数据"""
    for result in results:
        # 转换为字典用于统计
        result_dict = result.model_dump() if hasattr(result, "model_dump") else result
        stats.add_result(result_dict)


def print_comparison_table(stats_list: List[BenchmarkStats]) -> None:
    """打印对比表格"""
    headers = ["Scenario", "Total Calls", "Avg Score"]
    rows = []

    for stats in stats_list:
        summary = stats.get_summary()
        rows.append(
            [
                summary["name"],
                summary["total_calls"],
                summary["avg_score"],
            ],
        )

    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS COMPARISON")
    print("=" * 100)
    print("Note: Avg Score = average quality score")
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print("=" * 100)


def calculate_improvements(baseline_stats: BenchmarkStats, improved_stats: BenchmarkStats) -> Dict[str, float]:
    """计算改进百分比"""
    baseline = baseline_stats.get_summary()
    improved = improved_stats.get_summary()

    improvements = {}

    # 平均分数改进（相对提升百分比）
    if baseline["avg_score"] > 0:
        improvements["avg_score"] = (improved["avg_score"] - baseline["avg_score"]) / baseline["avg_score"] * 100
    else:
        improvements["avg_score"] = 0.0

    return improvements


def print_improvements(improvements: Dict[str, float]) -> None:
    """打印改进情况"""
    print("\n" + "=" * 100)
    print("IMPROVEMENTS WITH TOOL MEMORY (Baseline: Test without memory)")
    print("=" * 100)

    metric_labels = {
        "avg_score": "Average Score",
    }

    for metric, improvement in improvements.items():
        label = metric_labels.get(metric, metric)
        direction = "↑" if improvement > 0 else "↓"
        print(f"{label:25s}: {improvement:+7.2f}% {direction}")

    print("=" * 100)


def save_results(results: Dict[str, Any], filename: str = "benchmark_results.json") -> None:
    """保存结果到文件"""
    output_path = Path(__file__).parent / filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_path}")


def run_single_epoch(epoch_num: int, train_queries: List[str], test_queries: List[str]) -> Dict[str, Any]:
    """运行单个epoch的benchmark

    Args:
        epoch_num: epoch编号（从1开始）
        train_queries: 训练查询列表
        test_queries: 测试查询列表

    Returns:
        包含该epoch统计结果的字典
    """
    logger.info("\n" + "=" * 100)
    logger.info(f"EPOCH {epoch_num} - START")
    logger.info("=" * 100)

    # 初始化统计收集器
    train_no_memory_stats = BenchmarkStats(f"Epoch{epoch_num} - Train (No Memory)")
    test_no_memory_stats = BenchmarkStats(f"Epoch{epoch_num} - Test (No Memory)")
    test_with_memory_stats = BenchmarkStats(f"Epoch{epoch_num} - Test (With Memory)")

    all_results = {}

    # ==================== 步骤1: 无记忆在train上的效果 ====================
    print("\n" + "=" * 100)
    print(f"[EPOCH {epoch_num}] [STEP 1/5] Running on TRAIN without memory...")
    print("=" * 100)
    logger.info("Deleting workspace and starting fresh...")
    delete_workspace(TRAIN_WORKSPACE)
    time.sleep(2)
    prompt_template = "必须选择一个工具来回答问题\n 问题\n{query}"
    train_results_no_memory = run_use_mock_search(TRAIN_WORKSPACE, train_queries, prompt_template)

    # 添加结果到记忆库并获取带评分的结果
    train_scored_results = add_tool_call_results(TRAIN_WORKSPACE, train_results_no_memory)
    time.sleep(2)

    # 使用带评分的结果进行统计（如果有的话）
    if train_scored_results:
        logger.info(f"Using {len(train_scored_results)} scored results for statistics")
        all_results["train_no_memory"] = train_scored_results
        collect_statistics(train_scored_results, train_no_memory_stats)
    else:
        logger.warning("No scored results returned, using original results")
        all_results["train_no_memory"] = train_results_no_memory
        collect_statistics(train_results_no_memory, train_no_memory_stats)

    print(f"✓ Train (no memory) completed: {len(train_results_no_memory)}/{len(train_queries)} results collected")
    summary = train_no_memory_stats.get_summary()
    print(f"  Avg Score: {summary['avg_score']:.3f}")

    # ==================== 步骤2: 无记忆在test上的效果 ====================
    print("\n" + "=" * 100)
    print(f"[EPOCH {epoch_num}] [STEP 2/5] Running on TEST without memory...")
    print("=" * 100)
    logger.info("Deleting workspace and starting fresh...")
    delete_workspace(TEST_WORKSPACE)
    time.sleep(2)

    prompt_template = "必须选择一个工具来回答问题\n 问题\n{query}"
    test_results_no_memory = run_use_mock_search(TEST_WORKSPACE, test_queries, prompt_template)

    # 添加结果到记忆库并获取带评分的结果
    # 注意：这些结果会作为TEST_WORKSPACE的初始记忆，在步骤4会被复用
    test_scored_results_no_memory = add_tool_call_results(TEST_WORKSPACE, test_results_no_memory)
    time.sleep(2)

    # 使用带评分的结果进行统计（如果有的话）
    if test_scored_results_no_memory:
        logger.info(f"Using {len(test_scored_results_no_memory)} scored results for statistics")
        all_results["test_no_memory"] = test_scored_results_no_memory
        collect_statistics(test_scored_results_no_memory, test_no_memory_stats)
    else:
        logger.warning("No scored results returned, using original results")
        all_results["test_no_memory"] = test_results_no_memory
        collect_statistics(test_results_no_memory, test_no_memory_stats)

    print(f"✓ Test (no memory) completed: {len(test_results_no_memory)}/{len(test_queries)} results collected")
    summary = test_no_memory_stats.get_summary()
    print(f"  Avg Score: {summary['avg_score']:.3f}")

    # ==================== 步骤3: 总结train的工具记忆 ====================
    print("\n" + "=" * 100)
    print(f"[EPOCH {epoch_num}] [STEP 3/5] Summarizing tool memory from TRAIN...")
    print("=" * 100)

    # 获取所有工具名称（使用带评分的结果）
    tool_names_set = set()
    results_to_use = train_scored_results if train_scored_results else train_results_no_memory
    for result in results_to_use:
        tool_name = result.tool_name if hasattr(result, "tool_name") else None
        if tool_name:
            tool_names_set.add(tool_name)

    tool_names_str = ",".join(sorted(tool_names_set))
    print(f"Tools to summarize: {tool_names_str}")

    success = summarize_tool_memory(TRAIN_WORKSPACE, tool_names_str)
    if not success:
        logger.error("Failed to summarize tool memory")
        return {}

    time.sleep(3)

    print("✓ Tool memory summarized successfully")

    # 检索工具记忆内容
    memories = retrieve_tool_memory(TRAIN_WORKSPACE, tool_names_str)
    if not memories:
        logger.error("Failed to retrieve tool memory content")
        return {}

    logger.info(f"Retrieved tool memory content, total length: {len(memories)}")
    print("\n" + "-" * 100)
    print("Retrieved Tool Memory Content:")
    print("-" * 100)
    print(memories)
    print("-" * 100)

    # ==================== 步骤4: 有记忆在test上的效果 ====================
    print("\n" + "=" * 100)
    print(f"[EPOCH {epoch_num}] [STEP 4/5] Running on TEST with memory (after clearing existing memory)...")
    print("=" * 100)

    # 先清理TEST_WORKSPACE中已有的记忆记录（Step 2的60条结果）
    print("Deleting existing memory records from TEST workspace...")
    delete_workspace(TEST_WORKSPACE)
    time.sleep(2)  # 等待删除完成
    print("✓ TEST workspace memory cleared")

    # 通过prompt注入train阶段总结的记忆，测量记忆增强的效果
    # 注意：此时workspace是空的，只通过prompt提供记忆信息

    prompt_template = f"工具信息\n{memories}\n必须选择一个工具来回答问题\n 问题\n" + "{query}"
    test_results_with_memory = run_use_mock_search(TEST_WORKSPACE, test_queries, prompt_template)

    # 添加这些结果到记忆库并获取带评分的结果
    # 此时workspace已清空，返回的就是本次新增的60条结果
    test_all_results_with_memory = add_tool_call_results(TEST_WORKSPACE, test_results_with_memory)
    time.sleep(2)

    # workspace已清空，所有返回的结果都是本次新增的
    if test_all_results_with_memory:
        test_scored_results_with_memory = test_all_results_with_memory
        logger.info(f"Using {len(test_scored_results_with_memory)} scored results for statistics")
        all_results["test_with_memory"] = test_scored_results_with_memory
        collect_statistics(test_scored_results_with_memory, test_with_memory_stats)
    else:
        logger.warning("No scored results returned, using original results")
        all_results["test_with_memory"] = test_results_with_memory
        collect_statistics(test_results_with_memory, test_with_memory_stats)

    print(f"✓ Test (with memory) completed: {len(test_results_with_memory)}/{len(test_queries)} results collected")
    summary = test_with_memory_stats.get_summary()
    print(f"  Avg Score: {summary['avg_score']:.3f}")

    # ==================== 步骤5: 打印对比结果 ====================
    print("\n" + "=" * 100)
    print(f"[EPOCH {epoch_num}] [STEP 5/5] Generating comparison report and analysis...")
    print("=" * 100)

    # 打印统计表格
    print_comparison_table([train_no_memory_stats, test_no_memory_stats, test_with_memory_stats])

    # 计算并打印改进情况
    improvements = calculate_improvements(test_no_memory_stats, test_with_memory_stats)
    print_improvements(improvements)

    logger.info(f"EPOCH {epoch_num} - COMPLETE")

    return {
        "epoch": epoch_num,
        "statistics": {
            "train_no_memory": train_no_memory_stats.get_summary(),
            "test_no_memory": test_no_memory_stats.get_summary(),
            "test_with_memory": test_with_memory_stats.get_summary(),
        },
        "improvements": improvements,
    }


def main(test_mode: bool = False, run_epoch: int = 3):
    """主函数：运行完整的benchmark流程

    Args:
        test_mode: 如果为True，只使用每个难度级别的前3个查询进行快速测试
        run_epoch: 运行的epoch数量，默认为3
    """
    logger.info("=" * 100)
    logger.info("TOOL MEMORY BENCHMARK - START")
    if test_mode:
        logger.info("Running in TEST MODE (limited queries)")
    logger.info(f"Total Epochs: {run_epoch}")
    logger.info("=" * 100)

    # 加载查询数据
    queries_data = load_queries()
    train_queries = []
    test_queries = []

    # 合并所有难度级别的查询
    for difficulty in ["simple", "moderate", "complex"]:
        train_data = queries_data["train"].get(difficulty, [])
        test_data = queries_data["test"].get(difficulty, [])

        if test_mode:
            train_queries.extend(train_data[:5])
            test_queries.extend(test_data[:5])
        else:
            train_queries.extend(train_data)
            test_queries.extend(test_data)

    logger.info(f"Loaded {len(train_queries)} train queries and {len(test_queries)} test queries")

    # 运行多个epoch并收集结果
    all_epoch_results = []

    for epoch in range(1, run_epoch + 1):
        epoch_result = run_single_epoch(epoch, train_queries, test_queries)
        if epoch_result:
            all_epoch_results.append(epoch_result)
        else:
            logger.error(f"Epoch {epoch} failed, skipping...")

    # ==================== 计算多轮平均效果 ====================
    if not all_epoch_results:
        logger.error("No successful epochs, cannot calculate averages")
        return

    print("\n" + "=" * 100)
    print("MULTI-EPOCH AVERAGE RESULTS")
    print("=" * 100)

    # 计算每个场景的平均分数
    avg_train_no_memory = sum(e["statistics"]["train_no_memory"]["avg_score"] for e in all_epoch_results) / len(
        all_epoch_results,
    )
    avg_test_no_memory = sum(e["statistics"]["test_no_memory"]["avg_score"] for e in all_epoch_results) / len(
        all_epoch_results,
    )
    avg_test_with_memory = sum(e["statistics"]["test_with_memory"]["avg_score"] for e in all_epoch_results) / len(
        all_epoch_results,
    )

    # 计算平均改进
    avg_improvement = sum(e["improvements"]["avg_score"] for e in all_epoch_results) / len(all_epoch_results)

    # 打印汇总表格
    headers = ["Scenario", "Avg Score (across epochs)"]
    rows = [
        ["Train (No Memory)", f"{avg_train_no_memory:.3f}"],
        ["Test (No Memory)", f"{avg_test_no_memory:.3f}"],
        ["Test (With Memory)", f"{avg_test_with_memory:.3f}"],
    ]
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    print("\n" + "-" * 100)
    print(f"Average Improvement (Test with memory vs without): {avg_improvement:+.2f}%")
    print("-" * 100)

    # 打印每个epoch的详细结果
    print("\n" + "=" * 100)
    print("PER-EPOCH BREAKDOWN")
    print("=" * 100)

    headers = ["Epoch", "Train (No Mem)", "Test (No Mem)", "Test (With Mem)", "Improvement %"]
    rows = []
    for e in all_epoch_results:
        rows.append(
            [
                f"Epoch {e['epoch']}",
                f"{e['statistics']['train_no_memory']['avg_score']:.3f}",
                f"{e['statistics']['test_no_memory']['avg_score']:.3f}",
                f"{e['statistics']['test_with_memory']['avg_score']:.3f}",
                f"{e['improvements']['avg_score']:+.2f}%",
            ],
        )

    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # 保存最终结果
    benchmark_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_epochs": run_epoch,
        "successful_epochs": len(all_epoch_results),
        "average_results": {
            "train_no_memory": avg_train_no_memory,
            "test_no_memory": avg_test_no_memory,
            "test_with_memory": avg_test_with_memory,
            "improvement": avg_improvement,
        },
        "per_epoch_results": all_epoch_results,
    }

    save_results(benchmark_summary, "tool_memory_benchmark_results.json")

    logger.info("\n" + "=" * 100)
    logger.info("TOOL MEMORY BENCHMARK - COMPLETE")
    logger.info(f"Successfully completed {len(all_epoch_results)}/{run_epoch} epochs")
    logger.info("=" * 100)


if __name__ == "__main__":
    main(test_mode=False, run_epoch=3)
