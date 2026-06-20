"""
ReMeLight (file-based) evaluation script for LoCoMo benchmark.

This script evaluates the file-based memory system (ReMeLight) on the
LoCoMo benchmark, using the same evaluation protocol as the vector-based
eval_reme.py but adapted for ReMeLight's file-based API.

Pipeline:
1. Load LoCoMo data
2. For each user conversation:
   a. Initialize ReMeLight with per-user working_dir
   b. Process all sessions via summary_memory() -> writes memory/*.md
   c. Answer questions via memory_search() -> LLM generates answer
   d. Judge answers via LLM-as-Judge (GPT-4o-mini)
3. Aggregate and report metrics

Usage:
    python benchmark/locomo/eval_reme_light.py \
        --data_path locomo10.json \
        --top_k 20 --user_num 5 --max_concurrency 2
"""

import asyncio
import json
import os
import re
import shutil
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Any

from agentscope.message import Msg
from loguru import logger

from reme.reme_light import ReMeLight


# ==================== Configuration ====================
@dataclass
class EvalConfig:
    """Evaluation configuration parameters."""

    data_path: str = ""
    top_k: int = 20
    user_num: int = 1
    max_concurrency: int = 2
    batch_size: int = 40
    output_dir: str = "bench_results/reme_light"
    reme_model_name: str = "qwen-flash"
    eval_model_name: str = "qwen3-max"
    # Time to wait for FileWatcher to re-index after file writes (seconds)
    index_wait_seconds: int = 5
    # Skip summarization, go straight to QA (reuse existing working_dir)
    resume: bool = False


# ==================== Utilities ====================


class DataLoader:
    """Handles loading and parsing of LoCoMo data."""

    @staticmethod
    def load_json(file_path: str) -> dict:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def format_dialogue_messages(
        dialogue: list[dict],
        speaker_a: str,
        base_timestamp: datetime,
        time_interval: int,
    ) -> list[Msg]:
        """Format LoCoMo dialogue into agentscope Msg objects for ReMeLight."""
        messages: list[Msg] = []
        for idx, turn in enumerate(dialogue):
            role = "user" if turn["speaker"] == speaker_a else "assistant"
            ts = (base_timestamp + timedelta(seconds=idx * time_interval)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            msg = Msg(
                name=turn["speaker"],
                content=turn["text"],
                role=role,
                metadata={"time_created": ts},
            )
            messages.append(msg)
        return messages


class FileManager:
    """Manages file I/O operations for eval results."""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_user_dir(self, user_name: str) -> Path:
        user_dir = self.base_dir / user_name
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir

    def get_session_file(self, user_name: str, session_id: int) -> Path:
        return self.get_user_dir(user_name) / f"session_{session_id}.json"

    def save_session(self, user_name: str, session_id: int, data: dict):
        file_path = self.get_session_file(user_name, session_id)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def user_has_cache(self, user_name: str) -> bool:
        user_dir = self.get_user_dir(user_name)
        has_sessions = any(
            f.name.startswith("session_") and f.suffix == ".json"
            for f in user_dir.iterdir()
        )
        has_questions = (user_dir / "questions.json").exists()
        return has_sessions and has_questions

    def combine_results(self, output_file: str):
        """Combine all user session files into a single JSONL results file."""
        with open(output_file, "w", encoding="utf-8") as f_out:
            for user_dir in sorted(self.base_dir.iterdir()):
                if not user_dir.is_dir():
                    continue

                session_files = sorted(
                    f for f in user_dir.iterdir()
                    if f.name.startswith("session_") and f.suffix == ".json"
                )
                if not session_files:
                    continue

                with open(session_files[0], "r", encoding="utf-8") as f_in:
                    first_session = json.load(f_in)

                user_data = {
                    "uuid": first_session["uuid"],
                    "user_name": first_session["user_name"],
                    "sessions": [],
                }

                for sf in session_files:
                    with open(sf, "r", encoding="utf-8") as f_in:
                        session_data = json.load(f_in)
                        session_data.pop("uuid", None)
                        session_data.pop("user_name", None)
                        user_data["sessions"].append(session_data)

                question_file = user_dir / "questions.json"
                if question_file.exists():
                    with open(question_file, "r", encoding="utf-8") as f_in:
                        user_data["evaluation_results"] = json.load(f_in)

                f_out.write(json.dumps(user_data, ensure_ascii=False) + "\n")


# ==================== Memory Operations ====================


class ReMeLightMemoryProcessor:
    """Handles ReMeLight memory operations for eval."""

    def __init__(
        self,
        eval_model_name: str = "qwen3-max",
        index_wait_seconds: int = 5,
    ):
        self.eval_model_name = eval_model_name
        self.index_wait_seconds = index_wait_seconds

    # 每 0.5 秒检查 file_store 有没有索引文件，有了立刻返回
    async def _wait_for_index(self, reme: ReMeLight, timeout: int = 10):
        """Poll file_store until indexed files appear, instead of blind sleep."""
        from reme.core.enumeration import MemorySource
        fs = reme.service_context.file_stores["default"]
        for _ in range(timeout * 2):  # 每 0.5 秒检查一次
            files = await fs.list_files(MemorySource.MEMORY)
            if files:
                return
            await asyncio.sleep(0.5)
        logger.warning("FileWatcher indexing timed out, proceeding anyway")

    async def add_memories(
        self,
        reme: ReMeLight,
        messages: list[Msg],
        batch_size: int = 10000,
    ) -> tuple[str, float]:
        """Process session messages and persist to memory files.

        Returns:
            tuple: (summary_text, duration_ms)
        """
        start = time.time()

        summary_text = ""
        for i in range(0, len(messages), batch_size):
            batch = messages[i : i + batch_size]
            try:
                result = await reme.summary_memory(
                    messages=batch,
                    language="en",
                )
                summary_text += result
            except Exception as e:
                logger.error(f"summary_memory failed for batch: {e}")

        # 轮询等 FileWatcher 重建索引，替代 sleep(5)
        await self._wait_for_index(reme)

        duration_ms = (time.time() - start) * 1000
        return summary_text, duration_ms

    # LLM 生成 4 个变体问题，逐个搜，按 (path, line) 去重合并
    async def _multi_query_search(
        self, reme: ReMeLight, question: str, top_k: int
    ) -> list[dict]:
        """Generate multiple query variations and merge search results."""
        llm = reme.service_context.as_llms.get("default")
        if llm is None:
            return []

        # 让 LLM 生成查询变体
        prompt = _QUERY_VARIATIONS_PROMPT.format(question=question, n=4)
        try:
            resp = await asyncio.wait_for(
                llm(messages=[{"role": "user", "content": prompt}]), timeout=30
            )
            text = "".join(
                b["text"] if isinstance(b, dict) else getattr(b, "text", "")
                for b in (resp.content or [])
            )
            # 按行解析变体
            variations = [q.strip("- ").strip() for q in text.split("\n") if q.strip("- ").strip()]
        except Exception:
            variations = []

        # 原始问题 + 变体，去重
        all_queries = list(dict.fromkeys([question] + variations[:4]))
        logger.info(f"  Multi-query: {len(all_queries)} queries")

        # 逐个搜索，按 merge_key 去重合并
        seen = set()
        merged: dict[str, dict] = {}
        for q in all_queries:
            try:
                sr = await reme.memory_search(query=q, max_results=top_k, min_score=0.1)
                for r in _parse_search_results(sr):
                    key = f"{r.get('path','')}:{r.get('start_line','')}"
                    if key not in seen:
                        seen.add(key)
                        merged[key] = r
            except Exception:
                continue

        results = sorted(merged.values(), key=lambda r: r.get("score", 0), reverse=True)
        return results[:top_k]

    # 多轮检索 最多 3 轮, LLM 判断信息够不够，不够生成新查询再搜
    async def _multi_round_search(
        self, reme: ReMeLight, question: str, top_k: int, max_rounds: int = 3
    ) -> list[dict]:
        """Multi-round retrieval: search, check sufficiency, refine query if needed."""
        all_results = await self._multi_query_search(reme, question, top_k)
        if not all_results:
            return []

        llm = reme.service_context.as_llms.get("default")
        if llm is None or max_rounds <= 1:
            return all_results

        # 后续轮次
        seen_keys = {f"{r.get('path','')}:{r.get('start_line','')}" for r in all_results}
        for round_idx in range(1, max_rounds):
            # LLM 判断是否足够，不够则给新查询
            context = _format_search_results_for_prompt(all_results[:10])
            check_prompt = _SUFFICIENCY_CHECK_PROMPT.format(
                question=question, context=context
            )
            try:
                resp = await asyncio.wait_for(
                    llm(messages=[{"role": "user", "content": check_prompt}]), timeout=60
                )
                text = "".join(
                    b["text"] if isinstance(b, dict) else getattr(b, "text", "")
                    for b in (resp.content or [])
                )
            except Exception:
                break

            # 解析 LLM 决策
            if "SUFFICIENT" in text.upper() and "INSUFFICIENT" not in text.upper():
                break  # 够了就停

            # 提取新查询
            new_query = ""
            for line in text.split("\n"):
                if "NEW_QUERY:" in line.upper():
                    new_query = line.split(":", 1)[-1].strip()
                    break
            if not new_query:
                break

            logger.info(f"  Round {round_idx + 1}: refined query -> {new_query[:60]}...")
            new_results = await self._multi_query_search(reme, new_query, top_k // 2)
            for r in new_results:
                key = f"{r.get('path','')}:{r.get('start_line','')}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_results.append(r)

        return sorted(all_results, key=lambda r: r.get("score", 0), reverse=True)[:top_k]

    async def search_memory(
        self,
        reme: ReMeLight,
        query: str,
        top_k: int = 20,
    ) -> tuple[dict, list, float]:
        """Multi-round, multi-query memory search with LLM answer generation.

        Returns:
            tuple: (answer_dict, raw_search_results, duration_ms)
        """
        start = time.time()

        # 多轮 + 多查询检索
        raw_results = await self._multi_round_search(reme, query, top_k)

        # LLM 基于搜到的记忆生成结构化回答
        answer_dict = await _answer_question_with_memories(
            reme=reme,
            question=query,
            search_results=raw_results,
            model_name=self.eval_model_name,
        )

        duration_ms = (time.time() - start) * 1000
        return answer_dict, raw_results, duration_ms


def _parse_search_results(search_result) -> list[dict]:
    """Parse ReMeLight memory_search ToolResponse into list of result dicts."""
    try:
        if not search_result.content:
            return []
        # ToolResponse.content is a list of dicts, each with 'type' and 'text' keys
        block = search_result.content[0]
        if isinstance(block, dict):
            text = block.get("text", "[]")
        elif hasattr(block, "text"):
            text = block.text
        else:
            return []
        return json.loads(text)
    except (json.JSONDecodeError, AttributeError, IndexError, TypeError):
        logger.warning("Failed to parse memory_search results")
        return []


def _format_search_results_for_prompt(results: list[dict]) -> str:
    """Format raw search results into a prompt-friendly string."""
    if not results:
        return "No relevant memories found."

    lines = []
    for i, r in enumerate(results, 1):
        path = r.get("path", "unknown")
        snippet = r.get("snippet", r.get("content", ""))
        score = r.get("score", 0)
        lines.append(f"[{i}] {path} (score={score:.2f}):\n{snippet}")
    return "\n\n".join(lines)


async def _answer_question_with_memories(
    reme: ReMeLight,
    question: str,
    search_results: list[dict],
    model_name: str = "qwen3-max",
) -> dict:
    # 把搜索结果格式化成 prompt 上下文
    memories_text = _format_search_results_for_prompt(search_results)
    context = f"Memories from file-based memory system:\n{memories_text}"
    # 填入 prompt 模板
    prompt = _PROMPT_MEMZERO_JSON.format(context=context, question=question)

    llm = reme.service_context.as_llms.get("default")
    if llm is None:
        logger.error("No default LLM available")
        return {"reasoning": "LLM not available", "answer": ""}

    try:
        response = await asyncio.wait_for(
            llm(messages=[{"role": "user", "content": prompt}]),
            timeout=120,  # 2 minutes per LLM call
        )
        text = ""
        for b in (response.content or []):
            t = b["text"] if isinstance(b, dict) else getattr(b, "text", "")
            if t:
                text += t
        # Try parsing JSON from the response
        json_match = re.search(r"\{[^}]+\}", text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return {"reasoning": text, "answer": text}
    except asyncio.TimeoutError:
        logger.error("LLM answer generation timed out")
        return {"reasoning": "LLM timeout", "answer": ""}
    except Exception as e:
        logger.error(f"LLM answer generation failed: {e}")
        return {"reasoning": str(e), "answer": ""}

# 每个问题被裁判两次: 1. LLM 整理后的回答 vs 标准答案; 2.原始搜出来的记忆片段 vs 标准答案（衡量检索本身的质量）
async def _evaluation_for_question(
    reme: ReMeLight,
    question: str,
    golden_answer: str, # 数据集标准答案
    generated_answer: str, # LLM 生成的回答
    model_name: str = "qwen3-max",
) -> dict:
    """LLM-as-Judge: compare generated answer with golden answer."""
    await asyncio.sleep(2)  # Rate limiting

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        question=question,
        golden_answer=golden_answer,
        generated_answer=generated_answer,
    )

    llm = reme.service_context.as_llms.get("default")
    if llm is None:
        return {"reasoning": "LLM not available", "evaluation_result": False}

    try:
        # 调 LLM 当裁判，2 分钟超时
        response = await asyncio.wait_for(
            llm(messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]),
            timeout=120,
        )

        content = ""
        for b in (response.content or []):
            t = b["text"] if isinstance(b, dict) else getattr(b, "text", "")
            if t:
                content += t

        match = re.search(r'"label"\s*:\s*"([^"]*?)"', content)
        if match:
            label = match.group(1)
        else:
            # Fallback: look for CORRECT or WRONG anywhere in response
            if "CORRECT" in content.upper():
                label = "CORRECT"
            else:
                label = "WRONG"

        return {
            "reasoning": content,
            "evaluation_result": label.strip().upper() == "CORRECT",
        }
    except asyncio.TimeoutError:
        logger.error("Evaluation LLM call timed out")
        return {"reasoning": "LLM timeout", "evaluation_result": False}
    except Exception as e:
        logger.error(f"Evaluation LLM call failed: {e}")
        return {"reasoning": str(e), "evaluation_result": False}


# ==================== Evaluation Prompt Templates ====================

_SYSTEM_PROMPT = (
    "You are an expert grader that determines if answers to questions "
    "match a gold standard answer"
)

_USER_PROMPT_TEMPLATE = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
  (1) a question (posed by one user to another user),
  (2) a 'gold' (ground truth) answer,
  (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references, but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT.

Now it's time for the real question:
Question: {question}
Gold answer: {golden_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response.

Just return the label CORRECT or WRONG in a json format with the key as "label"."""

_PROMPT_MEMZERO_JSON = """# CONTEXT:
{context}

# CONTEXT PRIORITY:
When the context contains information from multiple sources, follow this strict priority order:
1. **Historical Dialogue** (highest priority) - Direct conversation content
2. **Extracted Memories** (medium priority) - Summarized memory points
3. **User Profile** (lowest priority) - General user information

# Question:
{question}

 # INSTRUCTIONS:
    1. Carefully analyze all provided memories (facts and entities)
    2. Pay special attention to the timestamps to determine when events occurred
    3. If the question asks about a specific event or fact, look for direct evidence
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. Always convert relative time references to specific dates, months, or years
    6. Be as specific as possible when talking about people, places, and events

# OUTPUT FORMAT:
Please provide your response in the following JSON format:

```json
{{
  "reasoning": "reasoning content",
  "answer": "Provide a detailed answer"
}}
```"""

_QUERY_VARIATIONS_PROMPT = """Generate {n} search query variations for the question below. Each variation should use different wording, focus on different entities, or approach from a different angle. Output one query per line, starting with "- ".

Question: {question}

Queries:"""

_SUFFICIENCY_CHECK_PROMPT = """You are evaluating whether retrieved memories contain enough information to answer a question.

Question: {question}

Retrieved memories:
{context}

If the memories contain sufficient information to answer the question, reply:
SUFFICIENT

If more information is needed, reply:
INSUFFICIENT
NEW_QUERY: <a refined search query to find the missing information>

Reply:"""


# ==================== Evaluation Classes ====================


class QuestionAnsweringEvaluator:
    """Evaluates question answering performance using ReMeLight."""

    def __init__(
        self,
        memory_processor: ReMeLightMemoryProcessor,
        eval_model_name: str = "qwen3-max",
    ):
        self.memory_processor = memory_processor
        self.eval_model_name = eval_model_name

    async def evaluate_questions(
        self,
        reme: ReMeLight,
        questions: list[dict],
        user_name: str,
        uuid: str,
        top_k: int = 20,
    ) -> list[dict]:
        """Evaluate all questions for one user."""
        results = []

        total = len(questions)
        for qi, qa in enumerate(questions):
            if qa.get("category") == 5:
                continue

            logger.info(f"  QA {qi+1}/{total}: {qa['question'][:80]}...")
            print(f"  QA {qi+1}/{total}: {qa['question'][:60]}...", flush=True)
            answer_dict, raw_results, duration_ms = (
                await self.memory_processor.search_memory(
                    reme=reme,
                    query=qa["question"],
                    top_k=top_k,
                )
            )

            system_answer = answer_dict.get("answer", "")
            system_reasoning = answer_dict.get("reasoning", "")

            # Evaluate LLM-generated answer
            eval_result = await _evaluation_for_question(
                reme=reme,
                question=qa["question"],
                golden_answer=qa["answer"],
                generated_answer=system_answer,
                model_name=self.eval_model_name,
            )

            # Also evaluate raw search results
            raw_memories_text = _format_search_results_for_prompt(raw_results)
            eval_raw = await _evaluation_for_question(
                reme=reme,
                question=qa["question"],
                golden_answer=qa["answer"],
                generated_answer=raw_memories_text,
                model_name=self.eval_model_name,
            )

            qa_result = {
                **qa,
                "uuid": uuid,
                "system_response": system_answer,
                "system_reasoning": system_reasoning,
                "retrieved_memories": raw_memories_text,
                "raw_search_results": raw_results,
                "search_duration_ms": duration_ms,
                "result_type": eval_result.get("evaluation_result"),
                "question_answering_reasoning": eval_result.get("reasoning", ""),
                "original_result_type": eval_raw.get("evaluation_result"),
                "original_question_answering_reasoning": eval_raw.get("reasoning", ""),
            }
            results.append(qa_result)

        return results


class MetricsAggregator:
    """Aggregates evaluation metrics (same as vector-based eval)."""

    @staticmethod
    def _compute_single_metric(
        qa_records: list[dict], result_key: str
    ) -> dict[str, Any]:
        total = len(qa_records)
        if total == 0:
            return {
                "correct_qa_ratio(all)": 0,
                "correct_qa_ratio(valid)": 0,
                "qa_valid_num": 0,
                "qa_num": 0,
                "category_1_accuracy": 0.0,
                "category_2_accuracy": 0.0,
                "category_3_accuracy": 0.0,
                "category_4_accuracy": 0.0,
            }

        correct = 0
        valid = 0
        cat_correct = {1: 0, 2: 0, 3: 0, 4: 0}
        cat_total = {1: 0, 2: 0, 3: 0, 4: 0}

        for qa in qa_records:
            cat = qa.get("category", 0)
            if cat in cat_total:
                cat_total[cat] += 1

            result_type = qa.get(result_key, "")
            if result_type is not None and cat in (1, 2, 3, 4):
                valid += 1
                if result_type is True:
                    correct += 1
                    if cat in cat_correct:
                        cat_correct[cat] += 1

        metrics = {
            "correct_qa_ratio(all)": correct / total,
            "correct_qa_ratio(valid)": correct / valid if valid > 0 else 0,
            "qa_valid_num": valid,
            "qa_num": total,
        }
        for cat in (1, 2, 3, 4):
            metrics[f"category_{cat}_accuracy"] = (
                cat_correct[cat] / cat_total[cat] if cat_total[cat] > 0 else 0.0
            )

        return metrics

    @staticmethod
    def compute_qa_metrics(qa_records: list[dict]) -> dict[str, Any]:
        return {
            "with_llm_answer": MetricsAggregator._compute_single_metric(
                qa_records, "result_type"
            ),
            "with_original_memories": MetricsAggregator._compute_single_metric(
                qa_records, "original_result_type"
            ),
        }

    @staticmethod
    def compute_time_metrics(eval_results_file: str) -> dict[str, float]:
        add_duration = 0
        search_duration = 0

        with open(eval_results_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                user_data = json.loads(line)
                for session in user_data.get("sessions", []):
                    add_duration += session.get("add_dialogue_duration_ms", 0)
                for qa in user_data.get("evaluation_results", {}).get(
                    "question_answering_records", []
                ):
                    search_duration += qa.get("search_duration_ms", 0)

        return {
            "add_dialogue_duration_time": add_duration / 1000 / 60,
            "search_memory_duration_time": search_duration / 1000 / 60,
            "total_duration_time": (add_duration + search_duration) / 1000 / 60,
        }


def parse_locomo_timestamp(timestamp_str: str) -> datetime | None:
    """Parse LoCoMo timestamp format: '6:07 pm on 13 January, 2023'."""
    timestamp_str = timestamp_str.replace("\\s+", " ").strip()

    if timestamp_str.lower() == "unknown" or not timestamp_str:
        return None

    try:
        return datetime.strptime(timestamp_str, "%I:%M %p on %d %B, %Y")
    except ValueError:
        logger.warning(f"Failed to parse timestamp: {timestamp_str}")
        return None


# ==================== Main Evaluator ====================


class LocomoReMeLightEvaluator:
    """Main evaluation orchestrator for ReMeLight on LoCoMo benchmark."""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.file_manager = FileManager(config.output_dir)
        self.memory_processor = ReMeLightMemoryProcessor(
            eval_model_name=config.eval_model_name,
            index_wait_seconds=config.index_wait_seconds,
        )
        self.qa_evaluator = QuestionAnsweringEvaluator(
            memory_processor=self.memory_processor,
            eval_model_name=config.eval_model_name,
        )
        self.data_loader = DataLoader()
        self._update_lock: asyncio.Lock | None = None
        self._output_file: str | None = None
        self._reme_instances: list[ReMeLight] = []

    async def create_reme(self, working_dir: str) -> ReMeLight:
        """Create a ReMeLight instance with eval configuration."""
        reme = ReMeLight(
            working_dir=working_dir, # 每个用户独立目录
            default_as_llm_config={  # 摘要用的 LLM
                "model_name": self.config.reme_model_name,
                "backend": "openai",
                "stream": False,
            },
            default_embedding_model_config={ # embedding
                "model_name": "text-embedding-v4",
                "backend": "openai",
            },
            default_file_store_config={ # 开启混合搜索
                "fts_enabled": True,
                "vector_enabled": True,
            },
            enable_load_env=True,
        )
        await reme.start()
        self._reme_instances.append(reme)
        return reme

    # 单个用户的完整评测流水线
    async def process_user(self, user_data: dict) -> dict:
        """Process all sessions for one user conversation."""
        conv = user_data["conversation"]
        speaker_a = conv["speaker_a"]
        speaker_b = conv["speaker_b"]
        uuid = f"{speaker_a}_{speaker_b}"
        user_name = [speaker_a, speaker_b]
        user_file_name = f"{speaker_a}_{speaker_b}"

        working_dir = str(
            Path(self.config.output_dir) / "working_dirs" / user_file_name
        )

        if self.config.resume:
            if not Path(working_dir).exists():
                logger.error(f"Resume mode: working_dir not found: {working_dir}")
                return {"uuid": uuid, "user_name": user_file_name, "status": "no_working_dir"}
            logger.info(f"Resume mode: reusing {working_dir}")
        else:
            if Path(working_dir).exists():
                shutil.rmtree(working_dir)

        # 初始化 ReMeLight，配 LLM + Embedding + FileWatcher
        reme = await self.create_reme(working_dir)

        session_num = (
            19 if uuid == "Caroline_Melanie"
            else int(len(conv) / 2 - 1)
        )
        time_interval = 60

        logger.info(
            f"Processing user {user_name}: {session_num} sessions, "
            f"working_dir={working_dir}"
        )

        if not self.config.resume:
            # 循环 19 个 session: 调 summary_memory() 写 memory/*.md
            for idx in range(session_num):
                logger.info(
                    f"  Session {idx + 1}/{session_num} for {user_file_name}"
                )
                session_data = {
                    "uuid": uuid,
                    "user_name": user_file_name,
                    "timestamp": conv[f"session_{idx + 1}_date_time"],
                    "session": conv[f"session_{idx + 1}"],
                }

                dialogue = conv[f"session_{idx + 1}"]
                base_timestamp = parse_locomo_timestamp(session_data["timestamp"])
                if base_timestamp is None:
                    base_timestamp = datetime(2023, 1, 1)

                formatted_messages = self.data_loader.format_dialogue_messages(
                    dialogue, speaker_a, base_timestamp, time_interval
                )

                # 调 summary_memory() 写 memory/*.md
                summary_text, duration_ms = await self.memory_processor.add_memories(
                    reme=reme,
                    messages=formatted_messages,
                    batch_size=self.config.batch_size,
                )

                session_data.update({
                    "dialogue": dialogue,
                    "summary_text": summary_text,
                    "add_dialogue_duration_ms": duration_ms,
                })

                self.file_manager.save_session(user_file_name, idx, session_data)

        qas = user_data.get("qa", [])
        # 逐条 QA：搜记忆 → LLM 回答 → LLM 裁判
        qa_results = await self.qa_evaluator.evaluate_questions(
            reme=reme,
            questions=qas,
            user_name=user_file_name,
            uuid=uuid,
            top_k=self.config.top_k,
        )

        question_file = self.file_manager.get_user_dir(user_file_name) / "questions.json"
        with open(question_file, "w", encoding="utf-8") as f:
            json.dump({"question_answering_records": qa_results}, f, ensure_ascii=False, indent=2)

        await reme.close()

        return {"uuid": uuid, "user_name": user_file_name, "status": "ok"}

    # 整个评测的调度中心
    async def run_evaluation(self):
        """Run the complete evaluation pipeline."""
        start_time = time.time()

        all_users = self.data_loader.load_json(self.config.data_path)
        users_to_process = all_users[: self.config.user_num]

        print("\n" + "=" * 80)
        print("LOCOMO EVALUATION - ReMeLight (FILE-BASED)")
        print(f"Users: {len(users_to_process)} | Concurrency: {self.config.max_concurrency}")
        print(f"Output: {self.config.output_dir}")
        print("=" * 80 + "\n")

        self._output_file = os.path.join(self.config.output_dir, "eval_results.jsonl")
        self._update_lock = asyncio.Lock()

        semaphore = asyncio.Semaphore(self.config.max_concurrency)

        async def process_with_cache(idx: int, user_data: dict):
            async with semaphore:
                user_name = (
                    f"{user_data['conversation']['speaker_a']}_"
                    f"{user_data['conversation']['speaker_b']}"
                )

                if self.file_manager.user_has_cache(user_name):
                    logger.info(f"[{idx}/{len(users_to_process)}] Skipping {user_name} (cached)")
                    return {"user_name": user_name, "status": "cached"}

                logger.info(f"[{idx}/{len(users_to_process)}] Processing {user_name}...")
                result = await self.process_user(user_data)
                logger.info(f"[{idx}/{len(users_to_process)}] Completed {user_name}")

                await self._trigger_update()
                return result

        tasks = [
            process_with_cache(idx, user)
            for idx, user in enumerate(users_to_process, 1)
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = time.time() - start_time
        print(f"\nProcessing completed in {elapsed:.2f}s")
        print(f"Results: {self._output_file}\n")

        await self._aggregate_and_report(self._output_file)

    async def _trigger_update(self):
        if self._update_lock is None or self._output_file is None:
            return
        async with self._update_lock:
            self.file_manager.combine_results(self._output_file)
            self._update_statistics(self._output_file)

    def _update_statistics(self, results_file: str):
        if not os.path.exists(results_file):
            return

        qa_records = []
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    user_data = json.loads(line)
                    eval_results = user_data.get("evaluation_results", {})
                    qa_records.extend(
                        eval_results.get("question_answering_records", [])
                    )
        except (json.JSONDecodeError, KeyError):
            return

        if not qa_records:
            return

        qa_metrics = MetricsAggregator.compute_qa_metrics(qa_records)
        time_metrics = MetricsAggregator.compute_time_metrics(results_file)

        final_results = {
            "overall_score": {
                "question_answering": qa_metrics,
                "time_consuming": time_metrics,
            },
        }

        report_file = os.path.join(self.config.output_dir, "eval_statistics.json")
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)

    async def _aggregate_and_report(self, results_file: str):
        print("=" * 80)
        print("AGGREGATING METRICS")
        print("=" * 80 + "\n")

        qa_records = []
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                user_data = json.loads(line)
                eval_results = user_data.get("evaluation_results", {})
                qa_records.extend(
                    eval_results.get("question_answering_records", [])
                )

        qa_metrics = MetricsAggregator.compute_qa_metrics(qa_records)
        time_metrics = MetricsAggregator.compute_time_metrics(results_file)

        final_results = {
            "overall_score": {
                "question_answering": qa_metrics,
                "time_consuming": time_metrics,
            },
            "question_answering_records": qa_records,
        }

        report_file = os.path.join(self.config.output_dir, "eval_statistics.json")
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)

        self._print_summary(qa_metrics, time_metrics)

    def _print_summary(self, qa_metrics: dict, time_metrics: dict):
        print("=" * 80)
        print("EVALUATION SUMMARY - ReMeLight (FILE-BASED)")
        print("=" * 80 + "\n")

        llm_metrics = qa_metrics["with_llm_answer"]
        print("Question Answering (with LLM answer):")
        print(f"  Correct (all):   {llm_metrics['correct_qa_ratio(all)']:.4f}")
        print(f"  Correct (valid): {llm_metrics['correct_qa_ratio(valid)']:.4f}")
        print(f"  Valid/Total:     {llm_metrics['qa_valid_num']}/{llm_metrics['qa_num']}")
        print(f"  Category 1 Accuracy: {llm_metrics['category_1_accuracy']:.4f}")
        print(f"  Category 2 Accuracy: {llm_metrics['category_2_accuracy']:.4f}")
        print(f"  Category 3 Accuracy: {llm_metrics['category_3_accuracy']:.4f}")
        print(f"  Category 4 Accuracy: {llm_metrics['category_4_accuracy']:.4f}")

        orig_metrics = qa_metrics["with_original_memories"]
        print("\nQuestion Answering (with original memories):")
        print(f"  Correct (all):   {orig_metrics['correct_qa_ratio(all)']:.4f}")
        print(f"  Correct (valid): {orig_metrics['correct_qa_ratio(valid)']:.4f}")
        print(f"  Valid/Total:     {orig_metrics['qa_valid_num']}/{orig_metrics['qa_num']}")

        print("\nTime Metrics:")
        print(f"  Memory Addition:  {time_metrics['add_dialogue_duration_time']:.2f} min")
        print(f"  Memory Search:    {time_metrics['search_memory_duration_time']:.2f} min")
        print(f"  Total:            {time_metrics['total_duration_time']:.2f} min")
        print("\n" + "=" * 80)


# ==================== Main ====================


async def main_async(
    data_path: str,
    top_k: int = 20,
    user_num: int = 1,
    max_concurrency: int = 2,
    reme_model_name: str = "qwen-flash",
    eval_model_name: str = "qwen3-max",
    output_dir: str = "bench_results/reme_light",
    index_wait_seconds: int = 5,
    resume: bool = False,
):
    config = EvalConfig(
        data_path=data_path,
        top_k=top_k,
        user_num=user_num,
        max_concurrency=max_concurrency,
        reme_model_name=reme_model_name,
        eval_model_name=eval_model_name,
        output_dir=output_dir,
        index_wait_seconds=index_wait_seconds,
        resume=resume,
    )

    evaluator = LocomoReMeLightEvaluator(config)
    await evaluator.run_evaluation()


def main(
    data_path: str,
    top_k: int = 20,
    user_num: int = 1,
    max_concurrency: int = 2,
    reme_model_name: str = "qwen-flash",
    eval_model_name: str = "qwen3-max",
    output_dir: str = "bench_results/reme_light",
    index_wait_seconds: int = 5,
    resume: bool = False,
):
    asyncio.run(
        main_async(
            data_path=data_path,
            top_k=top_k,
            user_num=user_num,
            max_concurrency=max_concurrency,
            reme_model_name=reme_model_name,
            eval_model_name=eval_model_name,
            output_dir=output_dir,
            index_wait_seconds=index_wait_seconds,
            resume=resume,
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ReMeLight (file-based) evaluation on LoCoMo benchmark"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="benchmark/locomo/data/locomo10.json",
        help="Path to LoCoMo data file (default: benchmark/locomo/data/locomo10.json)",
    )
    parser.add_argument(
        "--top_k", type=int, default=20, help="Max memory search results (default: 20)"
    )
    parser.add_argument(
        "--user_num", type=int, default=1, help="Number of users to evaluate"
    )
    parser.add_argument(
        "--max_concurrency", type=int, default=2, help="Max concurrent users"
    )
    parser.add_argument(
        "--reme_model_name",
        type=str,
        default="qwen-flash",
        help="Model for ReMeLight summarization",
    )
    parser.add_argument(
        "--eval_model_name",
        type=str,
        default="qwen3-max",
        help="Model for LLM-as-Judge evaluation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="bench_results/reme_light",
        help="Output directory for results",
    )
    parser.add_argument(
        "--index_wait_seconds",
        type=int,
        default=5,
        help="Seconds to wait for FileWatcher re-indexing after summarization",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Skip summarization, go straight to QA (reuse existing working_dir)",
    )

    args = parser.parse_args()
    print(f"Args: {args}")

    # Check if data file exists, print helpful instructions if not
    if not os.path.exists(args.data_path):
        print(f"\n  Data file not found: {args.data_path}\n")
        print("To download the LoCoMo dataset:")
        print("  mkdir -p benchmark/locomo/data")
        print("  git clone https://github.com/luyanhexay/locomo-dynamemory.git /tmp/locomo-dynamemory")
        print("  cp /tmp/locomo-dynamemory/data/locomo10.json benchmark/locomo/data/\n")
        print("Or specify a custom path:")
        print("  python benchmark/locomo/eval_reme_light.py --data_path /path/to/locomo10.json\n")
        import sys
        sys.exit(1)

    main(
        data_path=args.data_path,
        top_k=args.top_k,
        user_num=args.user_num,
        max_concurrency=args.max_concurrency,
        reme_model_name=args.reme_model_name,
        eval_model_name=args.eval_model_name,
        output_dir=args.output_dir,
        index_wait_seconds=args.index_wait_seconds,
        resume=args.resume,
    )
