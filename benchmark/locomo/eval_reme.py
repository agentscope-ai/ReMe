"""
Simplified evaluation script for ReMe on Locomo benchmark.

This script performs a simplified evaluation pipeline:
1. Load Locomo data
2. Process each user's sessions with ReMe (summary + retrieve)
3. Evaluate question answering
4. Generate metrics and statistics

Usage:
    python bench/halumem/eval_reme_simple.py --data_path locomo10.json \
        --top_k 20 --user_num 100 --max_concurrency 20
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
import yaml
from loguru import logger
from reme.core.enumeration import Role
from reme.core.schema import Message


from reme.reme import ReMe


# ==================== Configuration ====================
@dataclass
class EvalConfig:
    """Evaluation configuration parameters."""

    data_path: str
    top_k: int = 20
    user_num: int = 1
    max_concurrency: int = 2
    batch_size: int = 40
    output_dir: str = "bench_results/reme"
    reme_model_name: str = "qwen-flash"
    eval_model_name: str = "qwen3-max"
    algo_version: str = "locomo"
    enable_thinking_params: bool = False


# ==================== Utilities ====================


class DataLoader:
    """Handles loading and parsing of HaluMem data."""

    @staticmethod
    def load_jsonl(file_path: str) -> list[dict]:
        """Load all entries from a JSONL file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line.strip()) for line in f if line.strip()]

    @staticmethod
    def load_json(file_path: str) -> dict:
        """Load dict from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def format_dialogue_messages(
        dialogue: list[dict],
        speaker_a: str,
        base_timestamp: datetime,
        time_interval: int,
    ) -> list[dict]:
        """Format dialogue into ReMe message format with conversation_time."""

        return [
            {
                "role": "user" if turn["speaker"] == speaker_a else "assistant",
                "name": turn["speaker"],
                "content": turn["text"],
                "time_created": (base_timestamp + timedelta(seconds=idx * time_interval)).strftime("%Y-%m-%d %H:%M:%S"),
            }
            for idx, turn in enumerate(dialogue)
        ]

    @staticmethod
    def format_dialogue_for_eval(dialogue: list[dict], user_name: str = None) -> str:
        """Format dialogue into string for evaluation."""
        formatted_turns = []
        for turn in dialogue:
            timestamp = (
                datetime.strptime(
                    turn["timestamp"],
                    "%b %d, %Y, %H:%M:%S",
                )
                .replace(tzinfo=timezone.utc)
                .strftime("%Y-%m-%d %H:%M:%S")
            )

            # Use user_name if role is 'user' and user_name is provided
            role = user_name if turn["role"] == "user" and user_name else turn["role"]

            formatted_turns.append(
                f"Role: {role}\n" f"Content: {turn['content']}\n" f"Time: {timestamp}",
            )
        return "\n\n".join(formatted_turns)


class FileManager:
    """Manages file I/O operations."""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.tmp_dir = self.base_dir
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def get_user_dir(self, user_name: str) -> Path:
        """Get the directory path for a user."""
        user_dir = self.tmp_dir / user_name
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir

    def get_session_file(self, user_name: str, session_id: int) -> Path:
        """Get the file path for a specific session."""
        return self.get_user_dir(user_name) / f"session_{session_id}.json"

    def get_question_file(self, user_name: str) -> Path:
        """Get the file path for a specific question."""
        return self.get_user_dir(user_name) / "questions.json"

    def save_session(self, user_name: str, session_id: int, data: dict):
        """Save session data to file."""
        file_path = self.get_session_file(user_name, session_id)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Saved session {session_id} to {file_path}")

    def save_question(self, user_name: str, data: dict):
        """Save question data to file"""
        file_path = self.get_question_file(user_name)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Saved question to {file_path}")

    def load_session(self, user_name: str, session_id: int) -> dict | None:
        """Load session data from file."""
        file_path = self.get_session_file(user_name, session_id)
        if not file_path.exists():
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def user_has_cache(self, user_name: str) -> bool:
        """Check if user has cached results."""
        user_dir = self.get_user_dir(user_name)
        return any(f.name.startswith("session_") and f.suffix == ".json" for f in user_dir.iterdir())

    def combine_results(self, output_file: str):
        """Combine all user session files into a single JSONL file."""
        with open(output_file, "w", encoding="utf-8") as f_out:
            for user_dir in self.tmp_dir.iterdir():
                if not user_dir.is_dir():
                    continue

                session_files = sorted(
                    [f for f in user_dir.iterdir() if f.name.startswith("session_") and f.suffix == ".json"],
                )

                if not session_files:
                    continue

                # Load first session to get user metadata
                with open(session_files[0], "r", encoding="utf-8") as f_in:
                    first_session = json.load(f_in)

                user_data = {
                    "uuid": first_session["uuid"],
                    "user_name": first_session["user_name"],
                    "sessions": [],
                }

                # Load all sessions
                for session_file in session_files:
                    with open(session_file, "r", encoding="utf-8") as f_in:
                        session_data = json.load(f_in)
                        # Remove redundant user metadata
                        session_data.pop("uuid", None)
                        session_data.pop("user_name", None)
                        user_data["sessions"].append(session_data)

                question_file = user_dir / "questions.json"
                if not question_file.exists():
                    continue
                with open(question_file, "r", encoding="utf-8") as f_in:
                    question_data = json.load(f_in)
                    user_data["evaluation_results"] = {
                        "question_answering_records": question_data,
                    }

                f_out.write(json.dumps(user_data, ensure_ascii=False) + "\n")


# ==================== Memory Operations ====================


class MemoryProcessor:
    """Handles ReMe memory operations."""

    def __init__(
        self,
        reme: ReMe,
        eval_model_name: str = "qwen3-max",
        algo_version: str = "locomo",
        enable_thinking_params: bool = False,
    ):
        self.reme = reme
        self.eval_model_name = eval_model_name
        self.algo_version = algo_version
        self.enable_thinking_params = enable_thinking_params

    async def add_memories(
        self,
        user_id: str,
        messages: list[dict],
        batch_size: int = 10000,
    ) -> tuple[list[str], list, float]:
        """
        Add memories in batches using ReMe and return extracted memory contents.

        Returns:
            tuple: (extracted_memories, agent_messages, total_duration_ms)
        """
        extracted_memories = []
        summary_messages = []
        total_duration_ms = 0

        for i in range(0, len(messages), batch_size):
            batch = messages[i : i + batch_size]
            start = time.time()

            # Use new summary API
            result = await self.reme.summarize_memory(
                messages=batch,
                user_name=user_id,
                version=self.algo_version,
                return_dict=True,
                enable_time_filter=True,
                enable_thinking_params=self.enable_thinking_params,
            )

            duration_ms = (time.time() - start) * 1000
            total_duration_ms += duration_ms

            extracted_memories.extend([m.model_dump(exclude_none=True) for m in result["answer"]])
            summary_messages.extend([m.simple_dump(enable_argument_dict=True) for m in result["messages"]])

        return extracted_memories, summary_messages, total_duration_ms

    async def search_memory(
        self,
        query: str,
        user_id: str,
        top_k: int = 20,
    ) -> tuple[dict, list, float]:
        """
        Search memory using ReMe and return structured answer with reasoning.

        Returns:
            tuple: (answer_dict, agent_messages, duration_ms)
                answer_dict contains: {"reasoning": str, "answer": str, "memories": str}
        """
        start = time.time()

        # Retrieve memories from ReMe using new API
        result = await self.reme.retrieve_memory(
            query=query,
            retrieve_top_k=top_k,
            user_name=user_id,
            version=self.algo_version,
            return_dict=True,
            enable_time_filter=True,
            enable_thinking_params=self.enable_thinking_params,
        )

        # Extract memories from response
        memories = result["answer"]
        agent_messages = [x.simple_dump(enable_argument_dict=True) for x in result["messages"]]
        retrieved_nodes = [x.model_dump(exclude_none=True) for x in result["retrieved_nodes"]]

        # Use LLM to generate structured answer from memories
        answer_result = await answer_question_with_memories(
            reme=self.reme,
            question=query,
            memories=memories,
            user_id=user_id,
            model_name=self.eval_model_name,
        )

        # Add original memories to the result
        answer_result["memories"] = memories
        answer_result["retrieved_nodes"] = retrieved_nodes

        duration_ms = (time.time() - start) * 1000
        return answer_result, agent_messages, duration_ms


# ==================== Evaluation Functions ====================


async def answer_question_with_memories(
    reme: ReMe,
    question: str,
    memories: str,
    user_id: str = None,
    model_name: str = "qwen3-30b-a3b-instruct-2507",
):
    """
    Answer a question using retrieved memories with PROMPT_MEMZERO_JSON template.

    Args:
        reme: ReMe instance with default_llm and prompt_handler
        question: The question to answer
        memories: The retrieved memories (formatted as context)
        user_id: Optional user ID for context formatting
        model_name: Model name to use for LLM request

    Returns:
        dict with 'reasoning' and 'answer' fields
    """
    # Format context with memories
    if user_id:
        context = reme.prompt_handler.prompt_format(
            "TEMPLATE_MEMOS",
            user_id=user_id,
            memories=memories,
        )
    else:
        context = f"Memories:\n{memories}"

    # Use PROMPT_MEMZERO_JSON template for structured JSON response
    prompt = reme.prompt_handler.prompt_format(
        "PROMPT_MEMZERO_JSON",
        context=context,
        question=question,
    )

    result = await reme.get_llm("qwen3_max_instruct").simple_request_for_json(
        prompt=prompt,
        model_name=model_name,
    )

    return result


async def evaluation_for_question(
    reme: ReMe,
    question: str,
    golden_answer: str,
    generated_answer: str,
    model_name: str = "qwen3-max",
):
    """
    Question-Answering Evaluation with optional Dialogue Context.

    Args:
        reme: ReMe instance with default_llm and prompt_handler
        question: The question string to be evaluated.
        golden_answer: The reference (gold-standard) answer.
        generated_answer: The answer produced by the memory system.
        model_name: Model name to use for LLM request

    Returns:
        dict with 'reasoning' and 'evaluation_result' fields
    """
    await asyncio.sleep(10)
    # Use configured prompts
    system_prompt = reme.prompt_handler.prompt_format(
        "SYSTEM_PROMPT",
    )
    user_prompt = reme.prompt_handler.prompt_format(
        "USER_PROMPT",
        question=question,
        golden_answer=golden_answer,
        generated_answer=generated_answer,
    )

    reme_result = await reme.get_llm("qwen3_max_instruct").chat(
        messages=[
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=user_prompt),
        ],
        model_name=model_name,
    )

    content = reme_result.content
    match = re.search(r'"label"\s*:\s*"([^"]*?)"', content)
    if match:
        label = match.group(1)
    else:
        label = "WRONG"
    result = {
        "reasoning": content,
        "evaluation_result": label.strip().upper() == "CORRECT",
    }
    return result


# ==================== Evaluation ====================


class QuestionAnsweringEvaluator:
    """Evaluates question answering performance."""

    def __init__(self, memory_processor: MemoryProcessor, reme: ReMe, top_k: int, eval_model_name: str = "qwen3-max"):
        self.memory_processor = memory_processor
        self.reme = reme
        self.top_k = top_k
        self.eval_model_name = eval_model_name

    async def evaluate_questions(
        self,
        questions: list[dict],
        user_name: str,
        uuid: str,
    ) -> list[dict]:
        """Evaluate all questions for a conversation."""
        results = []

        for qa in questions:
            if qa["category"] == 5:
                continue
            answer_dict, agent_messages, duration_ms = await self.memory_processor.search_memory(
                query=qa["question"],
                user_id=user_name,
                top_k=self.top_k,
            )

            # Extract answer and reasoning from the structured response
            system_answer = answer_dict.get("answer", "")
            system_reasoning = answer_dict.get("reasoning", "")
            retrieved_memories = answer_dict.get("memories", "")
            retrieved_nodes = answer_dict.get("retrieved_nodes", "")

            # Evaluate response
            eval_result = await evaluation_for_question(
                reme=self.reme,
                question=qa["question"],
                golden_answer=qa["answer"],
                generated_answer=system_answer,
                model_name=self.eval_model_name,
            )

            eval_result_original_answer = await evaluation_for_question(
                reme=self.reme,
                question=qa["question"],
                golden_answer=qa["answer"],
                generated_answer=retrieved_memories,
                model_name=self.eval_model_name,
            )

            # Build result record
            qa_result = {
                **qa,
                "uuid": uuid,
                "system_response": system_answer,
                "system_reasoning": system_reasoning,
                "retrieved_memories": retrieved_memories,
                "retrieved_nodes": retrieved_nodes,
                "retrieve_messages": agent_messages,
                "search_duration_ms": duration_ms,
                "result_type": eval_result.get("evaluation_result"),
                "question_answering_reasoning": eval_result.get("reasoning", ""),
                "original_result_type": eval_result_original_answer.get("evaluation_result"),
                "original_question_answering_reasoning": eval_result_original_answer.get("reasoning", ""),
            }
            results.append(qa_result)

        return results


class MetricsAggregator:
    """Aggregates evaluation metrics."""

    @staticmethod
    def _compute_single_metric(qa_records: list[dict], result_key: str) -> dict[str, Any]:
        """Compute metrics for a single result type key."""
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

        category_1_correct = 0
        category_1_num = 0
        category_1_valid = 0
        category_2_correct = 0
        category_2_num = 0
        category_2_valid = 0
        category_3_correct = 0
        category_3_num = 0
        category_3_valid = 0
        category_4_correct = 0
        category_4_num = 0
        category_4_valid = 0

        for qa in qa_records:
            result_type = qa.get(result_key, "")
            category = qa.get("category", 0)
            if category == 1:
                category_1_num += 1
            elif category == 2:
                category_2_num += 1
            elif category == 3:
                category_3_num += 1
            elif category == 4:
                category_4_num += 1

            if result_type is not None and category in [1, 2, 3, 4]:
                valid += 1
                if result_type is True:
                    correct += 1

                if category == 1:
                    category_1_valid += 1
                    if result_type is True:
                        category_1_correct += 1
                elif category == 2:
                    category_2_valid += 1
                    if result_type is True:
                        category_2_correct += 1
                elif category == 3:
                    category_3_valid += 1
                    if result_type is True:
                        category_3_correct += 1
                elif category == 4:
                    category_4_valid += 1
                    if result_type is True:
                        category_4_correct += 1

        metrics = {
            "correct_qa_ratio(all)": correct / total,
            "qa_valid_num": valid,
            "qa_num": total,
            "category_1_accuracy": category_1_correct / category_1_num if category_1_num > 0 else 0,
            "category_1_num": category_1_num,
            "category_1_valid_num": category_1_valid,
            "category_2_accuracy": category_2_correct / category_2_num if category_2_num > 0 else 0,
            "category_2_num": category_2_num,
            "category_2_valid_num": category_2_valid,
            "category_3_accuracy": category_3_correct / category_3_num if category_3_num > 0 else 0,
            "category_3_num": category_3_num,
            "category_3_valid_num": category_3_valid,
            "category_4_accuracy": category_4_correct / category_4_num if category_4_num > 0 else 0,
            "category_4_num": category_4_num,
            "category_4_valid_num": category_4_valid,
        }

        if valid > 0:
            metrics.update(
                {
                    "correct_qa_ratio(valid)": correct / valid,
                },
            )
        else:
            metrics.update(
                {
                    "correct_qa_ratio(valid)": 0,
                },
            )

        return metrics

    @staticmethod
    def compute_qa_metrics(qa_records: list[dict]) -> dict[str, Any]:
        """Compute question answering metrics for both result_type and original_result_type."""
        return {
            "with_llm_answer": MetricsAggregator._compute_single_metric(qa_records, "result_type"),
            "with_original_memories": MetricsAggregator._compute_single_metric(qa_records, "original_result_type"),
        }

    @staticmethod
    def compute_time_metrics(eval_results_file: str) -> dict[str, float]:
        """Compute timing metrics from evaluation results."""
        add_duration = 0
        search_duration = 0

        with open(eval_results_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                user_data = json.loads(line)

                for session in user_data["sessions"]:
                    add_duration += session.get("add_dialogue_duration_ms", 0)

                eval_results = user_data.get("evaluation_results", {})
                for qa in eval_results.get("question_answering_records", []):
                    search_duration += qa.get("search_duration_ms", 0)

        # Convert to minutes
        return {
            "add_dialogue_duration_time": add_duration / 1000 / 60,
            "search_memory_duration_time": search_duration / 1000 / 60,
            "total_duration_time": (add_duration + search_duration) / 1000 / 60,
        }


# ==================== Evaluator ====================


class LocomoEvaluator:
    """
    LOCOMO 评估器核心类
    用于评估 MemAgent 的记忆完整性、记忆准确性和问答准确性
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        with open("eval_reme.yaml", "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
            self.summary_prompt_1 = data["user_message_summary_1"]
            self.summary_prompt_2 = data["user_message_summary_2"]
            self.retriever_prompt = data["user_message_retrieve"]

        ops_dict = {
            "personal_summarizer": {
                "prompt_dict": {
                    "user_message_s1": self.summary_prompt_1,
                    "user_message_s2": self.summary_prompt_2,
                },
            },
            "personal_retriever": {
                "prompt_dict": {
                    "user_message": self.retriever_prompt,
                },
                "params": {
                    "return_memory_nodes": True,
                },
            },
        }

        self.reme = ReMe(
            default_llm_config={
                "model_name": self.config.reme_model_name,
            },
            ops=ops_dict,
        )

        # Load evaluation prompts into ReMe's prompt handler
        prompts_yaml_path = Path(__file__).parent / "eval_reme.yaml"
        self.reme.prompt_handler.load_prompt_by_file(prompts_yaml_path)

        self.file_manager = FileManager(config.output_dir)
        self.memory_processor = MemoryProcessor(
            self.reme,
            config.eval_model_name,
            config.algo_version,
            config.enable_thinking_params,
        )
        self.qa_evaluator = QuestionAnsweringEvaluator(
            self.memory_processor,
            self.reme,
            config.top_k,
            config.eval_model_name,
        )
        self.data_loader = DataLoader()

        # For real-time updates
        self._update_lock: asyncio.Lock | None = None
        self._output_file: str | None = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.reme.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.reme.close()
        return False

    async def process_user(self, user_data: dict) -> dict:
        """Process all sessions for a user."""
        speaker_a = user_data["conversation"]["speaker_a"]
        speaker_b = user_data["conversation"]["speaker_b"]
        uuid = f"{speaker_a}_{speaker_b}"
        user_name = [speaker_a, speaker_b]
        user_file_name = f"{speaker_a}_{speaker_b}"

        new_user_data = {
            "uuid": f"{user_data['conversation']['speaker_a']}_{user_data['conversation']['speaker_b']}",
            "user_name": user_name,
            "sessions": [],
            "qas": [],
            "eval_results": {},
        }
        logger.info(f"Processing user: {speaker_a} and {speaker_b}")
        session_num = 19 if uuid == "Caroline_Melanie" else int(len(user_data["conversation"]) / 2 - 1)
        time_interval = 60

        # Process conversation
        for idx in range(session_num):
            conversation = user_data["conversation"]
            logger.info(f"Processing user {user_name}: session {idx+1}/{session_num}")
            session_data = {
                "uuid": uuid,
                "user_name": user_file_name,
                "timestamp": conversation[f"session_{idx+1}_date_time"],
                "session": conversation[f"session_{idx+1}"],
            }

            # Format dialogue
            dialogue = conversation[f"session_{idx+1}"]
            base_timestamp = parse_locomo_timestamp(session_data["timestamp"])
            formatted_messages = self.data_loader.format_dialogue_messages(
                dialogue,
                speaker_a,
                base_timestamp,
                time_interval,
            )
            extracted_memories, agent_messages, duration_ms = await self.memory_processor.add_memories(
                user_id=user_name,
                messages=formatted_messages,
                batch_size=self.config.batch_size,
            )
            session_data.update(
                {
                    "dialogue": dialogue,
                    "extracted_memories": extracted_memories,
                    "summary_messages": agent_messages,
                    "add_dialogue_duration_ms": duration_ms,
                },
            )

            self.file_manager.save_session(user_file_name, idx, session_data)

        # Process questions
        qas = user_data["qa"]
        qa_results = await self.qa_evaluator.evaluate_questions(
            questions=qas,
            user_name=user_name,
            uuid=uuid,
        )

        new_user_data["evaluation_results"] = {
            "question_answering_records": qa_results,
        }
        self.file_manager.save_question(user_file_name, qa_results)

        # Update results file after each conversation completes
        await self._trigger_update()

        return {"uuid": uuid, "user_name": user_name, "status": "ok"}

    async def _trigger_update(self):
        """Trigger real-time update of results and statistics."""
        if self._update_lock is None or self._output_file is None:
            return

        async with self._update_lock:
            self.file_manager.combine_results(self._output_file)
            self._update_statistics(self._output_file)

    async def run_evaluation(self):
        """Run the complete evaluation pipeline using ReMe."""
        start_time = time.time()

        # Load user data first to get user names
        all_users = self.data_loader.load_json(self.config.data_path)
        users_to_process = all_users[: self.config.user_num]

        # Extract all user names and delete all profiles
        all_user_names = [
            f"{user_data['conversation']['speaker_a']}_&_{user_data['conversation']['speaker_b']}"
            for user_data in all_users
        ]
        if all_user_names:
            for user_name in all_user_names:
                self.reme.get_profile_handler(user_name).delete_all()
            logger.info(f"Deleted all profiles for {len(all_user_names)} users")

        # Clear existing data
        await self.reme.default_vector_store.delete_all()

        # Clear meta_memory directory
        meta_memory_path = Path(f"meta_memory/{self.reme.default_vector_store.collection_name}")
        if meta_memory_path.exists():
            shutil.rmtree(meta_memory_path)
            logger.info(f"Cleared meta_memory directory: {meta_memory_path}")
        meta_memory_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print("LOCOMO EVALUATION - REME - QUESTION ANSWERING")
        print(f"Users: {len(users_to_process)} | Concurrency: {self.config.max_concurrency}")
        print("=" * 80 + "\n")

        # Output file path for real-time updates
        self._output_file = os.path.join(self.config.output_dir, "eval_results.jsonl")

        # Lock for thread-safe file updates
        self._update_lock = asyncio.Lock()

        # Process users with concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrency)

        async def process_with_cache_check(idx: int, user_data: dict):
            async with semaphore:
                user_name = f"{user_data['conversation']['speaker_a']}_{user_data['conversation']['speaker_b']}"

                # Check cache
                if self.file_manager.user_has_cache(user_name):
                    print(f"⚡ [{idx}/{len(users_to_process)}] Skipping {user_name} (cached)")
                    result = {"user_name": user_name, "status": "cached"}
                    # Also trigger update for cached users
                    await self._trigger_update()
                else:
                    print(f"🔄 [{idx}/{len(users_to_process)}] Processing {user_name}...")
                    result = await self.process_user(user_data)
                    print(f"✅ [{idx}/{len(users_to_process)}] Completed {user_name}")

                return result

        tasks = [process_with_cache_check(idx, user) for idx, user in enumerate(users_to_process, 1)]
        await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = time.time() - start_time
        print(f"\n✅ Processing completed in {elapsed:.2f}s")
        print(f"📁 Results: {self._output_file}\n")

        # Final aggregation and report
        await self.aggregate_and_report(self._output_file)

    def _update_statistics(self, results_file: str):
        """Update statistics file based on current results (for real-time monitoring)."""
        if not os.path.exists(results_file):
            return

        # Collect all QA records
        qa_records = []
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    user_data = json.loads(line)
                    eval_results = user_data.get("evaluation_results", {})
                    qa_records.extend(
                        eval_results.get("question_answering_records", []),
                    )
        except (json.JSONDecodeError, KeyError):
            return

        if not qa_records:
            return

        # Compute metrics
        qa_metrics = MetricsAggregator.compute_qa_metrics(qa_records)
        time_metrics = MetricsAggregator.compute_time_metrics(results_file)

        final_results = {
            "overall_score": {
                "question_answering": qa_metrics,
                "time_consuming": time_metrics,
            },
            "question_answering_records": qa_records,
        }

        # Save statistics
        report_file = os.path.join(self.config.output_dir, "eval_statistics.json")
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)

    async def aggregate_and_report(self, results_file: str):
        """Aggregate results and generate final report."""
        print("=" * 80)
        print("AGGREGATING METRICS")
        print("=" * 80 + "\n")

        # Collect all QA records
        qa_records = []
        print(results_file)
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                user_data = json.loads(line)
                print(user_data)
                eval_results = user_data.get("evaluation_results", {})
                qa_records.extend(
                    eval_results.get("question_answering_records", []),
                )

        # Compute metrics
        qa_metrics = MetricsAggregator.compute_qa_metrics(qa_records)
        time_metrics = MetricsAggregator.compute_time_metrics(results_file)

        final_results = {
            "overall_score": {
                "question_answering": qa_metrics,
                "time_consuming": time_metrics,
            },
            "question_answering_records": qa_records,
        }

        # Save final report
        report_file = os.path.join(self.config.output_dir, "eval_statistics.json")
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)

        print(f"📊 Statistics saved to: {report_file}\n")

        # Print summary
        self._print_summary(qa_metrics, time_metrics)

    def _print_summary(self, qa_metrics: dict, time_metrics: dict):
        """Print evaluation summary."""
        print("=" * 80)
        print("EVALUATION SUMMARY - REME")
        print("=" * 80 + "\n")

        # Print metrics for LLM-generated answer (result_type)
        llm_metrics = qa_metrics["with_llm_answer"]
        print(llm_metrics)
        print("📊 Question Answering (with LLM answer):")
        print(f"  Correct (all):       {llm_metrics['correct_qa_ratio(all)']:.4f}")
        print(f"  Correct (valid):     {llm_metrics['correct_qa_ratio(valid)']:.4f}")
        print(f"  Valid/Total:         {llm_metrics['qa_valid_num']}/{llm_metrics['qa_num']}")
        print(f"  Category 1 Accuracy: {llm_metrics['category_1_accuracy']:.4f}")
        print(f"  Category 2 Accuracy: {llm_metrics['category_2_accuracy']:.4f}")
        print(f"  Category 3 Accuracy: {llm_metrics['category_3_accuracy']:.4f}")
        print(f"  Category 4 Accuracy: {llm_metrics['category_4_accuracy']:.4f}")

        # Print metrics for original retrieved memories (original_result_type)
        orig_metrics = qa_metrics["with_original_memories"]
        print("\n📊 Question Answering (with original memories):")
        print(f"  Correct (all):       {orig_metrics['correct_qa_ratio(all)']:.4f}")
        print(f"  Correct (valid):     {orig_metrics['correct_qa_ratio(valid)']:.4f}")
        print(f"  Valid/Total:         {orig_metrics['qa_valid_num']}/{orig_metrics['qa_num']}")
        print(f"  Category 1 Accuracy: {orig_metrics['category_1_accuracy']:.4f}")
        print(f"  Category 2 Accuracy: {orig_metrics['category_2_accuracy']:.4f}")
        print(f"  Category 3 Accuracy: {orig_metrics['category_3_accuracy']:.4f}")
        print(f"  Category 4 Accuracy: {orig_metrics['category_4_accuracy']:.4f}")

        print("\n⏱️  Time Metrics:")
        print(f"  Memory Addition:  {time_metrics['add_dialogue_duration_time']:.2f} min")
        print(f"  Memory Search:    {time_metrics['search_memory_duration_time']:.2f} min")
        print(f"  Total:            {time_metrics['total_duration_time']:.2f} min")
        print("\n" + "=" * 80)


def parse_locomo_timestamp(timestamp_str: str):
    """
    Parse LoCoMo timestamp format.

    Input format: "6:07 pm on 13 January, 2023"
    Special value: "Unknown" or unparseable returns None
    Output: datetime object or None
    """
    # Clean string
    timestamp_str = timestamp_str.replace("\\s+", " ").strip()

    # Handle special cases: Unknown or empty string
    if timestamp_str.lower() == "unknown" or not timestamp_str:
        # No time information, return None
        return None

    try:
        return datetime.strptime(timestamp_str, "%I:%M %p on %d %B, %Y")
    except ValueError:
        # If parse fails, return None and print warning
        print(f"⚠️  Warning: Failed to parse timestamp '{timestamp_str}', no timestamp will be set")
        return None


# ==================== Main Pipeline ====================


async def main_async(
    data_path: str,
    top_k: int,
    user_num: int,
    max_concurrency: int,
    reme_model_name: str = "qwen-flash",
    eval_model_name: str = "qwen3-max",
    algo_version: str = "halumem",
    enable_thinking_params: bool = False,
):
    """Main async entry point for ReMe evaluation with proper resource cleanup."""
    config = EvalConfig(
        data_path=data_path,
        top_k=top_k,
        user_num=user_num,
        max_concurrency=max_concurrency,
        reme_model_name=reme_model_name,
        eval_model_name=eval_model_name,
        algo_version=algo_version,
        enable_thinking_params=enable_thinking_params,
    )

    # Use async context manager for automatic cleanup
    async with LocomoEvaluator(config) as evaluator:
        await evaluator.run_evaluation()


def main(
    data_path: str,
    top_k: int = 20,
    user_num: int = 1,
    max_concurrency: int = 2,
    reme_model_name: str = "qwen-flash",
    eval_model_name: str = "qwen3-max",
    algo_version: str = "halumem",
    enable_thinking_params: bool = False,
):
    """Synchronous entry point."""
    asyncio.run(
        main_async(
            data_path=data_path,
            top_k=top_k,
            user_num=user_num,
            max_concurrency=max_concurrency,
            reme_model_name=reme_model_name,
            eval_model_name=eval_model_name,
            algo_version=algo_version,
            enable_thinking_params=enable_thinking_params,
        ),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simplified evaluation for ReMe on Locomo benchmark")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to Locomo data file (e.g., locomo10.jsonl)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of top memories to retrieve (default: 20)",
    )
    parser.add_argument(
        "--user_num",
        type=int,
        default=1,
        help="Number of users to evaluate (default: 1)",
    )
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=2,
        help="Maximum concurrency for processing (default: 2)",
    )
    parser.add_argument(
        "--reme_model_name",
        type=str,
        default="qwen-flash",
        help="Model name for ReMe (default: qwen-flash)",
    )
    parser.add_argument(
        "--eval_model_name",
        type=str,
        default="qwen3-max",
        help="Model name for evaluation (default: qwen3-max)",
    )
    parser.add_argument(
        "--algo_version",
        type=str,
        default="default",
        help="Algorithm version for summary and retrieval (default: halumem)",
    )
    parser.add_argument(
        "--enable_thinking_params",
        action="store_true",
        default=True,
        help="Enable thinking parameters for summary and retrieval (default: False)",
    )

    args = parser.parse_args()
    print(f"args={args}!")

    main(
        data_path=args.data_path,
        top_k=args.top_k,
        user_num=args.user_num,
        max_concurrency=args.max_concurrency,
        reme_model_name=args.reme_model_name,
        eval_model_name=args.eval_model_name,
        algo_version=args.algo_version,
        enable_thinking_params=args.enable_thinking_params,
    )
