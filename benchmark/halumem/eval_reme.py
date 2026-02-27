"""
HaluMem Benchmark Evaluator for ReMe - Question Answering

A modular evaluation pipeline that:
1. Loads HaluMem benchmark data
2. Processes user sessions through ReMe (summarization + retrieval)
3. Evaluates question answering performance
4. Generates comprehensive metrics

Usage:
    python benchmark/halumem/eval_reme.py \
        --data_path /Users/yuli/workspace/HaluMem/data/HaluMem-Medium.jsonl \
        --top_k 20 --user_num 100 --max_concurrency 20
"""

import asyncio
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from reme.reme import ReMe


# ==================== Configuration ====================


@dataclass
class EvalConfig:
    """Evaluation configuration parameters."""

    data_path: str
    top_k: int = 20
    user_num: int = 1
    max_concurrency: int = 1
    batch_size: int = 40
    output_dir: str = "bench_results/reme"
    reme_model_name: str = "qwen-flash"
    eval_model_name: str = "qwen3-max"
    algo_version: str = "v1"
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
    def extract_user_name(persona_info: str) -> str:
        """Extract user name from persona info string."""
        match = re.search(r"Name:\s*(.*?); Gender:", persona_info)
        if not match:
            raise ValueError(f"No name found in persona_info: {persona_info}")
        return match.group(1).strip()

    @staticmethod
    def format_dialogue_messages(dialogue: list[dict]) -> list[dict]:
        """Format dialogue into ReMe message format with conversation_time (user messages only)."""
        return [
            {
                "role": turn["role"],
                "content": turn["content"],
                "time_created": datetime.strptime(
                    turn["timestamp"],
                    "%b %d, %Y, %H:%M:%S",
                )
                .replace(tzinfo=timezone.utc)
                .strftime("%Y-%m-%d %H:%M:%S"),
            }
            for turn in dialogue
            if turn["role"] == "user"  # Only include user messages
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

    def save_session(self, user_name: str, session_id: int, data: dict):
        """Save session data to file."""
        file_path = self.get_session_file(user_name, session_id)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Saved session {session_id} to {file_path}")

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

                f_out.write(json.dumps(user_data, ensure_ascii=False) + "\n")


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
    reference_answer: str,
    key_memory_points: str,
    response: str,
    dialogue: str = None,
    model_name: str = "qwen3-max",
):
    """
    Question-Answering Evaluation with optional Dialogue Context.

    Args:
        reme: ReMe instance with default_llm and prompt_handler
        question: The question string to be evaluated.
        reference_answer: The reference (gold-standard) answer.
        key_memory_points: The memory points used to derive the reference answer.
        response: The answer produced by the memory system.
        dialogue: Optional formatted dialogue history (role, content, time_created).
        model_name: Model name to use for LLM request

    Returns:
        dict with 'reasoning' and 'evaluation_result' fields
    """
    prompt = reme.prompt_handler.prompt_format(
        "EVALUATION_PROMPT_FOR_QUESTION2",
        question=question,
        reference_answer=reference_answer,
        key_memory_points=key_memory_points,
        response=response,
        dialogue=dialogue if dialogue else "",
    )

    result = await reme.get_llm("qwen3_max_instruct").simple_request_for_json(
        prompt=prompt,
        model_name=model_name,
    )

    return result


# ==================== Memory Operations ====================


class MemoryProcessor:
    """Handles ReMe memory operations."""

    def __init__(
        self,
        reme: ReMe,
        reme_model_name: str = "qwen3-max",
        eval_model_name: str = "qwen3-max",
        algo_version: str = "halumem",
        enable_thinking_params: bool = False,
    ):
        self.reme = reme
        self.reme_model_name = reme_model_name
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
        session_id: int,
        formatted_dialogue: str,
    ) -> list[dict]:
        """Evaluate all questions for a session."""
        results = []

        for qa in questions:
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
            evidence_text = "\n".join([e["memory_content"] for e in qa["evidence"]])
            eval_result = await evaluation_for_question(
                reme=self.reme,
                question=qa["question"],
                reference_answer=qa["answer"],
                key_memory_points=evidence_text,
                response=system_answer,
                dialogue=formatted_dialogue,
                model_name=self.eval_model_name,
            )

            eval_result_original_answer = await evaluation_for_question(
                reme=self.reme,
                question=qa["question"],
                reference_answer=qa["answer"],
                key_memory_points=evidence_text,
                response=retrieved_memories,
                dialogue=formatted_dialogue,
                model_name=self.eval_model_name,
            )

            # Build result record
            qa_result = {
                **qa,
                "uuid": uuid,
                "session_id": session_id,
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
                "hallucination_qa_ratio(all)": 0,
                "omission_qa_ratio(all)": 0,
                "correct_qa_ratio(valid)": 0,
                "hallucination_qa_ratio(valid)": 0,
                "omission_qa_ratio(valid)": 0,
                "qa_valid_num": 0,
                "qa_num": 0,
            }

        correct = 0
        hallucination = 0
        omission = 0
        valid = 0

        for qa in qa_records:
            result_type = qa.get(result_key, "")

            if result_type in ["Correct", "Hallucination", "Omission"]:
                valid += 1
                if result_type == "Correct":
                    correct += 1
                elif result_type == "Hallucination":
                    hallucination += 1
                elif result_type == "Omission":
                    omission += 1

        metrics = {
            "correct_qa_ratio(all)": correct / total,
            "hallucination_qa_ratio(all)": hallucination / total,
            "omission_qa_ratio(all)": omission / total,
            "qa_valid_num": valid,
            "qa_num": total,
        }

        if valid > 0:
            metrics.update(
                {
                    "correct_qa_ratio(valid)": correct / valid,
                    "hallucination_qa_ratio(valid)": hallucination / valid,
                    "omission_qa_ratio(valid)": omission / valid,
                },
            )
        else:
            metrics.update(
                {
                    "correct_qa_ratio(valid)": 0,
                    "hallucination_qa_ratio(valid)": 0,
                    "omission_qa_ratio(valid)": 0,
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

                    eval_results = session.get("evaluation_results", {})
                    for qa in eval_results.get("question_answering_records", []):
                        search_duration += qa.get("search_duration_ms", 0)

        # Convert to minutes
        return {
            "add_dialogue_duration_time": add_duration / 1000 / 60,
            "search_memory_duration_time": search_duration / 1000 / 60,
            "total_duration_time": (add_duration + search_duration) / 1000 / 60,
        }


# ==================== Main Pipeline ====================


class HaluMemEvaluator:
    """HaluMem evaluator with proper resource management."""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.reme = ReMe(
            default_llm_config={
                "model_name": self.config.reme_model_name,
            },
        )

        # Load evaluation prompts into ReMe's prompt handler
        prompts_yaml_path = Path(__file__).parent / "eval_reme.yaml"
        self.reme.prompt_handler.load_prompt_by_file(prompts_yaml_path)

        self.file_manager = FileManager(config.output_dir)
        self.memory_processor = MemoryProcessor(
            self.reme,
            config.reme_model_name,
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

    async def process_session(
        self,
        session: dict,
        session_id: int,
        user_name: str,
        uuid: str,
    ) -> dict:
        """Process a single session using ReMe."""
        session_data = {
            "uuid": uuid,
            "user_name": user_name,
            "session_id": session_id,
            "memory_points": session["memory_points"],
        }

        # Skip generated QA sessions
        if session.get("is_generated_qa_session", False):
            session_data["is_generated_qa_session"] = True
            return session_data

        dialogue = session["dialogue"]
        formatted_messages = self.data_loader.format_dialogue_messages(dialogue)

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

        # Evaluate questions if present
        if "questions" in session:
            formatted_dialogue = self.data_loader.format_dialogue_for_eval(dialogue, user_name)
            qa_results = await self.qa_evaluator.evaluate_questions(
                questions=session["questions"],
                user_name=user_name,
                uuid=uuid,
                session_id=session_id,
                formatted_dialogue=formatted_dialogue,
            )

            session_data["evaluation_results"] = {
                "question_answering_records": qa_results,
            }

        return session_data

    async def process_user(self, user_data: dict) -> dict:
        """Process all sessions for a user."""
        user_name = self.data_loader.extract_user_name(user_data["persona_info"])
        uuid = user_data["uuid"]

        logger.info(f"Processing user: {user_name}")

        for idx, session in enumerate(user_data["sessions"]):
            logger.info(f"  Session {idx + 1}/{len(user_data['sessions'])}")

            session_data = await self.process_session(
                session=session,
                session_id=idx,
                user_name=user_name,
                uuid=uuid,
            )

            self.file_manager.save_session(user_name, idx, session_data)

            # Update results file after each session completes
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
        all_users = self.data_loader.load_jsonl(self.config.data_path)
        users_to_process = all_users[: self.config.user_num]

        # Extract all user names and delete all profiles
        all_user_names = [self.data_loader.extract_user_name(user_data["persona_info"]) for user_data in all_users]
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
        print("HALUMEM EVALUATION - REME - QUESTION ANSWERING")
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
                user_name = self.data_loader.extract_user_name(user_data["persona_info"])

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
        await asyncio.gather(*tasks)

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

                    for session in user_data["sessions"]:
                        if session.get("is_generated_qa_session"):
                            continue

                        eval_results = session.get("evaluation_results", {})
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
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                user_data = json.loads(line)

                for session in user_data["sessions"]:
                    if session.get("is_generated_qa_session"):
                        continue

                    eval_results = session.get("evaluation_results", {})
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
        print("📊 Question Answering (with LLM answer):")
        print(f"  Correct (all):       {llm_metrics['correct_qa_ratio(all)']:.4f}")
        print(f"  Hallucination (all): {llm_metrics['hallucination_qa_ratio(all)']:.4f}")
        print(f"  Omission (all):      {llm_metrics['omission_qa_ratio(all)']:.4f}")
        print(f"  Correct (valid):     {llm_metrics['correct_qa_ratio(valid)']:.4f}")
        print(f"  Hallucination (valid): {llm_metrics['hallucination_qa_ratio(valid)']:.4f}")
        print(f"  Omission (valid):    {llm_metrics['omission_qa_ratio(valid)']:.4f}")
        print(f"  Valid/Total:         {llm_metrics['qa_valid_num']}/{llm_metrics['qa_num']}")

        # Print metrics for original retrieved memories (original_result_type)
        orig_metrics = qa_metrics["with_original_memories"]
        print("\n📊 Question Answering (with original memories):")
        print(f"  Correct (all):       {orig_metrics['correct_qa_ratio(all)']:.4f}")
        print(f"  Hallucination (all): {orig_metrics['hallucination_qa_ratio(all)']:.4f}")
        print(f"  Omission (all):      {orig_metrics['omission_qa_ratio(all)']:.4f}")
        print(f"  Correct (valid):     {orig_metrics['correct_qa_ratio(valid)']:.4f}")
        print(f"  Hallucination (valid): {orig_metrics['hallucination_qa_ratio(valid)']:.4f}")
        print(f"  Omission (valid):    {orig_metrics['omission_qa_ratio(valid)']:.4f}")
        print(f"  Valid/Total:         {orig_metrics['qa_valid_num']}/{orig_metrics['qa_num']}")

        print("\n⏱️  Time Metrics:")
        print(f"  Memory Addition:  {time_metrics['add_dialogue_duration_time']:.2f} min")
        print(f"  Memory Search:    {time_metrics['search_memory_duration_time']:.2f} min")
        print(f"  Total:            {time_metrics['total_duration_time']:.2f} min")
        print("\n" + "=" * 80)


# ==================== Entry Point ====================


async def main_async(
    data_path: str,
    top_k: int,
    batch_size: int,
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
        batch_size=batch_size,
        user_num=user_num,
        max_concurrency=max_concurrency,
        reme_model_name=reme_model_name,
        eval_model_name=eval_model_name,
        algo_version=algo_version,
        enable_thinking_params=enable_thinking_params,
    )

    # Use async context manager for automatic cleanup
    async with HaluMemEvaluator(config) as evaluator:
        await evaluator.run_evaluation()


def main(
    data_path: str,
    top_k: int,
    batch_size: int,
    user_num: int,
    max_concurrency: int,
    reme_model_name: str = "qwen-flash",
    eval_model_name: str = "qwen3-max",
    algo_version: str = "halumem",
    enable_thinking_params: bool = False,
):
    """Main entry point for ReMe evaluation."""
    asyncio.run(
        main_async(
            data_path=data_path,
            top_k=top_k,
            batch_size=batch_size,
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

    parser = argparse.ArgumentParser(
        description="Evaluate ReMe on HaluMem benchmark (Question Answering)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        # required=True,
        default="/Users/zhouwk/PycharmProjects/MemAgent/dataset/halumem/HaluMem-Medium.jsonl",
        help="Path to HaluMem JSONL file",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of memories to retrieve (default: 20)",
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
        default=1,
        help="Maximum concurrent user processing (default: 100)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=40,
        help="Batch size for memory summary processing of each conversation (default: 40)",
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
        default="v1",
        help="Algorithm version for summary and retrieval (default: v1)",
    )
    parser.add_argument(
        "--enable_thinking_params",
        action="store_true",
        default=False,
        help="Enable thinking parameters for summary and retrieval (default: False)",
    )

    args = parser.parse_args()
    print(f"args={args}!")

    main(
        data_path=args.data_path,
        top_k=args.top_k,
        batch_size=args.batch_size,
        user_num=args.user_num,
        max_concurrency=args.max_concurrency,
        reme_model_name=args.reme_model_name,
        eval_model_name=args.eval_model_name,
        algo_version=args.algo_version,
        enable_thinking_params=args.enable_thinking_params,
    )
