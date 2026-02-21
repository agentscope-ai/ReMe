"""
LongMemEval Benchmark Evaluator for ReMe - Retrieve Only

A simplified evaluation pipeline that only runs the retrieve and judge phases:
1. Loads LongMemEval benchmark data
2. Skips memory summarization (assumes memories are already in vector store)
3. Uses questions to query memory and generate answers
4. Uses LLM to judge answer correctness
5. Generates comprehensive metrics

This is useful for debugging/tuning the retrieve phase without re-running summary.

Usage:
    python benchmark/longmemeval/eval_longmemeval_reme_retrieve.py \
        --data_path dataset/longmemeval/longmemeval_s_cleaned.json \
        --top_k 20 --start_index 0 --end_index 10
"""

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from reme.reme import ReMe


# ==================== Configuration ====================


@dataclass
class RetrieveEvalConfig:
    """Evaluation configuration parameters for retrieve-only mode."""

    data_path: str
    top_k: int = 10
    start_index: int = 0
    end_index: Optional[int] = None
    max_concurrency: int = 1
    output_dir: str = "cache/bench_results/longmemeval_reme_retrieve"
    reme_model_name: str = "qwen-flash"
    eval_model_name: str = "qwen3-max"
    algo_version: str = "v1"
    samples_per_type: int = -1  # Number of samples per question type, -1 for all
    enable_thinking_params: bool = False
    # Optional: path to previous summary results to reload memories
    summary_results_dir: Optional[str] = None


# ==================== Answer Judge Prompts ====================


def get_anscheck_prompt(task: str, question: str, answer: str, response: str, abstention: bool = False) -> str:
    """Generate the answer checking prompt based on question type.

    Args:
        task: Question type, e.g. 'single-session-user', 'multi-session', 'temporal-reasoning'
        question: The question content
        answer: The reference answer
        response: The model's response
        abstention: Whether this is an unanswerable question

    Returns:
        Prompt for judging answer correctness
    """
    if not abstention:
        if task in ["single-session-user", "single-session-assistant", "multi-session"]:
            template = (
                "I will give you a question, a correct answer, and a response from a model. Please answer yes i"
                "f the response contains the correct answer. Otherwise, answer no. If the response is equival"
                "ent to the correct answer or contains all the intermediate steps to get the correct answer, "
                "you should also answer yes. If the response only contains a subset of the information required"
                " by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs"
                " the model response correct? Answer yes or no only."
            )
            prompt = template.format(question, answer, response)
        elif task == "temporal-reasoning":
            template = (
                "I will give you a question, a correct answer, and a response from a model. Please answer yes"
                " if the response contains the correct answer. Otherwise, answer no. If the response is equiv"
                "alent to the correct answer or contains all the intermediate steps to get the correct answer"
                ", you should also answer yes. If the response only contains a subset of the information requ"
                "ired by the answer, answer no. In addition, do not penalize off-by-one errors for the numbe"
                "r of days. If the question asks for the number of days/weeks/months, etc., and the model ma"
                "kes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's respon"
                "se is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs th"
                "e model response correct? Answer yes or no only."
            )
            prompt = template.format(question, answer, response)
        elif task == "knowledge-update":
            template = (
                "I will give you a question, a correct answer, and a response from a model. Please answer yes "
                "if the response contains the correct answer. Otherwise, answer no. If the response contains "
                "some previous information along with an updated answer, the response should be considered "
                "as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrec"
                "t Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            )
            prompt = template.format(question, answer, response)
        elif task == "single-session-preference":
            template = (
                "I will give you a question, a rubric for desired personalized response, and a response from a"
                " model. Please answer yes if the response satisfies the desired response. Otherwise, answer"
                " no. The model does not need to reflect all the points in the rubric. The response is corr"
                "ect as long as it recalls and utilizes the user's personal information correctly.\n\nQuest"
                "ion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes o"
                "r no only."
            )
            prompt = template.format(question, answer, response)
        else:
            # Default template
            template = (
                "I will give you a question, a correct answer, and a response from a model. Please answer y"
                "es if the response contains the correct answer. Otherwise, answer no. If the response is "
                "equivalent to the correct answer or contains all the intermediate steps to get the correc"
                "t answer, you should also answer yes. If the response only contains a subset of the infor"
                "mation required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel"
                " Response: {}\n\nIs the model response correct? Answer yes or no only."
            )
            prompt = template.format(question, answer, response)

    else:
        template = (
            "I will give you an unanswerable question, an explanation, and a response from a model. Please "
            "answer yes if the model correctly identifies the question as unanswerable. The model could say "
            "that the information is incomplete, or some other information is given but the asked informati"
            "on is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correct"
            "ly identify the question as unanswerable? Answer yes or no only."
        )
        prompt = template.format(question, answer, response)
    return prompt


# ==================== Utilities ====================


class DataLoader:
    """Handles loading and parsing of LongMemEval data."""

    @staticmethod
    def load_json(file_path: str) -> list[dict]:
        """Load all entries from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def filter_by_type(data: list[dict], samples_per_type: int = -1) -> list[tuple[int, dict]]:
        """Filter data by question type with specified number of samples per type.

        Args:
            data: List of question entries
            samples_per_type: Number of samples per type, -1 for all

        Returns:
            List of tuples (original_index, entry) for selected samples
        """
        if samples_per_type == -1:
            # Return all with original indices
            return list(enumerate(data))

        # Group by question type
        type_groups: dict[str, list[tuple[int, dict]]] = {}
        for i, entry in enumerate(data):
            qtype = entry.get("question_type", "unknown")
            if qtype not in type_groups:
                type_groups[qtype] = []
            type_groups[qtype].append((i, entry))

        # Select samples from each type
        selected = []
        for qtype, entries in type_groups.items():
            count = min(samples_per_type, len(entries))
            selected.extend(entries[:count])
            logger.info(f"  {qtype}: selected {count}/{len(entries)} samples")

        # Sort by original index to maintain order
        selected.sort(key=lambda x: x[0])
        return selected


class FileManager:
    """Manages file I/O operations."""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_question_result(self, idx: int, question_id: str, data: dict):
        """Save result for a single question."""
        file_path = self.base_dir / f"question_{idx:04d}_{question_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"✅ Saved question result to {file_path}")

    def load_question_result(self, idx: int, question_id: str) -> Optional[dict]:
        """Load result for a single question if exists."""
        file_path = self.base_dir / f"question_{idx:04d}_{question_id}.json"
        if not file_path.exists():
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_summary(self, results: list[dict]):
        """Save summary of all results."""
        file_path = self.base_dir / "summary.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        logger.info(f"✅ Saved summary to {file_path}")


# ==================== Evaluation Functions ====================


async def answer_question_with_memories(
    reme: ReMe,
    question: str,
    memories: str,
    user_id: str = None,
    model_name: str = "qwen3-max",
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

    result = await reme.get_llm(name=model_name).simple_request_for_json(
        prompt=prompt,
        model_name=None,
    )

    return result


# ==================== Memory Operations ====================


class RetrieveProcessor:
    """Handles ReMe memory retrieve operations only."""

    def __init__(
        self,
        reme: ReMe,
        reme_model_name: str = "qwen-flash",
        eval_model_name: str = "qwen3-max",
        algo_version: str = "v1",
        enable_thinking_params: bool = False,
    ):
        self.reme = reme
        self.reme_model_name = reme_model_name
        self.eval_model_name = eval_model_name
        self.algo_version = algo_version
        self.enable_thinking_params = enable_thinking_params

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
            llm_config_name="qwen3-max",
            query=query,
            retrieve_top_k=top_k,
            user_name=user_id,
            version=self.algo_version,
            return_dict=True,
            enable_time_filter=True,
            enable_thinking_params=True,
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


# ==================== Answer Judge ====================


class LongMemEvalJudge:
    """LongMemEval answer judge using LLM."""

    def __init__(self, reme: ReMe, model: str = "qwen3-max"):
        self.reme = reme
        self.model = model

    async def judge_answer(
        self,
        question_type: str,
        question: str,
        answer: str,
        response: str,
        abstention: bool = False,
    ) -> dict:
        """
        Judge if the model's response is correct.

        Returns:
            dict with is_correct, llm_response, and judge_prompt
        """
        prompt = get_anscheck_prompt(question_type, question, answer, response, abstention)

        try:
            llm_response = await self.reme.get_llm("default").simple_request(
                prompt=prompt,
                model_name=self.model,
            )
            llm_response_lower = llm_response.strip().lower()
            is_correct = llm_response_lower.startswith("yes")

            return {
                "is_correct": is_correct,
                "llm_response": llm_response,
                "judge_prompt": prompt,
            }
        except Exception as e:
            return {
                "is_correct": None,
                "error": str(e),
                "judge_prompt": prompt,
            }


# ==================== Metrics ====================


class MetricsAggregator:
    """Aggregates evaluation metrics for LongMemEval."""

    @staticmethod
    def compute_metrics(results: list[dict]) -> dict[str, Any]:
        """Compute overall and per-type metrics."""
        total = len(results)
        correct = sum(1 for r in results if r.get("judgment", {}).get("is_correct") is True)
        incorrect = sum(1 for r in results if r.get("judgment", {}).get("is_correct") is False)
        error = total - correct - incorrect

        metrics = {
            "total": total,
            "correct": correct,
            "incorrect": incorrect,
            "error": error,
            "accuracy": correct / total if total > 0 else 0,
            "accuracy_valid": correct / (correct + incorrect) if (correct + incorrect) > 0 else 0,
        }

        # Per question type statistics
        type_stats = {}
        for r in results:
            qtype = r.get("question_type", "unknown")
            if qtype not in type_stats:
                type_stats[qtype] = {"total": 0, "correct": 0, "incorrect": 0}
            type_stats[qtype]["total"] += 1
            if r.get("judgment", {}).get("is_correct") is True:
                type_stats[qtype]["correct"] += 1
            elif r.get("judgment", {}).get("is_correct") is False:
                type_stats[qtype]["incorrect"] += 1

        metrics["by_question_type"] = {
            qtype: {
                **stats,
                "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
                "accuracy_valid": (
                    stats["correct"] / (stats["correct"] + stats["incorrect"])
                    if (stats["correct"] + stats["incorrect"]) > 0
                    else 0
                ),
            }
            for qtype, stats in type_stats.items()
        }

        return metrics

    @staticmethod
    def compute_timing_stats(results: list[dict]) -> dict[str, Any]:
        """Compute timing statistics."""
        retrieve_times = []

        for r in results:
            retrieve_ms = r.get("retrieve_duration_ms", 0)
            if retrieve_ms > 0:
                retrieve_times.append(retrieve_ms)

        def compute_stats(times: list[float]) -> dict:
            if not times:
                return {"count": 0, "total_ms": 0, "avg_ms": 0, "min_ms": 0, "max_ms": 0}

            return {
                "count": len(times),
                "total_ms": sum(times),
                "total_min": sum(times) / 1000 / 60,
                "avg_ms": sum(times) / len(times),
                "min_ms": min(times),
                "max_ms": max(times),
            }

        return {
            "retrieve": compute_stats(retrieve_times),
            "total_time_min": sum(retrieve_times) / 1000 / 60,
        }


# ==================== Main Pipeline ====================


class LongMemEvalRetrieveEvaluator:
    """Retrieve-only evaluator for LongMemEval benchmark using ReMe."""

    def __init__(self, config: RetrieveEvalConfig):
        self.config = config
        self.reme = ReMe(
            default_llm_config={
                "model_name": self.config.reme_model_name,
            },
            llms={
                "qwen-plus-think": {
                    "backend": "openai",
                    "model_name": "qwen-plus",
                    "extra_body": {
                        "enable_thinking": True,
                    },
                },
                "qwen3-max-think": {
                    "backend": "openai",
                    "model_name": "qwen3-max",
                    "extra_body": {
                        "enable_thinking": True,
                    },
                },
                "qwen3-max": {
                    "backend": "openai",
                    "model_name": "qwen3-max",
                    "extra_body": {
                        "enable_thinking": False,
                    },
                },
            },
        )

        # Load evaluation prompts into ReMe's prompt handler
        prompts_yaml_path = Path(__file__).parent / "eval_reme.yaml"
        self.reme.prompt_handler.load_prompt_by_file(prompts_yaml_path)

        self.file_manager = FileManager(config.output_dir)
        self.retrieve_processor = RetrieveProcessor(
            self.reme,
            config.reme_model_name,
            config.eval_model_name,
            config.algo_version,
            config.enable_thinking_params,
        )
        self.judge = LongMemEvalJudge(self.reme, config.eval_model_name)
        self.data_loader = DataLoader()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.reme.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.reme.close()
        return False

    async def process_question_entry(self, entry: dict, idx: int) -> dict:
        """Process a single question entry (retrieve + judge only).

        Args:
            entry: A question entry from LongMemEval dataset
            idx: Index of the question

        Returns:
            Result dictionary
        """
        question_id = entry["question_id"]
        question = entry["question"]
        answer = entry["answer"]
        question_type = entry["question_type"]
        question_date = entry.get("question_date", "")
        haystack_dates = entry["haystack_dates"]
        haystack_session_ids = entry["haystack_session_ids"]
        haystack_sessions = entry["haystack_sessions"]

        # Use question_id as user_id for isolation (same as full eval)
        user_id = f"longmemeval_{question_id}"

        logger.info(f"\n{'='*60}")
        logger.info(f"Question ID: {question_id}")
        logger.info(f"Question Type: {question_type}")
        logger.info(f"Question: {question}")
        logger.info(f"Question_date: {question_date}")
        logger.info(f"Answer: {answer}")
        logger.info(f"Number of sessions: {len(haystack_sessions)}")
        logger.info(f"{'='*60}")

        # Skip summary phase - directly search memory and answer question
        logger.info("  Retrieving and answering question using ReMe...")
        answer_dict, retrieve_messages, retrieve_duration_ms = await self.retrieve_processor.search_memory(
            query=f"[Question_date: {question_date} | Question_type: {question_type}] " + question,
            user_id=user_id,
            top_k=self.config.top_k,
        )

        # Extract answer and reasoning from the structured response
        model_response = answer_dict.get("answer", "")
        model_reasoning = answer_dict.get("reasoning", "")
        retrieved_memories = answer_dict.get("memories", "")
        retrieved_nodes = answer_dict.get("retrieved_nodes", [])

        # Judge answer correctness
        logger.info("  Judging answer correctness...")
        judgment = await self.judge.judge_answer(
            question_type=question_type,
            question=question,
            answer=answer,
            response=model_response,
        )

        is_correct = judgment.get("is_correct")
        logger.info(
            f"  → Answer judgment: {'Correct' if is_correct else 'Incorrect' if is_correct is False else 'Error'}",
        )

        result = {
            "question_id": question_id,
            "question_type": question_type,
            "question": question,
            "answer": answer,
            "question_date": question_date,
            "haystack_dates": haystack_dates,
            "haystack_session_ids": haystack_session_ids,
            "num_sessions": len(haystack_sessions),
            "model_response": model_response,
            "model_reasoning": model_reasoning,
            "retrieved_memories": retrieved_memories,
            "retrieved_nodes": retrieved_nodes,
            "judgment": judgment,
            "retrieve_duration_ms": retrieve_duration_ms,
            "retrieve_messages": retrieve_messages,
        }

        # Save individual result
        self.file_manager.save_question_result(idx, question_id, result)

        logger.info(f"  Question {question_id} - Completed")

        return result

    async def run_evaluation(self):
        """Run the retrieve-only evaluation pipeline with parallel processing."""
        start_time = time.time()

        # NOTE: Do NOT clear vector store - we assume memories are already there from previous summary run

        # Load dataset
        logger.info(f"Loading dataset from: {self.config.data_path}")
        all_data = self.data_loader.load_json(self.config.data_path)
        logger.info(f"Total questions in dataset: {len(all_data)}")

        # Filter by question type
        logger.info(f"Filtering by type (samples_per_type={self.config.samples_per_type}):")
        filtered_data = self.data_loader.filter_by_type(all_data, self.config.samples_per_type)
        logger.info(f"Selected {len(filtered_data)} questions after filtering")

        # Apply start_index and end_index on filtered data
        end_index = self.config.end_index or len(filtered_data)
        start_index = self.config.start_index
        end_index = min(end_index, len(filtered_data))

        # Get the slice we want to process
        data_to_process = filtered_data[start_index:end_index]
        total_questions = len(data_to_process)

        logger.info(f"Processing {total_questions} questions (index {start_index} to {end_index - 1})")

        print("\n" + "=" * 80)
        print("LONGMEMEVAL EVALUATION - REME (RETRIEVE ONLY)")
        print(f"Samples per type: {self.config.samples_per_type} (-1 = all)")
        print(f"Questions to process: {total_questions} | Top-K: {self.config.top_k}")
        print(f"Max Concurrency: {self.config.max_concurrency}")
        print(f"ReMe Model: {self.config.reme_model_name} | Eval Model: {self.config.eval_model_name}")
        print(f"Algo Version: {self.config.algo_version}")
        print("⚠️  NOTE: Assumes memories are already in vector store from previous summary run")
        print("=" * 80 + "\n")

        # Use semaphore to control concurrency
        semaphore = asyncio.Semaphore(self.config.max_concurrency)

        async def process_with_semaphore(idx: int, original_idx: int, entry: dict) -> Optional[dict]:
            """Process a question with semaphore for concurrency control."""
            async with semaphore:
                question_id = entry["question_id"]

                # Check cache first (use original index for cache file naming)
                cached_result = self.file_manager.load_question_result(original_idx, question_id)
                if cached_result:
                    print(f"⚡ [{idx}/{total_questions}] Skipping question {original_idx} (cached)")
                    return cached_result

                print(f"\n{'#'*60}")
                print(f"### [{idx}/{total_questions}] Processing Question {original_idx} ###")
                print(f"{'#'*60}")

                try:
                    result = await self.process_question_entry(entry, original_idx)
                    print(f"✅ [{idx}/{total_questions}] Completed question {original_idx}")
                    return result
                except Exception as e:
                    logger.error(f"❌ Error processing question {original_idx}: {e}")
                    import traceback

                    traceback.print_exc()
                    return {
                        "question_id": question_id,
                        "error": str(e),
                        "question_type": entry.get("question_type", "unknown"),
                        "question": entry.get("question", ""),
                        "answer": entry.get("answer", ""),
                        "judgment": {"is_correct": None, "error": str(e)},
                    }

        # Create all tasks from filtered data (each item is a tuple of (original_idx, entry))
        tasks = [
            process_with_semaphore(idx + 1, original_idx, entry)
            for idx, (original_idx, entry) in enumerate(data_to_process)
        ]

        # Execute in parallel with controlled concurrency
        all_results = await asyncio.gather(*tasks, return_exceptions=False)

        # Filter out None results if any
        all_results = [r for r in all_results if r is not None]

        # Save summary
        self.file_manager.save_summary(all_results)

        elapsed = time.time() - start_time
        print(f"\n✅ Processing completed in {elapsed:.2f}s")
        if total_questions > 0:
            print(f"   Average time per question: {elapsed / total_questions:.2f}s")

        # Compute and report metrics
        self._report_metrics(all_results)

        return all_results

    def _report_metrics(self, results: list[dict]):
        """Report evaluation metrics."""
        metrics = MetricsAggregator.compute_metrics(results)
        timing_stats = MetricsAggregator.compute_timing_stats(results)

        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY - LONGMEMEVAL - REME (RETRIEVE ONLY)")
        print("=" * 80 + "\n")

        print("📊 Overall Results:")
        print(f"  ✅ Correct: {metrics['correct']}/{metrics['total']} ({100*metrics['accuracy']:.2f}%)")
        print(
            f"  ❌ Incorrect: {metrics['incorrect']}/{metrics['total']}"
            f" ({100*metrics['incorrect']/metrics['total'] if metrics['total'] > 0 else 0:.2f}%)",
        )
        if metrics["error"] > 0:
            print(f"  ⚠️ Error: {metrics['error']}/{metrics['total']} ({100*metrics['error']/metrics['total']:.2f}%)")
        print(f"  Accuracy (valid): {100*metrics['accuracy_valid']:.2f}%")

        print("\n📊 Accuracy by Question Type:")
        print("-" * 60)
        print(f"{'Question Type':<30} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
        print("-" * 60)
        for qtype in sorted(metrics["by_question_type"].keys()):
            stats = metrics["by_question_type"][qtype]
            print(f"{qtype:<30} {stats['correct']:<10} {stats['total']:<10} {100*stats['accuracy']:.2f}%")
        print("-" * 60)

        print("\n⏱️ Timing Statistics (Retrieve Only):")
        retrieve = timing_stats["retrieve"]
        print("  Memory Retrieval:")
        print(f"    Total Time:  {retrieve['total_min']:.2f} min")
        print(f"    Avg per Q:   {retrieve['avg_ms']:.0f} ms")
        print(f"  Total Time:    {timing_stats['total_time_min']:.2f} min")

        # Save metrics
        final_results = {
            "accuracy": metrics,
            "timing": timing_stats,
        }
        metrics_file = self.file_manager.base_dir / "eval_statistics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=4, ensure_ascii=False)
        print(f"\n📁 Statistics saved to: {metrics_file}")

        print("\n" + "=" * 80)


# ==================== Entry Point ====================


async def main_async(
    data_path: str,
    top_k: int = 20,
    start_index: int = 0,
    end_index: Optional[int] = None,
    max_concurrency: int = 1,
    output_dir: str = "bench_results/longmemeval_reme_retrieve",
    reme_model_name: str = "qwen-flash",
    eval_model_name: str = "qwen3-max",
    algo_version: str = "v1",
    samples_per_type: int = -1,
    enable_thinking_params: bool = False,
    summary_results_dir: Optional[str] = None,
):
    """Main async entry point for LongMemEval retrieve-only evaluation with proper resource cleanup."""
    config = RetrieveEvalConfig(
        data_path=data_path,
        top_k=top_k,
        start_index=start_index,
        end_index=end_index,
        max_concurrency=max_concurrency,
        output_dir=output_dir,
        reme_model_name=reme_model_name,
        eval_model_name=eval_model_name,
        algo_version=algo_version,
        samples_per_type=samples_per_type,
        enable_thinking_params=enable_thinking_params,
        summary_results_dir=summary_results_dir,
    )

    # Use async context manager for automatic cleanup
    async with LongMemEvalRetrieveEvaluator(config) as evaluator:
        await evaluator.run_evaluation()


def main(
    data_path: str,
    top_k: int = 20,
    start_index: int = 0,
    end_index: Optional[int] = None,
    max_concurrency: int = 1,
    output_dir: str = "bench_results/longmemeval_reme_retrieve",
    reme_model_name: str = "qwen-flash",
    eval_model_name: str = "qwen3-max",
    algo_version: str = "v1",
    samples_per_type: int = -1,
    enable_thinking_params: bool = False,
    summary_results_dir: Optional[str] = None,
):
    """Main entry point for LongMemEval retrieve-only evaluation."""
    asyncio.run(
        main_async(
            data_path=data_path,
            top_k=top_k,
            start_index=start_index,
            end_index=end_index,
            max_concurrency=max_concurrency,
            output_dir=output_dir,
            reme_model_name=reme_model_name,
            eval_model_name=eval_model_name,
            algo_version=algo_version,
            samples_per_type=samples_per_type,
            enable_thinking_params=enable_thinking_params,
            summary_results_dir=summary_results_dir,
        ),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate ReMe on LongMemEval benchmark (Retrieve Phase Only)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        # default="/Users/zhouwk/PycharmProjects/MemAgent/dataset/longmemeval/longmemeval_s_cleaned.json",
        default="/Users/zhouwk/PycharmProjects/MemAgent/dataset/longmemeval/longmemeval_oracle.json",
        help="Path to LongMemEval JSON file",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of memories to retrieve (default: 10)",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start index for processing questions (default: 0)",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="End index for processing questions (default: None, process all)",
    )
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=8,
        help="Maximum concurrent question processing (default: 1)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="bench_results/longmemeval_reme_retrieve_gpt4",
        help="Output directory for results",
    )
    parser.add_argument(
        "--reme_model_name",
        type=str,
        default="gpt-4o-mini-2024-07-18",
        help="Model name for ReMe operations (default: gpt-4o-mini-2024-07-18)",
    )
    parser.add_argument(
        "--eval_model_name",
        type=str,
        default="gpt-4o-mini-2024-07-18",
        help="Model name for evaluation/judgment (default: gpt-4o-mini-2024-07-18)",
    )
    parser.add_argument(
        "--algo_version",
        type=str,
        default="longmemeval",
        help="Algorithm version for retrieval (default: longmemeval)",
    )
    parser.add_argument(
        "--samples_per_type",
        type=int,
        default=4,
        help="Number of samples per question type, -1 for all (default: 4)",
    )
    parser.add_argument(
        "--enable_thinking_params",
        action="store_true",
        default=False,
        help="Enable thinking parameters for retrieval (default: False)",
    )
    parser.add_argument(
        "--summary_results_dir",
        type=str,
        default="/Users/zhouwk/PycharmProjects/ReMe/benchmark/longmemeval/bench_results",
        help="Optional: path to previous summary results directory (for reference)",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        default=False,
        help="Ignore cached results and re-run all questions (default: False)",
    )

    args = parser.parse_args()
    print(f"args={args}!")

    main(
        data_path=args.data_path,
        top_k=args.top_k,
        start_index=args.start_index,
        end_index=args.end_index,
        max_concurrency=args.max_concurrency,
        output_dir=args.output_dir,
        reme_model_name=args.reme_model_name,
        eval_model_name=args.eval_model_name,
        algo_version=args.algo_version,
        samples_per_type=args.samples_per_type,
        enable_thinking_params=args.enable_thinking_params,
        summary_results_dir=args.summary_results_dir,
    )
