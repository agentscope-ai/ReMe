# flake8: noqa: E402
# pylint: disable=too-many-return-statements
"""A minimal ReAct Agent for BFCL-v3(multi-turn) tasks."""

import re
import os
import time
import json
import warnings
import tempfile
import datetime
from pathlib import Path
from typing import Dict, List, Any

import ray
import requests
from tqdm import tqdm
from loguru import logger
from openai import OpenAI
from dotenv import load_dotenv

from bfcl_utils import (
    load_test_case,
    handle_user_turn,
    handle_tool_calls,
    extract_tool_schema,
    extract_single_turn_response,
    extract_multi_turn_responses,
    capture_and_print_score_files,
    create_error_response,
)
from bfcl_eval.model_handler.api_inference.qwen import QwenAPIHandler
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    is_empty_execute_response,
)
from bfcl_eval.eval_checker.eval_runner import (
    multi_turn_runner,
    ast_file_runner,
)
from bfcl_eval.eval_checker.eval_runner_helper import record_cost_latency
from bfcl_eval.utils import (
    is_multi_turn,
    is_relevance_or_irrelevance,
    find_file_with_suffix,
    load_file,
)

os.environ["BFCL_DATA_PATH"] = "data/multiturn_data_base_val.jsonl"
os.environ["BFCL_ANSWER_PATH"] = "data/possible_answer"
load_dotenv("../../.env")


@ray.remote
class BFCLAgent:
    """A minimal ReAct Agent for BFCL-v3(multi-turn) tasks."""

    def __init__(
        self,
        index: int,
        task_ids: List[str],
        experiment_name: str,
        data_path: str = os.getenv("BFCL_DATA_PATH"),
        answer_path: Path = Path(os.getenv("BFCL_ANSWER_PATH")),
        model_name: str = "qwen3-8b",
        temperature: float = 0.9,
        max_interactions: int = 30,
        max_response_size: int = 2000,
        num_trials: int = 1,
        enable_thinking: bool = False,
        use_memory: bool = False,
        use_memory_addition: bool = False,
        use_memory_deletion: bool = False,
        delete_freq: int = 10,
        freq_threshold: int = 5,
        utility_threshold: float = 0.5,
        memory_base_url: str = "http://0.0.0.0:8002/",
    ):

        self.index: int = index
        self.task_ids: List[str] = task_ids
        self.categories: List[str] = [task_id.rsplit("_", 1)[0] if "_" in task_id else task_id for task_id in task_ids]
        self.experiment_name: str = experiment_name
        self.data_path: str = data_path
        self.answer_path: Path = answer_path
        self.model_name: str = model_name
        self.temperature: float = temperature
        self.max_interactions: int = max_interactions
        self.max_response_size: int = max_response_size
        self.num_trials: int = num_trials
        self.enable_thinking: bool = enable_thinking
        self.use_memory: bool = use_memory
        self.use_memory_addition: bool = use_memory_addition if use_memory else False
        self.use_memory_deletion: bool = use_memory_deletion if use_memory else False
        self.delete_freq: int = delete_freq
        self.freq_threshold: int = freq_threshold
        self.utility_threshold: float = utility_threshold
        self.memory_base_url: str = memory_base_url

        self.llm_client = OpenAI()

        self.history: List[List[List[dict]]] = [[] for _ in range(num_trials)]
        self.retrieved_memory_list: List[List[List[Any]]] = [[] for _ in range(num_trials)]
        self.test_entry: List[List[Dict[str, Any]]] = [[] for _ in range(num_trials)]
        self.original_test_entry: List[List[Dict[str, Any]]] = [[] for _ in range(num_trials)]
        self.tool_schema: List[List[List[dict]]] = [[] for _ in range(num_trials)]
        self.current_turn = [[0 for _ in range(len(task_ids))] for _ in range(num_trials)]

        for run_id in range(num_trials):
            for task_index in range(len(task_ids)):
                self.init_state(run_id, task_index)

    def init_state(self, run_id, i) -> Dict[str, Any]:
        """Initialize the state of the agent."""
        self.test_entry[run_id].append(load_test_case(self.data_path, self.task_ids[i]))
        self.original_test_entry[run_id].append(self.test_entry[run_id][i].get("extra", {}))
        self.tool_schema[run_id].append(extract_tool_schema(self.test_entry[run_id][i].get("tools", [{}])))

        msg = self.test_entry[run_id][i].get("messages", [])
        self.history[run_id].append(msg)
        self.retrieved_memory_list[run_id].append([])
        self.current_turn[run_id][i] = 1

    def update_task_history_with_memory(self, run_id, task_index, previous_memories: None):
        """Update the task history with memory."""
        query = self.history[run_id][task_index][0]["content"]
        if len(previous_memories) == 0:
            response = self.get_memory(query)
            if response and "memory_list" in response["metadata"]:
                self.retrieved_memory_list[run_id][task_index] = response["metadata"]["memory_list"]
                task_memory = re.sub(r"\bMemory\s*(\d+)\s*[:]", r"Experience \1 :", response["answer"])
                logger.info(f"loaded task_memory: {task_memory}")
                self.history[run_id][task_index][0] = self.get_query_with_memory(query, task_memory)
        else:
            formatted_memories = []
            for i, memory in enumerate(previous_memories, 1):
                condition = memory["when_to_use"]
                memory_content = memory["content"]
                memory_text = f"Experience {i} :\n When to use: {condition}\n Content: {memory_content}\n"
                formatted_memories.append(memory_text)
            self.history[run_id][task_index][0] = self.get_query_with_memory(query, "\n".join(formatted_memories))

    def get_query_with_memory(self, query: str, memory: str):
        """Get the query with memory."""
        return {
            "role": "user",
            "content": "Task:\n" + query + "\n\nSome Related Experience to help you to complete the task:\n" + memory,
        }

    def get_query_without_experience(self, query: str):
        """Get the query without experience."""
        if "\n\nSome Related Experience" in query:
            query = query.split("\n\nSome Related Experience")[0].split("Task:\n")[-1]
        return query

    def get_traj_from_task_history(self, task_id: str, task_history: list, reward: float):
        """Get the trajectory from the task history."""
        return {
            "task_id": task_id,
            "messages": task_history,
            "score": reward,
        }

    def handle_api_response(self, response: requests.Response):
        """Handle API response with proper error checking"""
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None

        return response.json()

    def get_memory(self, query: str):
        """Retrieve relevant task memories based on a query"""
        response = requests.post(
            url=f"{self.memory_base_url}retrieve_task_memory",
            json={
                "query": query,
                "enable_llm_rerank": False,
                "enable_score_filter": False,
                "top_k": 5,
                "enable_llm_rewrite": False,
            },
        )

        result = self.handle_api_response(response)
        if not result:
            return None

        logger.info(f"query: {query}, response: {result}")
        return result

    def summary_memory(self, trajectories):
        """Generate a summary of conversation messages and create task memories"""
        response = requests.post(
            url=f"{self.memory_base_url}summary_task_memory",
            json={
                "trajectories": trajectories,
                "success_threshold": 1.0,
                "enable_soft_comparison": True,
                "validation_threshold": 0.5,
            },
        )

        result = self.handle_api_response(response)
        if not result:
            return []

        # Extract memory list from response
        memory_list = result.get("metadata", {}).get("memory_list", [])
        logger.info(f"add new memories: {memory_list}")
        return memory_list

    def add_memory(self, memory_list):
        """Add the memory to the memory pool."""
        response = requests.post(
            url=f"{self.memory_base_url}add_task_memory",
            json={
                "memory_list": memory_list,
            },
        )
        response.raise_for_status()

    def update_memory_information(self, memory_list, update_utility: bool = False):
        """Update the memory information."""
        response = requests.post(
            url=f"{self.memory_base_url}record_task_memory",
            json={
                "memory_list": memory_list,
                "update_utility": update_utility,
            },
        )
        response.raise_for_status()
        logger.info(response.json())

    def delete_memory(self):
        """Delete the memory from the memory pool."""
        response = requests.post(
            url=f"{self.memory_base_url}delete_task_memory",
            json={
                "freq_threshold": self.freq_threshold,
                "utility_threshold": self.utility_threshold,
            },
        )
        response.raise_for_status()

    def call_llm(self, messages: list, tool_schemas: list[dict]) -> str:
        """Call the LLM."""
        for i in range(100):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tool_schemas,
                    temperature=self.temperature,
                    seed=0,
                    extra_body={"enable_thinking": self.enable_thinking},
                    stream=self.enable_thinking,
                    parallel_tool_calls=True,
                )
                if not self.enable_thinking:
                    out_msg = response.choices[0].message
                    return out_msg.model_dump(exclude_unset=True, exclude_none=True)
                else:
                    reasoning_content = ""  # Complete reasoning process
                    answer_content = ""  # Define complete response
                    tool_info = []  # Store tool invocation information
                    is_answering = (
                        False  # Determine whether the reasoning process has finished and response has started
                    )

                    for chunk in response:
                        if not chunk.choices:
                            # Handle usage information
                            continue

                        delta = chunk.choices[0].delta
                        # Handle AI's thought process (chain reasoning)
                        if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                            reasoning_content += delta.reasoning_content
                        # Handle final response content
                        else:
                            if not is_answering:  # Print title when entering the response phase for the first time
                                is_answering = True
                            if delta.content is not None:
                                answer_content += delta.content

                            # Handle tool invocation information (support parallel tool calls)
                            if delta.tool_calls is not None:
                                for tool_call in delta.tool_calls:
                                    index = tool_call.index  # Tool call index, used for parallel calls

                                    # Dynamically expand tool information storage list
                                    while len(tool_info) <= index:
                                        tool_info.append(
                                            {
                                                "id": "",
                                                "type": "function",
                                                "index": index,
                                                "function": {"name": "", "arguments": ""},
                                            },
                                        )

                                    # Collect tool call ID (used for subsequent function calls)
                                    if tool_call.id:
                                        tool_info[index]["id"] += tool_call.id

                                    # Collect function name (used for subsequent routing to specific functions)
                                    if tool_call.function and tool_call.function.name:
                                        tool_info[index]["function"]["name"] += tool_call.function.name

                                    # Collect function parameters (in JSON string format, need subsequent parsing)
                                    if tool_call.function and tool_call.function.arguments:
                                        tool_info[index]["function"]["arguments"] += tool_call.function.arguments
                    msg = {
                        "role": "assistant",
                        "content": answer_content,
                        "reasoning_content": reasoning_content,
                    }
                    if tool_info:
                        msg["tool_calls"] = tool_info
                    return msg
            except Exception as e:
                logger.exception(f"encounter error with {e.args}")
                time.sleep(1 + i * 10)

        return "call llm error"

    def env_step(self, run_id: int, index: int, messages: str) -> str:
        """
        Process one step in the conversation.
        Both single turn and multi turn are supported.

        Args:
            messages: List of conversation messages, with the last one being assistant response
            test_entry: Test entry containing initial_config, involved_classes, question etc.
            **kwargs: Additional arguments for compatibility

        Returns:
            Dict containing next message and tools if applicable
        """
        try:
            if not messages:
                return handle_user_turn(self.original_test_entry[run_id][index], self.current_turn[run_id][index])

            if messages[-1]["role"] != "assistant":
                return create_error_response(
                    "Last message must be from assistant",
                )

            if "tool_calls" in messages[-1] and len(messages[-1]["tool_calls"]) > 0:
                try:
                    tool_calls = messages[-1]["tool_calls"]
                    decoded_calls = self._convert_tool_calls_to_execution_format(
                        tool_calls,
                    )
                    # decoded_calls:[function(param=xxx)]
                    print(f"decoded_calls: {decoded_calls}")
                    if is_empty_execute_response(decoded_calls):
                        warnings.warn(
                            f"is_empty_execute_response: {is_empty_execute_response(decoded_calls)}",
                        )
                        return handle_user_turn(
                            self.original_test_entry[run_id][index],
                            self.current_turn[run_id][index],
                        )
                    return handle_tool_calls(
                        tool_calls,
                        decoded_calls,
                        self.original_test_entry[run_id][index],
                        self.current_turn[run_id][index],
                    )
                except Exception as e:
                    warnings.warn(f"Errors during tool invocation: {str(e)}")
                    return handle_user_turn(self.original_test_entry[run_id][index], self.current_turn[run_id][index])
            else:
                return handle_user_turn(self.original_test_entry[run_id][index], self.current_turn[run_id][index])

        except Exception as e:
            return create_error_response(f"Failed to process request: {str(e)}")

    def _convert_tool_calls_to_execution_format(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Convert OpenAI format tool calls to execution format.

        Args:
            tool_calls: List of tool calls in OpenAI format

        Returns:
            List of function calls in string format
        """
        execution_list = []

        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            function_name = function.get("name", "")

            try:
                arguments = function.get("arguments", "{}")
                if isinstance(arguments, str):
                    args_dict = json.loads(arguments)
                else:
                    args_dict = arguments

                args_str = ", ".join([f"{k}={repr(v)}" for k, v in args_dict.items()])
                execution_list.append(f"{function_name}({args_str})")

            except Exception:
                execution_list.append(f"{function_name}()")

        return execution_list

    def get_reward(self, run_id, index) -> float:
        """Get the reward."""
        try:
            if not self.history[run_id][index] or not self.original_test_entry[run_id][index]:
                return 0.0

            model_name = "env_handler"
            handler = QwenAPIHandler(
                model_name,
                temperature=1.0,
            )  # FIXME: magic number

            model_result_data = self._convert_conversation_to_eval_format(run_id, index)

            prompt_data = [self.original_test_entry[run_id][index]]

            state = {"leaderboard_table": {}}
            record_cost_latency(
                state["leaderboard_table"],
                model_name,
                [model_result_data],
            )

            if is_relevance_or_irrelevance(self.categories[index]):
                accuracy, _ = self._eval_relevance_test(
                    handler,
                    model_result_data,
                    prompt_data,
                    model_name,
                    self.category,
                )
            else:
                # Find the corresponding possible answer file

                possible_answer_file = find_file_with_suffix(
                    self.answer_path,
                    self.categories[index],
                )
                possible_answer = load_file(possible_answer_file, sort_by_id=True)
                possible_answer = [item for item in possible_answer if item["id"] == self.task_ids[index]]
                if is_multi_turn(self.categories[index]):
                    accuracy, _ = self._eval_multi_turn_test(
                        handler,
                        model_result_data,
                        prompt_data,
                        possible_answer,
                        model_name,
                        self.categories[index],
                    )
                else:
                    accuracy, _ = self._eval_single_turn_test(
                        handler,
                        model_result_data,
                        prompt_data,
                        possible_answer,
                        model_name,
                        self.categories[index],
                    )
            print(f"model_result_data: {model_result_data}")
            if possible_answer:
                print(f"possible_answer: {possible_answer}")
            else:
                print("possible_answer: None")

            return accuracy

        except Exception:
            import traceback

            traceback.print_exc()
            return 0

    def _convert_conversation_to_eval_format(self, run_id, index) -> Dict[str, Any]:
        """
        Convert conversation history to evaluation format.

        Args:
            conversation_result: Result from run_conversation
            original_test_entry: Original test entry data

        Returns:
            Data in format expected by multi_turn_runner or other runners
        """
        if is_multi_turn(self.categories[index]):
            turns_data = extract_multi_turn_responses(self.history[run_id][index])
        else:
            turns_data = extract_single_turn_response(self.history[run_id][index])

        model_result_data = {
            "id": self.task_ids[index],
            "result": turns_data,
            "latency": 0,
            "input_token_count": 0,
            "output_token_count": 0,
        }

        return model_result_data

    def _eval_multi_turn_test(
        self,
        handler,
        model_result_data,
        prompt_data,
        possible_answer,
        model_name,
        test_category,
    ):
        """
        Evaluate multi-turn test.

        Args:
            handler: Model handler instance
            model_result_data: Model result data
            prompt_data: Prompt data
            possible_answer: Possible answer data
            model_name: Name of the model
            test_category: Category of the test

        Returns:
            Tuple of (accuracy, total_count)
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            score_dir = Path(temp_dir)
            accuracy, total_count = multi_turn_runner(
                handler=handler,
                model_result=[model_result_data],
                prompt=prompt_data,
                possible_answer=possible_answer,
                model_name=model_name,
                test_category=test_category,
                score_dir=score_dir,
            )
            capture_and_print_score_files(
                score_dir,
                model_name,
                test_category,
                "multi_turn",
            )
            return accuracy, total_count

    def _eval_single_turn_test(
        self,
        handler,
        model_result_data,
        prompt_data,
        possible_answer,
        model_name,
        test_category,
    ):
        """
        Evaluate single-turn AST test.

        Args:
            handler: Model handler instance
            model_result_data: Model result data
            prompt_data: Prompt data
            possible_answer: Possible answer data
            model_name: Name of the model
            test_category: Category of the test

        Returns:
            Tuple of (accuracy, total_count)
        """
        language = "Python"
        if "java" in test_category.lower():
            language = "Java"
        elif "js" in test_category.lower() or "javascript" in test_category.lower():
            language = "JavaScript"

        with tempfile.TemporaryDirectory() as temp_dir:
            score_dir = Path(temp_dir)
            accuracy, total_count = ast_file_runner(
                handler=handler,
                model_result=[model_result_data],
                prompt=prompt_data,
                possible_answer=possible_answer,
                language=language,
                test_category=test_category,
                model_name=model_name,
                score_dir=score_dir,
            )
            capture_and_print_score_files(
                score_dir,
                model_name,
                test_category,
                "single_turn",
            )
            return accuracy, total_count

    def execute(self):
        """Execute the agent."""
        result = []
        counter = 0
        for task_index, task_id in enumerate(tqdm(self.task_ids, desc=f"ray_index={self.index}")):
            t_result = None
            previous_memories = []
            for run_id in range(self.num_trials):
                try:
                    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    for i in range(self.max_interactions):
                        if self.use_memory and i == 0:
                            self.update_task_history_with_memory(run_id, task_index, previous_memories)
                        llm_output = self.call_llm(
                            self.history[run_id][task_index],
                            self.tool_schema[run_id][task_index],
                        )
                        self.history[run_id][task_index].append(llm_output)

                        env_output = self.env_step(run_id, task_index, self.history[run_id][task_index])
                        # Possible env_output returns after environment interaction:
                        # 1. Triggers a query with available tools list:
                        #    {"messages": [{"role": "user", "content": user_query}], "tools": tools}
                        # 2. Returns tool invocation result: {"messages":
                        #    [{"role": "tool", "content": {<exec_results>}, 'tool_call_id': 'chatcmpl-tool-xxx'}]}
                        #    <exec_results>: when success, returns result dicts, e.g., {"travel_cost_list": [x]},
                        #    when error, returns error message,
                        #    e.g., {"error": "cd: temporary: No such directory. You cannot use path ..."}
                        # 3. Conversation completion:
                        #    {"messages": [{"role": "env", "content": "[CONVERSATION_COMPLETED]"}]}
                        # 4. Program error: {"messages": [{"role": "env", "content": f"[ERROR] {error_message}"}]}

                        # tool_list update
                        if "tools" in env_output:
                            self.tool_schema[run_id][task_index] = extract_tool_schema(env_output["tools"])

                        new_tool_calls = []
                        new_tool_call_ids = []
                        next_user_msg = ""
                        for idx, msg in enumerate(env_output.get("messages", [])):
                            if msg["role"] == "tool" and len(msg["content"]) > 0:
                                new_tool_calls.append(msg.get("content", ""))
                                new_tool_call_ids.append(msg.get("tool_call_id", ""))
                            elif msg["role"] == "user":
                                next_user_msg = msg.get("content", "")
                                self.current_turn[run_id][task_index] += 1
                            else:  # for env role messages
                                next_user_msg = msg.get("content", "")

                        if new_tool_calls:
                            for idx, call in enumerate(new_tool_calls):
                                self.history[run_id][task_index].append(
                                    {"role": "tool", "content": str(call), "tool_call_id": new_tool_call_ids[idx]},
                                )
                        else:
                            self.history[run_id][task_index].append({"role": "user", "content": next_user_msg})

                        logger.info(f"index={self.index} task_id={task_id} iteration={i}")

                        if self.task_completed(run_id, task_index):
                            break

                    reward = self.get_reward(run_id, task_index)
                    if self.use_memory:
                        if self.use_memory_addition:
                            new_traj_list = [
                                self.get_traj_from_task_history(task_id, self.history[run_id][task_index], reward),
                            ]
                            previous_memories = self.summary_memory(new_traj_list)
                            if reward == 1:
                                self.add_memory(previous_memories)

                        # update the freq & utility attributes of retrieved memories
                        update_utility: bool = reward == 1
                        self.update_memory_information(self.retrieved_memory_list[run_id][task_index], update_utility)

                    counter += 1
                    if self.use_memory_deletion and counter % self.delete_freq == 0:
                        self.delete_memory()

                    t_result = {
                        "run_id": run_id,
                        "task_id": self.task_ids[task_index],
                        "experiment_name": self.experiment_name,
                        "task_completed": self.task_completed(run_id, task_index),
                        "reward": reward,
                        "task_history": self.history[run_id][task_index],
                        "task_start_time": start_time,
                    }
                    if reward == 1:
                        break

                except Exception as e:
                    logger.exception(f"encounter error with {e.args}")
            result.append(t_result)
        return result

    def task_completed(self, run_id, index):
        """
        Check if task is completed.

        Returns:
            True if task is completed, False otherwise
        """
        return self.history[run_id][index][-1]["content"] == "[CONVERSATION_COMPLETED]"


def main():
    """Main function to run the BFCLAgent."""
    with open(os.getenv("BFCL_DATA_PATH"), "r", encoding="utf-8") as f:
        task_ids = [json.loads(l)["id"] for l in f]
    dataset_name = "dev"
    agent = BFCLAgent(
        index=0,
        task_ids=[task_ids[0]],
        experiment_name=f"qwen3_8b_{dataset_name}",
    )
    result = agent.execute()
    logger.info(f"result={json.dumps(result)}")


if __name__ == "__main__":
    main()
