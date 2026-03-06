# flake8: noqa: E402, E501
import os
from typing import List, Any

from tqdm import tqdm

os.environ["APPWORLD_ROOT"] = "."
from dotenv import load_dotenv

load_dotenv("../../.env")

import re
import time
import json
import ray
import requests
import datetime

from appworld import AppWorld, load_task_ids
from jinja2 import Template
from loguru import logger
from openai import OpenAI

from prompt import NEW_PROMPT_TEMPLATE


@ray.remote
class AppworldReactAgent:
    """A minimal ReAct Agent for AppWorld tasks."""

    def __init__(
        self,
        index: int,
        task_ids: List[str],
        experiment_name: str,
        model_name: str = "qwen3-8b",
        temperature: float = 0.9,
        max_interactions: int = 30,
        max_response_size: int = 129024,
        num_trials: int = 1,
        use_memory: bool = False,
        use_memory_addition: bool = False,
        use_memory_deletion: bool = False,
        delete_freq: int = 10,
        freq_threshold: int = 5,
        utility_threshold: float = 0.5,
        memory_base_url: str = "http://0.0.0.0:8002/",
        memory_workspace_id: str = "appworld_v1",
    ):

        self.index: int = index
        self.task_ids: List[str] = task_ids
        self.experiment_name: str = experiment_name
        self.model_name: str = model_name
        self.temperature: float = temperature
        self.max_interactions: int = max_interactions
        self.max_response_size: int = max_response_size
        self.num_trials: int = num_trials
        self.use_memory: bool = use_memory
        self.use_memory_addition: bool = use_memory_addition if use_memory else False
        self.use_memory_deletion: bool = use_memory_deletion if use_memory else False
        self.delete_freq: int = delete_freq
        self.freq_threshold: int = freq_threshold
        self.utility_threshold: float = utility_threshold
        self.memory_base_url: str = memory_base_url
        self.memory_workspace_id: str = memory_workspace_id

        self.llm_client = OpenAI()

        self.history: List[List[List[dict]]] = [[] for _ in range(num_trials)]
        self.retrieved_memory_list: List[List[List[Any]]] = [[] for _ in range(num_trials)]

        for run_id in range(num_trials):
            for _ in range(len(task_ids)):
                self.retrieved_memory_list[run_id].append([])
                self.history[run_id].append([])

    def call_llm(self, messages: list) -> str:
        for i in range(100):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    extra_body={"enable_thinking": False},
                    seed=0,
                )

                return response.choices[0].message.content

            except Exception as e:
                logger.exception(f"encounter error with {e.args}")
                time.sleep(1 + i * 10)

        return "call llm error"

    def prompt_messages(self, run_id, task_index, previous_memories: None, world: AppWorld):
        app_descriptions = json.dumps(
            [{"name": k, "description": v} for (k, v) in world.task.app_descriptions.items()],
            indent=1,
        )
        dictionary = {"supervisor": world.task.supervisor, "app_descriptions": app_descriptions}
        sys_prompt = Template(NEW_PROMPT_TEMPLATE.lstrip()).render(dictionary)
        query = world.task.instruction
        if self.use_memory:
            if len(previous_memories) == 0:
                response = self.get_memory(world.task.instruction)
                if response and "memory_list" in response["metadata"]:
                    self.retrieved_memory_list[run_id][task_index] = response["metadata"]["memory_list"]
                    task_memory = response["answer"]
                    logger.info(f"loaded task_memory: {task_memory}")
                    query = (
                        "Task:\n"
                        + query
                        + "\n\nSome Related Experience to help you to complete the task:\n"
                        + re.sub(r"(?i)\bMemory\s*(\d+)\s*[:]", r"Experience \1:", task_memory)
                    )
            else:
                formatted_memories = []
                for i, memory in enumerate(previous_memories, 1):
                    condition = memory["when_to_use"]
                    memory_content = memory["content"]
                    memory_text = f"Experience {i}:\n When to use: {condition}\n Content: {memory_content}\n"
                    formatted_memories.append(memory_text)
                query = (
                    "Task:\n"
                    + query
                    + "\n\nSome Related Experience to help you to complete the task:\n"
                    + "\n".join(formatted_memories)
                )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": query},
        ]
        self.history[run_id][task_index] = messages

    @staticmethod
    def get_reward(world) -> float:
        tracker = world.evaluate()
        num_passes = len(tracker.passes)
        num_failures = len(tracker.failures)
        return num_passes / (num_passes + num_failures)

    def extract_code_and_fix_content(
        self,
        text: str,
        ignore_multiple_calls=True,
    ) -> tuple[str, str]:
        full_code_regex = r"```python\n(.*?)```"
        partial_code_regex = r".*```python\n(.*)"

        original_text = text
        output_code = ""
        match_end = 0
        # Handle multiple calls
        for re_match in re.finditer(full_code_regex, original_text, flags=re.DOTALL):
            code = re_match.group(1).strip()
            if ignore_multiple_calls:
                text = original_text[: re_match.end()]
                return code, text
            output_code += code + "\n"
            match_end = re_match.end()
        # check for partial code match at end (no terminating ```)  following the last match
        partial_match = re.match(
            partial_code_regex,
            original_text[match_end:],
            flags=re.DOTALL,
        )
        if partial_match:
            output_code += partial_match.group(1).strip()
            # terminated due to stop condition. Add stop condition to output.
            if not text.endswith("\n"):
                text = text + "\n"
            text = text + "```"
        if len(output_code) == 0:
            return text, text
        else:
            return output_code, text

    def execute(self):
        result = []
        counter = 0
        for task_index, task_id in enumerate(tqdm(self.task_ids, desc=f"ray_index={self.index}")):
            t_result = None
            previous_memories = []
            # Run each task num_trials times
            for run_id in range(self.num_trials):
                start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with AppWorld(task_id=task_id, experiment_name=f"{self.experiment_name}_run_{run_id}") as world:
                    before_score = self.get_reward(world)
                    for i in range(self.max_interactions):
                        if i == 0:
                            self.prompt_messages(
                                run_id=run_id,
                                task_index=task_index,
                                previous_memories=previous_memories,
                                world=world,
                            )
                        code_msg = self.call_llm(self.history[run_id][task_index])
                        code, text = self.extract_code_and_fix_content(code_msg)
                        self.history[run_id][task_index].append({"role": "assistant", "content": code})

                        output = world.execute(code)
                        # if len(output) > self.max_response_size:
                        #     # logger.warning(f"output exceed max size={len(output)}")
                        #     output = output[: self.max_response_size]
                        self.history[run_id][task_index].append(
                            {"role": "user", "content": "Output:\n```\n" + output + "```\n\n"},
                        )

                        if world.task_completed():
                            break

                    after_score = self.get_reward(world)
                    uplift_score = after_score - before_score

                    if self.use_memory:
                        if self.use_memory_addition:
                            new_traj_list = [
                                self.get_traj_from_task_history(task_id, self.history[run_id][task_index], after_score),
                            ]
                            previous_memories = self.add_memory(new_traj_list)
                            if after_score != 1:
                                self.delete_memory_by_ids([mem["memory_id"] for mem in previous_memories])

                        # update the freq & utility attributes of retrieved memories
                        update_utility: bool = after_score == 1
                        self.update_memory_information(self.retrieved_memory_list[run_id][task_index], update_utility)

                    counter += 1
                    if self.use_memory_deletion:  # and counter % self.delete_freq == 0:
                        self.delete_memory()

                    t_result = {
                        "task_id": world.task_id,
                        "run_id": run_id,
                        "experiment_name": self.experiment_name,
                        "task_completed": world.task_completed(),
                        "before_score": before_score,
                        "after_score": after_score,
                        "uplift_score": uplift_score,
                        "task_history": self.history[run_id][task_index],
                        "task_start_time": start_time,
                    }
                    if after_score == 1:
                        break
            result.append(t_result)

        return result

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
                "workspace_id": self.memory_workspace_id,
                "query": query,
            },
        )

        result = self.handle_api_response(response)
        if not result:
            return None

        logger.info(f"query: {query}, response: {result}")
        return result

    def get_traj_from_task_history(self, task_id: str, task_history: list, reward: float):
        pattern = r"\n\nSome Related Experience to help you to complete the task:.*"
        task_history[1]["content"] = re.sub(pattern, "", task_history[1]["content"], flags=re.DOTALL)
        return {
            "task_id": task_id,
            "messages": task_history,
            "score": reward,
        }

    def add_memory(self, trajectories):
        """Generate a summary of conversation messages and create task memories"""

        response = requests.post(
            url=f"{self.memory_base_url}summary_task_memory",
            json={
                "workspace_id": self.memory_workspace_id,
                "trajectories": trajectories,
            },
        )

        result = self.handle_api_response(response)
        if not result:
            return []

        # Extract memory list from response
        memory_list = result.get("metadata", {}).get("memory_list", [])
        print(f"Task memory list created: {len(memory_list)} memories")
        return memory_list

    def delete_memory_by_ids(self, memory_ids):
        response = requests.post(
            url=f"{self.memory_base_url}vector_store",
            json={
                "workspace_id": self.memory_workspace_id,
                "action": "delete_ids",
                "memory_ids": memory_ids,
            },
        )
        response.raise_for_status()

    def update_memory_information(self, memory_list, update_utility: bool = False):
        response = requests.post(
            url=f"{self.memory_base_url}record_task_memory",
            json={
                "workspace_id": self.memory_workspace_id,
                "memory_dicts": memory_list,
                "update_utility": update_utility,
            },
        )
        response.raise_for_status()
        logger.info(response.json())

    def delete_memory(self):
        response = requests.post(
            url=f"{self.memory_base_url}delete_task_memory",
            json={
                "workspace_id": self.memory_workspace_id,
                "freq_threshold": self.freq_threshold,
                "utility_threshold": self.utility_threshold,
            },
        )
        response.raise_for_status()


def main():
    dataset_name = "train"
    task_ids = load_task_ids(dataset_name)
    agent = AppworldReactAgent(index=0, task_ids=task_ids[0:1], experiment_name=dataset_name, num_trials=1)
    result = agent.execute()
    logger.info(f"result={json.dumps(result)}")


if __name__ == "__main__":
    main()
