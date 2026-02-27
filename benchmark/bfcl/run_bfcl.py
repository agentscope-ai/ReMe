"""Run evaluation on BFCL-V3-Multi-Turn-Base dataset."""

import time
import json
from pathlib import Path

import ray
import requests
from loguru import logger
from dotenv import load_dotenv
from bfcl_agent import BFCLAgent

load_dotenv("../../.env")


def run_agent(
    max_workers: int,
    dataset_name: str,
    experiment_suffix: str,
    model_name: str = "qwen3-8b",
    enable_thinking: bool = False,
    data_path: str = "data/multiturn_data_base_val.jsonl",
    answer_path: Path = Path("data/possible_answer"),
    num_trials: int = 1,
    use_memory: bool = False,
    memory_base_url: str = "http://0.0.0.0:8002/",
    use_memory_addition: bool = True,
    use_memory_deletion: bool = False,
    delete_freq: int = 10,
    freq_threshold: int = 5,
    utility_threshold: float = 0.5,
):
    """Run the agent"""
    experiment_name = dataset_name + "_" + experiment_suffix
    path: Path = Path(
        f"./exp_result/{model_name}/with_think" if enable_thinking else f"./exp_result/{model_name}/no_think",
    )
    path.mkdir(parents=True, exist_ok=True)

    with open(data_path, "r", encoding="utf-8") as f:
        task_ids = [json.loads(line)["id"] for line in f]

    result: list = []

    def dump_file():
        with open(path / f"{experiment_name}.jsonl", "a", encoding="utf-8") as f:
            for x in result:
                f.write(json.dumps(x) + "\n")

    future_list: list = []
    for i in range(max_workers):
        actor = BFCLAgent.remote(
            index=i,
            model_name=model_name,
            task_ids=task_ids[i::max_workers],
            experiment_name=experiment_name,
            data_path=data_path,
            answer_path=answer_path,
            num_trials=num_trials,
            use_memory=use_memory,
            memory_base_url=memory_base_url,
            use_memory_addition=use_memory_addition,
            use_memory_deletion=use_memory_deletion,
            delete_freq=delete_freq,
            freq_threshold=freq_threshold,
            utility_threshold=utility_threshold,
            enable_thinking=enable_thinking,
        )
        future = actor.execute.remote()
        future_list.append(future)
        time.sleep(1)
    logger.info("submit complete")

    for i, future in enumerate(future_list):
        t_result = ray.get(future)
        if t_result:
            if isinstance(t_result, list):
                result.extend(t_result)
            else:
                result.append(t_result)

        logger.info(f"{i + 1}/{len(task_ids)} complete")
    dump_file()


def handle_api_response(response: requests.Response):
    """Handle API response with proper error checking"""
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

    return response.json()


def load_memory(path: str = "docs/library", api_url: str = "http://0.0.0.0:8002/"):
    """Load memories from disk into the vector store"""
    response = requests.post(
        url=f"{api_url}load_memory",
        json={
            "load_file_path": path,
            "clear_existing": True,
        },
    )

    result = handle_api_response(response)
    if result:
        print(f"Memory loaded from {path}")


def main():
    """Main function"""
    max_workers = 4
    if max_workers > 1:
        ray.init(num_cpus=max_workers)

    num_runs = 4
    num_trials = 1
    model_name = "qwen3-8b"
    enable_thinking = True
    use_memory = True
    use_memory_addition = False
    use_memory_deletion = False
    memory_base_url = "http://0.0.0.0:8003/"

    if use_memory:
        load_file_path = "docs/library/paper_data/task/bfcl_qwen3_8b.jsonl"
        load_memory(load_file_path, memory_base_url)

    for _ in range(num_runs):
        run_agent(
            max_workers=max_workers,
            model_name=model_name,
            dataset_name="bfcl-multi-turn-base",
            experiment_suffix="w-fixed-memory",
            data_path="data/multiturn_data_base_val.jsonl",
            answer_path=Path("data/possible_answer"),
            enable_thinking=enable_thinking,
            num_trials=num_trials,
            use_memory=use_memory,
            memory_base_url=memory_base_url,
            use_memory_addition=use_memory_addition,
            use_memory_deletion=use_memory_deletion,
            delete_freq=5,
            freq_threshold=5,
            utility_threshold=0.5,
        )


if __name__ == "__main__":
    main()
