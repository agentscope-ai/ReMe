import json
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import ray
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from loguru import logger

from frozenlake_react_agent import FrozenLakeReactAgent
from map_manager import MapManager


def generate_training_configs(num_maps: int = 20, map_size: int = 4, is_slippery: bool = False) -> List[Dict]:
    """Generate random maps for training/task memory generation"""
    configs = []

    for i in range(num_maps):
        # Generate both slippery and non-slippery versions
        random_map = generate_random_map(size=map_size)

        config = {
            "task_type": "training",
            "map_desc": random_map,
            "map_size": map_size,
            "is_slippery": is_slippery,
            "task_id": f"train_{i}_{is_slippery}",
        }
        configs.append(config)

    return configs


def generate_test_configs(num_test_maps: int = 100, is_slippery: bool = False) -> List[Dict]:
    """Generate test configurations using MapManager"""
    logger.info(f"📋 Generating test configurations for {num_test_maps} maps")

    # Initialize MapManager and get test maps
    map_manager = MapManager()
    maps_data = map_manager.get_or_create_test_maps(num_maps=num_test_maps, map_size=4)

    configs = []

    for map_data in maps_data:
        map_desc = np.array([list(row) for row in map_data["map_desc"]], dtype="c")
        map_id = map_data["map_id"]

        for use_memory in [True, False]:
            config = {
                "task_type": "test",
                "map_desc": map_desc,
                "map_size": 4,
                "is_slippery": is_slippery,
                "use_task_memory": use_memory,
                "map_id": map_id,
                "task_id": f"test_map{map_id}_slip{is_slippery}_mem{use_memory}",
            }
            configs.append(config)

    logger.info(f"✅ Generated {len(configs)} test configurations")
    return configs


def train(
    experiment_name: str,
    max_workers: int = 2,
    num_runs: int = 3,
    num_training_maps=15,
    is_slippery: bool = False,
) -> None:
    """Phase 1: Generate task memory from random maps"""
    logger.info("🎯 Starting Training Phase - Generating Task Memory")
    logger.info("=" * 60)

    training_configs = generate_training_configs(num_maps=num_training_maps, map_size=4, is_slippery=is_slippery)
    path = Path("./exp_result")
    path.mkdir(parents=True, exist_ok=True)

    results = []

    def dump_results():
        output_file = path / f"{experiment_name}_training.jsonl"
        with open(output_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        logger.info(f"Training results saved to {output_file}")

    if max_workers > 1:
        # Distributed training
        future_list = []
        for i in range(max_workers):
            worker_configs = training_configs[i::max_workers]
            if worker_configs:  # Only create worker if it has tasks
                agent = FrozenLakeReactAgent.remote(
                    index=i,
                    task_configs=worker_configs,
                    experiment_name=experiment_name,
                    num_runs=num_runs,
                    use_task_memory=False,  # No task memory in training phase
                    make_task_memory=True,  # Generate task memory
                )
                future = agent.execute.remote()
                future_list.append(future)
                time.sleep(1)

        logger.info(f"Started {len(future_list)} training workers")

        for i, future in enumerate(future_list):
            worker_results = ray.get(future)
            if worker_results:
                results.extend(worker_results)
                logger.info(f"results: {results[0]}")
            logger.info(f"Training worker {i + 1}/{len(future_list)} completed")
            dump_results()

    else:
        # Single process training
        agent = FrozenLakeReactAgent(
            index=0,
            task_configs=training_configs,
            experiment_name=experiment_name,
            num_runs=num_runs,
            use_task_memory=False,
            make_task_memory=True,
        )
        results = agent.execute()
        dump_results()

    # Calculate training statistics
    successful_runs = [r for r in results if r["success"]]
    total_runs = len(results)
    success_rate = len(successful_runs) / total_runs if total_runs > 0 else 0

    logger.info(f"Training completed: {len(successful_runs)}/{total_runs} successful ({success_rate:.2%})")
    return results


def test(
    experiment_name: str,
    max_workers: int = 2,
    num_runs: int = 5,
    num_test_maps: int = 100,
    is_slippery: bool = False,
) -> None:
    """Phase 2: Test on fixed maps with/without task memory"""
    logger.info("🧪 Starting Test Phase - Evaluating Performance")
    logger.info(f"📊 Testing on {num_test_maps} maps with {num_runs} runs each")
    logger.info("=" * 60)

    test_configs = generate_test_configs(num_test_maps=num_test_maps, is_slippery=is_slippery)
    path = Path("./exp_result")
    path.mkdir(parents=True, exist_ok=True)

    # Group configs by task memory usage for separate experiments
    memory_configs = [c for c in test_configs if c.get("use_task_memory", False)]
    no_memory_configs = [c for c in test_configs if not c.get("use_task_memory", False)]

    logger.info(f"📝 Configs without task memory: {len(no_memory_configs)}")
    logger.info(f"📝 Configs with task memory: {len(memory_configs)}")

    def dump_results(suffix: str):
        output_file = path / f"{experiment_name}_test_{suffix}.jsonl"
        with open(output_file, "w") as f:
            for result in all_results:
                f.write(json.dumps(result) + "\n")
        logger.info(f"💾 Test results saved to {output_file}")

    # Test without task memory first
    logger.info("🚫 Testing WITHOUT task memory...")
    all_results = []
    results_no_memory = run_test_configs(
        configs=no_memory_configs,
        experiment_name=experiment_name,
        max_workers=max_workers,
        num_runs=num_runs,
        use_task_memory=False,
    )
    all_results.extend(results_no_memory)
    dump_results("no_memory")

    # Test with task memory
    logger.info("✅ Testing WITH task memory...")
    all_results = []
    results_with_memory = run_test_configs(
        configs=memory_configs,
        experiment_name=experiment_name,
        max_workers=max_workers,
        num_runs=num_runs,
        use_task_memory=True,
    )
    all_results.extend(results_with_memory)
    dump_results("with_memory")

    return all_results


def run_test_configs(
    configs: List[Dict],
    experiment_name: str,
    max_workers: int,
    num_runs: int,
    use_task_memory: bool,
) -> List[Dict]:
    """Run a set of test configurations"""
    results = []

    if max_workers > 1:
        future_list = []
        for i in range(max_workers):
            worker_configs = configs[i::max_workers]
            if worker_configs:
                agent = FrozenLakeReactAgent.remote(
                    index=i,
                    task_configs=worker_configs,
                    experiment_name=experiment_name,
                    num_runs=num_runs,
                    use_task_memory=use_task_memory,
                    make_task_memory=False,
                )
                future = agent.execute.remote()
                future_list.append(future)
                time.sleep(1)

        for i, future in enumerate(future_list):
            worker_results = ray.get(future)
            if worker_results:
                results.extend(worker_results)
            logger.info(f"Test worker {i + 1}/{len(future_list)} completed")

    else:
        agent = FrozenLakeReactAgent(
            index=0,
            task_configs=configs,
            experiment_name=experiment_name,
            num_runs=num_runs,
            use_task_memory=use_task_memory,
            make_task_memory=False,
        )
        results = agent.execute()

    return results


def main():
    """Main execution function"""
    experiment_name = "frozenlake_no_slippery"
    max_workers = 4
    training_runs = 4  # Runs per training map
    num_training_maps = 50
    test_runs = 1  # Runs per test configuration
    num_test_maps = 100  # Number of test maps to use
    is_slippery = False
    # model_name = "qwen-max-latest"

    # Initialize Ray if using multiple workers
    if max_workers > 1:
        ray.init(num_cpus=max_workers)

    try:
        # Phase 1: Training (Experience Generation)
        logger.info("🚀 Starting FrozenLake Experiment")
        logger.info(f"🎯 Experiment: {experiment_name}")
        logger.info(f"🏃 Workers: {max_workers}")
        logger.info(f"📊 Test maps: {num_test_maps}")
        logger.info(f"🔄 Test runs per map: {test_runs}")

        training_results = train(
            experiment_name=experiment_name,
            max_workers=max_workers,
            num_runs=training_runs,
            num_training_maps=num_training_maps,
            is_slippery=is_slippery,
        )

        # Wait a bit for task memory service to process
        logger.info("⏰ Waiting for task memory service to process data...")
        time.sleep(10)

        # Phase 2: Testing (Performance Evaluation)
        test_results = test(
            experiment_name=experiment_name,
            max_workers=max_workers,
            num_runs=test_runs,
            num_test_maps=num_test_maps,
            is_slippery=is_slippery,
        )

        # Summary
        logger.info("🎉 Experiment completed!")
        logger.info(f"📈 Training results: {len(training_results)} episodes")
        logger.info(f"📈 Test results: {len(test_results)} episodes")

        # Quick statistics
        successful_training = sum(1 for r in training_results if r.get("success", False))
        training_success_rate = successful_training / len(training_results) if training_results else 0

        successful_test = sum(1 for r in test_results if r.get("success", False))
        test_success_rate = successful_test / len(test_results) if test_results else 0

        logger.info(f"📊 Training success rate: {training_success_rate:.2%}")
        logger.info(f"📊 Test success rate: {test_success_rate:.2%}")

    finally:
        if max_workers > 1:
            ray.shutdown()


if __name__ == "__main__":
    main()
