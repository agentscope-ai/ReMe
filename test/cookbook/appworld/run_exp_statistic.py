import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from loguru import logger


def calculate_best_at_k(scores: list, k: int) -> float:
    """
    Calculate best@k
    Divide scores into groups of size k, take the maximum value in each group,
    then average these maximum values

    Args:
        scores: List of after_score values for all runs of a task
        k: Group size

    Returns:
        best@k value
    """
    if len(scores) % k != 0:
        raise ValueError(f"Length of scores ({len(scores)}) must be divisible by k ({k})")

    group_maxs = []
    for i in range(0, len(scores), k):
        group = scores[i : i + k]
        group_maxs.append(max(group))

    return sum(group_maxs) / len(group_maxs)


def calculate_pass_at_k(scores: list, k: int) -> float:
    if len(scores) % k != 0:
        raise ValueError(f"Length of scores ({len(scores)}) must be divisible by k ({k})")

    group_maxs = []
    for i in range(0, len(scores), k):
        group = scores[i : i + k]
        is_pass = 1.0 if max(group) >= 1.0 else 0.0
        group_maxs.append(is_pass)

    return sum(group_maxs) / len(group_maxs)


def get_possible_k_values(total_runs: int) -> list:
    """
    Get all possible k values (factors of total_runs)

    Args:
        total_runs: Total number of runs

    Returns:
        List of k values in descending order
    """
    k_values = []
    for k in range(1, total_runs + 1):
        if total_runs % k == 0:
            k_values.append(k)
    return sorted(k_values, reverse=True)  # Sort from large to small


def run_exp_statistic():
    path: Path = Path("./exp_result")

    # Store results for all experiments
    all_results = {}

    for file in [f for f in path.glob("*.jsonl") if not f.stem[-1].isdigit()]:
        # Group results by task_id
        task_results = defaultdict(list)

        with open(file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)

                if isinstance(data, list):
                    for part_data in data:
                        task_id = part_data["task_id"]
                        after_score = part_data["after_score"]
                        task_results[task_id].append(after_score)
                else:
                    task_id = data["task_id"]
                    after_score = data["after_score"]
                    task_results[task_id].append(after_score)

        if not task_results:
            logger.warning(f"No valid data found in file {file}")
            continue

        # Check if each task has consistent number of runs
        run_counts = [len(scores) for scores in task_results.values()]
        if len(set(run_counts)) > 1:
            logger.warning(f"Inconsistent number of runs for different tasks in file {file}: {set(run_counts)}")
            continue

        num_runs = run_counts[0]
        logger.info(f"File {file}: {len(task_results)} tasks, {num_runs} runs per task")

        # Get all possible k values
        k_values = get_possible_k_values(num_runs)
        logger.info(f"Calculable best@k values: {k_values}")

        # Calculate various best@k values
        file_results = {"file": file.name}

        for k in k_values:
            best_at_k_scores = []
            pass_at_k_scores = []
            for task_id, scores in task_results.items():
                try:
                    best_k_score = calculate_best_at_k(scores, k)
                    pass_at_k_score = calculate_pass_at_k(scores, k)
                    pass_at_k_scores.append(pass_at_k_score)
                    best_at_k_scores.append(best_k_score)
                except ValueError as e:
                    logger.error(f"Error calculating best@{k} for task {task_id}: {e}")
                    continue

            if best_at_k_scores:
                avg_best_at_k = sum(best_at_k_scores) / len(best_at_k_scores)
                file_results[f"best@{k}"] = avg_best_at_k
                logger.info(f"file={file.name} best@{k}={avg_best_at_k:.4f}")

            if pass_at_k_scores:
                avg_pass_at_k = sum(pass_at_k_scores) / len(pass_at_k_scores)
                file_results[f"pass@{k}"] = avg_pass_at_k
                logger.info(f"file={file.name} pass@{k}={avg_pass_at_k:.4f}")

        all_results[file.name] = file_results

    # Create and display table
    if all_results:
        df = pd.DataFrame(list(all_results.values()))
        df = df.set_index("file")

        # Sort columns by the number in column name (best@8, best@4, best@2, best@1)
        pass_columns = [col for col in df.columns if col.startswith("pass@")]
        # best_columns = [col for col in df.columns]
        pass_columns.sort(key=lambda x: x, reverse=False)
        df = df[pass_columns]

        print("\n" + "=" * 80)
        print("Experiment Results Summary Table")
        print("=" * 80)
        print(df.round(4))
        print("=" * 80)

        # Save table to CSV
        output_path = path / "experiment_summary.csv"
        df.to_csv(output_path)
        logger.info(f"Results table saved to: {output_path}")
    else:
        logger.warning("No valid experiment results found")


if __name__ == "__main__":
    run_exp_statistic()
