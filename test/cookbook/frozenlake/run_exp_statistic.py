import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from loguru import logger


def calculate_best_at_k(scores: List[float], k: int) -> float:
    """
    Calculate best@k metric.
    Divide scores into groups of size k, take the maximum value in each group,
    then average these maximum values.

    Args:
        scores: List of success scores (0 or 1) for all runs of a task
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


def get_possible_k_values(total_runs: int) -> List[int]:
    """Get all possible k values (divisors of total_runs)"""
    k_values = []
    for k in range(1, total_runs + 1):
        if total_runs % k == 0:
            k_values.append(k)
    return sorted(k_values, reverse=True)


def parse_task_config(task_id: str, map_config: Dict) -> Tuple[str, bool, bool]:
    """
    Parse task configuration from task_id and map_config.

    Returns:
        (condition, is_slippery, use_experience)
    """
    is_slippery = map_config.get("is_slippery", True)
    use_experience = map_config.get("use_experience", False)

    # Create condition string
    slip_str = "slippery" if is_slippery else "no_slip"
    exp_str = "with_exp" if use_experience else "no_exp"
    condition = f"{slip_str}_{exp_str}"

    return condition, is_slippery, use_experience


def analyze_frozenlake_results():
    """Analyze FrozenLake experiment results"""
    path = Path("./exp_result")

    if not path.exists():
        logger.error("Experiment results directory not found!")
        return

    all_results = {}

    # Process all result files
    for file in path.glob("*test*.jsonl"):
        logger.info(f"Processing {file.name}")

        # Group results by condition and map
        condition_results = defaultdict(lambda: defaultdict(list))

        with open(file, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)

                    if isinstance(data, list):
                        for item in data:
                            process_single_result(item, condition_results)
                    else:
                        process_single_result(data, condition_results)

                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in {file.name}: {e}")
                    continue

        if not condition_results:
            logger.warning(f"No valid data found in {file.name}")
            continue

        # Calculate metrics for this file
        file_metrics = calculate_file_metrics(condition_results, file.name)
        all_results[file.name] = file_metrics

    # Generate comprehensive report
    if all_results:
        generate_analysis_report(all_results)
    else:
        logger.warning("No valid results found!")


def process_single_result(data: Dict, condition_results: Dict):
    """Process a single result entry"""
    map_config = data.get("map_config", {})
    task_id = data.get("task_id", "unknown")
    success = data.get("success", False)

    # Parse condition
    condition, is_slippery, use_experience = parse_task_config(task_id, map_config)

    # Extract map identifier - prefer map_id from map_config
    map_id = map_config.get("map_id", "unknown")
    if map_id == "unknown" and "test_map" in task_id:
        # Fallback to parsing from task_id
        parts = task_id.split("_")
        for part in parts:
            if part.startswith("map"):
                try:
                    # Extract number from "mapXX"
                    map_num = "".join(filter(str.isdigit, part))
                    if map_num:
                        map_id = int(map_num)
                        break
                except:
                    pass

    # Store result
    success_score = 1.0 if success else 0.0
    condition_results[condition][f"map_{map_id}"].append(success_score)


def calculate_file_metrics(condition_results: Dict, filename: str) -> Dict:
    """Calculate metrics for a single file"""
    file_metrics = {"file": filename}

    for condition, map_results in condition_results.items():
        condition_scores = []

        # Collect all scores for this condition
        for map_id, scores in map_results.items():
            condition_scores.extend(scores)

        if not condition_scores:
            continue

        # Check if all maps have the same number of runs
        run_counts = [len(scores) for scores in map_results.values()]
        if len(set(run_counts)) > 1:
            logger.warning(f"Inconsistent runs for {condition}: {set(run_counts)}")
            continue

        num_runs = run_counts[0] if run_counts else 0
        if num_runs == 0:
            continue

        # Calculate overall success rate
        overall_success = sum(condition_scores) / len(condition_scores)
        file_metrics[f"{condition}_success_rate"] = overall_success

        # Calculate best@k metrics
        k_values = get_possible_k_values(num_runs)
        for k in k_values:
            try:
                # Calculate best@k for each map, then average
                map_best_k_scores = []
                for map_id, scores in map_results.items():
                    map_best_k = calculate_best_at_k(scores, k)
                    map_best_k_scores.append(map_best_k)

                avg_best_k = sum(map_best_k_scores) / len(map_best_k_scores)
                file_metrics[f"{condition}_best@{k}"] = avg_best_k

            except ValueError as e:
                logger.warning(f"Error calculating best@{k} for {condition}: {e}")

        # Map-level analysis
        map_success_rates = {}
        for map_id, scores in map_results.items():
            map_success_rate = sum(scores) / len(scores)
            map_success_rates[map_id] = map_success_rate

        file_metrics[f"{condition}_map_details"] = map_success_rates

        logger.info(
            f"{filename} - {condition}: {overall_success:.3f} success rate, "
            f"{len(map_results)} maps, {num_runs} runs each",
        )

    return file_metrics


def generate_analysis_report(all_results: Dict):
    """Generate comprehensive analysis report"""
    logger.info("Generating comprehensive analysis report...")

    # 1. Create summary table
    summary_data = []
    for file_name, metrics in all_results.items():
        row = {"file": file_name}

        # Extract success rates and best@k metrics
        for key, value in metrics.items():
            if key != "file" and not key.endswith("_map_details"):
                row[key] = value

        summary_data.append(row)

    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.set_index("file")

        print("\n" + "=" * 100)
        print("FROZENLAKE EXPERIMENT RESULTS SUMMARY")
        print("=" * 100)
        print(df_summary.round(4))
        print("=" * 100)

        # Save summary table
        output_path = Path("./exp_result") / "frozenlake_summary.csv"
        df_summary.to_csv(output_path)
        logger.info(f"Summary table saved to: {output_path}")

    # 2. Condition comparison
    print("\n" + "=" * 80)
    print("CONDITION COMPARISON")
    print("=" * 80)

    condition_comparison = defaultdict(list)

    for file_name, metrics in all_results.items():
        for key, value in metrics.items():
            if "_success_rate" in key:
                condition = key.replace("_success_rate", "")
                condition_comparison[condition].append(value)

    # Calculate average performance per condition
    condition_avg = {}
    for condition, scores in condition_comparison.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            condition_avg[condition] = avg_score
            print(f"{condition:20s}: {avg_score:.4f} (±{pd.Series(scores).std():.4f})")

    # 3. Experience effect analysis
    print("\n" + "=" * 80)
    print("EXPERIENCE EFFECT ANALYSIS")
    print("=" * 80)

    experience_analysis = analyze_experience_effect(condition_avg)
    for analysis_line in experience_analysis:
        print(analysis_line)

    # 4. Map difficulty analysis
    print("\n" + "=" * 80)
    print("MAP DIFFICULTY ANALYSIS")
    print("=" * 80)

    map_analysis = analyze_map_difficulty(all_results)
    for map_id, difficulty in map_analysis.items():
        print(f"{map_id:10s}: {difficulty:.4f} average success rate")

    # 5. Detailed statistics
    print("\n" + "=" * 80)
    print("DETAILED STATISTICS")
    print("=" * 80)

    generate_detailed_stats(all_results)


def analyze_experience_effect(condition_avg: Dict[str, float]) -> List[str]:
    """Analyze the effect of experience on performance"""
    analysis = []

    # Compare with/without experience for each slippery condition
    slippery_no_exp = condition_avg.get("slippery_no_exp", 0)
    slippery_with_exp = condition_avg.get("slippery_with_exp", 0)
    no_slip_no_exp = condition_avg.get("no_slip_no_exp", 0)
    no_slip_with_exp = condition_avg.get("no_slip_with_exp", 0)

    if slippery_no_exp > 0 and slippery_with_exp > 0:
        improvement_slippery = (slippery_with_exp - slippery_no_exp) / slippery_no_exp * 100
        analysis.append(f"Slippery condition - Experience effect: {improvement_slippery:+.1f}%")
        analysis.append(f"  Without exp: {slippery_no_exp:.4f}")
        analysis.append(f"  With exp:    {slippery_with_exp:.4f}")

    if no_slip_no_exp > 0 and no_slip_with_exp > 0:
        improvement_no_slip = (no_slip_with_exp - no_slip_no_exp) / no_slip_no_exp * 100
        analysis.append(f"No-slip condition - Experience effect: {improvement_no_slip:+.1f}%")
        analysis.append(f"  Without exp: {no_slip_no_exp:.4f}")
        analysis.append(f"  With exp:    {no_slip_with_exp:.4f}")

    # Overall experience effect
    exp_conditions = [v for k, v in condition_avg.items() if "with_exp" in k]
    no_exp_conditions = [v for k, v in condition_avg.items() if "no_exp" in k]

    if exp_conditions and no_exp_conditions:
        avg_with_exp = sum(exp_conditions) / len(exp_conditions)
        avg_without_exp = sum(no_exp_conditions) / len(no_exp_conditions)
        overall_improvement = (avg_with_exp - avg_without_exp) / avg_without_exp * 100
        analysis.append(f"Overall experience effect: {overall_improvement:+.1f}%")

    return analysis


def analyze_map_difficulty(all_results: Dict) -> Dict[str, float]:
    """Analyze difficulty of different maps"""
    map_scores = defaultdict(list)

    for file_name, metrics in all_results.items():
        for key, value in metrics.items():
            if key.endswith("_map_details") and isinstance(value, dict):
                for map_id, success_rate in value.items():
                    map_scores[map_id].append(success_rate)

    # Calculate average difficulty per map
    map_difficulty = {}
    for map_id, scores in map_scores.items():
        if scores:
            avg_success = sum(scores) / len(scores)
            map_difficulty[map_id] = avg_success

    # Sort by difficulty (hardest first)
    return dict(sorted(map_difficulty.items(), key=lambda x: x[1]))


def generate_detailed_stats(all_results: Dict):
    """Generate detailed statistics"""
    total_experiments = len(all_results)
    total_conditions = set()

    for metrics in all_results.values():
        for key in metrics.keys():
            if "_success_rate" in key:
                condition = key.replace("_success_rate", "")
                total_conditions.add(condition)

    print(f"Total experiment files: {total_experiments}")
    print(f"Total conditions tested: {len(total_conditions)}")
    print(f"Conditions: {', '.join(sorted(total_conditions))}")

    # Best performing conditions
    all_success_rates = []
    for metrics in all_results.values():
        for key, value in metrics.items():
            if "_success_rate" in key and isinstance(value, (int, float)):
                all_success_rates.append((key.replace("_success_rate", ""), value))

    if all_success_rates:
        best_condition = max(all_success_rates, key=lambda x: x[1])
        worst_condition = min(all_success_rates, key=lambda x: x[1])

        print(f"Best performance: {best_condition[0]} ({best_condition[1]:.4f})")
        print(f"Worst performance: {worst_condition[0]} ({worst_condition[1]:.4f})")


def main():
    """Main function for statistics analysis"""
    logger.info("🔍 Starting FrozenLake Results Analysis")
    analyze_frozenlake_results()
    logger.info("📊 Analysis completed!")


if __name__ == "__main__":
    main()
