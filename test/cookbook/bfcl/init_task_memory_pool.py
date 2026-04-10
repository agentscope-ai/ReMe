import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any

import requests


def load_task_case(data_path: str, task_id: str | None) -> Dict[str, Any]:
    """
    load training cases by id
    """
    if not Path(data_path).exists():
        raise FileNotFoundError(f"BFCL data file '{data_path}' not found")

    if task_id is None:
        raise ValueError("task_id is required")

    with open(data_path, "r", encoding="utf-8") as f:
        if str(task_id).isdigit():
            idx = int(task_id)
            for line_no, line in enumerate(f):
                if line_no == idx:
                    return json.loads(line)
            raise ValueError(f"Task case index {idx} not found in {data_path}")
        else:
            for line in f:
                data = json.loads(line)
                if data.get("id") == task_id:
                    return data
            raise ValueError(f"Task case id '{task_id}' not found in {data_path}")


def get_tool_prompt(tools):
    tool_prompt = "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
    for tool in tools:
        tool_prompt += "\n" + json.dumps(tool)
    tool_prompt += '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
    return tool_prompt


def group_trajectories_by_task_id(jsonl_entries: List[Dict[str, Any]]) -> List[List[Any]]:
    """
    group trajectories by task_id

    Args:
        jsonl_entries: JSONL entry list

    Returns:
        List[List[Any]]: trajectory list grouped by task_id
    """
    grouped = defaultdict(list)

    for entry in jsonl_entries:
        task_id = entry.get("task_id", "")
        taks_case = load_task_case("data/multiturn_data_base.jsonl", task_id)
        tools = taks_case.get("tools", [{}])
        from bfcl_utils import extract_tool_schema

        tool_schema = extract_tool_schema(tools)
        entry["task_history"][0]["content"] += get_tool_prompt(tool_schema)
        grouped[task_id].append(entry)

    # retain only the two with the highest and lowest rewards
    filtered_groups = []
    for key, trajectories in grouped.items():
        if len(trajectories) == 1:
            # when only one trajectory, retain it
            filtered_groups.append(trajectories)
        elif len(trajectories) == 2:
            # when there are two trajectories, retain them
            filtered_groups.append(trajectories)
        else:
            # when there are more than two trajectories, choose the two with the highest and lowest rewards
            trajectories.sort(key=lambda t: t["reward"])
            min_reward_traj = trajectories[0]  # highest reward
            max_reward_traj = trajectories[-1]  # lowest reward
            filtered_groups.append([min_reward_traj, max_reward_traj])

    return filtered_groups


def post_to_summarizer(trajectories: List[Any], service_url: str, workspace_id: str) -> Dict[str, Any]:
    trajectory_dicts = [
        {
            "task_id": traj["task_id"],
            "messages": traj["task_history"],
            "score": traj["reward"],
        }
        for traj in trajectories
    ]

    request_data = {
        "trajectories": trajectory_dicts,
        "workspace_id": workspace_id,
    }

    try:
        response = requests.post(f"{service_url}/summary_task_memory", json=request_data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e), "trajectories_count": len(trajectories)}


def process_trajectories_with_threads(
    grouped_trajectories: List[List[Any]],
    service_url: str,
    workspace_id: str,
    n_threads: int = 4,
) -> List[Dict[str, Any]]:
    """
    use threads to process trajectories

    Args:
        grouped_trajectories: group trajectory list by task_id
        service_url: memory summarizer service URL
        workspace_id: workspace ID
        n_threads: number of threads

    Returns:
        all results
    """
    results = []

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        future_to_group = {
            executor.submit(post_to_summarizer, group, service_url, workspace_id): i
            for i, group in enumerate(grouped_trajectories)
        }

        for future in as_completed(future_to_group):
            group_index = future_to_group[future]
            try:
                result = future.result()
                result["group_index"] = group_index
                result["group_size"] = len(grouped_trajectories[group_index])
                results.append(result)
                print(
                    f'✅ Group {group_index} processed: {result["metadata"].get("memory_list", 0) if "memory_list" in result["metadata"] else "error"}',
                )
            except Exception as e:
                error_result = {
                    "group_index": group_index,
                    "group_size": len(grouped_trajectories[group_index]),
                    "error": str(e),
                }
                results.append(error_result)
                print(f"❌ Group {group_index} failed: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL to memories using ReMe service")
    parser.add_argument("--jsonl_file", type=str, required=True, help="Path to the JSONL file")
    parser.add_argument("--service_url", type=str, default="http://localhost:8001", help="ReMe service URL")
    parser.add_argument("--workspace_id", type=str, required=True, help="Workspace ID for the task memory pool")
    parser.add_argument("--output_file", type=str, help="Output file to save results (optional)")
    parser.add_argument("--n_threads", type=int, default=4, help="Number of threads for processing")

    args = parser.parse_args()

    print(f"Processing JSONL file: {args.jsonl_file}")
    print(f"Service URL: {args.service_url}")
    print(f"Workspace ID: {args.workspace_id}")
    print(f"Threads: {args.n_threads}")

    with open(args.jsonl_file, "r") as f:
        data = [json.loads(line) for line in f]
    print(f"Loaded {len(data)} entries from JSONL file")

    grouped_trajectories = group_trajectories_by_task_id(data)
    print(f"Total groups: {len(grouped_trajectories)}")

    results = process_trajectories_with_threads(
        grouped_trajectories,
        args.service_url,
        args.workspace_id,
        n_threads=args.n_threads,
    )

    print(f"Processed {len(results)} groups")

    success_count = sum(1 for r in results if "error" not in r)
    error_count = len(results) - success_count
    total_memories = sum(len(r["metadata"].get("memory_list", [])) for r in results if "memory_list" in r["metadata"])

    print(f"✅ Success: {success_count}")
    print(f"❌ Errors: {error_count}")
    print(f"📊 Total task memories created: {total_memories}")

    if args.output_file:
        try:
            summary = {
                "workspace_id": args.workspace_id,
                "jsonl_file": args.jsonl_file,
                "total_groups": len(grouped_trajectories),
                "success_count": success_count,
                "error_count": error_count,
                "total_task_memories": total_memories,
                "results": results,
            }

            with open(args.output_file, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Results saved to: {args.output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        main()
    else:
        print("Running in compatibility mode...")
        with open("exp_result/qwen3-8b/no_think/bfcl-multi-turn-base_wo-exp.jsonl", "r") as f:
            data = [json.loads(line) for line in f]

        grouped_trajectories = group_trajectories_by_task_id(data)
        print(f"Total groups: {len(grouped_trajectories)}")

        results = process_trajectories_with_threads(
            grouped_trajectories,
            "http://localhost:8001",
            "bfcl_v3",
            n_threads=4,
        )
        print(f"Processed {len(results)} groups")
