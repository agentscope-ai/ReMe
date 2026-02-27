# pylint: disable=W0621
"""Preprocess multi-turn test cases"""

import json


from pathlib import Path
from bfcl_eval.model_handler.model_style import ModelStyle
from bfcl_eval.eval_checker.eval_runner_helper import load_file
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.constants.eval_config import MULTI_TURN_FUNC_DOC_PATH
from bfcl_eval.constants.category_mapping import MULTI_TURN_FUNC_DOC_FILE_MAPPING
from bfcl_eval.model_handler.utils import (
    convert_to_tool,
    func_doc_language_specific_pre_processing,
)


def process_multi_turn_test_case(file_path, output_path):
    """
    Multi-turn test cases don't have the function doc in the prompt. We need to add them here.
    """
    test_cases = []
    with open(output_path, "w", encoding="utf-8") as outf:
        with open(file_path, encoding="utf-8") as f:
            file = f.readlines()
            for line in file:
                entry = json.loads(line)
                if "multi_turn" not in entry["id"]:
                    continue
                test_category: str = entry["id"].rsplit("_", 1)[0]
                involved_classes = entry["involved_classes"]
                entry["function"] = []
                for func_collection in involved_classes:
                    # func_doc is a list of dict
                    func_doc = load_file(
                        MULTI_TURN_FUNC_DOC_PATH / MULTI_TURN_FUNC_DOC_FILE_MAPPING[func_collection],
                    )
                    entry["function"].extend(func_doc)

                # Handle Miss Func category; we need to remove the holdout function doc
                if "missed_function" in entry:
                    for turn_index, missed_func_names in entry["missed_function"].items():
                        entry["missed_function"][turn_index] = []
                        for missed_func_name in missed_func_names:
                            for i, func_doc in enumerate(entry["function"]):
                                if func_doc["name"] == missed_func_name:
                                    # Add the missed function doc to the missed_function list
                                    entry["missed_function"][turn_index].append(func_doc)
                                    # Remove it from the function list
                                    entry["function"].pop(i)
                                    break

                functions = func_doc_language_specific_pre_processing(entry["function"], test_category)
                tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, ModelStyle.OpenAI_Completions)

                test_cases.append(
                    {
                        "id": entry["id"],
                        "messages": entry["question"][0],
                        "tools": tools,
                        "extra": entry,
                    },
                )
                outf.write(json.dumps(test_cases[-1], ensure_ascii=False) + "\n")

    return test_cases


if __name__ == "__main__":
    file_path = Path("./gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v3_multi_turn_base.json")
    output_path = "data/multiturn_data_base.jsonl"
    preprocessed_test_cases = process_multi_turn_test_case(file_path, output_path)
