import json
from pathlib import Path
from typing import Dict, List, Any

from bfcl_eval.constants.default_prompts import (
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
)
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
)
from bfcl_eval.model_handler.model_style import ModelStyle
from bfcl_eval.model_handler.utils import (
    convert_to_tool,
    default_decode_execute_prompting,
    func_doc_language_specific_pre_processing,
)


def load_test_case(data_path: str, test_id: str | None) -> Dict[str, Any]:
    if not Path(data_path).exists():
        raise FileNotFoundError(f"BFCL data file '{data_path}' not found")

    if test_id is None:
        raise ValueError("task_id is required")

    with open(data_path, "r", encoding="utf-8") as f:
        if str(test_id).isdigit():
            idx = int(test_id)
            for line_no, line in enumerate(f):
                if line_no == idx:
                    return json.loads(line)
            raise ValueError(f"Test case index {idx} not found in {data_path}")
        else:
            for line in f:
                data = json.loads(line)
                if data.get("id") == test_id:
                    return data
            raise ValueError(f"Test case id '{test_id}' not found in {data_path}")


def handle_user_turn(
    test_entry: Dict[str, Any],
    current_turn: int,
) -> Dict[str, Any]:
    """
    Handle user turn by returning appropriate content from test_entry["question"].
    For non-first turns, processes user query and tools.

    Args:
        test_entry: Test entry containing conversation data
        current_turn: Current turn number

    Returns:
        Response containing next user message and tools
    """
    try:
        current_turn_message = []
        tools = compile_tools(test_entry)
        questions = test_entry.get("question", [])
        holdout_function = test_entry.get("holdout_function", {})

        if str(current_turn) in holdout_function:
            test_entry["function"].extend(holdout_function[str(current_turn)])
            tools = compile_tools(test_entry)
            assert len(questions[current_turn]) == 0, "Holdout turn should not have user message."
            current_turn_message = [
                {
                    "role": "user",
                    "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
                },
            ]
            return create_user_response(current_turn_message, tools)
        if current_turn >= len(questions):
            return create_completion_response()

        current_turn_message = questions[current_turn]

        return create_user_response(current_turn_message, tools)

    except Exception as e:
        return create_error_response(f"Failed to process user message: {str(e)}")


def handle_tool_calls(
    tool_calls: List[Dict[str, Any]],
    decoded_calls: list[str],
    test_entry: Dict[str, Any],
    current_turn: int,
) -> Dict[str, Any]:
    """
    Handle tool calls from assistant.

    Args:
        tool_calls: List of tool calls in OpenAI format
        decoded_calls: List of decoded function calls
        test_entry: Test entry containing environment data
        current_turn: Current turn number

    Returns:
        Response containing tool execution results
    """
    execution_results, _ = execute_multi_turn_func_call(
        func_call_list=decoded_calls,
        initial_config=test_entry["initial_config"],
        involved_classes=test_entry["involved_classes"],
        model_name="env_handler",
        test_entry_id=test_entry["id"],
        long_context=("long_context" in test_entry["id"] or "composite" in test_entry["id"]),
        is_evaL_run=False,
    )
    # print('execution_results in handler_tool_calls:', execution_results)

    return create_tool_response(tool_calls, execution_results)


def compile_tools(test_entry: dict) -> list:
    """
    Compile functions into tools format.

    Args:
        test_entry: Test entry containing functions

    Returns:
        List of tools in OpenAI format
    """
    functions: list = test_entry["function"]
    test_category: str = test_entry["id"].rsplit("_", 1)[0]

    functions = func_doc_language_specific_pre_processing(functions, test_category)
    tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, ModelStyle.OpenAI_Completions)

    return tools


def create_tool_response(
    tool_calls: List[Dict[str, Any]],
    execution_results: List[str],
) -> Dict[str, Any]:
    """
    Create response for tool calls.

    Args:
        tool_calls: List of tool calls
        execution_results: List of execution results

    Returns:
        Response containing tool execution results
    """
    tool_messages = []
    for i, (tool_call, result) in enumerate(zip(tool_calls, execution_results)):
        tool_messages.append(
            {
                "role": "tool",
                "content": result,
                "tool_call_id": tool_call.get("id", f"call_{i}"),
            },
        )

    return {"messages": tool_messages}


def create_user_response(
    question_turn: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Create response containing user message.

    Args:
        question_turn: List of messages for current turn
        tools: List of available tools

    Returns:
        Response containing user message and tools
    """
    user_content = ""
    for msg in question_turn:
        if msg["role"] == "user":
            user_content = msg["content"]
            break

    return {"messages": [{"role": "user", "content": user_content}], "tools": tools}


def create_completion_response() -> Dict[str, Any]:
    """
    Create response indicating conversation completion.

    Returns:
        Response with completion message
    """
    return {"messages": [{"role": "env", "content": "[CONVERSATION_COMPLETED]"}]}


def create_error_response(error_message: str) -> Dict[str, Any]:
    """
    Create response for error conditions.

    Args:
        error_message: Error message to include

    Returns:
        Response containing error message
    """
    return {"messages": [{"role": "env", "content": f"[ERROR] {error_message}"}]}


def decode_execute(result):
    """
    Decode execute results for compatibility with evaluation framework.

    Args:
        result: Result to decode

    Returns:
        List of decoded function calls
    """
    return default_decode_execute_prompting(result)


def extract_single_turn_response(messages: List[Dict[str, Any]]) -> str:
    """
    Extract single-turn response from conversation messages.

    Args:
        messages: List of conversation messages

    Returns:
        String representation of the response
    """
    for message in reversed(messages):
        if message["role"] == "assistant":
            if "tool_calls" in message and message["tool_calls"]:
                formatted_calls = []
                for tool_call in message["tool_calls"]:
                    formatted_call = format_single_tool_call_for_eval(
                        tool_call,
                    )
                    if formatted_call:
                        formatted_calls.append(formatted_call)
                return "\n".join(formatted_calls) if formatted_calls else ""
            elif message.get("content"):
                return message["content"]

    return ""


def extract_multi_turn_responses(
    messages: List[Dict[str, Any]],
) -> List[List[str]]:
    """
    Extract multi-turn responses from conversation messages.

    Args:
        messages: List of conversation messages

    Returns:
        List of turns, each turn is a list of function call strings
    """
    turns_data = []
    current_turn_responses = []

    i = 0
    while i < len(messages):
        message = messages[i]

        if message["role"] == "user":
            if current_turn_responses:
                turns_data.append(current_turn_responses)
                current_turn_responses = []

            i += 1
            while i < len(messages) and messages[i]["role"] == "assistant":
                assistant_msg = messages[i]

                if "tool_calls" in assistant_msg and assistant_msg["tool_calls"]:
                    for tool_call in assistant_msg["tool_calls"]:
                        formatted_call = format_single_tool_call_for_eval(
                            tool_call,
                        )
                        if formatted_call:
                            current_turn_responses.append(formatted_call)

                i += 1

                while i < len(messages) and messages[i]["role"] == "tool":
                    i += 1
        else:
            i += 1

    if current_turn_responses:
        turns_data.append(current_turn_responses)

    return turns_data


def format_single_tool_call_for_eval(tool_call: Dict[str, Any]) -> str:
    """
    Format a single tool call into string representation for evaluation.

    Args:
        tool_call: Single tool call in OpenAI format

    Returns:
        Formatted string representation
    """
    function = tool_call.get("function", {})
    function_name = function.get("name", "")

    try:
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, str):
            args_dict = json.loads(arguments)
        else:
            args_dict = arguments

        args_str = ", ".join([f"{k}={repr(v)}" for k, v in args_dict.items()])
        return f"{function_name}({args_str})"

    except Exception:
        return f"{function_name}()"


def capture_and_print_score_files(
    score_dir: Path,
    model_name: str,
    test_category: str,
    eval_type: str,
):
    """
    Capture and print contents of score files written to score_dir.

    Args:
        score_dir: Directory containing score files
        model_name: Name of the model
        test_category: Category of the test
        eval_type: Type of evaluation (relevance/multi_turn/single_turn)
    """
    try:
        print(f"\n=== {eval_type.upper()} Evaluation Result Files ===")
        print(f"Model: {model_name}")
        print(f"Test Category: {test_category}")
        print(f"Evaluation Type: {eval_type}")

        for file_path in score_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(score_dir)
                print(f"\n--- File: {relative_path} ---")

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    if (
                        file_path.suffix == ".json"
                        or content.strip().startswith("{")
                        or content.strip().startswith("[")
                    ):
                        try:
                            import json

                            lines = content.strip().split("\n")
                            formatted_lines = []
                            for line in lines:
                                if line.strip():
                                    parsed = json.loads(line)
                                    formatted_lines.append(
                                        json.dumps(
                                            parsed,
                                            ensure_ascii=False,
                                            indent=2,
                                        ),
                                    )
                            content = "\n".join(formatted_lines)
                        except json.JSONDecodeError:
                            pass

                    print(content)

                except UnicodeDecodeError:
                    print(f"[Binary file, size: {file_path.stat().st_size} bytes]")
                except Exception as e:
                    print(f"[Error reading file: {str(e)}]")

        print(f"=== {eval_type.upper()} Evaluation Result Files End ===\n")

    except Exception as e:
        print(f"Error capturing evaluation result files: {str(e)}")


def extract_tool_schema(tools):
    for i in range(len(tools)):
        tools[i]["function"].pop("response")
    return tools
