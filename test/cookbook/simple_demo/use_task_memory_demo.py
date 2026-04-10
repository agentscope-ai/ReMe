#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task Memory Demo for MemoryScope

This script demonstrates how to use the task memory capabilities of MemoryScope.
It shows how to run an agent, summarize conversations, retrieve memories, and
manage the memory workspace.
"""

import json
import time
from typing import List, Dict, Any, Optional

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API configuration
BASE_URL = "http://0.0.0.0:8002/"
WORKSPACE_ID = "test_workspace"


def handle_api_response(response: requests.Response) -> Optional[Dict[str, Any]]:
    """
    Handle API response with proper error checking

    Args:
        response: Response object from requests

    Returns:
        Response JSON if successful, None otherwise
    """
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

    return response.json()


def delete_workspace() -> None:
    """
    Delete the current workspace from the vector store

    Returns:
        None
    """
    response = requests.post(
        url=f"{BASE_URL}vector_store",
        json={
            "workspace_id": WORKSPACE_ID,
            "action": "delete",
        },
    )

    result = handle_api_response(response)
    if result:
        print(f"Workspace '{WORKSPACE_ID}' deleted successfully")


def run_agent(query: str, dump_messages: bool = False) -> List[Dict[str, Any]]:
    """
    Run the agent with a specific query

    Args:
        query: The query to send to the agent
        dump_messages: Whether to save messages to a file

    Returns:
        List of message objects from the conversation
    """
    response = requests.post(
        url=f"{BASE_URL}react",
        json={"query": query},
    )

    result = handle_api_response(response)
    if not result:
        return []

    # Extract and display the answer
    answer = result.get("answer", "")
    print(f"Agent response: {answer}")

    # Get the conversation messages
    messages = result.get("messages", [])

    # Optionally save messages to file
    if dump_messages and messages:
        with open("task_messages.jsonl", "w") as f:
            f.write(json.dumps(messages, indent=2, ensure_ascii=False))
        print(f"Messages saved to messages.jsonl")

    return messages


def run_summary(messages: List[Dict[str, Any]], enable_dump_memory: bool = True) -> None:
    """
    Generate a summary of conversation messages and create task memories

    Args:
        messages: List of message objects from a conversation
        enable_dump_memory: Whether to save memory list to a file

    Returns:
        None
    """
    if not messages:
        print("No messages to summarize")
        return

    response = requests.post(
        # url=f"{BASE_URL}summary_task_memory_simple",
        url=f"{BASE_URL}summary_task_memory",
        json={
            "workspace_id": WORKSPACE_ID,
            "trajectories": [
                {"messages": messages, "score": 1.0},
            ],
        },
    )

    result = handle_api_response(response)
    if not result:
        return

    # Extract memory list from response
    memory_list = result.get("metadata", {}).get("memory_list", [])
    print(f"Memory list: {memory_list}")

    # Optionally save memory list to file
    if enable_dump_memory and memory_list:
        with open("task_memory.jsonl", "w") as f:
            f.write(json.dumps(memory_list, indent=2, ensure_ascii=False))
        print(f"Memory saved to memory.jsonl")


def run_retrieve(query: str) -> str:
    """
    Retrieve relevant task memories based on a query

    Args:
        query: The query to retrieve relevant memories

    Returns:
        String containing the retrieved memory answer
    """
    response = requests.post(
        # url=f"{BASE_URL}retrieve_task_memory_simple",
        url=f"{BASE_URL}retrieve_task_memory",
        json={
            "workspace_id": WORKSPACE_ID,
            "query": query,
        },
    )

    result = handle_api_response(response)
    if not result:
        return ""

    # Extract and return the answer
    answer = result.get("answer", "")
    print(f"Retrieved memory: {answer}")
    return answer


def run_agent_with_memory(query_first: str, query_second: str, enable_dump_memory: bool = True) -> List[Dict[str, Any]]:
    """
    Run the agent with memory augmentation

    This function demonstrates how to use task memory to enhance agent responses:
    1. First run the agent with the second query to build memory
    2. Then summarize the conversation to create memories
    3. Retrieve relevant memories for the first query
    4. Run the agent with the first query augmented with retrieved memories

    Args:
        query_first: The query to run with memory augmentation
        query_second: The query to build initial memories
        enable_dump_memory: Whether to save memory list to a file

    Returns:
        List of message objects from the final conversation
    """
    # Run agent with second query to build initial memories
    print(f"\n--- Building memories with query: '{query_second}' ---")
    messages = run_agent(query=query_second)

    # Summarize conversation to create memories
    print("\n--- Summarizing conversation to create memories ---")
    run_summary(messages, enable_dump_memory)
    time.sleep(1)

    # Retrieve relevant memories for the first query
    print(f"\n--- Retrieving memories for query: '{query_first}' ---")
    retrieved_memory = run_retrieve(query_first)

    # Run agent with first query augmented with retrieved memories
    print(f"\n--- Running agent with memory-augmented query ---")
    augmented_query = f"{retrieved_memory}\n\nUser Question:\n{query_first}"
    print(f"Augmented query: {augmented_query}")
    messages = run_agent(query=augmented_query)

    return messages


def dump_memory(path: str = "./") -> None:
    """
    Dump the vector store memories to disk

    Args:
        path: Directory path to save the memories

    Returns:
        None
    """
    response = requests.post(
        url=f"{BASE_URL}vector_store",
        json={
            "workspace_id": WORKSPACE_ID,
            "action": "dump",
            "path": path,
        },
    )

    result = handle_api_response(response)
    if result:
        print(f"Memory dumped to {path}")


def load_memory(path: str = "./") -> None:
    """
    Load memories from disk into the vector store

    Args:
        path: Directory path to load the memories from

    Returns:
        None
    """
    response = requests.post(
        url=f"{BASE_URL}vector_store",
        json={
            "workspace_id": WORKSPACE_ID,
            "action": "load",
            "path": path,
        },
    )

    result = handle_api_response(response)
    if result:
        print(f"Memory loaded from {path}")


def main() -> None:
    """
    Main function to demonstrate task memory workflow
    """
    # Define example queries
    query1 = "Analyze Xiaomi Corporation"
    query2 = "Analyze the company Tesla."

    print("=== Task Memory Demo ===")

    # Step 1: Clean up workspace
    print("\n1. Deleting workspace...")
    delete_workspace()

    # Step 2: Run agent with first query and save messages
    print("\n2. Running agent with first query...")
    run_agent(query=query1, dump_messages=True)

    # Step 3: Demonstrate memory-augmented agent
    print("\n3. Running memory-augmented agent workflow...")
    run_agent_with_memory(query_first=query1, query_second=query2)

    # Step 4: Demonstrate memory persistence
    print("\n4. Dumping memory to disk...")
    dump_memory()

    print("\n5. Loading memory from disk...")
    load_memory()

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
