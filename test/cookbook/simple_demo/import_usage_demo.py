import asyncio

from reme_ai import ReMeApp


# ============================================
# Task Memory Management Examples
# ============================================


async def summary_task_memory(app: ReMeApp):
    """
    Experience Summarizer: Learn from execution trajectories

    curl -X POST http://localhost:8002/summary_task_memory \
      -H "Content-Type: application/json" \
      -d '{
        "workspace_id": "task_workspace",
        "trajectories": [
          {"messages": [{"role": "user", "content": "Help me create a project plan"}], "score": 1.0}
        ]
      }'
    """
    result = await app.async_execute(
        name="summary_task_memory",
        workspace_id="task_workspace",
        trajectories=[
            {
                "messages": [
                    {"role": "user", "content": "Help me create a project plan"},
                ],
                "score": 1.0,
            },
        ],
    )
    print("Summary Task Memory Result:")
    print(result["answer"])


async def retrieve_task_memory(app: ReMeApp):
    """
    Retriever: Get relevant memories

    curl -X POST http://localhost:8002/retrieve_task_memory \
      -H "Content-Type: application/json" \
      -d '{
        "workspace_id": "task_workspace",
        "query": "How to efficiently manage project progress?",
        "top_k": 1
      }'
    """
    result = await app.async_execute(
        name="retrieve_task_memory",
        workspace_id="task_workspace",
        query="How to efficiently manage project progress?",
        top_k=1,
    )
    print("Retrieve Task Memory Result:")
    print(result["answer"])


# ============================================
# Personal Memory Management Examples
# ============================================


async def summary_personal_memory(app: ReMeApp):
    """
    Memory Integration: Learn from user interactions

    curl -X POST http://localhost:8002/summary_personal_memory \
      -H "Content-Type: application/json" \
      -d '{
        "workspace_id": "task_workspace",
        "trajectories": [
          {"messages": [
            {"role": "user", "content": "I like to drink coffee while working in the morning"},
            {"role": "assistant", "content": "I understand, you prefer to start your workday with coffee to stay energized"}
          ]}
        ]
      }'
    """
    result = await app.async_execute(
        name="summary_personal_memory",
        workspace_id="task_workspace",
        trajectories=[
            {
                "messages": [
                    {"role": "user", "content": "I like to drink coffee while working in the morning"},
                    {
                        "role": "assistant",
                        "content": "I understand, you prefer to start your workday with coffee to stay energized",
                    },
                ],
            },
        ],
    )
    print("Summary Personal Memory Result:")
    print(result["answer"])


async def retrieve_personal_memory(app: ReMeApp):
    """
    Memory Retrieval: Get personal memory fragments

    curl -X POST http://localhost:8002/retrieve_personal_memory \
      -H "Content-Type: application/json" \
      -d '{
        "workspace_id": "task_workspace",
        "query": "What are the users work habits?",
        "top_k": 5
      }'
    """
    result = await app.async_execute(
        name="retrieve_personal_memory",
        workspace_id="task_workspace",
        query="What are the user's work habits?",
        top_k=5,
    )
    print("Retrieve Personal Memory Result:")
    print(result["answer"])


# ============================================
# Tool Memory Management Examples
# ============================================


async def add_tool_call_result(app: ReMeApp):
    """
    Record tool execution results

    curl -X POST http://localhost:8002/add_tool_call_result \
      -H "Content-Type: application/json" \
      -d '{
        "workspace_id": "tool_workspace",
        "tool_call_results": [
          {
            "create_time": "2025-10-21 10:30:00",
            "tool_name": "web_search",
            "input": {"query": "Python asyncio tutorial", "max_results": 10},
            "output": "Found 10 relevant results...",
            "token_cost": 150,
            "success": true,
            "time_cost": 2.3
          }
        ]
      }'
    """
    result = await app.async_execute(
        name="add_tool_call_result",
        workspace_id="tool_workspace",
        tool_call_results=[
            {
                "create_time": "2025-10-21 10:30:00",
                "tool_name": "web_search",
                "input": {"query": "Python asyncio tutorial", "max_results": 10},
                "output": "Found 10 relevant results...",
                "token_cost": 150,
                "success": True,
                "time_cost": 2.3,
            },
        ],
    )
    print("Add Tool Call Result:")
    print(result["answer"])


async def summary_tool_memory(app: ReMeApp):
    """
    Generate usage guidelines from history

    curl -X POST http://localhost:8002/summary_tool_memory \
      -H "Content-Type: application/json" \
      -d '{
        "workspace_id": "tool_workspace",
        "tool_names": "web_search"
      }'
    """
    result = await app.async_execute(
        name="summary_tool_memory",
        workspace_id="tool_workspace",
        tool_names="web_search",
    )
    print("Summary Tool Memory Result:")
    print(result["answer"])


async def retrieve_tool_memory(app: ReMeApp):
    """
    Retrieve tool guidelines before use

    curl -X POST http://localhost:8002/retrieve_tool_memory \
      -H "Content-Type: application/json" \
      -d '{
        "workspace_id": "tool_workspace",
        "tool_names": "web_search"
      }'
    """
    result = await app.async_execute(
        name="retrieve_tool_memory",
        workspace_id="tool_workspace",
        tool_names="web_search",
    )
    print("Retrieve Tool Memory Result:")
    print(result["answer"])


# ============================================
# Vector Store Management Example
# ============================================


async def load_vector_store(app: ReMeApp):
    """
    Load pre-built memories

    curl -X POST http://localhost:8002/vector_store \
      -H "Content-Type: application/json" \
      -d '{
        "workspace_id": "appworld",
        "action": "load",
        "path": "./docs/library/"
      }'
    """
    result = await app.async_execute(
        name="vector_store",
        workspace_id="appworld",
        action="load",
        path="./docs/library/",
    )
    print("Load Vector Store Result:")
    print(result["answer"])


# ============================================
# Main Execution
# ============================================


async def main():
    """Run all examples"""
    async with ReMeApp(
        "llm.default.model_name=qwen3-30b-a3b-thinking-2507",
        "embedding_model.default.model_name=text-embedding-v4",
        "vector_store.default.backend=memory",
    ) as app:
        print("=" * 60)
        print("Task Memory Examples")
        print("=" * 60)
        await summary_task_memory(app)
        print("\n")
        await retrieve_task_memory(app)

        print("\n" + "=" * 60)
        print("Personal Memory Examples")
        print("=" * 60)
        await summary_personal_memory(app)
        print("\n")
        await retrieve_personal_memory(app)

        print("\n" + "=" * 60)
        print("Tool Memory Examples")
        print("=" * 60)
        await add_tool_call_result(app)
        print("\n")
        await summary_tool_memory(app)
        print("\n")
        await retrieve_tool_memory(app)


if __name__ == "__main__":
    asyncio.run(main())
