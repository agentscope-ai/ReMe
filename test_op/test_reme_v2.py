import asyncio

from reme_ai import ReMeApp
from reme_ai.core.agent import SimpleSummaryAgentOp, SimpleRetrieveAgentOp
from reme_ai.core.tool import AddMemoryOp, DeleteMemoryOp, ReadHistoryOp, UpdateMemoryOp, VectorRetrieveMemoryOp


async def test_summary_and_retrieve():
    """Test summary agent first, then retrieve agent."""


    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "My name is Alice and I love programming in Python."
        },
        {
            "role": "assistant",
            "content": "Nice to meet you, Alice! Python is a great programming language."
        },
        {
            "role": "user",
            "content": "I also enjoy machine learning and deep learning."
        },
        {
            "role": "assistant",
            "content": "That's wonderful! Machine learning and deep learning are fascinating fields."
        },
    ]

    async with ReMeApp(
            "llm.default.model_name=qwen3-max",
            "vector_store.default.backend=local",
    ) as _:
        # app.service_config.llm["default"].model_name = "qwen3-max"
        # app.service_config.vector_store["default"].backend = "local"

        # Step 1: Summary agent to summarize the conversation
        summary_agent = SimpleSummaryAgentOp() << [
            AddMemoryOp(),
            DeleteMemoryOp(),
            UpdateMemoryOp(),
            VectorRetrieveMemoryOp(top_k=20),
        ]
        await summary_agent.async_call(
            messages=messages,
            workspace_id="test_workspace",
            memory_target="Alice",
            ref_memory_id="test_ref_001",
        )
        print(f"Summary result: {summary_agent.output}")

        # Step 2: Retrieve agent to retrieve relevant memories
        retrieve_agent = SimpleRetrieveAgentOp() << [
            VectorRetrieveMemoryOp(enable_summary_memory=True, top_k=10),
            ReadHistoryOp(),
        ]
        await retrieve_agent.async_call(
            query="What does Alice like?",
            workspace_id="test_workspace",
            memory_target="Alice",
        )
        print(f"Retrieve result: {retrieve_agent.output}")


if __name__ == "__main__":
    asyncio.run(test_summary_and_retrieve())
