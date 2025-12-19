import asyncio

from reme_ai import ReMeApp
from reme_ai.core.agent.personal import SimpleSummaryAgentOp, SimpleRetrieveAgentOp
from reme_ai.core.tool import AddMemoryOp, DeleteMemoryOp, ReadHistoryMemoryOp, UpdateMemoryOp, RetrieveMemoryOp


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
            RetrieveMemoryOp(top_k=20),
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
            RetrieveMemoryOp(enable_summary_memory=True, top_k=10),
            ReadHistoryMemoryOp(),
        ]
        await retrieve_agent.async_call(
            query="What does Alice like?",
            workspace_id="test_workspace",
            memory_target="Alice",
        )
        print(f"Retrieve result: {retrieve_agent.output}")


if __name__ == "__main__":
    asyncio.run(test_summary_and_retrieve())


"""
## summary memory
load_meta_skills()
summary_context(thinking, summary)
hand_off_agent(thinking, type, target)

add_memory()
delete_memory()
update_memory()
retrieve_memory()

update_tool_memory()
update_identity_memory(memory_content)

## retrieve memory
load_meta_memory() << read_identity_memory()
retrieve_personal_memory(think, target, query)
retrieve_procedural_memory(think, target, query)
retrieve_tool_memory(think, target, tool_name)
read_history()

# summary agent
meta-summarizer
identity_summary_agent
personal_summary_agent
procedural_summary_agent
tool_summary_agent

# retrieve agent
meta-retrieve
llm_agent


# summary working memory
compact
compress
auto

# reload working memory
grep_content(query, -5, 5, 0, 10)


## retrieve skills
load_meta_skills()
load_skills()
load_reference_skills()
execute_shell(skill_name, parameters)



meta-summarizer << [
    summary_context(thinking, summary)
    hand_off_agent(thinking, type, target)
]

hand_off_agent << [
    identity_summary_agent << [update_identity_memory]
    personal_summary_agent << [add_memory, delete_memory, update_memory, retrieve_memory]
    procedural_summary_agent << [add_memory, delete_memory, update_memory, retrieve_memory]
    tool_summary_agent << [update_tool_memory]
]

meta-retrieve << [
    retrieve_personal_memory(think, target, query)
    retrieve_procedural_memory(think, target, query)
    retrieve_tool_memory(think, target, tool_name)
    retrieve_history(think, id)
]

llm_agent << [
    read_identity_memory() 主动读取
    ...
]



1. memory: layers
2. op: agent + tools
2. retrieve: skills渐进式agentic
    retriever << [
        load_meta_memory(),
            ```
            格式："- <memory_type>(<memory_target>): <description>"
            personal jinli xxxxx
            personal jiaji xxxxx
            personal jinli&jiaji xxxxx
            procedural appworld xxxxx
            procedural bfcl-v3 xxxxx
            tool tool_guidelines  xxxxx
            identity self   xxxxx
            ```
        RetrieveMemory(list(memory_type, memory_target, query))), layer1+layer2
        ReadHistory(ref_memory_id), layer3
    ]


personal ref_memory_id  -> history


3. summary:
    messages -> add_history_memory -> db (layer1 memory)
    meta-summarizer << [
        load_meta_memory(),
        
        add_meta_memory(list(memory_type, memory_target)),
        
        add_summary_memory(summary_memory),  -> layer2 memory
        
        hands_off_agent(list(memory_type, memory_target)) ,
    ]
    personal_summary_agent << [add_memory, update_memory, delete_memory, VectorRetrieveMemoryOp(only layer1)]
    procedural_summary_agent << [add_memory, update_memory, delete_memory, VectorRetrieveMemoryOp(only layer1)]  # todo
    tool_summary_agent << [add_memory, update_memory, VectorRetrieveMemoryOp(only layer1)]
    identity_summary_agent << [read_identity_memory, update_identity_memory]

5. thinking实验

6. multi模式实验

7. dialog_agent / hooking


kit/system
what is reme
key diff
develop
file system


experimental / v1 / v2 ?
"""
