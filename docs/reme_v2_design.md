# ReMeV2 design
一些问题：
1. 只focus在python import的方案？
2. 旧的workflow & agentic 方案都按照这个接口？
3. 是否需要异步？单独构建，不在本期迭代中

保留user_id的设计

## Long Term Memory Basic Usage

```python
import os
from reme_ai import ReMe

# os.environ["OPENAI_API_KEY"] = "sk-..."
# os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

os.environ["REME_LLM_API_KEY"] = "sk-..."
os.environ["REME_LLM_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
os.environ["REME_EMBEDDING_API_KEY"] = "sk-..."
os.environ["REME_EMBEDDING_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

memory = ReMe(
    memory_space="remy",  # workspace
    llm={"backend": "openai", "model": "qwen-plus", "temperature": 0.6, },
    embedding={"backend": "openai", "model": "text-embedding-v4", "dimension": 1024},
    vector_store={"backend": "local_file"},   # 支持的其他vector_store包括xxxx
)

# memory.update/delete/list/  <=>

result = await memory.summary(
    messages=[
        {"role": "user", "content": "I'm travelling to SF"},
        {"role": "assistant", "content": "That's great to hear!"}
    ],
    user_id="Alice",  # user_id 
    # memory_type="auto" # 默认是auto
    # **kwargs  所有参数都放到这里
)

memories = await memory.retrieve(
    query="what is your travel plan?",  # or messages
    limit=3,
    user_id="Alice",
    # memory_type="auto"   # 默认是auto
)
memories_str = "\n".join(f"- {m['memory']}" for m in memories["results"])
print(memories_str)
```

## Long Term Memory Cli Chat / APP体验（都保留）
```python
import os

from reme_ai import ReMe
from openai import OpenAI

os.environ["REME_LLM_API_KEY"] = "sk-..."
os.environ["REME_LLM_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
os.environ["REME_EMBEDDING_API_KEY"] = "sk-..."
os.environ["REME_EMBEDDING_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

memory = ReMe(
    memory_space="remy",  # workspace
    llm={"backend": "openai", "model": "qwen-plus", "temperature": 0.6, },
    embedding={"backend": "openai", "model": "text-embedding-v4", "dimension": 1024},
    vector_store={"backend": "local_file"},   # 支持的其他vector_store包括xxxx
)

openai_client = OpenAI()

os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

def chat_with_memories(query: str, history_messages: list[dict], user_name: str = "", start_summary_size: int = 2, keep_size: int = 0) -> str:
    memories = memory.retrieve(query=query, user_id=user_name, limit=3)
    system_prompt = f"You are a helpful AI named `Remy`. Use the user memories to answer the question. If you don't know the answer, just say you don't know. Don't try to make up an answer. Answer the question based on query and memories.\n"
    if memories:
        memories_str = "\n".join(f"- {m['memory']}" for m in memories["results"])
        system_prompt += f"User Memories:\n{memories_str}\n"
    
    system_message = {"role": "system", "content": system_prompt}
    history_messages.append({"role": "user", "content": query})
    response = openai_client.chat.completions.create(model="qwen-plus", messages=[system_message] + history_messages)
    history_messages.append({"role": "assistant", "content": response.choices[0].message.content})
    
    if history_messages and len(history_messages) >= start_summary_size:
        memory.summary(history_messages[:-keep_size], user_id=user_name)
        print("current memories: " + memory.list_memories(user_id=user_name))
        history_messages = history_messages[-keep_size:]
        
    return history_messages[-1]["content"]

def main():
    user_name = input("user_name: ").strip()
    print("Chat with Remy (type 'exit' to quit)")
    
    messages = []
    while True:
        user_input = input(f"{user_name}: ").strip()
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        print(f"Remy: {chat_with_memories(user_input, messages, user_name)}")
        
    memory.delete_all_memories(user_id=user_name)
    print("All memories deleted")
    
if __name__ == "__main__":
    main()  
```

## Long Term Memory Advance Usage
```python
import os

from reme_ai import ReMe
from reme_ai.retriever import FlowRetriever, AgenticRetriever
from reme_ai.summarizer import FlowSummarizer, AgenticSummarizer
from reme_ai.ops import AOp, BOP, COP
from reme_ai.tools import ATool, BTool, CTool

os.environ["REME_LLM_API_KEY"] = "sk-..."
os.environ["REME_LLM_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
os.environ["REME_EMBEDDING_API_KEY"] = "sk-..."
os.environ["REME_EMBEDDING_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

memory = ReMe(
    memory_space="remy",  # workspace
    llm={"backend": "openai", "model": "qwen-plus", "temperature": 0.6, },
    embedding={"backend": "openai", "model": "text-embedding-v4", "dimension": 1024},
    vector_store={"backend": "local_file"},   # 支持的其他vector_store包括xxxx
    use_agentic_mode=True,
)

# 只暴露agentic方式进行构建，或者就不暴露接口
memory.set_retriever(AgenticRetriever(tools=[ATool(), BTool(), CTool]), system_prompt="xxxx")  # 后面应该会把Op去掉
memory.set_summarizer(AgenticSummarizer(tools=[ATool(), BTool(), CTool()]))

# memory.set_retriever(Pipeline(ops=[Router1Op(), Router2Op()]))
# memory.set_retriever(Router1Op() >> Router2Op())
# memory.set_summarizer(FlowSummarizer(AOp(top_k=5) >> BOP() >> COP()))

result = memory.summary(
    desc="user(Alice) & AI的对话",
    messages=[
        {"role": "user", "content": "I'm travelling to SF"},
        {"role": "assistant", "content": "That's great to hear!"}
    ],
    memory_type="auto", # auto, personal, procedural, tool
    **kwargs  # top_k 表格有哪一些参数是可以修改的，包括memory_type
)

memories = memory.retrieve(
    query="what is your travel plan?",
    limit=3,
    memory_type="auto", # auto, personal, procedural, tool
    **kwargs,
)
memories_str = "\n".join(f"- {m['memory']}" for m in memories["results"])
print(memories_str)
```


## Short Term Memory Basic Usage
```python
import os
from reme_ai import ReMe

os.environ["REME_LLM_API_KEY"] = "sk-..."
os.environ["REME_LLM_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
os.environ["REME_EMBEDDING_API_KEY"] = "sk-..."
os.environ["REME_EMBEDDING_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

memory = ReMe(
    memory_space="remy",  # workspace
    llm={"backend": "openai", "model": "qwen-plus", "temperature": 0.6, },
    embedding={"backend": "openai", "model": "text-embedding-v4", "dimension": 1024},
    vector_store={"backend": "local_file"},   # 支持的其他vector_store包括xxxx
)

result = memory.offload_context(
    messages=[
        {"role": "user", "content": "I'm travelling to SF"},
        {"role": "assistant", "content": "That's great to hear!"}
    ],
    **kwargs
)

# agentic
memories = memory.reload_context(
    query="what is your travel plan?",
    limit=3,
    **kwargs
)
memories_str = "\n".join(f"- {m['memory']}" for m in memories["results"])
print(memories_str)
```


## Short Term Memory with ReactAgent
集成agentscope langchain

## Long Term Memory with ReactAgent
集成agentscope langchain

装饰器直接用

1. retrieve / reload 是否合并？

1. todo 对比5家，出参，入参，竞品对比的逻辑


2. session调研新的接口，带session id的设计