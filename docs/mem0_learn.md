# ReMeV2的设计文档

## 竞品调研（开源框架+商业化Memory+论文）
### mem0

开源方案 vs 闭源方案
1. from mem0.client.main import AsyncMemoryClient, MemoryClient
2. from mem0.memory.main import AsyncMemory, Memory
https://docs.mem0.ai/platform/platform-vs-oss

问题：之前对比的是闭源方案？

memory分类【短期记忆与长期记忆】：
https://docs.mem0.ai/core-concepts/memory-types
短期记忆使当前的对话保持连贯性。它包括：
- Conversation history – recent turns in order so the agent remembers what was just said.
- Working memory – temporary state such as tool outputs or intermediate calculations.
- Attention context – the immediate focus of the assistant, similar to what a person holds in mind mid-sentence.
长期记忆能够将知识在不同阶段间保存下来。它记录了：
- Factual memory – user preferences, account details, and domain facts.
- Episodic memory – summaries of past interactions or completed tasks.
- Semantic memory – relationships between concepts so agents can reason about them later.
``` ?
class MemoryType(Enum):
    SEMANTIC = "semantic_memory"  -> personal
    EPISODIC = "episodic_memory"   -> summary 
    PROCEDURAL = "procedural_memory"  -> procedural
```
完全没有SEMANTIC和EPISODIC的引用，只有PROCEDURAL

提供了哪些接口：
https://docs.mem0.ai/core-concepts/memory-operations/add
1. add 对应 summary workflow
2. search 对应 retrieve workflow
3. update 原子操作
4. delete 原子操作
接口层面：开源版本和闭源版本是对齐的

MCP接口，
1. add_memory	Save text or conversation history for a user/agent
2. search_memories	Semantic search across existing memories with filters 
3. get_memories	List memories with structured filters and pagination
4. get_memory	Retrieve one memory by its memory_id
5. update_memory	Overwrite a memory’s text after confirming the ID
6. delete_memory	Delete a single memory by memory_id
7. delete_all_memories	Bulk delete all memories in scope
8. delete_entities	Delete a user/agent/app/run entity and its memories
9. list_entities	Enumerate users/agents/apps/runs stored in Mem0
前4个对齐的是闭源版本的python接口，add增加了text


user_id概念
user_id="alice" <==> memory_target
agent_id/session_id可选
多个user_id # 带讨论，库隔离

### letta(MemGPT)  @weikang
TODO



## 我们相比竞品的优势
1. 算法层面：workflow -> agentic & 渐进式方案
2. 模型层面：mem-agent-rl
3. 支持用户友好的接口，只用用户二开

``` 同步方案
from reme_ai import xxx_summarizer, xxx_retriever # v2的agent
from reme_ai import xxx_tool # v2的agent
from reme_ai import xxx_op # v1的op
from reme_ai import OpenAILLM, OpenAIEMBEDDING
from reme_ai import VectorStore

def main():
    reme = ReMe(
        llm={},
        embedding={},
        vector_store={},
        retriever=Retriever(ops=Opa(top_k=5) >> OpB()]
        retriever=Retriever(tools=[ToolA(), ToolB()],
        summarizer=Summarizer(ops=Opa(top_k=5) >> OpB()],
        summarizer=Summarizer(tools=[ToolA(), ToolB()],
    )
    
    # 服务模式
    reme.serve()
    
    # 直接调用
    result = reme.retrieve(
        "food preferences",
        filters={"user_id": "alice"}, top_k=10)
        
    result = reme.summary(
        user_id="alice",
        messages=[{"role": "user", "content": "Help me create a project plan"}])

    reme.close()

if __name__ == "__main__":
    main()
```
Reme提供接口
1. 顶层：提供 summary 和 retrieve 的python使用 & http接口
2. 底层db：http接口 / mcp接口（和http保持一致）


### 引用的库希望不是外部一个小库
合并flowllm部分逻辑回来

### vector_store
1. 复用之前的，有不断开发的需求
2. 复用mem0，没有异步接口
3. 使用langchain和llama-index
   1. 需要额外封装
   2. 之前碰到es同步接口底层是es异步封装，外部要异步很复杂

增加下载量
