# ReMeV2 深度设计文档：渐进式 Agentic Memory 方案

## 一、 背景与现状分析

### 1.1 当前面临的挑战

* **外功修炼（接口易用性）**：现有的 `server-client` 模式对新手开发者不够友好，集成成本高，需要更直观、纯 Pythonic 的调用方式。
* **内功修炼（架构深度）**：受 `skills` 和 `agentic memory` 启发，现有的存储检索较为机械。我们需要一种基于**渐进式检索（Progressive Retrieval）**与**渐进式总结（Progressive Summarization）**的智能体记忆方案。

### 1.2 核心目标

1. **极简开发体验**：开发者友好，全异步接口，支持本地直接运行与 CLI 体验。
2. **认知架构升级**：引入 渐进式检索 & 渐进式总结 的 Agentic 模式，融合多种记忆，让记忆的存取具备“思考”过程。
3. **生态融合**：原生支持 AgentScope、LangChain 等主流框架。

---

## 二、 竞品调研与启示

### 2.1 主流竞品深度对比

| 产品          | 设计哲学     | 核心优势                                        | 局限性               |
|-------------|----------|---------------------------------------------|-------------------|
| **mem0**    | 智能便签本    | 原子事实提取，极高 Token 效率。                         | 缺乏对复杂逻辑链条的支持。     |
| **Letta**   | 带硬盘的 CPU | 模拟计算机三级存储（Core/Recall/Archival），Agent 自主控存。 | 状态机管理相对复杂。        |
| **MIRIX**   | 认知架构图谱   | 实体-关系双引擎，支持记忆“进化”与“固化”。                     | 侧重研究，落地集成门槛较高。    |
| **LangMem** | 用户档案系统   | 异步 Compaction（压缩），Schema 驱动，强一致性。           | 偏向 SaaS 应用，灵活性略逊。 |

### 2.2 mem0
- https://github.com/mem0ai/mem0
- https://docs.mem0.ai/core-concepts/memory-operations/add
- https://docs.mem0.ai/core-concepts/memory-operations/search
- https://docs.mem0.ai/core-concepts/memory-operations/update
- https://docs.mem0.ai/core-concepts/memory-operations/delete

#### 2.2.1 API Reference
| 接口名称 | 核心输入参数 (Inputs) | 核心输出 (Outputs) | 背后逻辑 (Internal Logic) |
| --- | --- | --- | --- |
| **Add** | `messages` (文本/对话), `user_id`, `metadata` | `id`, `event` (ADD/UPDATE), `data` | **提取与合并**：LLM 提取事实，自动去重并更新已有记忆，而非简单堆叠。 |
| **Search** | `query` (自然语言), `filters`, `limit` | `id`, `memory` (事实文本), `score`, `metadata` | **语义检索**：基于向量相似度查找最相关的“原子事实”，支持多维过滤。 |
| **Update** | `memory_id` (必填), `data` (新内容) | 操作状态 (Success/Fail) | **手动干预**：允许开发者对特定的事实进行精确修正。 |
| **Delete** | `memory_id` 或 `user_id` (清空) | 操作状态 (Success/Fail) | **遗忘机制**：物理删除或逻辑移除不再需要的信息。 |

#### 2.2.2 Tech Strategy & Benefits
| 维度 | 技术方案 (Technical Solution) | 核心优势 (Key Advantages) |
| --- | --- | --- |
| **存储架构** | **混合存储**：向量数据库 (Vector) + 图数据库 (Graph) + 关系型元数据。 | **多维关联**：不仅能搜到相似内容，还能理解实体间的逻辑关系（如“父子”、“因果”）。 |
| **数据处理** | **原子化事实提取**：利用 LLM 将长篇对话压缩为简短的 Fact。 | **极高 Token 效率**：注入 Prompt 的内容更精炼，减少 90% 以上的冗余信息，大幅降本。 |
| **管理层级** | **多级联动**：User (长期)  Agent (专业)  Session (短期)。 | **个性化定制**：实现跨会话的“长效记忆”，AI 能记住用户一个月前说过的偏好。 |
| **冲突处理** | **自适应更新算法**：新信息进入时自动比对旧记忆。 | **数据一致性**：自动处理矛盾信息（如用户更换了住址），确保记忆库始终是“最新真理”。 |
| **兼容性** | **解耦设计**：支持多种 Embedding 模型与向量数据库后端。 | **快速集成**：几行代码即可为现有 LLM 应用增加记忆层，适配各种生产环境。 |


---

### 2.3 Letta
- https://github.com/letta-ai/letta
- https://docs.letta.com/guides/agents/archival-memory/
- https://docs.letta.com/guides/agents/archival-search/

#### 2.3.1 存储架构层级 (Memory Tiering)

Letta 将记忆分为三个物理/逻辑层，模拟计算机的存储架构：

| 记忆层级 | 存储介质 | 访问方式 | 核心作用 |
| --- | --- | --- | --- |
| **Core Memory** | **上下文窗口 (Prompt)** | 直接读写 | **即时意识**：包含 `Persona`（AI 设定）和 `Human`（用户信息）。Agent 随时可见，响应最快。 |
| **Recall Memory** | **关系型数据库 (SQL)** | 分页检索 | **短期/历史回顾**：存储完整的对话流（Messages）。用于回答“你刚才说了什么”。 |
| **Archival Memory** | **向量数据库 (Vector)** | 语义搜索 | **长期知识库**：存储海量事实或文档。Agent 通过工具自主检索或存入。 |

#### 2.3.2 核心操作接口 (API & Tool Reference)

在 Letta 中，记忆的操作通常封装为 **Tools**，由 Agent 根据推理需求主动调用。

| 接口/工具名称 | 输入参数 (Inputs) | 核心输出 (Outputs) | 背后逻辑 (Internal Logic) |
| --- | --- | --- | --- |
| **`core_memory_update`** | `section`, `new_content` | 更新后的段落内容 | **原子替换**：直接修改 System Prompt 中的特定块（如：更新用户的职业或 AI 的性格偏好）。 |
| **`archival_memory_insert`** | `content` (字符串) | 写入状态/ID | **知识沉淀**：将当前对话中的重要信息或外部文件片段“持久化”到向量数据库。 |
| **`archival_memory_search`** | `query`, `page` | 匹配的文本块列表 | **主动 RAG**：Agent 意识到知识不足时，自主发起向量检索，并将结果拉入临时上下文。 |
| **`conversation_search`** | `query`, `start_date` | 历史消息记录 | **全文检索**：在 Recall Memory 中根据关键词或时间戳查找历史对话详情。 |
| **`send_message`** | `message`, `agent_id` | 响应流/状态更新 | **状态循环**：这是主入口，触发 Agent 的“思考-行动-观察”循环，自动处理内存同步。 |

#### 2.3.3 技术策略与核心优势 (Tech Strategy & Benefits)

| 维度 | 技术方案 (Technical Solution) | 核心优势 (Key Advantages) |
| --- | --- | --- |
| **状态持久化** | **Agent State Snapshot**：将 Agent 的所有内存、工具定义和历史记录打包存入数据库。 | **无限存续**：Agent 不再是无状态的 API 调用。重启服务器后，Agent 依然记得所有细节。 |
| **自主演进** | **Self-Editing Loop**：Agent 拥有修改自己 Core Memory 的权限（通过函数调用）。 | **认知闭环**：AI 能在交流中发现矛盾并自我更正，例如发现用户搬家后自动更新 `Human` 模块。 |
| **算力调度** | **OOC (Out-of-Context) 管理**：当对话过长，系统自动将旧消息从 Core 移入 Recall。 | **突破 Context 限制**：在 8k 窗口的模型上也能处理相当于 1M 窗口的逻辑量，且成本更低。 |
| **多代理协同** | **Letta Server 中控**：统一管理多个 Agent 的状态机与资源访问权限。 | **企业级扩展**：支持创建 Agent 团队，每个 Agent 拥有独立的记忆空间但可共享 Archival 库。 |
| **解耦灵活性** | **Provider Agnostic**：后端支持 Postgres/Chroma，前端支持 OpenAI/Anthropic/Local LLMs。 | **无缝迁移**：不绑定特定模型，开发者可以根据成本或能力随时更换底座。 |

#### 2.3.4 与 mem0 的深度对比

* **设计哲学**：
* **mem0** 像是一个**“智能记事本”**，它在后台默默地帮你总结事实。
* **Letta** 像是一个**“带硬盘的 CPU”**，它把记忆管理完全交给了 Agent 自己的逻辑推理。


* **交互模式**：
* **mem0** 通常是外部干预（Add/Search）。
* **Letta** 强调 **Agentic Control**（Agent 意识到需要搜索时才去搜索），这种模式更接近人类的思维过程。

---

### 2.4 MIRIX
- https://github.com/Mirix-AI/MIRIX
- https://docs.mirix.io/

#### 2.4.1 API Reference

| 接口名称 | 核心输入参数 (Inputs) | 核心输出 (Outputs) | 背后逻辑 (Internal Logic) |
| --- | --- | --- | --- |
| **Add** | `content` (观察/对话), `agent_id`, `context_type` (如任务/闲聊) | `memory_id`, `graph_nodes`, `status` | **实体建模**：不只是提取事实，而是将信息拆解为实体（Entities）与关系（Relations），并挂载到智能体的知识图谱中。 |
| **Query** | `query` (意图), `scope` (全局/局部), `top_k` | `retrieved_memories`, `relation_paths`, `score` | **混合检索**：结合向量（Vector）的语义相关性和图（Graph）的拓扑连接性，寻找具有逻辑深度背景的记忆。 |
| **Evolve** | `target_memories` (可选), `agent_id` | `optimized_structure`, `merged_nodes` | **记忆固化/压缩**：模仿人类大脑的“睡眠”机制，自动合并碎片化记忆，将短期经验转化为长期的结构化知识。 |
| **Observe** | `interaction_stream`, `feedback` | `insights`, `priority_update` | **实时学习**：根据用户反馈或环境变化，动态调整记忆的权重（Importance）和置信度。 |

#### 2.4.2 Tech Strategy & Benefits

| 维度 | 技术方案 (Technical Solution) | 核心优势 (Key Advantages) |
| --- | --- | --- |
| **存储架构** | **语义-关系双引擎**：向量索引（Vector Index）+ 属性图（Property Graph）。 | **深度上下文**：不仅知道“是什么”，还能通过图路径推理出“为什么”，有效解决 LLM 幻觉问题。 |
| **记忆层级** | **三层架构**：感知记忆 (Perception) -> 语义记忆 (Semantic) -> 经验记忆 (Episodic)。 | **任务适应性**：不同任务自动匹配不同的记忆深度，短期任务关注细节，长期任务关注模式。 |
| **演化机制** | **自主固化 (Self-Consolidation)**：通过 LLM 定期对冗余、矛盾信息进行清洗和逻辑抽象。 | **永久生命力**：解决随时间推移记忆库膨胀导致的检索噪声，确保记忆库“越用越聪明”。 |
| **推理增强** | **基于记忆的 RAG+**：在检索到的事实基础上，额外提供关联的逻辑链条（Logic Chains）。 | **辅助决策**：为 Agent 提供决策支撑，使其在处理复杂流程时具备类似“长期经验值”的直觉。 |
| **多代理协同** | **内存共享协议**：支持 Agent 之间的记忆交换与知识同步。 | **群体智能**：多个 Agent 可以共享同一套底层知识体系，同时保留各自的私有工作记忆。 |

#### 2.4.3 与 mem0 的主要区别

* **Mem0** 侧重于**个性化偏好存储**（Personalization），核心是记住“用户喜欢什么”。
* **MIRIX** 侧重于**智能体认知架构**（Agent Cognition），核心是让 Agent 具备类似人类的“知识归纳”和“逻辑推理”记忆能力。

---

### 2.5 LangMem
- https://github.com/langchain-ai/langmem
- https://langchain-ai.github.io/langmem/

#### 2.5.1 API Reference

| 接口名称 | 核心输入参数 (Inputs) | 核心输出 (Outputs) | 背后逻辑 (Internal Logic) |
| --- | --- | --- | --- |
| **Add Messages** | `thread_id`, `messages` (List), `user_id` | 操作确认 / 任务 ID | **流式注入**：将原始对话追加到指定的 Thread。LangMem 会自动关联用户上下文，准备进行后续的异步处理。 |
| **Query Memory** | `user_id`, `query` (语义描述), `namespace` | 结构化记忆对象 (JSON / Text) | **多维检索**：不仅支持向量相似度搜索，还能根据定义的 Schema 返回结构化的用户画像或知识状态。 |
| **Trigger Logic** | `thread_id`, `memory_type` | 更新后的 Memory State | **异步固化**：后台启动 LLM 任务，将长篇对话“压缩”并“提取”到长期存储中。支持自定义提取逻辑（如更新用户信息）。 |
| **Manage State** | `user_id`, `patch_data` (增量更新) | 成功/失败 状态 | **精确受控**：开发者可以直接修改持久化的状态（State），支持类似于 Git 的状态管理。 |

#### 2.5.2 Tech Strategy & Benefits

| 维度 | 技术方案 (Technical Solution) | 核心优势 (Key Advantages) |
| --- | --- | --- |
| **存储架构** | **Stateful Persistence**：基于关系型数据库 (Postgres) + 向量索引。 | **强一致性**：利用数据库事务确保记忆更新的可靠性，支持复杂的结构化查询与过滤。 |
| **数据处理** | **异步化 Compaction (压缩)**：在对话间隙通过后台 Worker 提取知识。 | **无感延迟**：核心对话流程不被记忆提取阻塞，通过定时或事件驱动完成“记忆固化”，优化用户体验。 |
| **管理层级** | **Thread -> User -> Organization**：三层级联记忆。 | **上下文隔离**：完美适配 SaaS 应用场景，既能记住单次对话（Thread），也能沉淀用户习惯（User）。 |
| **逻辑引擎** | **Schema-Driven (模式驱动)**：允许定义 JSON Schema 来规范记忆内容。 | **高度可预测**：输出不再是散乱的句子，而是结构化的字段，方便下游程序直接调用逻辑（如自动填充表单）。 |
| **集成生态** | **LangGraph 原生集成**：作为 Checkpointer 或存储节点直接接入。 | **生态协同**：如果你已经在用 LangChain，LangMem 可以无缝接管状态流转，无需重写底层存储逻辑。 |

#### 2.5.3 与 mem0 的核心差异

* **mem0** 像是一个**“便签本”**：它擅长从每一句话里抠出零散的事实（如“我喜欢吃苹果”），然后把它们存成一条条语义片段。
* **LangMem** 像是一个**“用户档案系统”**：它更擅长分析一整段对话，然后更新一个复杂的 JSON 档案（如更新用户的偏好模型、性格标签、历史任务状态）。

---

## 三、 ReMeV2 API 接口设计

### 3.1 Long-Term Memory (长期记忆)

#### 3.1.1 Basic Usage (基础用法)

The most straightforward way to use ReMe for long-term memory management. Supports basic summary and retrieval operations.

```python
import os
from reme_ai import ReMe

os.environ["REME_LLM_API_KEY"] = "sk-..."
os.environ["REME_LLM_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
os.environ["REME_EMBEDDING_API_KEY"] = "sk-..."
os.environ["REME_EMBEDDING_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

memory = ReMe(
    memory_space="remy",  # workspace identifier
    llm={"backend": "openai", "model": "qwen-plus", "temperature": 0.6},
    embedding={"backend": "openai", "model": "text-embedding-v4", "dimension": 1024},
    vector_store={"backend": "local_file"},  # supported: local_file, chromadb, qdrant, etc.
)

# Summarize conversation into memory
result = await memory.summary(
    messages=[
        {"role": "user", "content": "I'm travelling to SF"},
        {"role": "assistant", "content": "That's great to hear!"}
    ],
    user_id="Alice",
    # memory_type="auto"  # default: auto (auto, personal, procedural, tool)
)

# Retrieve relevant memories
memories = await memory.retrieve(
    query="what is your travel plan?",
    limit=3,
    user_id="Alice",
    # memory_type="auto"  # default: auto
)
memories_str = "\n".join(f"- {m['memory']}" for m in memories["results"])
print(memories_str)
```

#### 3.1.2 CLI Chat Application (命令行聊天应用)

A complete example demonstrating how to build a memory-enhanced chatbot with CLI interface.

```python
import os
from reme_ai import ReMe
from openai import OpenAI

os.environ["REME_LLM_API_KEY"] = "sk-..."
os.environ["REME_LLM_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
os.environ["REME_EMBEDDING_API_KEY"] = "sk-..."
os.environ["REME_EMBEDDING_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

memory = ReMe(
    memory_space="remy",
    llm={"backend": "openai", "model": "qwen-plus", "temperature": 0.6},
    embedding={"backend": "openai", "model": "text-embedding-v4", "dimension": 1024},
    vector_store={"backend": "local_file"},
)

os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
openai_client = OpenAI()

def chat_with_memories(
    query: str,
    history_messages: list[dict],
    user_name: str = "",
    start_summary_size: int = 2,
    keep_size: int = 0
) -> str:
    # Retrieve relevant memories for the query
    memories = memory.retrieve(query=query, user_id=user_name, limit=3)

    # Build system prompt with memories
    system_prompt = (
        "You are a helpful AI named `Remy`. Use the user memories to answer the question. "
        "If you don't know the answer, just say you don't know. Don't try to make up an answer.\n"
    )
    if memories:
        memories_str = "\n".join(f"- {m['memory']}" for m in memories["results"])
        system_prompt += f"User Memories:\n{memories_str}\n"

    # Generate response
    system_message = {"role": "system", "content": system_prompt}
    history_messages.append({"role": "user", "content": query})
    response = openai_client.chat.completions.create(
        model="qwen-plus",
        messages=[system_message] + history_messages
    )
    history_messages.append({"role": "assistant", "content": response.choices[0].message.content})

    # Summarize history when it gets too long
    if len(history_messages) >= start_summary_size:
        memory.summary(history_messages[:-keep_size], user_id=user_name)
        print("Current memories: " + memory.list_memories(user_id=user_name))
        history_messages = history_messages[-keep_size:]

    return history_messages[-1]["content"]

def main():
    user_name = input("Enter your name: ").strip()
    print("Chat with Remy (type 'exit' to quit)")

    messages = []
    while True:
        user_input = input(f"{user_name}: ").strip()
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        print(f"Remy: {chat_with_memories(user_input, messages, user_name)}")

    # Cleanup
    memory.delete_all_memories(user_id=user_name)
    print("All memories deleted")

if __name__ == "__main__":
    main()
```

#### 3.1.3 Advanced Usage (高级用法)

For advanced users who want to customize retriever and summarizer behavior with Agentic mode.

```python
import os
from reme_ai import ReMe
from reme_ai.retriever import AgenticRetriever
from reme_ai.summarizer import AgenticSummarizer
from reme_ai.tools import ATool, BTool, CTool

os.environ["REME_LLM_API_KEY"] = "sk-..."
os.environ["REME_LLM_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
os.environ["REME_EMBEDDING_API_KEY"] = "sk-..."
os.environ["REME_EMBEDDING_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

memory = ReMe(
    memory_space="remy",
    llm={"backend": "openai", "model": "qwen-plus", "temperature": 0.6},
    embedding={"backend": "openai", "model": "text-embedding-v4", "dimension": 1024},
    vector_store={"backend": "local_file"},
    use_agentic_mode=True,
)

# Customize retriever and summarizer with custom tools and prompts
memory.set_retriever(
    AgenticRetriever(tools=[ATool(), BTool(), CTool()]),
    system_prompt="Custom retrieval instructions..."
)
memory.set_summarizer(
    AgenticSummarizer(tools=[ATool(), BTool(), CTool()])
)

# Use the customized memory system
result = memory.summary(
    messages=[
        {"role": "user", "content": "I'm travelling to SF"},
        {"role": "assistant", "content": "That's great to hear!"}
    ],
    user_id="Alice",
    memory_type="auto",  # auto, personal, procedural, tool
)

memories = memory.retrieve(
    query="what is your travel plan?",
    limit=3,
    user_id="Alice",
    memory_type="auto",
)
memories_str = "\n".join(f"- {m['memory']}" for m in memories["results"])
print(memories_str)
```

### 3.2 Short-Term Memory (短期记忆)

#### 3.2.1 Basic Usage (基础用法)

Context offload/reload API for managing short-term conversational memory within a session.

```python
import os
from reme_ai import ReMe

os.environ["REME_LLM_API_KEY"] = "sk-..."
os.environ["REME_LLM_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
os.environ["REME_EMBEDDING_API_KEY"] = "sk-..."
os.environ["REME_EMBEDDING_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

memory = ReMe(
    memory_space="remy",
    llm={"backend": "openai", "model": "qwen-plus", "temperature": 0.6},
    embedding={"backend": "openai", "model": "text-embedding-v4", "dimension": 1024},
    vector_store={"backend": "local_file"},
)

# Offload context when conversation gets too long
result = memory.offload_context(
    messages=[
        {"role": "user", "content": "I'm travelling to SF"},
        {"role": "assistant", "content": "That's great to hear!"}
    ],
)

# Reload relevant context when needed
memories = memory.reload_context(
    query="what is your travel plan?",
    limit=3,
)
memories_str = "\n".join(f"- {m['memory']}" for m in memories["results"])
print(memories_str)
```

### 3.3 Framework Integration (框架集成)

#### 3.3.1 Integration with AgentScope

Integration example for AgentScope ReActAgent with long-term memory support.

```python
# TODO: Provide AgentScope integration example
```

#### 3.3.2 Integration with LangChain

Integration example for LangChain agents with ReMe memory layer.

```python
# TODO: Provide LangChain integration example
```

### 3.4 OpenAI Compatible Interface

OpenAI-compatible API interface for seamless integration with existing OpenAI-based applications.

```python
# TODO: Research and implement OpenAI-compatible interface
# - Support for threads and assistants API
# - Compatible with OpenAI SDK
# - Support for streaming responses
```



---

## 四、核心方案设计

### 4.1 设计概述

ReMeV2 采用简洁的架构设计，核心理念为：**ReMeV2 = Tool(s) + Agent(s)**

- **Tool层**：提供原子化的记忆操作能力，包括增删改查、检索、元数据管理等基础操作
- **Agent层**：基于Tool层构建的智能代理，负责复杂的记忆管理逻辑，如分类总结、渐进式检索等
- **Runtime层**：内部调度机制，协调Tool和Agent的交互流程

### 4.2 Tool层设计

Tool层提供装饰器形式的记忆操作工具，每个工具类通过 `@tool` 装饰器注册，明确定义初始化参数和调用参数。

#### 4.2.1 基类：BaseMemoryToolOp

**初始化参数：**
- `enable_multiple` (bool): Enable multi-item operation mode. Default: `True`
- `enable_thinking_params` (bool): Include thinking parameter in tool schema for model reasoning. Default: `False`
- `memory_metadata_dir` (str): Directory path for storing memory metadata. Default: `"./memory_metadata"`

#### 4.2.2 Tool操作列表

以下是所有Tool操作的完整定义，包括继承关系、初始化参数和调用参数：

| Tool类                      | 继承自              | 初始化参数（除基类外）                                                                                                      | Tool Call参数（单项模式）                                                                                               | Tool Call参数（多项模式）                                                                                                     |
|----------------------------|------------------|------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| **AddMemoryOp**            | BaseMemoryToolOp | `add_when_to_use` (bool, 默认: False)<br>`add_metadata` (bool, 默认: True)                                           | `when_to_use` (str, 可选)<br>`memory_content` (str, 必需)<br>`metadata` (dict, 可选)                                  | `memories` (array, 必需):<br>  - `when_to_use` (str, 可选)<br>  - `memory_content` (str, 必需)<br>  - `metadata` (dict, 可选) |
| **UpdateMemoryOp**         | BaseMemoryToolOp | 无                                                                                                                | `memory_id` (str, 必需)<br>`memory_content` (str, 必需)<br>`metadata` (dict, 可选)                                    | `memories` (array, 必需):<br>  - `memory_id` (str, 必需)<br>  - `memory_content` (str, 必需)<br>  - `metadata` (dict, 可选)   |
| **DeleteMemoryOp**         | BaseMemoryToolOp | 无                                                                                                                | `memory_id` (str, 必需)                                                                                           | `memory_ids` (array[str], 必需)                                                                                         |
| **VectorRetrieveMemoryOp** | BaseMemoryToolOp | `enable_summary_memory` (bool, 默认: False)<br>`add_memory_type_target` (bool, 默认: False)<br>`top_k` (int, 默认: 20) | `query` (str, 必需)<br>`memory_type` (str, 可选, 枚举: [identity, personal, procedural])<br>`memory_target` (str, 可选) | `query_items` (array, 必需):<br>  - `query` (str, 必需)<br>  - `memory_type` (str, 可选)<br>  - `memory_target` (str, 可选)   |
| **AddMetaMemoryOp**        | BaseMemoryToolOp | 无                                                                                                                | `memory_type` (str, 必需, 枚举: [personal, procedural])<br>`memory_target` (str, 必需)                                | `meta_memories` (array, 必需):<br>  - `memory_type` (str, 必需)<br>  - `memory_target` (str, 必需)                          |
| **ReadMetaMemoryOp**       | BaseMemoryToolOp | `enable_tool_memory` (bool, 默认: False)<br>`enable_identity_memory` (bool, 默认: False)                             | 无（无输入schema）                                                                                                    | N/A (enable_multiple=False)                                                                                           |
| **AddHistoryMemoryOp**     | BaseMemoryToolOp | 无                                                                                                                | `messages` (array[object], 必需)                                                                                  | N/A (enable_multiple=False)                                                                                           |
| **ReadHistoryMemoryOp**    | BaseMemoryToolOp | 无                                                                                                                | `memory_id` (str, 必需)                                                                                           | `memory_ids` (array[str], 必需)                                                                                         |
| **AddSummaryMemoryOp**     | AddMemoryOp      | 无（继承自AddMemoryOp）                                                                                                | `summary_memory` (str, 必需)<br>`metadata` (dict, 可选)                                                             | N/A (enable_multiple=False)                                                                                           |
| **ReadIdentityMemoryOp**   | BaseMemoryToolOp | 无                                                                                                                | 无（无输入schema）                                                                                                    | N/A (enable_multiple=False)                                                                                           |
| **UpdateIdentityMemoryOp** | BaseMemoryToolOp | 无                                                                                                                | `identity_memory` (str, 必需)                                                                                     | N/A (enable_multiple=False)                                                                                           |
| **ThinkToolOp**            | BaseAsyncToolOp  | `add_output_reflection` (bool, 默认: False)                                                                        | `reflection` (str, 必需)                                                                                          | N/A                                                                                                                   |
| **HandsOffOp**             | BaseMemoryToolOp | 无                                                                                                                | `memory_type` (str, 必需, 枚举: [identity, personal, procedural, tool])<br>`memory_target` (str, 必需)                | `memory_tasks` (array, 必需):<br>  - `memory_type` (str, 必需)<br>  - `memory_target` (str, 必需)                           |

### 4.3 Agent层设计

#### 4.3.1 基类：BaseMemoryAgentOp

Agent层构建在Tool层之上，封装复杂的记忆管理逻辑。每个Agent通过组合多个Tool实现特定的记忆管理任务。

#### 4.3.2 Agent操作列表

以下是所有Agent操作的完整定义，包括初始化参数、调用参数和可用工具：

| Agent类                         | 继承自               | 初始化参数（基类外）                                                                         | Tool Call参数                                                                                                                                                     | 可用工具                                                                                                                     |
|--------------------------------|-------------------|------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| **PersonalSummaryAgentV1Op**   | BaseMemoryAgentOp | None                                                                               | `workspace_id` (str, required)<br>`memory_target` (str, required)<br>`query` (str, optional)<br>`messages` (array, optional)<br>`ref_memory_id` (str, required) | add_memory<br>update_memory<br>delete_memory<br>vector_retrieve_memory                                                   |
| **ProceduralSummaryAgentV1Op** | BaseMemoryAgentOp | None                                                                               | `workspace_id` (str, required)<br>`memory_target` (str, required)<br>`query` (str, optional)<br>`messages` (array, optional)<br>`ref_memory_id` (str, required) | add_memory<br>update_memory<br>delete_memory<br>vector_retrieve_memory                                                   |
| **ToolSummaryAgentV1Op**       | BaseMemoryAgentOp | None                                                                               | `workspace_id` (str, required)<br>`memory_target` (str, required)<br>`query` (str, optional)<br>`messages` (array, optional)<br>`ref_memory_id` (str, required) | add_memory<br>update_memory<br>vector_retrieve_memory                                                                    |
| **IdentitySummaryAgentV1Op**   | BaseMemoryAgentOp | None                                                                               | `workspace_id` (str, required)<br>`query` (str, optional)<br>`messages` (array, optional)                                                                       | read_identity_memory<br>update_identity_memory                                                                           |
| **ReMeSummaryAgentV1Op**       | BaseMemoryAgentOp | `enable_tool_memory` (bool, 默认: True)<br>`enable_identity_memory` (bool, 默认: True) | `workspace_id` (str, required)<br>`query` (str, optional)<br>`messages` (array, optional)                                                                       | add_meta_memory<br>add_summary_memory<br>hands_off<br>(内部调用: add_history_memory, read_identity_memory, read_meta_memory) |
| **ReMeRetrieveAgentV1Op**      | BaseMemoryAgentOp | `enable_tool_memory` (bool, 默认: True)                                              | `workspace_id` (str, required)<br>`query` (str, optional)<br>`messages` (array, optional)                                                                       | vector_retrieve_memory<br>read_history_memory<br>(内部调用: read_meta_memory)                                                |
| **ReMyAgentV1Op**              | BaseMemoryAgentOp | `enable_tool_memory` (bool, 默认: True)<br>`enable_identity_memory` (bool, 默认: True) | `workspace_id` (str, required)<br>`query` (str, optional)<br>`messages` (array, optional)                                                                       | vector_retrieve_memory<br>read_history_memory<br>(内部调用: read_identity_memory, read_meta_memory)                          |

### 4.4 Runtime层设计（内部实现）

Runtime层负责协调Tool和Agent的调用流程，实现记忆的渐进式处理。

#### 4.4.1 渐进式总结流程（Summary）

总结流程采用分层处理策略，首先保存历史对话，读取元信息，然后由主Agent协调多个专用Agent完成分类总结。

**流程结构：**

```python
# Step 1: Save conversation history
AddHistoryMemoryOp()

# Step 2: Load meta information (memory types and targets)
ReadMetaMemoryOp()

# Step 3: Progressive summarization with delegation
ReMeSummaryAgentV1Op(tools=[
  # Add meta memory entries for new memory types/targets
  AddMetaMemoryOp(list(memory_type, memory_target)),

  # Add general summary memory as fallback
  AddSummaryMemoryOp(summary_memory),

  # Delegate to specialized summary agents
  HandsOffOp(list(memory_type, memory_target), agents=[
    PersonalSummaryAgentV1Op,      # Summarize personal memories
    ProceduralSummaryAgentV1Op,    # Summarize procedural memories
    ToolSummaryAgentV1Op,          # Summarize tool-related memories
    IdentitySummaryAgentV1Op       # Update identity memory
  ]),
])

# Specialized agents and their available tools
PersonalSummaryAgentV1Op(tools=[AddMemoryOp, UpdateMemoryOp, DeleteMemoryOp, VectorRetrieveMemoryOp])
ProceduralSummaryAgentV1Op(tools=[AddMemoryOp, UpdateMemoryOp, DeleteMemoryOp, VectorRetrieveMemoryOp])
ToolSummaryAgentV1Op(tools=[AddMemoryOp, UpdateMemoryOp, VectorRetrieveMemoryOp])
IdentitySummaryAgentV1Op(tools=[ReadIdentityMemoryOp, UpdateIdentityMemoryOp])
```

#### 4.4.2 渐进式检索流程（Retrieve）

检索流程采用三层检索策略，类似于技能系统的加载机制，逐层加载和过滤记忆。

**流程结构：**

```python
# Progressive retrieval with three layers
ReMeRetrieveAgentV1Op(tools=[
  # Layer 0: Load meta memory (all available memory types and targets)
  ReadMetaMemoryOp(),
  # Output format example:
  # - personal(jinli): Information about Jinli's personal life and preferences
  # - personal(jiaji): Information about Jiaji's background and interests
  # - personal(jinli&jiaji): Shared memories between Jinli and Jiaji
  # - procedural(appworld): Procedural knowledge for AppWorld tasks
  # - procedural(bfcl-v3): Procedural knowledge for BFCL-v3 benchmark
  # - tool(tool_guidelines): Guidelines for tool usage
  # - identity(self): Agent's self-identity information

  # Layer 1+2: Vector-based retrieval on structured memories
  VectorRetrieveMemoryOp(list(memory_type, memory_target, query)),

  # Layer 3: Load full conversation history for specific memory
  ReadHistoryMemoryOp(ref_memory_id),
])
```

**与技能系统的类比：**

```python
# Skill system hierarchy (for reference)
load_meta_skills       # Load skill metadata
load_skills            # Load skill implementations
load_reference_skills  # Load detailed skill documentation
execute_shell          # Execute actual commands
```

## 五、扩展设计与实验方向

### 5.1 Summary Memory机制

Summary Memory作为通用维度的记忆类型，提供兜底的原始对话索引能力。

**工作流程示例：**

```txt
Step 1: Progressive summarization across sessions
           session1: List[Message] -> session2: List[Message] -> session3: List[Message] -> ...
summary    ✓ (always)                 ✓ (always)                 ✓ (always)
personal   ✗                          ✗                          ✓ (when applicable)
procedural ✗                          ✓ (when applicable)        ✗

Step 2: Retrieval with fallback strategy
vector_retrieve_memory(query, memory_type="personal", memory_target="jinli")
  -> Search in memory_type: ["personal", "summary"]  # Fallback to summary if personal not found
```

**设计优势：**
1. Provides a universal dimension for memory extraction across all memory types
2. Ensures fallback indexing of original conversations when specific meta memory is not available
3. Maintains conversation context even when specialized memory extraction fails

### 5.2 Thinking参数实验

探索不同的模型推理能力增强方案，受AgentScope和Claude启发。

#### 5.2.1 Thinking参数设计

```python
async def record_to_memory(
    self,
    thinking: str,
    content: list[str],
    **kwargs: Any,
) -> ToolResponse:
    """Use this function to record important information that you may
    need later. The target content should be specific and concise, e.g.
    who, when, where, do what, why, how, etc.

    Args:
        thinking (`str`):
            Your thinking and reasoning about what to record
        content (`list[str]`):
            The content to remember, which is a list of strings.
    """
```

#### 5.2.2 实验对比方案

| 方案类型                          | 说明                                       | 灵感来源       |
|-------------------------------|--------------------------------------------|----------------|
| Thinking Model                | Native reasoning-capable models (e.g., o1) | OpenAI         |
| Instruct Model                | Standard instruction-following models       | Baseline       |
| Instruct Model + Thinking Params | Add thinking parameter to tool schema     | AgentScope     |
| Instruct Model + Thinking Tool   | Dedicated thinking tool for explicit reasoning | Claude     |

### 5.3 多项操作模式实验

对比单次调用和批量调用的性能与准确性差异。

**两种模式对比：**

| 模式         | Tool调用方式               | Model调用次数 | 优势                         | 劣势                     |
|--------------|----------------------------|---------------|------------------------------|--------------------------|
| 单项模式     | Single-item per call       | Multiple      | Fine-grained control         | Higher latency, more tokens |
| 多项模式     | Batch multiple items       | Single        | Lower latency, fewer tokens  | Potential batch errors   |

**实验目标：**
- Evaluate accuracy: single vs. batch operations
- Measure latency and token efficiency
- Identify optimal use cases for each mode

### 5.4 多版本与扩展性

支持从基类继承实现自定义Agent，便于团队协作和功能迭代。

**扩展示例：**

```python
# Version 2 implementations by different team members
PersonalSummaryAgentV2Op / PersonalRetrieveAgentV2Op      # @weikang
ProceduralSummaryAgentV2Op / ProceduralRetrieveAgentV2Op  # @zouyin

# Inherit from BaseMemoryAgentOp
class PersonalSummaryAgentV2Op(BaseMemoryAgentOp):
    """Enhanced personal memory summarization with improved algorithms"""
    pass
```

### 5.5 文件系统集成（未来方向）

探索将文件操作能力集成到记忆系统中，支持基于文件的记忆管理。

**挑战与考虑：**

1. **操作适配性**：Current operations (retrieve/add/update/delete) need adaptation for file-based storage
2. **工具选择**：Consider file operation tools: `grep`, `glob`, `ls`, `read_file`, `write_file`, `edit_file`
3. **模型能力**：Base models have limited file operation capabilities; `qwen3-code` shows better performance

**潜在架构：**

```python
# File-based memory operations
FileMemoryOp(tools=[
    grep,           # Search within files
    glob,           # File pattern matching
    ls,             # List directory contents
    read_file,      # Read file contents
    write_file,     # Write new memory files
    edit_file,      # Update existing memory files
])
```

### 5.6 自我修改上下文

支持Agent动态修改自身的上下文状态，实现自适应记忆管理。

**实现方式：**

1. **Summary Agent 主动修改**：
   - `add_meta_memory` directly modifies agent context
   - Updates available memory types and targets during execution

2. **ReMy Agent 被动修改**：
   - Retrieves `identity_memory` at each interaction
   - Dynamically updates self-state based on retrieved identity
   - Enables adaptive behavior based on accumulated identity knowledge

## ReMe V2 开发路线图与实施计划

### 技术改造阶段
1. **代码整合与兼容**：合并flowllm中reme必要的代码，保留现在server-client的依赖，兼容现在各个仓库的依赖代码
2. **核心接口重构**：新的ReMe接口设计，支持summary，retrieve，context_offload, context_reload 4个核心接口
3. **Agentic算法升级**：新的agentic算法方案开发

### 评估验证阶段
4. **Benchmark测试**
   - halumem
   - locomo
   - longmemevel
   - personal-v2 ?
   - appworld/bfcl-v3

### 发布推广阶段
5. **技术报告**撰写与发布
6. **生态更新**：更新各个仓库的依赖代码
   - agentscope
   - agentscope-runtime
   - evotraders
   - alias(tool-memory)
   - agentscope-java
   - AgentEvolver
   - cookbook: reme procedural memory paper
   - tool-memory-upgrade（将要合并）

**里程碑目标**：春节前完成小版本发布

---

## ReMe V2 核心竞争优势

### 1. 渐进式 Agentic Memory 架构【核心创新】
融合了多种记忆的渐进式agentic方案，实现从短期到长期记忆的智能化演进

### 2. 全生命周期记忆管理
同时支持长期记忆（Long-term Memory）和短期记忆（Working Memory），完整覆盖Agent认知周期

### 3. 模型
提供开源小模型

### 4. 开发者友好生态
   1. **简洁接口**：提供简洁的接口设计，全异步接口
   2. **即开即用**：提供CLI工具，开箱即用的体验
   3. **生态融合**：提供和AgentScope、LangChain无缝集成的方案
   4. **高度可扩展**：支持Agentic算法的二次开发与定制