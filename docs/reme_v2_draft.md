# ReMeV2-重构的设计思路和代码细节

## 背景

### Claude Skills的启发
ReMeV2的设计受到Claude Skills的渐进式披露（Progressive Disclosure）架构启发。Claude Skills通过三层架构实现高效的上下文管理：

1. **元数据层（Metadata Layer）**：在启动时加载技能的名称和描述，让Agent了解可用的技能及其用途
2. **指令层（Instructions Layer）**：当技能被触发时，加载SKILL.md文件的主体部分，包含具体的工作流程、最佳实践和指导
3. **资源层（Resources Layer）**：按需加载附加资源，如代码示例、参考资料等

这种渐进式披露确保在任何给定时间，只有相关内容占据上下文窗口，避免重复提供相同指导，从而优化性能。

参考：https://docs.claude.com/en/docs/agents-and-tools/agent-skills/quickstart

### ReMeV2的设计理念
借鉴Claude Skills的思路，ReMeV2设计了基于**渐进式检索&总结**的记忆管理方案：

**渐进式检索（Progressive Retrieval）**：
- Layer 1: 加载元记忆（Meta Memory）- 记忆类型、目标和描述概览
- Layer 2: 检索具体记忆（Retrieve Memory）- 根据查询获取相关记忆内容
- Layer 3: 读取历史细节（Read History）- 按需加载完整的历史交互记录

**渐进式总结（Progressive Summarization）**：
- 通过专门的Summary Agents（personal/procedural/tool/identity）对不同类型的记忆进行分层总结
- 使用Meta-Summarizer协调各个Summary Agent，实现记忆的增量更新和压缩
- Summary Memory作为新的维度压缩Message，优化上下文使用

这种设计既保证了记忆检索的效率，又确保了记忆总结的准确性和可维护性。

## Memory
@MemoryType 三层架构

### Memory schema 设计
@MemoryNode的

## 代码设计
相对workflow更加简洁，激进？
ReMeV2 = tool(s) + agent(s)

### tool
@tool
列出所有的类初始化参数，tool_call参数

#### 基类：BaseMemoryToolOp

**初始化参数：**
- `enable_multiple` (bool): 是否启用多项操作模式。默认：`True`
- `enable_thinking_params` (bool): 是否在schema中包含thinking参数。默认：`False`
- `memory_metadata_dir` (str): 存储记忆元数据的目录路径。默认：`"./memory_metadata"`

#### Tool操作列表

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

**注意事项：**
1. 所有BaseMemoryToolOp子类自动继承 `enable_multiple`、`enable_thinking_params` 和 `memory_metadata_dir` 参数
2. 当 `enable_thinking_params=True` 时，会自动在tool call schema中添加 `thinking` 参数
3. 部分工具在 `__init__` 方法中强制设置 `enable_multiple=False`（AddHistoryMemoryOp、AddSummaryMemoryOp、ReadIdentityMemoryOp、UpdateIdentityMemoryOp、ReadMetaMemoryOp）
4. 通过 `self.context` 访问的上下文参数：`workspace_id`、`memory_type`、`memory_target`、`ref_memory_id`、`author`
5. VectorRetrieveMemoryOp：当 `add_memory_type_target=False` 时，memory_type和memory_target从context中获取，而非tool call参数

### agent
@agent/v1
列出所有的类初始化参数，tool_call参数

#### 基类：BaseMemoryAgentOp

**初始化参数：**
- `max_steps` (int)：ReAct循环的最大推理-执行步数。默认值：`20`
- `tool_call_interval` (float)：工具调用之间的间隔时间（秒）。默认值：`0`
- `add_think_tool` (bool)：是否为指令模型添加思考工具。默认值：`False`

**类属性：**
- `memory_type` (MemoryType | None)：Agent的记忆类型。默认值：`None`

**上下文参数（通过 `self.context` 访问）：**
- `workspace_id` (str, required)：工作空间标识符
- `query` (str, optional)：查询文本输入
- `messages` (array[object], optional)：消息输入（query的替代方式）
- `memory_target` (str, optional)：记忆操作的目标
- `ref_memory_id` (str, optional)：参考记忆ID
- `author` (str)：作者/模型名称（从LLM配置获取）

#### Agent操作

| Agent类                         | 继承自               | 初始化参数（基类外）                                                                         | Tool Call参数                                                                                                                                                     | 可用工具                                                                                                                     |
|--------------------------------|-------------------|------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| **PersonalSummaryAgentV1Op**   | BaseMemoryAgentOp | None                                                                               | `workspace_id` (str, required)<br>`memory_target` (str, required)<br>`query` (str, optional)<br>`messages` (array, optional)<br>`ref_memory_id` (str, required) | add_memory<br>update_memory<br>delete_memory<br>vector_retrieve_memory                                                   |
| **ProceduralSummaryAgentV1Op** | BaseMemoryAgentOp | None                                                                               | `workspace_id` (str, required)<br>`memory_target` (str, required)<br>`query` (str, optional)<br>`messages` (array, optional)<br>`ref_memory_id` (str, required) | add_memory<br>update_memory<br>delete_memory<br>vector_retrieve_memory                                                   |
| **ToolSummaryAgentV1Op**       | BaseMemoryAgentOp | None                                                                               | `workspace_id` (str, required)<br>`memory_target` (str, required)<br>`query` (str, optional)<br>`messages` (array, optional)<br>`ref_memory_id` (str, required) | add_memory<br>update_memory<br>vector_retrieve_memory                                                                    |
| **IdentitySummaryAgentV1Op**   | BaseMemoryAgentOp | None                                                                               | `workspace_id` (str, required)<br>`query` (str, optional)<br>`messages` (array, optional)                                                                       | read_identity_memory<br>update_identity_memory                                                                           |
| **ReMeSummaryAgentV1Op**       | BaseMemoryAgentOp | `enable_tool_memory` (bool, 默认: True)<br>`enable_identity_memory` (bool, 默认: True) | `workspace_id` (str, required)<br>`query` (str, optional)<br>`messages` (array, optional)                                                                       | add_meta_memory<br>add_summary_memory<br>hands_off<br>(内部调用: add_history_memory, read_identity_memory, read_meta_memory) |
| **ReMeRetrieveAgentV1Op**      | BaseMemoryAgentOp | `enable_tool_memory` (bool, 默认: True)                                              | `workspace_id` (str, required)<br>`query` (str, optional)<br>`messages` (array, optional)                                                                       | vector_retrieve_memory<br>read_history_memory<br>(内部调用: read_meta_memory)                                                |
| **ReMyAgentV1Op**              | BaseMemoryAgentOp | `enable_tool_memory` (bool, 默认: True)<br>`enable_identity_memory` (bool, 默认: True) | `workspace_id` (str, required)<br>`query` (str, optional)<br>`messages` (array, optional)                                                                       | vector_retrieve_memory<br>read_history_memory<br>(内部调用: read_identity_memory, read_meta_memory)                          |

**说明：**
1. 所有Agent都从 `BaseMemoryAgentOp` 继承 `max_steps`、`tool_call_interval` 和 `add_think_tool` 参数
2. Agent实现ReAct模式：推理步骤（带工具的LLM调用）→ 执行步骤（执行工具调用）→ 重复直到完成或达到max_steps
3. `PersonalSummaryAgentV1Op` 的 `memory_type = MemoryType.PERSONAL`
4. `ProceduralSummaryAgentV1Op` 的 `memory_type = MemoryType.PROCEDURAL`
5. `ToolSummaryAgentV1Op` 的 `memory_type = MemoryType.TOOL`
6. `IdentitySummaryAgentV1Op` 的 `memory_type = MemoryType.IDENTITY`
7. Summary agents（Personal/Procedural/Tool/Identity）通常由 `ReMeSummaryAgentV1Op` 通过 `hands_off` 工具调用
8. `ReMeSummaryAgentV1Op` 在每个推理步骤中动态更新系统提示中的meta_memory_info
9. 工具可用性由每个Agent初始化时注册的 `ops` 字典决定

## Runtime设计(内部设计)

### summary 渐进式总结
AddHistoryMemoryOp()
ReadMetaMemoryOp()

ReMeSummaryAgentV1Op << [
  AddMetaMemoryOp(list(memory_type, memory_target)),
  AddSummaryMemoryOp(summary_memory),
  HandsOffOp(list(memory_type, memory_target)) << [
    PersonalSummaryAgentV1Op,
    ProceduralSummaryAgentV1Op,
    ToolSummaryAgentV1Op,
    IdentitySummaryAgentV1Op
  ],
]
PersonalSummaryAgentV1Op << [AddMemoryOp, UpdateMemoryOp, DeleteMemoryOp, VectorRetrieveMemoryOp]
ProceduralSummaryAgentV1Op << [AddMemoryOp, UpdateMemoryOp, DeleteMemoryOp, VectorRetrieveMemoryOp]
ToolSummaryAgentV1Op << [AddMemoryOp, UpdateMemoryOp, VectorRetrieveMemoryOp]
IdentitySummaryAgentV1Op << [ReadIdentityMemoryOp, UpdateIdentityMemoryOp]

### retrieve: 渐进式检索【这里和skills检索很像】
``` skills
load_meta_skills
load_skills
load_reference_skills
execute_shell
```

ReMeRetrieveAgentV1Op << [
  ReadMetaMemoryOp(),
      ``` prompt
      格式："- <memory_type>(<memory_target>): <description>"
      personal jinli xxxxx
      personal jiaji xxxxx
      personal jinli&jiaji xxxxx
      procedural appworld xxxxx
      procedural bfcl-v3 xxxxx
      tool tool_guidelines  xxxxx
      identity self   xxxxx
      ```
  VectorRetrieveMemoryOp(list(memory_type, memory_target, query)), layer1+layer2
  ReadHistoryMemoryOp(ref_memory_id), layer3
]

## 额外的设计

### summary memory的作用
```txt
step1: summary jinli's dialog
           session1: List[Message] -> session2: List[Message] -> session3: List[Message] -> ...
summary    1                          1                          1
personal   0                          0                          1
procedural 0                          1                          0 

step2: retrieve
vector_retrieve_memory(query, memory_type="personal", memory_target="jinli") -> memory_type in ["personal", "summary"]
```
1. 相较于其他维度的memory，提供了一种通用维度的memory抽取逻辑
2. 确保如果不符合其他的meta memory的情况下，有一个兜底的原始对话的索引。

### 带thinking参数的实验
Inspired by Agentscope
```
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

对比
- thinking model
- instruct model 
- instruct model with thinking_params @Inspired by Agentscope
- instruct model with thinking_tool @Inspired by claude

### multi模式实验
- tool是单次调用，模型多次调用工具
- tool是多次调用，模型只需要调用一次工具

### 多版本-扩展性
从BaseMemoryAgentOp继承：
personal_summary_agent_v2/personal_retrieve_agent_v2 @weikang
procedural_summary_agent_v2/procedural_retrieve_agent_v2 @zouyin

### Hook模式/Agent
``` python
@memory_wrapper
async def call_agent(messages: List[Message]) -> List[Message]:
    await agent.achat(messages)
    history_messages: List[Message] = agent.history_messages
    return history_messages
```
or 
```
ReMyAgentV1Op << [ReadMetaMemoryOp, VectorRetrieveMemoryOp, ReadHistoryMemoryOp]
+ async call summary_service
```

### 项目组织
新的方案：experimental or v2 or core
老的方案：v1 or 废弃
- 文档&README如何组织这几部分
- CookBook
  - reme-agent
  - 使用mem-agent和agentscope、langchain结合
  - mini-reme
v1->v2->v3?
personal -> 多种记忆 -> 融合多种记忆+agentic+渐进式 -> 文件系统？
                    -> MemoryModel


### 文件系统-涉及后续的代码再次重写
难点：retrieve/add/update/delete 需要变化吗？
使用File工具? 
grep/grob/ls/read_file/write_file/edit_file?
基模操作这些的能力稍差，qwen3-code的能力相对可以。

### short-term-memory
TODO, 单独设计

### 自我修改上下文
summary_agent add_meta_memory直接修改上下文
remy_agent 可以每次通过retrieve_identity_memory修改自己的状态

### 论文
1. mem0
2. agentscope
3. datajuicer

### 模型
xxx

### hagging face建设
1. 金融？
2. bfcl/appworld


### 看下竞品的文档
看文档而不是看代码
直接让云鹏用


### 对外接口，对齐mem0
https://docs.mem0.ai/core-concepts/memory-operations/add

- import: 
  - basic: 去掉with，直接reme.memory.summary/retrieve，大幅降低developer的使用成本

- http[reme-http-server]：
  - 只有summary_memory & retrieve_memory两个接口
  - vector-db相关接口：包括workspace & memory的更加详细的操作接口（v1只提供了dump，load）

- mcp[reme-mcp-server]: 
  - 提供retrieve的[ReadMetaMemoryOp, VectorRetrieveMemoryOp, ReadHistoryMemoryOp]


####
是否要提供一个方便用户二开的接口。
reme-mini 学习文档


1. add不是原子操作，https://docs.mem0.ai/core-concepts/memory-operations/add
2. search/update/delete算原子操作
3. user_id是物理隔离+用户是谁两个概念；目前物理隔离是workspace_id, 用户是谁是memory_target（名字可换）
4. mem0支持了很多的vector_store，但是只有同步接口


是否需要重载运算符



一起调研竞品：
1. 开源框架
2. 商业化的Memory
3. 论文

我们相比竞品的优势：
1. agentic 渐进式
2. 支持用户开发：
    用户使用接口：
        3. 顶层接口：summary / retrieve <=> add / search 
        4. 底层接口（MCP）：add/delete/update/retrieve/
    用户二开接口：
        xxx
3. mem-agent 模型





6 llm = {provider: "openai", model_name: "gpt-4-turbo", temperature: 0.7}

import asyncio
from reme_ai import ReMeApp


async def main():

    reme = ReMe(
        # 这里之有golbal的配置
        llm = {provider: "openai", model_name: "gpt-4-turbo", temperature: 0.7},
        "embedding_model.default.model_name=text-embedding-v4",
        "vector_store.default.backend=memory"
        retrieve=Retriever(ops=Opa() >> OpB(),], # 修改只在自己的op内修改
        retrieve=Retriever(tools=[ToolA(), ToolB()],
        summarizer=Summarizer(tools=[ToolA(), ToolB()]      
    )
    
    # summary的参数优先基高于全局优先基
    result = await reme.summary(
        workspace_id="task_workspace",
        trajectories=[
            {
                "messages": [
                    {"role": "user", "content": "Help me create a project plan"}
                ],
                "score": 1.0
            }
        ]
    )


    reme.close()

if __name__ == "__main__":
    asyncio.run(main())


# 通过python代码启动。reme.serve()



class BasePipeline(object):
    pass


class GraphPipeline(BasePipeline):
    pass


class AgenticPipeline(BasePipeline):
    pass


retrieve_pipeline = Pipeline(
    ops=[
        BuildQueryOp(),
        RecallVectorStoreOp(),
        RerankMemoryOp(enable_llm_rerank=True, enable_score_filter=False, top_k=5),
        RewriteMemoryOp(enable_llm_rewrite=True)
    ],
    tools=[BuildQueryOp(), RecallVectorStoreOp()],
)


1. 现有的op参数变更
2. 新建自己的op
3. search的top_k 》初始化的优先级
4. react_agent.tools = [xxx, xxx, xxx]

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