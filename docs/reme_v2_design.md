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
列出所有的类初始化参数，tool_call参数，透传的参数

### agent
@agent/v1
列出所有的类初始化参数，tool_call参数，透传的参数

## Runtime设计

### summary 渐进式总结
add_history_memory()
meta-summarizer << [
  load_meta_memory(),    
  add_meta_memory(list(memory_type, memory_target)),
  add_summary_memory(summary_memory),
  hands_off_agent(list(memory_type, memory_target)) << [
    personal_summary_agent,
    procedural_summary_agent,
    tool_summary_agent,
    identity_summary_agent
  ],
]
personal_summary_agent << [add_memory, update_memory, delete_memory, vector_retrieve_memory]
procedural_summary_agent << [add_memory, update_memory, delete_memory, vector_retrieve_memory]
tool_summary_agent << [add_memory, update_memory, vector_retrieve_memory]
identity_summary_agent << [read_identity_memory, update_identity_memory]

### retrieve: 渐进式检索【这里和skills检索很像】
``` skills
load_meta_skills
load_skills
load_reference_skills
execute_shell
```

retriever << [
  load_meta_memory(),
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
  RetrieveMemory(list(memory_type, memory_target, query))), layer1+layer2
  ReadHistory(ref_memory_id), layer3
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
reme-agent << [load_meta_memory, RetrieveMemory, ReadHistory]
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

### 对外接口，对齐mem0
- http[reme-http-server]：
  - 只有summary_memory & retrieve_memory两个接口，大幅降低developer的使用成本
  - vector-db相关接口：包括workspace & memory的更加详细的操作接口（v1只提供了dump，load）
- import: 和http相同
- mcp[reme-mcp-server]: 
  - 提供retrieve的[load_meta_memory, RetrieveMemory, ReadHistory]

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


