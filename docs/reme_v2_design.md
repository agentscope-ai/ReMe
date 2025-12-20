# ReMeV2-重构的设计思路和代码细节

## 背景
Inspired by claude skills，设计基于渐进式检索&总结的方案。
https://docs.claude.com/en/docs/agents-and-tools/agent-skills/quickstart

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
作为新的维度压缩Message新

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
xxxx

### 论文
1. mem0
2. agentscope
3. datajuicer

### 模型
xxx

### hagging face建设
1. 金融？
2. bfcl/appworld


