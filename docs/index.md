---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# ReMe: Memory Management Kit for Agents
  <em>Remember Me, Refine Me.</em>

<div class="flex justify-center space-x-3">
  <a href="https://pypi.org/project/reme-ai/"><img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python Version"></a>
  <a href="https://pypi.org/project/reme-ai/"><img src="https://img.shields.io/badge/pypi-0.2.0.0-blue?logo=pypi" alt="PyPI Version"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-black" alt="License"></a>
  <a href="https://github.com/agentscope-ai/ReMe"><img src="https://img.shields.io/github/stars/modelscope/ReMe?style=social" alt="GitHub Stars"></a>
</div>

---

ReMe provides AI agents with a unified memory system—enabling the ability to extract, reuse, and share memories across
users, tasks, and agents.

Agent memory can be viewed as:

```text
Agent Memory = Long-Term Memory + Short-Term Memory
             = (Personal + Task + Tool) Memory + (Working Memory)
```

Personal memory helps "**understand user preferences**", task memory helps agents "**perform better**", and tool memory enables "**smarter tool usage**". Working memory provides **short-term contextual memory** by keeping recent reasoning and tool results compact and accessible without overflowing the model's context window.

## Architecture Design

<p align="center">
 <img src="_static/figure/reme_usage.jpg" alt="ReMe Logo" width="100%">
</p>

ReMe integrates three complementary memory capabilities:

:::{admonition} Task Memory/Experience
:class: note

Procedural knowledge reused across agents

- **Success Pattern Recognition**: Identify effective strategies and understand their underlying principles
- **Failure Analysis Learning**: Learn from mistakes and avoid repeating the same issues
- **Comparative Patterns**: Different sampling trajectories provide more valuable memories through comparison
- **Validation Patterns**: Confirm the effectiveness of extracted memories through validation modules

:::

Learn more about how to use task memory from [task memory](task_memory/task_memory.md)

:::{admonition} Personal Memory
:class: note

Contextualized memory for specific users

- **Individual Preferences**: User habits, preferences, and interaction styles
- **Contextual Adaptation**: Intelligent memory management based on time and context
- **Progressive Learning**: Gradually build deep understanding through long-term interaction
- **Time Awareness**: Time sensitivity in both retrieval and integration

:::

Learn more about how to use personal memory from [personal memory](personal_memory/personal_memory.md)

:::{admonition} Tool Memory
:class: note

Data-driven tool selection and usage optimization

- **Historical Performance Tracking**: Success rates, execution times, and token costs from real usage
- **LLM-as-Judge Evaluation**: Qualitative insights on why tools succeed or fail
- **Parameter Optimization**: Learn optimal parameter configurations from successful calls
- **Dynamic Guidelines**: Transform static tool descriptions into living, learned manuals

:::

Learn more about how to use tool memory from [tool memory](tool_memory/tool_memory.md)

:::{admonition} Working Memory
:class: note

Short‑term contextual memory for long‑running agents via **message offload & reload**:

- **Message Offload**: Compact large tool outputs to external files or LLM summaries
- **Message Reload**: Search (`grep_working_memory`) and read (`read_working_memory`) offloaded content on demand

**📖 Concept & API**:
- Message offload overview: [Message Offload](work_memory/message_offload.md)
- Offload / reload operators: [Message Offload Ops](work_memory/message_offload_ops.md), [Message Reload Ops](work_memory/message_reload_ops.md)

**💻 End‑to‑End Demo**:
- Working memory quick start: [Working Memory Quick Start](cookbook/working/quick_start.md)
- ReAct agent with working memory: [react_agent_with_working_memory.py](../cookbook/working_memory/react_agent_with_working_memory.py)
- Runnable demo: [work_memory_demo.py](../cookbook/working_memory/work_memory_demo.py)

:::

---

## 📦 Ready-to-Use Memories

ReMe provides pre-built memories that agents can immediately use with verified best practices:

### Available Memories

- **`appworld.jsonl`**: Memory library for Appworld agent interactions, covering complex task planning and execution
  patterns
- **`bfcl_v3.jsonl`**: Working memory library for BFCL tool calls

### Quick Usage

```{code-cell}
# Load pre-built memories
response = requests.post("http://localhost:8002/vector_store", json={
    "workspace_id": "appworld",
    "action": "load",
    "path": "./docs/library/"
})

# Query relevant memories
response = requests.post("http://localhost:8002/retrieve_task_memory", json={
    "workspace_id": "appworld",
    "query": "How to navigate to settings and update user profile?",
    "top_k": 1
})
```


## 📚 Resources

- **[Installation Guide](installation.md)**, **[Quick Start](quick_start.md)**: Get started quickly with practical examples
- **[Vector Storage Setup](vector_store_api_guide.md)**: Configure local, Elasticsearch, Qdrant, ChromaDB, or ObVec (OceanBase / seekdb via pyobvector) storage and usage
- **[MCP Guide](mcp_quick_start.md)**: Create MCP services
- **[Personal Memory](personal_memory/personal_memory.md)**, **[Task Memory](task_memory/task_memory.md)** & **[Tool Memory](tool_memory/tool_memory.md)**: Operators used in personal memory, task memory and tool memory. You can modify the config to customize the pipelines.
- **[Example Collection](./cookbook/appworld/quickstart.md)**: Real use cases and best practices

---

## Citation

```bibtex
@software{AgentscopeReMe2025,
  title = {AgentscopeReMe: Memory Management Kit for Agents},
  author = {Li Yu, Jiaji Deng, Zouying Cao},
  url = {https://reme.agentscope.io},
  year = {2025}
}
```
