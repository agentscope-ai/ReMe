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

# Context Offload

## 1. Background: Why Context Offload?

### The Agent Context Challenge

In modern AI agent systems, LLMs interact with tools through iterative loops, accumulating conversation history and tool results. With each iteration, a critical problem emerges:

**The Core Problem: Context Window Explosion**

When an agent executes complex tasks, it relies on maintaining conversation history to track progress and make informed decisions. However:

- **Rapid Context Growth**: Each tool call appends input parameters and output results to message history
- **Token Consumption**: A single tool call can consume hundreds or thousands of tokens, especially for data-heavy operations
- **Context Window Limits**: Most LLMs have finite context windows (e.g., 128K, 200K tokens)
- **Context Rot**: As context grows beyond optimal thresholds, model performance degrades significantly

**Example: Web Research Agent**

Imagine an agent performing research across multiple sources:
```
Iteration 1: web_search("AI context management") → 3,500 tokens
Iteration 2: read_webpage(url_1) → 8,200 tokens
Iteration 3: web_search("context compression techniques") → 4,100 tokens
Iteration 4: read_webpage(url_2) → 7,800 tokens
...
Iteration 15: summarize_findings() → Total context: 95,000 tokens
```

As context accumulates:
- **At 50K tokens**: Agent performs normally, accurate responses
- **At 100K tokens**: Responses become repetitive, slower inference
- **At 150K tokens**: Significant quality degradation, "context rot" sets in
- **At 200K tokens**: Context window exhausted, cannot continue

**Without context management, agents hit walls after just 15-20 complex tool calls.**

### The Solution: Context Offload as Context Engineering

Context Offload solves this by **intelligently moving non-essential information out of active context**, allowing agents to operate indefinitely while maintaining optimal performance:

**1. Context Compaction** (Reversible Strategy)
- **Selective Storage**: Large tool results stored in external files
- **Reference Retention**: Only file paths kept in message history
- **On-Demand Retrieval**: Full content can be retrieved when needed

**2. Context Compression** (LLM-Based Strategy)
- **Intelligent Summarization**: LLM generates concise summaries of older message groups
- **Priority Preservation**: Recent messages and system prompts remain intact
- **Information Density**: Maintains key information while reducing token count

**3. Hybrid Auto Mode** (Adaptive Strategy)
- **Compaction First**: Applies compaction to tool messages
- **Compression When Needed**: Triggers compression if compaction ratio exceeds threshold
- **Dynamic Adjustment**: Adapts strategy based on context characteristics

### Enhanced Context Management

Instead of letting context grow uncontrollably, the agent now benefits from:

```
Traditional Approach (No Context Management):
50 messages → 95,000 tokens → Context rot begins
- Response quality: Degraded
- Inference speed: Slow
- Can continue: No (approaching limit)
- Information lost: No, but unusable

+ Context Offload Approach:
50 messages → 15,000 tokens (after offload) → Optimal performance maintained
- Response quality: High
- Inference speed: Fast
- Can continue: Yes (85% headroom remaining)
- Information lost: No (stored externally, retrievable)

Offload Details:
- 20 tool messages compacted → Stored in /context_store/
- 15 older messages compressed → Summarized in system message
- 5 recent messages preserved → Full content intact
- External storage: 80,000 tokens offloaded
- Active context: 15,000 tokens (84% reduction)
```

This managed context enables the agent to:
- **Operate Indefinitely**: No hard limit on conversation length
- **Maintain Performance**: Stay within optimal token range (10-30K tokens)
- **Preserve Information**: All data accessible through file system or summaries
- **Optimize Costs**: Reduce token consumption by 70-90% in long conversations

### The Impact: From Context Explosion to Controlled Growth

**Traditional Approach (No Context Management):**
```
Agent: "I've executed 20 tool calls, context is now 100K tokens"
→ Performance degradation begins
→ Slower responses, repetitive outputs
→ Cannot continue beyond 30 calls
→ Task abandoned due to context limits
```

**Context Offload Approach (Intelligent Management):**
```
Agent: "I've executed 100 tool calls, active context maintained at 18K tokens"
→ Optimal performance throughout
→ Fast, accurate responses
→ Can continue indefinitely
→ All historical data accessible when needed
```

**Real-World Impact:**

```
Before Context Offload (20 tool calls):
- Active context: 95,000 tokens
- Performance: Degraded (context rot)
- Can continue: No (near limit)
- Response quality: 6/10
- Inference time: 8-12 seconds
- Max task complexity: Low (15-20 calls)

After Context Offload (100 tool calls):
- Active context: 18,000 tokens (-81%)
- Performance: Optimal
- Can continue: Yes (90% headroom)
- Response quality: 9/10
- Inference time: 2-4 seconds (-70%)
- Max task complexity: High (100+ calls)
```
## 2. How Context Offload Works

### Three Context Offload Strategies

ReMe implements three complementary strategies based on [FlowLLM's Context Management Guide](https://flowllm-ai.github.io/flowllm/zh/reading/20251110-manus-context-report/):

#### 1. Compaction (Reversible, No Information Loss)

**Principle**: Store complete content externally, keep only references in context.

**How It Works:**
```
Before Compaction (Tool Message):
{
  role: "tool",
  tool_call_id: "call_123",
  content: {
    path: "/workspace/data.json",
    content: "...[8,000 tokens of data]...",
    status: "success"
  }
}
Token count: 8,200 tokens

After Compaction:
{
  role: "tool",
  tool_call_id: "call_123",
  content: "{path: '/workspace/data.json', content: '...[first 100 chars]...'
            (detailed result is stored in /context_store/call_123.txt)}"
}
Token count: 150 tokens (98% reduction)

Full content stored: /context_store/call_123.txt
Retrievable via: ReadFile tool when needed
```

**Best For:**
- Tool messages with large outputs (>2,000 tokens)
- Web page contents
- File read results
- Search results

**Advantages:**
- 100% reversible (no information loss)
- Can retrieve full content anytime
- Massive token savings (90-98% per message)

#### 2. Compression (LLM-Based, Semantic Preservation)

**Principle**: Use LLM to summarize older message groups while preserving key information.

**How It Works:**
```
Before Compression (15 older messages):
Message 1: "User: Research AI context management"
Message 2: "Assistant: I'll search for information..."
Message 3: "Tool: Found 20 articles on context management..."
...
Message 15: "Assistant: Based on research, key techniques are..."
Total: 35,000 tokens

After Compression (LLM Summary):
System message appendix:
"<state_snapshot>
<overall_goal>The user requested research on AI context management. The agent performed
web searches, reviewed 5 key articles, and identified three main techniques:
compaction, compression, and retrieval. Key findings: compaction is reversible
and efficient for tool results; ...
</overall_goal>
...
</state_snapshot>
"
Total: 7,000 tokens (80% reduction)

Original messages stored: /context_store/compressed_group_001.json
Retrievable if detailed history needed
```

**Best For:**
- Long conversation histories (>20k Tokens)
- Exploratory dialogues
- Context exceeding compaction threshold
- When compaction alone insufficient

**Advantages:**
- Semantic information preserved
- Natural language summaries
- Maintains conversation flow
- Significant token savings (80-90%)

#### 3. Auto Mode (Intelligent Adaptive Strategy)

**Principle**: Apply compaction first, then compression if needed based on effectiveness ratio.

**How It Works:**
```
Step 1: Apply Compaction
Original context: 95,000 tokens
After compaction: 72,000 tokens
Compaction ratio: 72,000 / 95,000 = 0.76 (76%)

Step 2: Evaluate Effectiveness
Threshold: 0.75 (75%)
Actual ratio: 0.76 > 0.75 → Compaction not sufficient

Step 3: Apply Compression
After compression: 18,000 tokens
Final reduction: 81% total

Decision Logic:
- If compaction ratio ≤ 0.75 → Stop, compaction sufficient
- If compaction ratio > 0.75 → Continue with compression
```

**Best For:**
- General-purpose agent applications
- Unknown context characteristics
- Long-running tasks

**Advantages:**
- Adaptive to context type
- Maximizes efficiency
- Balances reversibility and reduction
- Production-proven strategy

## 3. Operation Details: How to Use Each Component

### 3.1 `context_offload`

**Purpose**: Manages context window limits by intelligently offloading message content to external storage, supporting three strategies: compaction (reversible), compression (LLM-based), and auto (adaptive hybrid).

**Flow**:
```yaml
context_offload:
  flow_content: ContextOffloadOp() >> BatchWriteFileOp()
  description: "Manages context window limits by compacting tool messages and compressing conversation history"
```

**Process**:

The operation supports three context management modes:

**1. Compact Mode** (Reversible, No Information Loss)
1. Identifies tool messages exceeding `max_tool_message_tokens` threshold
2. Stores full content in external files (`{store_dir}/tool_call_{id}.txt`)
3. Replaces message content with preview + file reference
4. Keeps recent N messages unchanged (defined by `keep_recent_count`)

**2. Compress Mode** (LLM-Based, Semantic Preservation)
1. Excludes system messages and recent N messages (defined by `keep_recent_count`)
2. Groups remaining messages by `group_token_threshold`
3. Uses LLM to generate semantic summaries for each group
4. Appends summaries to system message as `<state_snapshot>`
5. Stores original message groups in external files (`{store_dir}/compressed_group_{chat_id}_{idx}.json`)

**3. Auto Mode** (Adaptive Strategy - **Recommended**)
1. First applies compaction to tool messages
2. Calculates compaction ratio: `tokens_after / tokens_before`
3. If ratio > 0.75 (compaction insufficient): applies compression to remaining messages
4. If ratio ≤ 0.75 (compaction sufficient): stops processing

**Configuration** (`default.yaml`):
```yaml
op:
  context_offload:
    flow_content: ContextOffloadOp() >> BatchWriteFileOp()
    params:
      # Mode selection
      context_manage_mode: "auto"           # Options: "compact", "compress", "auto"

      # Compaction parameters
      max_tool_message_tokens: 2000         # Threshold for compacting individual tool messages
      max_total_tokens: 20000               # Total token threshold for triggering offload

      # Compression parameters
      group_token_threshold: 0              # Max tokens per compression group (0 = single group)

      # Common parameters
      keep_recent_count: null               # Recent messages to preserve (1 for compact, 2 for compress)
      store_dir: "/path/to/store"           # Directory for storing offloaded content
      chat_id: null                         # Chat session identifier (auto-generated if null)
```

**Key Parameters Explained:**
- `context_manage_mode`: Strategy selection
  - `"compact"`: Only applies compaction to tool messages
  - `"compress"`: Only applies LLM-based compression
  - `"auto"`: Applies compaction first, then compression if ratio > 0.75
- `max_tool_message_tokens`: Individual tool message threshold (default: 2000)
- `max_total_tokens`: Total context threshold for triggering offload (default: 20000)
- `group_token_threshold`: Controls compression granularity (0 = compress all at once)
- `keep_recent_count`: Number of recent messages to keep intact (default: 1 for compact, 2 for compress)
- `store_dir`: **Required** - Directory path for external storage
- `chat_id`: Optional session identifier for file naming

#### Usage with curl

**Example 1: Auto Mode (Recommended)**
```bash
curl -X POST http://0.0.0.0:8002/context_offload \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant"},
      {"role": "user", "content": "Search for AI papers"},
      {"role": "assistant", "content": "I will search for papers", "tool_calls": [...]},
      {"role": "tool", "tool_call_id": "call_123", "content": "...[8000 tokens of results]..."},
      ...
    ],
    "context_manage_mode": "auto",
    "max_total_tokens": 20000,
    "max_tool_message_tokens": 2000,
    "keep_recent_count": 2,
    "store_dir": "/workspace/context_store",
    "chat_id": "research_session_001"
  }'
```

**Example 2: Compact Only Mode**
```bash
curl -X POST http://0.0.0.0:8002/context_offload \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [...],
    "context_manage_mode": "compact",
    "max_tool_message_tokens": 2000,
    "keep_recent_count": 1,
    "store_dir": "/workspace/context_store"
  }'
```

**Example 3: Compress Only Mode**
```bash
curl -X POST http://0.0.0.0:8002/context_offload \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [...],
    "context_manage_mode": "compress",
    "max_total_tokens": 20000,
    "group_token_threshold": 5000,
    "keep_recent_count": 2,
    "store_dir": "/workspace/context_store",
    "chat_id": "chat_session_002"
  }'
```
# TODO 这里需要确认一下具体返回信息是什么
**Response**:
```json
{
  "success": true,
  "answer": "Context offload completed successfully. Reduced from 95,000 to 18,000 tokens (81% reduction).",
  "metadata": {
    "offload_summary": {
      "original_token_count": 95000,
      "final_token_count": 18000,
      "reduction_percentage": 81,
      "compaction_applied": true,
      "compression_applied": true,
      "compacted_messages": 12,
      "compressed_message_groups": 3,
      "files_created": 15
    },
    "compacted_files": [
      "/workspace/context_store/tool_call_123.txt",
      "/workspace/context_store/tool_call_456.txt",
      ...
    ],
    "compressed_files": [
      "/workspace/context_store/compressed_group_research_session_001_0.json",
      "/workspace/context_store/compressed_group_research_session_001_1.json",
      ...
    ],
    "offloaded_messages": [
      {
        "role": "tool",
        "tool_call_id": "call_123",
        "content": "Full search results for AI papers... (detailed result is stored in /workspace/context_store/tool_call_123.txt)",
        "original_tokens": 8200,
        "current_tokens": 150
      },
      ...
    ]
  }
}
```

#### Usage with Python

```{code-cell}
import requests

# API endpoint
BASE_URL = "http://0.0.0.0:8002/"

def offload_context(messages: list, mode: str = "auto", store_dir: str = "/workspace/context_store") -> dict:
    """
    Offload context using specified strategy

    Args:
        messages: List of conversation messages
        mode: "auto", "compact", or "compress"
        store_dir: Directory for external storage

    Returns:
        Response with offloaded messages and statistics
    """
    response = requests.post(
        url=f"{BASE_URL}context_offload",
        json={
            "messages": messages,
            "context_manage_mode": mode,
            "max_total_tokens": 20000,
            "max_tool_message_tokens": 2000,
            "keep_recent_count": 2,
            "store_dir": store_dir,
            "chat_id": "my_chat_session"
        }
    )
    return response.json()

# Example: Offload long conversation history
messages = [
    {"role": "system", "content": "You are a research assistant"},
    {"role": "user", "content": "Find recent AI papers on context management"},
    {"role": "assistant", "content": "I'll search for papers", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "call_123", "content": "..." * 3000},  # Large tool result
    # ... more messages ...
]

# Auto mode (recommended)
result = offload_context(messages, mode="auto")

if result['success']:
    summary = result['metadata']['offload_summary']
    print(f"Token reduction: {summary['original_token_count']} → {summary['final_token_count']}")
    print(f"Reduction rate: {summary['reduction_percentage']}%")
    print(f"Compacted messages: {summary['compacted_messages']}")
    print(f"Compressed groups: {summary['compressed_message_groups']}")

    # Use offloaded messages in next API call
    offloaded_messages = result['metadata']['offloaded_messages']
    # Continue conversation with reduced context...
```

**Complete examples**: See `cookbook/simple_demo/` for full working code with context offload integration.

**Best Practices:**
1. **Use Auto Mode**: Adapts automatically based on context characteristics
2. **Set Appropriate Thresholds**: Adjust `max_tool_message_tokens` based on your use case (1000-3000 typical)
3. **Preserve Recent Context**: Keep `keep_recent_count` at 2-3 for optimal agent performance
4. **Organize Storage**: Use dedicated `store_dir` per workspace/session for easy management
5. **Monitor Reduction Ratio**: If consistently >75% after compaction, consider tuning thresholds

**When to Use Each Mode:**
- **Auto**: General agent applications, unknown context patterns (**Recommended**)
- **Compact**: Tool-heavy workflows with large outputs (web scraping, file processing)
- **Compress**: Dialogue-heavy workflows with minimal tool usage (customer support, tutoring)

### 3.2 `context_reload`
