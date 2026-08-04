# beam result

## longmemeval版本的prompt

### 100K


agentscope==2.0.4.post1, conda reme 环境, 20 并发, 从头构建 memory
(2026-08-04, 20 cases / 400 Qs, 总耗时 61.8 min)

| 题型 | Agentic | Binary | input tok/q | output tok/q | total tok/q | tool calls/q |
|---|---|---|---|---|---|---|
| abstention | 0.500 | 0.500 | 90,460 | 1,121 | 91,581 | 4.15 |
| contradiction_resolution | 0.338 | 0.300 | 27,951 | 751 | 28,703 | 2.48 |
| event_ordering | 0.533 | 0.472 | 143,223 | 5,658 | 148,880 | 4.55 |
| information_extraction | 0.833 | 0.796 | 38,650 | 794 | 39,444 | 2.62 |
| instruction_following | 0.794 | 0.762 | 32,991 | 782 | 33,773 | 2.35 |
| knowledge_update | 0.688 | 0.675 | 26,218 | 716 | 26,934 | 2.08 |
| multi_session_reasoning | 0.644 | 0.584 | 76,239 | 3,229 | 79,469 | 4.00 |
| preference_following | 0.858 | 0.842 | 28,486 | 865 | 29,350 | 2.17 |
| summarization | 0.581 | 0.412 | 83,303 | 1,866 | 85,168 | 3.88 |
| temporal_reasoning | 0.581 | 0.550 | 35,142 | 1,213 | 36,354 | 2.42 |
| **OVERALL** | **0.635** | **0.589** | **58,266** | **1,699** | **59,966** | **3.07** |

**Memory Construction Token 消耗（default agent / qwen3.6-flash, per case）**

| 指标 | 均值 ± 标准差 |
|---|---|
| input_tokens | 2,172,316 ± 975,656 |
| output_tokens | 136,697 ± 37,484 |
| total_tokens | 2,309,013 ± 1,009,612 |

### 1M

| 题型 | Prompted(limit=15) | Prompted Binary | Agentic | Agentic Binary |
|---|---|---|---|---|
| abstention | 0.464 | 0.464 | 0.464 | 0.464 |
| contradiction_resolution | 0.079 | 0.068 | 0.379 | 0.346 |
| event_ordering | 0.455 | 0.334 | 0.535 | 0.442 |
| information_extraction | 0.653 | 0.589 | 0.795 | 0.758 |
| instruction_following | 0.541 | 0.524 | 0.774 | 0.758 |
| knowledge_update | 0.571 | 0.507 | 0.693 | 0.679 |
| multi_session_reasoning | 0.426 | 0.324 | 0.656 | 0.602 |
| preference_following | 0.718 | 0.676 | 0.805 | 0.796 |
| summarization | 0.516 | 0.303 | 0.672 | 0.504 |
| temporal_reasoning | 0.198 | 0.169 | 0.462 | 0.448 |
| **OVERALL** | **0.462** | **0.396** | **0.623** | **0.580** |

在auto-memory中加入Source Index机制

### 1M (2026-07-28, 35 cases / 700 Qs)

| 题型 | Agentic | Agentic Binary |
|---|---|---|
| abstention | 0.457 | 0.457 |
| contradiction_resolution | 0.405 | 0.371 |
| event_ordering | 0.566 | 0.478 |
| information_extraction | 0.782 | 0.742 |
| instruction_following | 0.860 | 0.843 |
| knowledge_update | 0.704 | 0.679 |
| multi_session_reasoning | 0.663 | 0.603 |
| preference_following | 0.827 | 0.807 |
| summarization | 0.686 | 0.532 |
| temporal_reasoning | 0.511 | 0.500 |
| **OVERALL** | **0.646** | **0.601** |