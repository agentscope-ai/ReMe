# beam result

## longmemeval版本的prompt

### 100K


agentscope==2.0.4.post1, conda reme 环境, 20 并发, eval-only（复用已构建 memory）
(2026-08-05, 20 cases / 400 Qs, 总耗时 46.0 min)

| 题型 | Agentic | Binary | input tok/q | output tok/q | total tok/q | tool calls/q |
|---|---|---|---|---|---|---|
| abstention | 0.550 | 0.550 | 96,031 | 1,070 | 97,101 | 4.58 |
| contradiction_resolution | 0.438 | 0.412 | 32,263 | 872 | 33,135 | 2.48 |
| event_ordering | 0.501 | 0.423 | 140,195 | 5,163 | 145,358 | 4.70 |
| information_extraction | 0.873 | 0.832 | 50,245 | 883 | 51,128 | 3.15 |
| instruction_following | 0.750 | 0.725 | 37,986 | 848 | 38,834 | 2.67 |
| knowledge_update | 0.688 | 0.675 | 31,198 | 651 | 31,849 | 2.27 |
| multi_session_reasoning | 0.626 | 0.584 | 85,038 | 4,563 | 89,601 | 4.28 |
| preference_following | 0.925 | 0.912 | 34,281 | 989 | 35,270 | 2.50 |
| summarization | 0.623 | 0.461 | 89,657 | 2,056 | 91,713 | 4.12 |
| temporal_reasoning | 0.637 | 0.625 | 34,563 | 1,049 | 35,612 | 2.52 |
| **OVERALL** | **0.661** | **0.620** | **63,146** | **1,814** | **64,960** | **3.33** |

Memory Construction 平均 token 消耗（default agent, 20 cases 全量构建）:

| Agent | input tok/case | output tok/case | total tok/case |
|---|---|---|---|
| default | 2,172,316 | 136,697 | 2,309,013 |

### 1M

agentscope==2.0.4.post1, conda reme 环境, 20 并发, 全量构建 memory
(2026-08-05, 35 cases / 700 Qs, 总耗时 459.2 min)

| 题型 | Agentic | Binary | input tok/q | output tok/q | total tok/q | tool calls/q |
|---|---|---|---|---|---|---|
| abstention | 0.429 | 0.429 | 118,707 | 1,178 | 119,886 | 4.20 |
| contradiction_resolution | 0.391 | 0.364 | 49,787 | 810 | 50,597 | 2.50 |
| event_ordering | 0.558 | 0.456 | 201,514 | 3,889 | 205,403 | 4.79 |
| information_extraction | 0.809 | 0.772 | 78,950 | 894 | 79,844 | 3.00 |
| instruction_following | 0.852 | 0.832 | 55,757 | 924 | 56,681 | 2.81 |
| knowledge_update | 0.779 | 0.771 | 45,981 | 665 | 46,646 | 2.37 |
| multi_session_reasoning | 0.658 | 0.612 | 138,133 | 2,873 | 141,006 | 4.40 |
| preference_following | 0.798 | 0.777 | 51,796 | 920 | 52,716 | 2.53 |
| summarization | 0.693 | 0.537 | 158,794 | 2,905 | 161,700 | 4.44 |
| temporal_reasoning | 0.536 | 0.536 | 100,176 | 3,148 | 103,324 | 3.90 |
| **OVERALL** | **0.650** | **0.609** | **99,959** | **1,821** | **101,780** | **3.49** |

Memory Construction 平均 token 消耗（default agent, 35 cases 全量构建）:

| Agent | input tok/case | output tok/case | total tok/case |
|---|---|---|---|
| default | 31,943,817 | 1,417,061 | 33,360,878 |