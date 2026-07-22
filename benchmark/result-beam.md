# beam result

## longmemeval版本的prompt

### 100K


| 题型 | Prompted(limit=15) | Agentic |
|---|---|---|
| abstention | 0.525 | 0.575 |
| contradiction_resolution | 0.100 | 0.384 |
| event_ordering | 0.403 | 0.465 |
| information_extraction | 0.618 | 0.884 |
| instruction_following | 0.481 | 0.719 |
| knowledge_update | 0.637 | 0.650 |
| multi_session_reasoning | 0.444 | 0.633 |
| preference_following | 0.706 | 0.829 |
| summarization | 0.423 | 0.617 |
| temporal_reasoning | 0.344 | 0.550 |
| **OVERALL** | **0.468** | **0.631** |

### 1M

| 题型 | Prompted(limit=15) | Prompted Binary | Agentic | Agentic Binary |
|---|---|---|---|---|
| abstention | 0.464 | 0.464 | 0.514 | 0.514 |
| contradiction_resolution | 0.079 | 0.068 | 0.373 | 0.339 |
| event_ordering | 0.455 | 0.334 | 0.547 | 0.450 |
| information_extraction | 0.653 | 0.589 | 0.818 | 0.764 |
| instruction_following | 0.541 | 0.524 | 0.765 | 0.745 |
| knowledge_update | 0.571 | 0.507 | 0.636 | 0.629 |
| multi_session_reasoning | 0.426 | 0.324 | 0.593 | 0.540 |
| preference_following | 0.718 | 0.676 | 0.838 | 0.824 |
| summarization | 0.516 | 0.303 | 0.661 | 0.478 |
| temporal_reasoning | 0.198 | 0.169 | 0.394 | 0.383 |
| **OVERALL** | **0.462** | **0.396** | **0.614** | **0.567** |