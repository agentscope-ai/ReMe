# LongMemEval 数据集测试结果

## cleaned-s

**basic settings**

1. 使用修改后的auto-memory prompt，关闭auto-dream机制
2. reme-memory中的全部session的时间一定早于question的时间

**results **

1. Agentic answer框架回答，每次最多调用5次search

| Category | Total | Correct | Wrong | Accuracy |
|---|---|---|---|---|
| single-session-user | 70 | 66 | 4 | 94.3% |
| single-session-assistant | 56 | 52 | 4 | 92.9% |
| knowledge-update | 78 | 60 | 18 | 76.9% |
| multi-session | 133 | 93 | 40 | 69.9% |
| temporal-reasoning | 133 | 78 | 55 | 58.6% |
| single-session-preference | 30 | 8 | 22 | 26.7% |
| **Overall** | **500** | **357** | **143** | **71.4%** |

2. prompted-based amswer，每次固定使用原始query召回10个fileChunk

| Category | Total | Correct | Wrong | Accuracy |
|---|---|---|---|---|
| single-session-assistant | 56 | 56 | 0 | 100.0% |
| single-session-user | 70 | 67 | 3 | 95.7% |
| knowledge-update | 78 | 69 | 9 | 88.5% |
| multi-session | 133 | 99 | 34 | 74.4% |
| temporal-reasoning | 133 | 83 | 50 | 62.4% |
| single-session-preference | 30 | 16 | 14 | 53.3% |
| **Overall** | **500** | **390** | **110** | **78.0%** |

3. golden session。 使用与prompt-based answer相似的方法，唯一区别是，输入的chunk是longMemEval提供的golden session。

| Category | Total | Correct | Wrong | Accuracy |
|---|---|---|---|---|
| single-session-assistant | 56 | 56 | 0 | 100.0% |
| single-session-user | 70 | 69 | 1 | 98.6% |
| knowledge-update | 78 | 74 | 4 | 94.9% |
| temporal-reasoning | 133 | 124 | 9 | 93.2% |
| multi-session | 133 | 117 | 16 | 88.0% |
| single-session-preference | 30 | 17 | 13 | 56.7% |
| **Overall** | **500** | **457** | **43** | **91.4%** |

4. golden session + time filter. 和上面一个实验的区别是，输入的golden被过滤了一次，要求输入session的时间戳必须早于question的时间才行。

一共被过滤掉了75个session，44个question受到了影响。temperal reasoning类型受影响最大。有20个case不包含任何一个groundtruth session。 根据golden session回答正确并且golden session非空，一共有424个case。

| Category | Total | Correct | Wrong | Accuracy |
|---|---|---|---|---|
| knowledge-update | 78 | 75 | 3 | 96.2% |
| single-session-user | 70 | 67 | 3 | 95.7% |
| multi-session | 133 | 122 | 11 | 91.7% |
| single-session-assistant | 56 | 55 | 1 | 98.2% |
| temporal-reasoning | 133 | 91 | 42 | 68.4% |
| single-session-preference | 30 | 16 | 14 | 53.3% |
| **Overall** | **500** | **426** | **74** | **85.2%** |

5. 关闭auto-memory机制，根据原始query一次性混合检索召回原始session，计算recall.

| Category | Total  | yes-judge | recall@5 / yes | recall@10 / yes |
|---|---|---|---|---|
| knowledge-update | 78 | 75 | 99.3% | 100% |
| single-session-user | 70 | 67 | 100% | 100% |
| multi-session | 133 | 122 | 91.8% | 95.8% |
| single-session-assistant | 56 | 55 | 100% | 100% |
| temporal-reasoning | 133 | 91 | 87.6% | 94.2% |
| single-session-preference | 30 | 16 | 100% | 100% |
| **Overall** | **500** | **426** | **87.6%** | **94.2%** |


## 最终groundtruth

### agentic + prompted（最终GT，2026-07-16）


| Category | Total | Agentic | Prompted limit=15 |
|---|---|---|---|
| single-session-assistant | 56 | 56/56 (100.0%) | 56/56 (100.0%) |
| single-session-user | 70 | 66/70 (94.3%) | 64/70 (91.4%) |
| knowledge-update | 78 | 73/78 (93.6%) | 68/78 (87.2%) |
| temporal-reasoning | 133 | 123/133 (92.5%) | 121/133 (91.0%) |
| multi-session | 133 | 111/133 (83.5%) | 102/133 (76.7%) |
| single-session-preference | 30 | 16/30 (53.3%) | 7/30 (23.3%) |
| **Overall** | **500** | **445/500 (89.0%)** | **418/500 (83.6%)** |

Prompted token 消耗：总 input 13,109,924 (平均 26,220/题)，总 output 316,327 (平均 633/题)。
平均 sessions_ingested: 44.8，dreams_triggered: 0。






现在的进展：
1. 初步修复了longmemeval，大约有56（11.2%）个case与原始不一致；答案为unknown的case从29(3.8%)增加到了67(13.4%)。temporal reasoning类型受影响最大。
2. 使用新的groundtruth测评，agentic和prompted分别提升到了88.8%和86.4%
3. 引入去重功能可以给agentic方法带来+1%的提升；日期过滤带来1.2%的提升；目前draft和python功能暂无提升
发现的问题：
1. 现在的架构难以处理计数类问题，auto-memory和原始session会导致统计重复
2. 容易被错误的auto-memory误导，原始session非常重要，longmemeval的推理难度并不低

TODO
1. grudntruth 还有些问题
2. 尝试新的架构，auto-memory功能改成构摘要，给LLM提供工具，通过auto-memory摘要读取链接到的原始session


现在还有8个分歧：37f165cf, dd2973ad, 09ba9854, gpt4_2f8be40d, bf659f65, 7405e8b1, 9ee3ecd6, 8550ddae