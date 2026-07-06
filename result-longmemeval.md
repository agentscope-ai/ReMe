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

一共被过滤掉了75个session，44个question受到了影响。temperal reasoning类型受影响最大。

| Category | Total | Correct | Wrong | Accuracy |
|---|---|---|---|---|
| knowledge-update | 78 | 75 | 3 | 96.2% |
| single-session-user | 70 | 67 | 3 | 95.7% |
| multi-session | 133 | 122 | 11 | 91.7% |
| single-session-assistant | 56 | 55 | 1 | 98.2% |
| temporal-reasoning | 133 | 91 | 42 | 68.4% |
| single-session-preference | 30 | 16 | 14 | 53.3% |
| **Overall** | **500** | **426** | **74** | **85.2%** |