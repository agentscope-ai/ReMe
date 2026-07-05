# LongMemEval 数据集测试结果
## Oracle数据集结果
**实验设置：** 在ReMe的基础上关闭dream机制，使用原始Auto-memory，把原始session添加到index中，回答问题可以索引到原始对话。React框架回答。

**实验结果：**

| QA类型 | 正确/总数 | 准确率 |
|------|-----------|--------|
| single-session-assistant | 46/56 | 82.1% |
| knowledge-update | 58/78 | 74.4% |
| single-session-user | 47/70 | 67.1% |
| multi-session | 78/133 | 58.6% |
| temporal-reasoning | 72/133 | 54.1% |
| single-session-preference | 7/30 | 23.3% |
| **Overall** | **308/500** | **61.6%** |


