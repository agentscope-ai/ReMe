# MobileMem-Omni 视觉推理评测

使用每个用户完整、按时间排序的原始对话，评测 ReMe Image 在官方 `visual_reasoning` 题目上的表现。
记忆与回答 Job 仅接收原始文本和图像，不接收数据集摘要、caption、标准答案、证据标注或题型。
官方 `multi_visual_reasoning` 不被重命名，也不纳入这个评测范围。

## 运行

安装两个插件，并按原目录结构解压官方图片：

```bash
pip install -e ".[as]" -e plugins/mobilemem -e plugins/mobilemem-judge
mkdir -p /path/to/omni/image
python -m zipfile -e /path/to/omni/image.zip /path/to/omni/image
export LLM_API_KEY="..."
export LLM_BASE_URL="..."
python -m benchmark.mobilemem.run --dataset /path/to/omni --output /path/to/exp-xxxx
```

[config.yaml](config.yaml) 默认使用 `qwen3.8-max`、4 个用户并发、16 个题目并发；
可通过 `--model`、`--user-concurrency`、`--query-concurrency` 覆盖。
每次运行创建 `<output>/mobilemem/<model>`，已有目录不会被覆盖。

## 评测设置

每个用户使用独立的新工作空间，按时间顺序通过多模态 `auto_memory` 摄入全部 session，然后更新一次索引。
每题使用新的 `agentic_answer` Agent，最多 30 轮，工具为 BM25 `search`、`read` 和 `read_image`。
检索范围包含 daily、digest 笔记及原始 session 正文；保存的原生图像附件与 Image ID 可供回答 Agent 重新读图。

[候选模型配置](../../plugins/mobilemem/src/reme_mobilemem/benchmark.yaml) 在记忆、回答和压缩阶段使用同一模型：
temperature 为 0，开启 thinking，输出预算 65,536，上下文为 1,000,000 token，最多 4,096 张图。
原生上下文比例保持 0.8/0.1。输入图片不缩放；MPO 保持既有的主图 PNG 转换。
数据集图片路径别名通过唯一 basename 对应原件。

问题使用官方 QA 模板，并提示通过证据工具作答。标准答案与证据仅交给独立的
[MobileMem Judge](../../plugins/mobilemem-judge/README.md)。默认采用 MemBase 可执行默认
`mobilemem-omni-judge` 评分模板，模型为 `qwen3.8-max`，temperature 为 0，关闭 thinking，输出预算 4,096。
报告包含二元 LLM-Judge 准确率及官方 token-overlap F1、BLEU-1。

## 结果

每个用户的 `workspace/` 保存笔记、原始 JSONL、图像及原生 AgentScope 轨迹。
`sessions/*.json` 与 `questions/*/` 保存响应、评分、指标和技术错误。
`summary.json` 汇总完成/应测题数，指标只对成功评分的结果求均值。
原生中断与失败 Job 保留为未评分的技术缺口，不作为零分答案；runner 不额外加入 fallback 或重试队列。

usage policy 第 2 版统计已观测的成功 HTTP 请求，包括压缩调用及摄入/回答最终失败前已经产生的请求。
记忆成本只汇总 `sessions/*.json` 和 `index-response.json`；回答成本只汇总 `answer-response.json`，
包括失败响应及直接异常时的 `response: null`。成功与失败的回答统一保存在这个文件中，
根据 `response.success` 决定是否进入评分；旧运行的 `answer.json` 副本、result 和 response metadata 不重复计费。
`memory_usage.json` 仍仅在完整历史摄入及索引更新成功后生成，不能将它误作部分摄入标记。
Judge 用量不计入方法成本；缺失字段保持未知，并记录每个字段的观测覆盖数。
`mean_tokens_per_user` / `mean_tokens_per_question` 是本次已观测成本按计划用户数/应测题数均摊，
summary 明确记录分母和尝试/完成数，不代表成功样本平均成本。旧的 selected-success-only 报告不改写口径。

错误记录包含失败阶段、用户/会话/题号、安全的图片相对路径、脱敏原因和调用栈位置，不保存凭据、请求 headers、
图片载荷或 frame locals；清理错误单独保留，`cleanup_error_count` 使本轮以状态码 2 退出，但不丢弃已评分结果。
`judge-settings.json` 从实际校验后的配置中保存 Judge 模型、参数和协议白名单，
包括可选的 `JUDGE_MODEL_NAME` 覆盖值，不导出凭据。

协议来源、数据版本和改编代码的完整 MIT 许可证见 [英文说明](README.md)。
