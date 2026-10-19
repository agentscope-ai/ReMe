# 在 MemLens-128K / 256K 上评测 ReMe Image

使用官方固定的 195 题 memory-agent 子集。128K、256K 分别读取自己的完整历史，
每题使用独立 workspace。

## 运行

```bash
pip install -e ".[as]" -e plugins/memlens -e plugins/memlens-judge
export MEMLENS_DATA=/path/to/MEMLENS
export EXPERIMENT_DIR=/path/to/new-experiment
python -m benchmark.memlens.run --model qwen3.8-max --context-length 256k
```

通过环境变量提供 `LLM_API_KEY` 和 `LLM_BASE_URL`，不要提交凭据。
数据目录包含 `dataset_128k.json`、`dataset_256k.json`、
`agent_subset_195.json` 和 `release_images/`。使用
`--context-length 128k` 切换长度；用 `--limit-scenes 1` 和独立的
`--output` 路径小批量验证，不裁剪场景历史。并发配置在 `config.yaml`。

## 评测设置

原始文本、时间、Image ID 和交错的原图输入原生 Auto Memory；
不使用附带的 BLIP caption 或答案标注。BM25 索引 daily、digest 和原始
session 正文。每题新建 30 轮回答 Agent，工具为 `search`、`read`、
`read_image`，不启用 embedding 或 auto-tag。

| 设置 | 记忆／压缩 | 回答 |
| --- | --- | --- |
| 模型 | 选定的候选模型 | 同一候选模型 |
| Thinking／temperature／stream | 开启／0／关闭 | 开启／0／关闭 |
| 输出预算 | 65,536 | 16,384 |
| 模型 context size | 200,000 | 400,000 |

[候选插件配置](../../plugins/memlens/src/reme_memlens/benchmark.yaml) 保留
30 轮预算、0.8／0.1 上下文触发／预留比例、50,000 token 的工具结果限制和
4,096 张图片保留上限。原生请求最多重试 3 次，间隔 60 秒，HTTP timeout
为 600 秒；不叠加 SDK／Agent 重试，也不修补模型输出。

独立的 [MemLens Judge](../../plugins/memlens-judge/README.md) 使用官方按
category／subtype 区分的二元评分：默认 Qwen3.8-Max、thinking 关闭、temperature
0、输出预算 256。辅助指标为官方 SubEM、F1。答案与题型字段仅进入评分。
`JUDGE_MODEL_NAME` 可独立覆盖 Judge 模型；`settings.json` 记录插件默认值和
环境变量展开后实际生效的 Judge 模型、生成参数与协议，不保存凭据或服务端点。

## 结果

结果保存到 `<output>/memlens-<length>/<model>/`。该目录的 JSON 记录配置与汇总；
`reme_image/scenes/<scene>/` 保存摄入响应、原生 workspace、回答、judge、
指标及完成状态。workspace 下的 `mem_session/` 保留原生 Agent JSONL 轨迹。
已有结果目录不会被覆盖。

新运行使用 `successful-http-requests-v2` 口径：统计记忆／回答作用域中观测到的
成功 HTTP 请求，包括压缩，以及所属 Job 或历史处理随后失败前已成功的请求；
每个作用域只计一次。不计 Judge 请求及未成功的 HTTP 尝试。这是实测请求用量，
并非完整样本成本估计或供应商账单。未知计数保持 `null`，`records` 与各计数的
`coverage` 标示观测覆盖；预期、已评分、未评分题数分别记录。

技术失败计为覆盖缺口，合法空答案与达到轮数上限的回答仍评分。失败诊断记录阶段、
场景／session／题目 ID、脱敏错误链和不含局部变量的堆栈位置，不保存凭据、
请求头、原始配置或图片数据。失败 Response 在检查前保存到独立的
`diagnostics/*-response.json`；`answer.json` 仍只表示成功回答，保留原始预测。
`diagnostics/*-usage.json` 在直接异常或取消时也保留已观测用量；
`status.json.memory_complete` 独立表示历史摄入和索引完成。
清理失败保留评分及用量、增加 `cleanup_error_count`，并使 CLI 返回非零状态。

历史报告沿用其原有的选定成功 Job／历史统计口径，不按新口径重写数据。

数据来源、上游版本和完整授权文本见[英文 README](README.md#sources-and-license)。
