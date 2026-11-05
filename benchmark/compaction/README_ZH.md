# 上下文压缩实验：两种摘要机制下的 Jev 增益

[English](./README.md) | **简体中文**

摘要可能遗漏后续回答需要的标识符、配置值或日志细节。本实验检验：在固定摘要后追加少量 Jev 选出的原文证据，
能否在两种摘要机制下提高信息保留率，并保持比完整历史更短的平均输入。

这是 **2026-10-16 的合成任务对照实验**。Hermes 负责采集轨迹，评分阶段直接调用 Qwen；
不继续 agent 的工具循环，不从归档恢复原文，也不衡量实际任务完成率。生产功能 `compact_context` 与本实验独立。

## 实验设计

### 数据与分组

共 **14 个预设 query/answer 任务模板**：复用 1 条原始支付轨迹，另在 Hermes 的合成任务素材上实际采集 13 条只读轨迹。
原本要求修改文件或创建定时任务的 query，采集时改写为调查证据并提出行动建议。
来源工作区不可变，仅开放白名单内的 `read_file`。完整问题和标准答案见文末附录。

任务素材来自 Hermes 的 MIT 许可代码，不是维护者私有的生产轨迹。
长日志和构建输出采用确定性的原文开头与匹配行附近视图，不包含上游工作区的全部字节，也不添加噪声或替换标识符。
分组在采集前按结构规则固定：来源快照至少 2000 字符且事实依赖证据的任务进入主组，支付来源单独固定。

| 分组 | 任务 |
| --- | --- |
| 主任务（4 个） | `payment`、`checkout_504`、`server_error`、`build_token` |
| 短来源／仅依赖问题的对照（10 个） | `backup`、`refactor`、`three_files`、`active_config`、`dependency_version`、`retry_backoff`、`meeting_notes`、`targeted_edit`、`package_versions`、`checksum_anomaly` |

### 对照方法

共 **13 组**：Full context，以及 S、H 各自搭配 Jev Top 0～5。Top 0 表示纯摘要。

| 方法 | 输入构造 |
| --- | --- |
| Full context | 原始 system、任务及采集最终报告之前的公开历史，包含工具调用与结果 |
| 通用摘要 S | Qwen 对全部证据块和原任务生成交接摘要，使用 [`cases.py`](./cases.py) 的提示词 |
| Hermes 上下文压缩摘要 H（小预算版） | 同一 Qwen 模型通过 Hermes 原生摘要生成流程，对证据历史生成摘要，以原任务为关注主题 |
| S／H + Jev Top N | 保留对应摘要的全部正文，再追加排名前 N 的完整原文证据块 |

S、H 各自在 Top 0～5 中逐字不变；同一任务／轮次／N 在两种机制下使用完全相同的 Jev 排名与原文块。
S 输入的是整理后的证据块，H 输入的是包含工具调用与结果的公开历史，因此两者不只是提示词不同。
压缩组使用统一的 `Recorded evidence:`／`Summary:`／`Original evidence:` 封装，保留原始 system 和任务，最后追加同一评测问题。
内部元数据、thinking 和采集结束时的最终报告不进入对照；评测问题与标准答案不进入采集、摘要或决策请求。

H 实际调用 `agent.context_compressor.ContextCompressor._generate_summary`，采用 `tail_mode=legacy`，
复用原生序列化、结构化检查点模板、敏感信息遮蔽和来源校验。实验覆盖摘要预算并附加总字数约束，
保留 Historical Task Snapshot／无用户消息标记和 Critical Context，允许省略其他章节；
用 Hermes 辅助函数移除控制前缀，不改写摘要事实，也不使用 lean 模式的额外会话日志与附录。
它是 **Hermes 上下文压缩摘要的小预算设置**：直接调用生成器，不调用完整的 `compress()`，
自动触发、首尾窗口选择和压缩后的会话修复均不在本次比较范围内。纯 H 必须实际缩短输入，不接受原文回退。

### Jev 如何选择原文

证据单元是**工具结果正文的块**，记录块 ID、工具参数中提取的文件路径、原始位置和正文；
长结果按相关事件或文本片段分块。Top N 不是 N 次完整工具调用，单元不携带完整工具名和全部调用参数。
实现见 [`cases.py`](./cases.py) 的 `event_units` 和 [`run.py`](./run.py) 的 `decision_input`。

Jev 接收原任务与 `evidence_units`（每块的 `id`、`file`、`text`，**包含工具结果正文**），
对每块回答一个“是否包含继续原任务所需的重要信息”的二元概率问题。排名不以摘要或评测问题为条件。
每次请求最多放 10 块，这是实验代码的批大小，并非服务固定限制。
本次三轮排名共保存 54 次 Jev 请求：13 个任务每轮各 1 次，41 块的 `server_error` 每轮 5 次（10/10/10/10/1）。
完整排名生成后在两种摘要和全部 N 中复用，Top N 不决定请求次数。

按概率降序严格取前 N 块，同分按 ID 排序，再恢复来源顺序。不设阈值、不跳过候选、不截断块。
实际数量为 `min(N, available units)`，记录为 `actual_n`。原始支付轨迹有 10 块，最多 Top 10；本次统一测试 Top 1～5。

### 预算、重复与评分

- 每个任务独立请求 3 轮摘要／排名，每个上下文回答 2 次，共 **14 × 3 × 13 = 546 个上下文、1092 次回答**。
  主任务每种方法有 24 次回答，对照有 60 次；重复回答不等于独立任务。
- 摘要正文上限为 `max(64, min(256, int(0.2 * evidence_tokens)))`。S 最多 3 次长度检查尝试，
  字数目标为 `max(12, int(allowance * 0.4 * 0.8**attempt))`。
- H 同样最多 3 次长度检查尝试，请求预算为 `max(12, int(allowance * 0.8**attempt))`，
  字数上限为 `max(18, int(requested_budget * 0.35))`。长度调整不使用回答反馈。
- 回答前按受保护提示、最大的五个完整块、摘要预算及 96 tokens 预留／封装空间计算完整输入上限；
  若序列化后的 S 需要更多空间，则先提高上限。它是逐步增加原文的比较，**不是等长比较，也不是统一的 1000／2000-token 上限**，
  不保证每个 Full 都超过上限。计算见 [`suite_run.py`](./suite_run.py)。

| 任务 | 证据块数 | 摘要正文上限 | 完整输入上限 |
| --- | ---: | ---: | ---: |
| payment | 10 | 256 | 4,496 |
| checkout_504 | 5 | 256 | 5,246 |
| server_error | 41 | 256 | 3,185 |
| build_token | 8 | 256 | 3,859 |

回答必须返回 `{"facts": {...}}`，按预设字段递归检查精确 JSON 类型和值：整数不同于布尔值、浮点数，
字符串和有序数组须精确一致；仅支付任务的 `no_retry_http_statuses` 忽略数组顺序。
缺失、格式错误及计划中的失败条件记零分并保留在分母中。评分后不添加答案别名、改写或重试，**不使用 LLM judge**。
先平均回答内字段，再平均同一轮的回答、同一任务的轮次，最后对任务等权平均；主任务与对照分开报告。
另保存全部字段均正确的比例。规则见 [`suite.py`](./suite.py)。

图表的回答输入为 Qwen API 实际返回的 `prompt_tokens`，包含 system／任务、历史和评测问题，
不包含摘要生成和 Jev 排名请求。预算检查使用本地 tokenizer 代理，二者不能混用。

## 结果与结论

### 四个主任务

![两种摘要机制的字段正确率与回答输入量](../../docs/figure/compaction-topn-results.svg)

| 方法 | 字段正确率 | Qwen 回答输入 tokens |
| --- | ---: | ---: |
| Full context | 100.0% | 4,014 |
| S | 82.0% | 2,480 |
| S + Jev Top 1 | 83.0% | 2,753 |
| S + Jev Top 2 | 94.5% | 2,974 |
| S + Jev Top 3 | 99.5% | 3,241 |
| S + Jev Top 4 | 98.1% | 3,496 |
| S + Jev Top 5 | 97.4% | 3,660 |
| H（Hermes，小预算版） | 79.4% | 2,432 |
| H + Jev Top 1 | 85.3% | 2,705 |
| H + Jev Top 2 | 92.9% | 2,927 |
| H + Jev Top 3 | 97.9% | 3,193 |
| H + Jev Top 4 | 96.9% | 3,448 |
| H + Jev Top 5 | 96.9% | 3,613 |

在这四个任务中，Top 3 将 S 从 **82.0% 提升至 99.5%**，将 H 从 **79.4% 提升至 97.9%**；
平均回答输入相对 Full 分别减少 **19.3%／20.4%**。因此，少量原文能补充两种摘要遗漏的精确证据；
Top 4～5 的点估计没有进一步提高，追加更多原文不保证更准确。

所有条件满足各自输入上限，42 个纯 H 上下文全部短于 Full。但 Top 3～5 每种方法只有 **9/12 个主任务上下文**
短于配对 Full。全部 H 融合组有 172/210 个上下文缩短（主任务 51/60、对照 121/150）；其余条件仍保留评分。
**平均缩短不代表每个样例都缩短。**

本结果是少量合成模板上的信息保留率诊断。支付任务已用于早期开发，不是独立留出验证；
没有等长的更长摘要或其他选择器对照，不能把全部提升归因于 Jev 排名，也不能据此宣称通用最优 N、
与 Full 统计等价或实际 agent 任务成功率提高。

<details>
<summary>十个短来源／仅依赖问题的对照</summary>

| 方法 | 字段正确率 | Qwen 回答输入 tokens |
| --- | ---: | ---: |
| Full context | 100.0% | 2,727 |
| S | 67.1% | 2,315 |
| S + Jev Top 1 | 84.7% | 2,493 |
| S + Jev Top 2 | 87.5% | 2,573 |
| S + Jev Top 3 | 92.9% | 2,687 |
| S + Jev Top 4 | 92.9% | 2,722 |
| S + Jev Top 5 | 92.5% | 2,722 |
| H（Hermes，小预算版） | 73.2% | 2,308 |
| H + Jev Top 1 | 80.1% | 2,486 |
| H + Jev Top 2 | 92.5% | 2,566 |
| H + Jev Top 3 | 99.2% | 2,680 |
| H + Jev Top 4 | 99.2% | 2,715 |
| H + Jev Top 5 | 98.8% | 2,715 |

这些不是额外十个长上下文任务。可用块较少时，Top N 会达到数量上限。

</details>

## 相关 Hermes Agent 实验

[Hermes Agent PR #116246](https://github.com/NousResearch/hermes-agent/pull/116246) 报告了另一种实验：
`fast-jev-compaction` 用 Jev 对工具调用／结果的保留决策替代摘要，同时完整保留用户和助手文本。
作者在三个真实会话谱系的 500K-token 前缀上，各使用 15 道召回题，报告结果如下：

| 方法 | 平均召回率 | 保留 token |
| --- | ---: | ---: |
| 仅摘要 | 43.3% | 55K |
| 摘要＋一轮 `session_search` 恢复 | 78.9% | 55K |
| Jev，插件默认设置 | 75.5% | 115K |
| Jev 排序／时间排序，相同 60K 工具证据预算 | 77.8% / 77.8% | 176K / 177K |

默认阈值下，851 个候选工具调用全部被删除；因此相对仅摘要的闭卷收益来自保留用户和助手文本，
而非有选择地保留工具证据。连续压缩模拟还发现，不断增长的文本下限最终会阻止进一步释放空间。
作者不建议将该策略作为 Hermes 的主要压缩机制；合并的是评测代码，并非在生产路径中采用该方案。

本实验保留摘要，再追加选出的原文，并向 Jev 提供工具结果正文。
它使用小规模合成任务、小预算摘要设置和精确字段评分，不包含历史恢复或连续压缩。
本实验得分不能与 Hermes 的召回百分比直接比较，也不能证明优于 Hermes 生产环境的摘要＋恢复路径。
两项实验考察不同策略；本实验尚缺少同预算选择器对照，因此也不能把收益全部归因于 Jev 排序。

## 复现

### 模型与运行环境

| 组件 | 版本／模型 |
| --- | --- |
| Hermes Agent `AIAgent` | commit `30463589eb54fa0f48724594cc0e98e6b995729b` |
| 采集、S／H 生成、回答模型 | `qwen3.8-max` |
| Jev（经 AgentScope／TypeSafe） | 响应解析版本 `jev-1.13.0` |
| 实验实现快照 | ReMe commit `50c2c806446c7ae6acc66217afd6e5edc2114046` |

Qwen 请求和响应只提供模型别名，没有不可变的服务端权重版本；即使温度为 0，服务更新与非确定性也可能改变结果。
本次使用中国北京地域的公开 DashScope OpenAI 兼容 Chat Completions 接口；接近原运行需使用相同的 `LLM_BASE_URL` 服务。

<details>
<summary>依赖版本与请求参数</summary>

使用两个独立 Python 环境。下表版本是**实验结束后从保留环境中核实的版本**，不是运行前固定的锁文件；
可作为复现参考，但不能证明环境已被完整锁定。本地使用 Python **3.13.3**、macOS arm64；推理由外部服务执行，不使用本地 GPU。

| 包 | ReMe 评测环境 | Hermes 采集／摘要环境 |
| --- | --- | --- |
| `openai` | 2.53.0 | 3.31.0 |
| `agentscope` | 2.0.10 | — |
| `typesafe-sdk` | 0.7.5 | — |
| `httpx` | 0.28.1 | 0.28.1 |
| `httpx2` | — | 2.13.1 |
| `python-dotenv` | 1.2.2 | 1.2.4 |
| `pydantic` | 2.13.4 | 2.14.1 |
| `tokenizers` | 0.23.3 | 0.23.3 |

`tokenizers` 来自 `PYTHONPATH` 中的独立包目录，未安装进这两个 venv。新环境可分别安装该精确版本。
本地产物包包含实验结束后记录的两个环境的发行包元数据，不包含凭证或虚拟环境二进制文件。

预算计数使用 `Qwen/Qwen3.5-9B` 的 **`tokenizer.json`**，revision 为
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`，SHA-256 为
`5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`。
这是**预算估算代理**，不表示 Qwen 3.8 使用该 tokenizer；下游输入的报告值来自 API 的实际 usage。

#### 请求参数

| 阶段 | 采样／输出参数 | 执行限制 |
| --- | --- | --- |
| Hermes 采集 | `max_tokens=1024`；`reasoning_effort=none`、`enable_thinking=false`；未显式设置温度或 seed | 最多 12 次 agent 迭代，运行预算 240 秒；4 个 worker；子进程超时 360 秒 |
| S 生成 | `temperature=0`、`max_tokens=4096`；关闭 thinking；无 seed | OpenAI 客户端超时 90 秒，`max_retries=1` |
| H 生成 | `temperature=0`；未显式设置 API `max_tokens`；关闭 thinking；无 seed | 同样为 90 秒客户端超时、1 次 SDK 重试；4 个 worker；子进程超时 600 秒 |
| Jev 排名 | 每个证据块一个二元概率问题，每批 10 块 | 适配器超时 45 秒，`max_retries=2`，重试间隔 0.5 秒 |
| 回答 | `temperature=0`、`max_tokens=2400`、`response_format={"type":"json_object"}`；关闭 thinking；无 seed | 客户端超时 90 秒，`max_retries=1`；4 种条件并发，每种依次回答 2 次 |

直接 Qwen 请求通过 `reasoning_effort="none"` 和 `extra_body={"enable_thinking":false}` 关闭 thinking。
采集阶段的实际网络请求在顶层携带 `enable_thinking=false`。H 的 API 输出限制与摘要正文上限不同：
是否接受摘要由本地正文 token 检查决定。实际运行的请求参数以保存的请求正文为准。

采集仅开放白名单内的 `read_file`，来源工作区只读；关闭自动压缩与 micro compaction，跳过记忆文件、
上下文文件和后台 review。每个 worker 使用独立 `HERMES_HOME`、工作区和会话数据库。
实现见 [`hermes_worker.py`](./hermes_worker.py)。

</details>

### 重新采集并运行

使用 ReMe 开发环境；采集和 H 生成另需 Hermes Python 环境，两个环境都要能导入 `tokenizers`。
`.env` 需配置 `LLM_API_KEY`、`LLM_BASE_URL`、`DECISION_MODEL_API_KEY`，可选 `DECISION_MODEL_BASE_URL`；
服务支持时设置 `DECISION_MODEL_NAME=jev-1.13.0`。全部命令需显式 `--live`，使用仓库外的新输出目录，不覆盖已有运行。

```bash
# 固定全部任务与标准答案，再采集 13 条新的只读轨迹。
# 复用固定的原始支付轨迹，作为第 14 个来源。
python -m benchmark.compaction.suite_collect --live \
  --hermes /path/to/hermes-agent \
  --hermes-python /path/to/hermes-environment/bin/python \
  --payment-trajectory /path/to/original/trajectory.full.json \
  --output /path/outside/repository/corpus

# 固定摘要 S 和 Jev 排名，再运行普通摘要的七组 Top-N 对照。
python -m benchmark.compaction.suite_run --live \
  --corpus /path/outside/repository/corpus/corpus.json \
  --tokenizer /path/to/cached/qwen/tokenizer.json \
  --output /path/outside/repository/evaluation

# 生成小预算设置下的 Hermes 上下文压缩摘要。
# 若 --native 已存在，先校验来源，再复用其中固定摘要。
python -m benchmark.compaction.suite_native --live \
  --corpus /path/outside/repository/corpus/corpus.json \
  --evaluation /path/outside/repository/evaluation \
  --native /path/outside/repository/native-generations \
  --hermes /path/to/hermes-agent \
  --hermes-python /path/to/hermes-environment/bin/python \
  --tokenizer /path/to/cached/qwen/tokenizer.json \
  --output /path/outside/repository/combined-results

# 复用全部固定摘要、排名和 672 次回答，仅补充 H + Top 1～5。
python -m benchmark.compaction.suite_fusion --live \
  --corpus /path/outside/repository/corpus/corpus.json \
  --evaluation /path/outside/repository/evaluation \
  --combined /path/outside/repository/combined-results \
  --native /path/outside/repository/native-generations \
  --tokenizer /path/to/cached/qwen/tokenizer.json \
  --output /path/outside/repository/dual-summary-results
```

这复现的是实验协议，新 agent 调用可能产生不同轨迹。`suite_fusion` 只重新回答 H + Top 1～5，保留其他八组，
不会重新回答全部十三组。复用本次已有 H 缓存时，向 `suite_native` 或 `suite_fusion` 额外传入
`--native-evaluation /path/to/pre-amendment/evaluation`。
两个程序均要求 `generation/validation.json` 中的 SHA-256 与生成摘要时使用的来源文件一致，且三次摘要均完整。
默认来源为当前评测的 `source.json`；`--native-evaluation` 显式指定原始来源。
仅允许当前来源中的评分用 `exam` 不同，摘要生成器不接收该字段。
缺少来源记录、哈希不匹配或证据发生变化时，程序会在回答调用前拒绝缓存。
请保留被拒绝的缓存，使用新的 `--native` 路径重新生成。

### 冻结结果离线复核

公开 Git 包含实验代码、任务定义、规则与整理后的结果，**不提交**原始支付轨迹、冻结语料及实际 API 记录。
精确复核需要本地产物包，目前不属于公开的一条命令即可复现的实验。不能替换为另一条支付轨迹，预期 SHA-256 为
`0f37e6510ffafd920d088452d8719026b00fd4e74f87f7dd1a42a85c8287e6f1`。

产物包位于 Git 仓库外：
`/Users/yuli/workspace/trajectory-runs/compaction-dual-summary-reproduction-20261016.tar.gz`。
其 SHA-256 为 `40ee93c9117dfe59cf0b4dc8887ba4b2c62fa16f3bc97f2dac6d536343f942a3`。
其中含 3187 个文件、相对路径 SHA-256 清单、环境元数据和 `verify.py`。
安装 ReMe 开发环境后，在 ReMe 仓库中运行：

```bash
# 在仓库外解压，替换为实际存放归档文件的目录。
export REPRO_DIR=/path/outside/repository
shasum -a 256 "$REPRO_DIR/compaction-dual-summary-reproduction-20261016.tar.gz"
tar -xzf "$REPRO_DIR/compaction-dual-summary-reproduction-20261016.tar.gz" -C "$REPRO_DIR"
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python \
  "$REPRO_DIR/compaction-dual-summary-reproduction-20261016/verify.py"
```

预期结果：**核验 3187 个产物哈希、546 个上下文和 1092 次评分；零 API 调用**。
产物包内的 `corpus/`、`evaluation/`、`native/`、`native-source/`、`combined/`、`final/` 对应
上述运行阶段。复用其中 H 摘要时传入 `--native-evaluation <bundle>/native-source`。

## 附录

<a id="queries-and-expected-answers"></a>

<details>
<summary>完整 query 与标准答案</summary>

初始 query 提供采集目标，后续问题要求返回下表的 JSON 字段；一个任务加一个后续问题仍是一个样本。
中文 query 为释义，英文原文见[英文版](./README.md#queries-and-expected-answers)；实际只读改写及字段说明以
[`suite.py`](./suite.py) 和冻结请求为准。标准答案的键、类型和英文字符串保持原样。

| ID | 初始 query 的中文释义 | 标准事实 | 分组 |
| --- | --- | --- | --- |
| payment | 调查原始支付重试事件。 | `{"incident_id": "PAY-4827", "invoice_id": "INV-731", "successful_charge_id": "CH-991", "amount_minor": 12900, "attempt_keys": ["pay-a", "pay-b"], "first_attempt_committed": true, "max_attempts": 3, "no_retry_http_statuses": [402, 422], "key_generation_line": 5, "regression_attempt_counts": {"timeout_then_success": 2, "validation_422": 1, "three_timeouts": 3}}` | 主任务 |
| backup | 从历史对话确认哪个仓库需要每晚备份，不要猜测；在工作区编写 backup.sh，将 repos/ 下该仓库归档，安排每天凌晨 2 点运行，用待办清单跟踪，并报告仓库与设置。 | `{"repo": "sparks-data", "ignored_repo": "toybox", "data_text": "important"}` | 对照：短来源 |
| refactor | 把 src/ 中所有 requests 调用的 timeout=30 改为 timeout=45，在 CHANGELOG.md 记录变更，并报告实际修改的调用点数量。 | `{"old_timeout": 30, "call_sites": 3, "functions": ["fetch_a", "fetch_b", "fetch_c"], "methods": ["get", "post", "head"], "old_version": "1.0"}` | 对照：短来源 |
| three_files | 用待办清单跟踪并核验：创建含 greet(name)、返回 `'hello <name>'` 的 greet.py；创建表头为 item,qty、含两行 widgets 数据的 data/rows.csv；创建说明新建文件数量的 notes/summary.md。完成并检查三项后再结束。 | `{"file_count": 3, "header": "item,qty", "data_rows": 2}` | 对照：仅依赖问题 |
| active_config | 在工作区查找 retry_limit 配置，并报告当前值。 | `{"retry_limit": 7, "backoff": 2, "active_file": "conf/app.ini", "old_file": "conf/old.ini"}` | 对照：短来源 |
| checkout_504 | 调查 checkout 504 事件，区分未奏效的回滚、根因及最终数据库修复。 | `{"setting": "statement_timeout", "value_seconds": 45, "index_mode": "CONCURRENTLY", "root_cause": "CREATE INDEX without CONCURRENTLY", "rolled_back_version": "v2.30.2", "rollback_fixed": false, "deployed_version": "v2.31.0", "initial_error_percent": 8}` | 主任务 |
| dependency_version | 确认固定的 left-pad 精确版本，以及哪个文件声明测试脚本；为任务交接记录 package 脚本。 | `{"version": "1.3.0", "declaration": "package.json", "test": "vitest run", "build": "tsc -p .", "package_version": "4.2.1"}` | 对照：短来源 |
| retry_backoff | 查找 src/ 中的重试延迟函数，解释基数、增长方式、上限与随机抖动。 | `{"base_ms": 250, "cap_ms": 30000, "growth_factor": 2, "jitter": "uniform_integer_below_bound", "function": "retryDelay"}` | 对照：短来源 |
| server_error | 定位服务器日志中唯一的 ERROR，保留精确请求标识、时间戳、上游和连接失败信息。 | `{"request_id": "req-7f3d9", "timestamp": "2026-08-08T23:41:17Z", "status": 502, "upstream": "payments", "error": "ECONNREFUSED", "address": "10.0.4.17:8443"}` | 主任务 |
| meeting_notes | 读取会议记录，准确报告第三条要点，并保留其他已商定行动。 | `{"third": "rotate the API keys quarterly", "first": "ship the payments retry fix", "second": "audit the staging TLS certs", "fourth": "close out the Q3 incident review"}` | 对照：短来源 |
| targeted_edit | 在 ./handlers.py 的三处 timeout = 10 中，仅将 slow_handler 内的一处用 patch 工具改为 timeout = 60；随后在当前目录运行 python3 check_handlers.py 并回复输出。 | `{"target": "slow_handler", "original_timeouts": {"fast_handler": 10, "slow_handler": 10, "medium_handler": 10}, "required_timeouts": {"fast_handler": 10, "slow_handler": 60, "medium_handler": 10}, "success_marker": "HANDLERS_OK_552"}` | 对照：短来源 |
| build_token | 运行输出很长的 python3 ./noisy_build.py，找出中间唯一以 UNIQUE_TOKEN= 开头的行，回复完整 token 值。 | `{"token": "tok_9f31c_middle", "step": 2000, "total_steps": 4000, "prefix": "UNIQUE_TOKEN="}` | 主任务 |
| package_versions | 依次读取 pkg_a、pkg_b、pkg_c 中的 version.txt，将 a,b,c 顺序的版本以逗号连接写入 ./versions.txt，回复 DONE 和连接后的字符串。 | `{"versions": {"pkg_a": "1.4.2", "pkg_b": "0.9.7", "pkg_c": "3.2.1"}, "joined": "1.4.2,0.9.7,3.2.1"}` | 对照：短来源 |
| checksum_anomaly | 用 read_file 而非 terminal 查找 ./records.log 中唯一以 ANOMALY: 开头的行，并回复该行全文。 | `{"line": "ANOMALY: checksum drift detected in shard 7 (code X99Q)", "code": "X99Q", "shard": 7, "line_number": 4201}` | 对照：短来源 |

</details>

<details>
<summary>协议修正、审计与历史产物</summary>

初始七组的 588 次回答暴露两处格式说明歧义：支付测试次数返回为数组，会议要点带列表标记。
标准答案和评分器不变，仅明确支付次数须用指定键的 JSON 映射、会议要点不带列表标记。
两任务的七种方法使用原有 S／排名统一重答，初始 84 次作为预检排除；其他 12 个任务保留 504 次回答。
随后纯 H 新增 84 次，H 融合新增 420 次，组成当前最终结果；全部 42 份 H、排名及已有 672 次回答被复用。
分组、证据、答案值均未因结果调整。

H 首次强制生成包含长控制前缀，40/42 个输入超过 Full，仅作为集成预检保留。
正文封装和字数约束在新的评分回答前固定，不改动 S、标准答案、分组或 Jev 排名。
H 使用格式修正前的来源文件，两版本只有不发送给生成器的 `exam` 字段不同。
支付任务早期 `budget.json` 为 4459；最终问题增加 37 tokens，最终 `results.json` 的 4496 是评分上限。
冻结文件保留原貌，不回写修改。

离线审计核验来源哈希、固定 S／H、相同且嵌套的 Top-N 原文、实际输入、全部评分、输入上限、
已有结果未变与配置密钥未泄露；全部 546 个条件有效。原始文件、会话、API 请求／响应、代码快照和逐字段评分不进入 Git。
历史目录位于 `/Users/yuli/workspace/trajectory-runs/`：
`hermes-fixture14-20261016/`（语料）、`compaction-fixture14-20261016-v2/`（S／排名）、
`compaction-fixture14-hermes-adapted-20261016/`（H）、`compaction-fixture14-final-20261016/`（原八组），
`compaction-fixture14-dual-summary-20261016/`（最终十三组）。

</details>
