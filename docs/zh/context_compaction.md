# 基于决策模型的上下文压缩

`compact_context` 为旧工具结果生成可恢复的预览建议：判断证据对当前任务是否仍有用，保留工具调用和不确定的证据。
原始历史由宿主持有。流程使用通用 `as_decision_model` 接口，当前 `agentscope` 适配器连接 TypeSafe/Jev。

![证据观测、保守选择、预算检查和原文恢复](../figure/decision-context-compaction.svg)

## 一次压缩如何工作

1. **宿主先归档。** 提供 AgentScope 2.x JSON 消息、`goal`、可选的 `next_action`、只读工具允许列表，以及
   `source_refs`：原始调用 ID 到归档来源的映射。引用是宿主定义的标识；ReMe 不打开、验证或创建归档。
   宿主必须在应用建议前保存原始历史，并提供能根据引用取回原文的工具。没有引用的配对不参与压缩。
2. **确定性保护。** 按唯一 ID 配对 `tool_call` 和 `tool_result`。第一条、system、最近和显式保护的消息及调用保留。
   未完成或失败的调用、非文本结果、非法输入、工具名不一致和不明确的配对保留。`eligible_tool_names` 默认为空。
3. **展示有界证据。** 每批只携带本批候选的工具名、输入、结果头尾片段、字符数和完整性标记，同时携带目标、
   下一步行动和有限的首条、system、保护及近期文本。thinking 不发送。预览可能包含源码等数据，宿主应在推理前处理敏感内容。
4. **每对一个问题。** 判断“这份证据是否包含继续正确执行任务所需的信息”，不再判断能否重新获取。
   截断或缺失信息代表不确定。单个证据放不进请求预算时保留，不移除观测内容后继续盲判。
5. **只替换明确无关的结果。** 概率 `<= drop_threshold`（默认 `0.1`）时，允许替换为原文头尾片段和恢复引用。
   更高概率、缺失或非法答案保留原文。供应商错误和超时保留受影响证据；外部取消在请求清理后向上传播。

工具调用和消息都不删除。新结果只有更短时才替换。用户与 assistant 文本、thinking、元数据和输入原文不改变。
不可丢失的约束、审计回执和关键证据应显式保护，不能只依赖概率。默认阈值是保守起点，不代表已经校准。

## 配置与调用

```bash
pip install "reme-ai[core,decision]"
export DECISION_MODEL_API_KEY=your-typesafe-api-key
export DECISION_MODEL_BACKEND=agentscope
# 可选：DECISION_MODEL_NAME、DECISION_MODEL_BASE_URL
```

CLI 也从 `.env` 读取 `DECISION_MODEL_*`，详见[决策模型配置](./configuration.md#决策模型配置)。
省略模型名或配置为 `null` 时使用 SDK 默认模型。

在一个终端启动服务：

```bash
reme start config=decision_compaction
```

在另一个终端调用，引用应已由宿主归档并能够解析：

```bash
reme compact_context \
  goal='修复支付重试' next_action='检查重试实现' \
  eligible_tool_names='["search_code"]' source_refs='{"c1":"archive/demo/c1"}' \
  options='{"preserve_recent_messages":1}' \
  messages='[{"role":"user","content":[{"type":"text","text":"修复重试"}]},{"role":"assistant","content":[{"type":"tool_call","id":"c1","name":"search_code","input":"{\"path\":\"icons\"}","state":"finished"},{"type":"tool_result","id":"c1","name":"search_code","output":"旧图标搜索结果","state":"success"}]},{"role":"user","content":[{"type":"text","text":"接下来检查支付代码"}]}]'
```

短结果可能因为恢复标记反而更长而保持不变。配置继承默认服务和后台 Job；嵌入宿主时，可参考
[集成测试](https://github.com/agentscope-ai/ReMe/blob/main/tests/integration/test_decision_compaction.py)只装配压缩 Job 与决策组件。

## 主模型 token 预算

决策请求的估算预算与压缩后主模型输入预算独立。使用 `max_output_tokens` 时，宿主必须在构造 Step 时注入同步函数
`token_counter(messages)`，按实际模型输入格式计算 token，包含 system、工具定义和消息封装。函数必须返回非负整数。
缺少计数函数时在推理前报错。ReMe 的检索分词器不能代替模型 token 计数器。

```python
step = CompactContextStep(as_decision_model=decision_model, token_counter=count_model_input)
response = await step(
    messages=snapshot,
    goal="修复重试",
    next_action="检查补丁",
    source_refs=archive_refs,
    eligible_tool_names=["read_file"],
    options={"max_output_tokens": 32000},
)
```

已经满足预算时不调用模型。否则只在低概率候选内按概率从低到高替换，满足预算即停止；替换必须同时减少文本大小和
宿主 token 计数。不会为了满足预算删除必要或不确定的证据。仍然超限时返回 `reason="budget_exceeded"`、
`budget_met=false`，宿主应先执行摘要等后续策略。未指定预算时处理所有符合条件的低概率结果，`budget_met=null`，
由宿主自行检查最终输入。

## 返回值与默认配置

标准 Response 的 `answer` 包含 `schema_version=2`、建议 `messages`、`changed`、`reason`、`budget_met`、
`decisions` 和 `stats`。每个决策记录原始 `call_id`、内部 `id`、动作 `keep` / `preview`、`probability`、
`reason` 和 `source_ref`。原因区分保护、不确定、未观测、非法答案、供应商错误、超时和没有节省。
统计包括前后大小及宿主 token 数、候选和保护数量、最大批次状态估算 token、请求数、usage 和耗时，也写入响应元数据。

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `drop_threshold` | `0.1` | 只有不高于此概率的结果允许替换为预览 |
| `preserve_recent_messages` | `6` | 保护最近的消息数 |
| `evidence_chars` | `2000` | 每个输入、结果预览或近期文本的字符上限 |
| `preview_chars` | `300` | 建议结果保留的头尾片段字符上限，不含恢复标记 |
| `max_state_tokens` | `25000` | 决策状态的估算上限 |
| `max_request_tokens` | `30000` | 状态和问题的估算上限，含 256 token 额外预留 |
| `max_output_tokens` | `null` | 可选的宿主计数预算 |
| `max_concurrency` | `4` | 最大并发请求数 |
| `total_timeout_seconds` | `30` | 包含排队的总体请求超时 |

## 验证

两个集成场景分别模拟支付重试排障和副本延迟调查，让重要源码、日志与已经结束的无关任务竞争。
配置凭证后显式运行：

```bash
REME_RUN_DECISION_INTEGRATION=1 pytest tests/integration/test_decision_compaction.py -v -s
```

只有模型请求调用外部服务，历史中的工具不会执行。这些用例验证行为，不证明通用任务成功率。
真实 trajectory 评测应对比近期保留和纯规则预览，同时测量恢复次数、继续执行的质量、时延和 token 节省。
ReMe 不会自动生成摘要或写入长期记忆。

## 两种摘要机制下的 Jev 增益

### 实验背景

摘要可以缩短历史，但可能丢失后续回答需要的精确标识符、配置值或日志细节。本实验考察：**在摘要保持不变的基础上，
追加 Jev 选出的少量原文，能否补回这些信息？这一增益能否同时出现在普通摘要和 Hermes 摘要中？**
同时报告字段正确率和 Qwen 回答输入量，观察信息保留与上下文长度之间的取舍。

### 实验设置

| 项目 | 本次设置 |
| --- | --- |
| 数据 | 14 个预设 query/answer 的合成任务模板：1 条已有支付轨迹，另用 Hermes 实际采集 13 条只读轨迹。按来源长度预先确定 4 个主任务：支付重试、checkout 504、服务器错误定位、构建日志 token 查找；10 个短任务／query-only 对照单独报告。 |
| 框架与模型 | Hermes Agent `AIAgent`，版本固定为 `30463589eb54`；采集、两种摘要和最终回答均使用 **`qwen3.8-max`**。Jev 为 **`jev-1.13.0`**，通过 AgentScope **2.0.10** 调用。采集使用 Hermes，评分阶段直接调用 Qwen。 |
| 两种摘要 | **S**：通用任务交接摘要提示词。**H**：Hermes 上下文压缩摘要（小预算版），复用其 `legacy` 摘要生成流程，调整预算和长度约束；实验直接生成摘要，不经过自动触发和首尾窗口选择。 |
| 13 个对照组 | **Full context** 保留完整的评测前历史；**S + Top 0～5** 和 **H + Top 0～5** 各 6 组。Top 0 是纯摘要；Top N 在原摘要后追加 N 个完整证据块。两条曲线使用相同 Jev 排名和原文，每轮的 S／H 均逐字固定。 |
| 压缩预算 | 主动压缩的小预算模拟：摘要正文上限按证据长度设为 **64～256 tokens**；完整回答输入上限按任务计算。N 增大时保留更多原文，实际输入随之变化，各组并非等长比较；Full 是无压缩参考，不要求每个任务都超过该上限。 |
| 评测 | 原任务和证据用于摘要／排序，评测问题与标准答案不提供给它们；最终采集报告也排除。回答需返回 JSON，按预设字段的严格类型和值规则评分，**不用 LLM judge**。分数先在任务内平均，再对任务等权平均。 |
| 重复与规模 | 每个任务 3 轮摘要／排名，每个上下文回答 2 次；共 **546 个上下文、1092 次回答**。主图只展示 4 个主任务，每种方法 24 次回答；重复回答不等于新增任务。摘要和回答温度为 0，关闭 thinking。 |

Jev 先依据原任务给全部证据块打分，再选择 Top N；N 只影响保留数量，不增加排名调用次数。
这是预设标准答案的**信息保留评测**，没有让 Hermes 在压缩后继续调用工具完成任务。

### 读图与结论

![Answer quality and downstream input across two summary strategies](../figure/compaction-topn-results.svg)

- **A：两种摘要都能从追加原文中获益。** Full 的字段正确率为 100%；Top 3 将 S 从 **82.0% 提升至 99.5%**，
  将 H 从 **79.4% 提升至 97.9%**。这支持“摘要 + 关键原文”在本组任务中补充摘要遗漏的信息，尚不能证明通用优势或与 Full 统计等价。
- **B：以更多上下文换取更高的信息保留率。** Top 3 的平均回答输入为 S **3241**、H **3193 tokens**，
  相比 Full 的 **4014 tokens** 分别减少 **19.3%／20.4%**，但都高于各自纯摘要。Top 4～5 未继续提分，更多原文不保证效果更好。

**结果边界：** 主结论来自 4 个合成任务模板；10 个短任务对照不混入主图。纯 H 的 42 个上下文均短于 Full，
但 Top 3～5 每种方法仅有 **9/12 个主任务上下文**实际缩短，全部样例均保留评分。严格字段评分包含格式要求，语义相同但格式不同也可能失分。
当前结果展示了信息保留与长度的取舍；尚未加入等长度的更长摘要或其他选择器对照，不能将全部增益归因于 Jev 排序本身。
agent 继续执行的成功率尚未验证。

<!-- 支付轨迹有 10 个证据块，最多 Top 10；实际 N 为 min(N, 可用证据块数)。 -->
[完整实验：设计、结果、标准答案与复现](https://github.com/jinliyl/ReMe/blob/74e1251ddcd212205788f21ccd6b5ebd82ade7c3/benchmark/compaction/README_ZH.md)

## 注入宿主管理的决策模型

嵌入式宿主可以复用已有的 AgentScope JEV 实例，与注入 LLM 的方式一致：

```python
# app 已配置 backend: agentscope 的 as_decision_model.compaction 组件。
await app.update_component("as_decision_model", "compaction", model=host_jev)
```

YAML 中的 `model` 仍表示供应商模型名称；运行时 `model` 属性接收可调用的模型实例，读取属性不会创建客户端。
在首次请求之前注入。ReMe 继续校验并转换 JSON 问题，但不再要求额外的 ReMe 凭据，也不会关闭借用的客户端。
宿主应等待 ReMe 调用结束后自行关闭实例。替换 ReMe 自建模型或存在进行中请求的实例之前，须先关闭组件。
关闭过程即使被取消，也会等待活动请求和自建客户端清理完成，再传播取消信号。组件关闭时会解除实例引用；
重新启动使用外部模型的组件前需要再次注入。独立运行时按凭据创建模型的行为保持不变。

自建客户端在线程中初始化，证书加载不会阻塞事件循环。构造期间取消请求时，组件会等待构造完成并保留客户端所有权，
由后续组件关闭负责释放；此时不会发送该请求。

宿主可从 `reme.steps.compaction` 导入 `CompactContextStep`、`ToolPair`、`collect_pairs` 和 `preview_result`。
配对识别与预览构造函数和压缩 Step 共享相同规则，供宿主校验提案使用；归档、引用解析和提交仍由宿主负责。
