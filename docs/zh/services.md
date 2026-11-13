---
title: 服务与部署
description: 使用 ReMe 的 HTTP、SSE、MCP 和 Studio 服务，并理解默认安全边界。
---

# 服务与部署

ReMe 可以作为本地 HTTP 服务、独立 MCP Server 或一次性 CLI Job 运行。默认 HTTP 服务监听
`127.0.0.1:2333`；同一进程提供 JSON API、SSE、MCP 与可选的 ReMe Studio。

## HTTP 服务

在一个终端启动服务并保持运行，以下启动命令任选一条：

```bash
reme start
reme start service.host=127.0.0.1 service.port=8181
```

主机地址也可以使用 `::1` 等 IPv6 字面量；HTTP 和 MCP client 会在连接时正确格式化该地址。
如果服务绑定到 IPv6 通配地址 `::`，ReMe 会向进程内 client 提供可连接的回环地址 `::1`。

在另一个终端通过 CLI client 调用 Job：

```bash
reme search query="用户偏好" limit=5
```

普通 Job 也暴露为 `POST /<job-name>`；以下 HTTP 示例使用默认端口 `2333`：

```bash
curl -s http://127.0.0.1:2333/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"用户偏好","limit":5}'
```

请求体允许 Job 参数位于顶层。响应遵循 `Response`：

```json
{
  "success": true,
  "answer": "...",
  "metadata": {}
}
```

未被 Step 捕获的错误会转换成 `success: false` 的响应。

## Streaming 与 SSE

`backend: stream` 的 Job 仍使用 `POST /<job-name>`，但响应类型是 `text/event-stream`。每个 chunk 使用统一的 streaming schema；失败时会发出错误 chunk，并以终止事件结束。

默认 `chat` 是 Stream Job：

```bash
curl -N http://127.0.0.1:2333/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"总结我的长期偏好"}'
```

Studio 的普通对话和 **Proactive Chat** 共用一套界面。默认配置只提供 `chat`，接收 `query` 和可选的 `session_id`，
各对话独立。使用 `config=proactive` 启动后，提供 `pa_*` Job，不再提供 `chat`。前端按服务实际开放的 Job 启用对应入口。

Proactive Chat 使用由 Application 管理当前会话的 `pa_chat`，发送时只传 `query`。`pa_session` 无参数，`answer` 返回
`{"session_id": "...", "status": "idle"}`；尚未创建会话时两个字段均为 null。`pa_new_session` 无参数，创建并切换空会话，
在 `answer` 返回新 ID，不触发模型生成。

先在一个终端运行 `reme start config=proactive`，再从另一个终端调用：

```bash
reme pa_session
reme pa_chat query="有哪些事项值得继续？"
reme pa_new_session
```

当前会话在同一 Application 内共享，重启后清空指针，会话文件保留。生成期间拒绝并发发送和新建，状态查询无需等待生成。
PA Job 目前只支持 `CodexAgentWrapper`，其他 wrapper 抛出 `NotImplementedError`。
当前原生会话不可用时，`pa_session` 返回 `status: "error"`，可通过新建会话恢复。
空会话创建一旦开始，会先完成当前会话的切换，再传播取消信号。

`proactive` 还提供只读的 `POST /pa_events`，供后台客户端订阅。服务启动后，在另一个终端调用：

```bash
curl -N http://127.0.0.1:2333/pa_events \
  -H 'Content-Type: application/json' \
  -d '{}'
```

每个 SSE `data:` 帧仍是 `StreamChunk`，其中 `chunk` 为有类型约束的
[`PaEvent`](../../reme/schema/pa_event.py)。首次连接收到 `snapshot`，包含当前会话 ID、合并后的 PA 活动状态和
可选的 `SessionSnapshot` 历史，不会创建会话。后续事件如下：

| 类型 | 客户端处理 |
| --- | --- |
| `session_changed` | 切换当前会话，清空上一会话的展示历史。 |
| `message_upsert` | 按 `(session_id, message.id)` 插入或替换完整的 `message`。 |
| `status_changed` | 更新活动状态；PA 操作释放串行锁之前持续显示忙碌。 |
| `reset` | 丢弃旧游标；同一连接随后发送新的 `snapshot`。 |
| `heartbeat` | 保持连接，不携带游标。 |

应用事件后保存其不透明的 `cursor`。重连时传 `{"cursor":"<已保存的游标>"}`，先重放缓冲中的遗漏事件，再接收实时变化。
进程或 runtime 重启、上游断开、客户端落后超过缓冲范围时，必须重新同步快照。快照读取与增量交接通过版本复核，
覆盖并发切换和消息提交。历史读取失败时，快照包含 `error`，不含历史和游标；客户端保留已有展示，服务在心跳间隔后重试，
成功取得快照后恢复增量投递。
客户端需要容忍重复 upsert；只有已完成的主动消息才进入提醒气泡队列，逐字回复继续使用 `pa_chat`。

消息事件在主动消息成功发布后，或聊天 turn 结束且原生历史可读后产生，也覆盖取消后的历史。恢复依据仍是会话文件，
事件缓冲和游标仅保存在内存中，不代表送达或已读回执。通过 `components.pa.default.max_events`、`max_buffer_bytes` 和
`heartbeat_seconds` 配置缓冲容量和心跳。`jobs.pa_events.stream_queue_size: 1` 限制 HTTP 输出队列，慢客户端通过历史
重新同步，避免无限积压。其他 Stream Job 默认行为保持不变，`stream_queue_size: 0` 表示无界队列。
自定义配置启用 `pa_events` 时，需要一并配置 `proactive.yaml` 中的 `pa` 组件；订阅跟随该组件配置的 Codex wrapper。

`pa` 组件从 `{metadata_dir}/pa/session_ids.json` 的最后一条恢复当前对话。选择会话时，先原子保存会话顺序，
再发布当前选择，同时保留读取游标。启动时将已有 `{session_dir}/pa.json` 的选择迁入列表，保存成功后移除旧文件。
原生对话历史仍归 Codex 所有，不删除历史；元数据无效或原生历史不可用时明确报错，不静默创建替代会话。
事件游标和活动状态在重启后重新创建，客户端通过快照恢复。

`proactive` 配置设置 `service.timeout_graceful_shutdown: 5`。HTTP 服务退出时，先给正在处理的请求五秒完成，
随后取消仍未结束的请求，再关闭应用组件，避免长订阅无限阻塞退出。自定义 HTTP 配置启用长订阅时，也应设置有限的
优雅退出超时。

三个 Agent 统一使用 `{workspace_dir}/{daily_dir}/personal/MEMORY.md`（默认是工作区内的 `daily/personal/MEMORY.md`）。
PA 的 Prompt 提供实际配置的工作区和该记忆路径；对话 Prompt 要求回复前先读记忆，
更新前先读取最新完整文件，再合并修改，不再将记忆正文注入会话 Prompt。新建会话或执行主动任务时，公共逻辑仅在文件
不存在时，用 `reme/steps/proactive/agents/base.yaml` 的 `default_memory` 初始化，通过逐路径锁和原子写入保护并发。
默认内容覆盖财经新闻和 AI 前沿，并要求只提醒有价值的内容，
非重要或重复信息不要打扰主人；已有文件（包括空文件）保持原样，
不再进行首次信息询问或根据空正文跳过任务。

`agents/chat.py` 提供对话和新建会话 Step，`api/sessions.py` 提供会话查询和选择；Remy 的人设、记忆维护与遗忘说明由
`PromptHandler` 从 `agents/chat.yaml` 加载，提供英文默认版本和中文版本。对外 Job 仍为 `pa_chat`、`pa_session`、`pa_new_session`。
proactive 默认 `language=zh`，可用 `language=en` 切换英文；Step 的 `language` 优先于 Application。
各角色使用简洁的中英文提示词与少量通用例子；默认模型与推理强度以 `reme/config/proactive.yaml` 为准。

`reme/steps/proactive/` 按职责组织：`agents/` 放模型执行与同名提示词，`tools/` 放 Agent 工具，
`data_connectors/` 放外部数据采集，`background/` 放调度与投递，`api/` 放会话和状态查询。
公共请求校验放在 `base.py`，跨请求状态、文件和会话归 `reme/components/pa/`；通用 JSON 校验归 `reme/steps/file_io/`。
Codex 客户端配置 1000000 token 的上下文窗口与 600000 token 的自动压缩阈值，按总 token 计量；配置须匹配所选提供方的能力。
原生长期记忆保持启用。`CODEX_MODEL_NAME` 可覆盖模型，切换型号时也应检查 model catalog 的能力描述。
配置文件变更在下次启动时加载，恢复后的会话在后续聊天中采用当前配置。
通过 Step 的 `prompt_dict` 覆盖对应语言的键（如 `system_prompt_zh`、`default_memory_zh`）；默认记忆覆盖仅影响缺失文件。
主人要求遗忘时，提示 Agent 删除对应内容，不转存，也不从旧会话或摘要中恢复；维护时清理确认过时的状态，保留有效的长期事实。
创建会话会初始化缺失的记忆，但不生成回复。每次聊天从当前配置生成 Prompt 和工具绑定，不在指针文件或内存会话状态中
缓存这些参数；wrapper 记录每次实际生效的配置，仅供后续 fork 继承，记忆路径始终指向最新文件。
路径无效或初始化失败时，新建失败并保留原来的当前会话。

智能主动 Agent 每轮从对话 Agent 当前选定的会话 fork，创建分支时就覆盖为主动 Agent 的 system prompt 和工具，
等待分支原生 compact 完成后，在同一分支最多执行一次工作 query。来源对话保持不变。
对话 Agent 尚无原生聊天轮次时创建新工作会话，跳过 fork 和 compact；读取已有会话失败仍明确报错。
这条 query 依次理解最新记忆、已完成对话和 inbox，处理明确配置要求，阅读周期结果和实际投递历史，筛选、去重、
按需调研并决定是否发送。实现和双语 Prompt 见
`reme/steps/proactive/agents/proactive.py` 及同名 YAML；不再逐步注入 query 或要求中间汇报。
调研预算仍由 Prompt 约定为最多三条线索、五次搜索；较新的明确反馈优先于继承背景和此前交接。
暂停、非活动时段、晚间收尾或配置无效时，均不启动主动模型，包括 compact；inbox 留待下次允许运行时处理。
关注事项、反馈和配置要求统一由对话 Agent 通过 `write_proactive_inbox` 排队，不触发额外模型调用。
`enabled=false` 或配置无效时，须先从外部重新启用或修复配置，排队消息才会继续处理。

提醒保留核心发现、必要背景、相关时间和已核实的原文链接，简洁完整；结合价值、新意、时机、频率及主人的回应判断
是否值得打扰，多次未回应则减少打扰。空内容是明确的静默决定。代码校验有效轮次、当前发送 query、单轮一次决定，
非空发送还须依据最新消息历史。已接受提醒继续独立后台投递，保留替换、去重、每日问候限制和晚间收尾规则。

工作 query 成功后，将最终的未完成事项摘要原子替换到 `handoff.md`，空内容清空交接；失败或中断时保留此前交接。
输入工具读取后即推进消费游标，失败后不自动重放已消费输入；源文件和原生对话历史仍可读取。
准备、compact 和执行共用活动窗口截止时间。临时任务到期后 fork 来源对话，继承其上下文、模型和工具；
周期任务复用各自会话。Runtime 负责协调，文件操作、会话消费和事件回放分别由专门对象管理。

PA 对话、会话和事件 Step 明确依赖配置中的 `PaRuntime`，不再兜底到 metadata 字典。
Runtime 管理会话生命周期及单个有类型的当前轮次，内部任务与提醒对象各自持有自己的锁；轮次结束不影响后台投递。
提醒提交必须绑定有效轮次。指令工具集中在 `tools/inbox.py`，消息工具集中在 `tools/messages.py`。
任务工具仅由 YAML 配置，默认是截图与财经采集。会话选择只从 `session_ids.json` 恢复，旧 `pa.json` 启动迁移已移除。

PA 组件在 Job 启动前创建 `{metadata_dir}/pa/` 下缺失的文件：

| 文件 | 初始内容 |
| --- | --- |
| `session_ids.json` | `[]`，最后一条表示当前对话 |
| `proactive.json` | `components.pa.default.default_proactive` |
| `cron.json` | `components.pa.default.default_tasks` |
| `proactive_inbox.md` | 空文件 |
| `proactive_inbox_receipts.jsonl` | 空的请求回执流水 |
| `temp_tasks.json` | `[]` |
| `handoff.md` | 空的主动 Agent 下轮交接 |
| `chat_inbox.jsonl` | 空文件 |
| `cron_runs.jsonl` | 空文件 |

Inbox 按接收方命名：`proactive_inbox.md` 接收对话 Agent 发给 Proactive 的指令，消费游标保存在
`proactive_inbox_cursor.json`，请求回执保存在 `proactive_inbox_receipts.jsonl`。
`chat_inbox.jsonl` 保存后台发给对话 Agent 的消息及投递状态。周期任务文档放在本 PA 目录的 `cron/{task_id}/`，
一次性任务文档放在 `temporary/{id}/`。

默认值在 `reme/config/proactive.yaml` 中配置：start_time 为字符串 "0700"、end_time 为字符串 "2330"、
interval_minutes 为 5。已有内容保留，包括空文本和空任务列表；已有 JSON 错误时明确报错，不用默认值覆盖。
每个 Agent 调用模型前通过 `agents/base.yaml` 的本地化默认内容初始化缺失的个人记忆。

唤醒使用应用时区，开始时间包含、结束时间不包含；支持跨午夜区间，开始结束相同表示全天。
活动时间内一轮完成后再等待指定间隔，活动时间外等待下次开始。等待期间最多每30秒检查 JSON 修改，无需重启；
配置错误时报告错误并暂停模型工作。旧 YAML 等待参数、内存临时覆盖和 set_next_wakeup 工具已移除。
资源清理仍使用独立 CronJob。
每轮准备前和工作 query 前重查活动时段，准备及执行过程到活动结束时间中断并保留此前交接；晚间收尾阻止后续普通轮次。轮询和规则判断不调用模型。

可选交流 hint 默认开启：在 `proactive.json` 设置 `hints_enabled: false` 可关闭抽样和注入；
`hint_probabilities` 分别配置 `morning`、`midday`、`evening`，默认依次为1.0、0.5、0.5，范围均为0到1。
初始值来自 YAML 的 `default_proactive`；已有 JSON 缺失这些字段或只配置部分时段时，其余使用默认值。
在活动时间内，早间06:00–12:00、午间12:00–14:00，以及傍晚结束前30分钟分别最多抽一次；
结束前窗口最早从18:00开始，跨午夜或全天活动则使用21:00后的晚间窗口，不把晚间 hint 投到早上。
未抽中也占用当天该时段的机会；有当前对话时才抽样，并在 `hint_state.json` 中保存结果，重启不重抽，尽量避开上次同类变体。
抽中的双语方向来自 `agents/proactive.yaml`，compact 完成后注入唯一的工作 query；Agent 可以忽略，正常的新信息提醒不受抽样影响。
共18个方向，围绕生活关心、问候和小彩蛋，只有一个晚间方向涉及已确认 TODO 的总结。
是否实际发送仍以 `chat_inbox.jsonl` 为准，随机方向不写入个人记忆，也不应作为后续轮次指令。
特殊提醒的 `kind` 为 `morning`、`midday` 或 `evening`，必须匹配本轮抽到的 hint，每个本地日期各限一次；
失败或废弃不重新取得名额，晚间收尾也不会因此重新开放，避免已注入但确认丢失时再次打扰。`normal` 用于新信息，禁止夹带明确的早安、午安、晚安问候。
任何非空 `evening` 决策都是当天收尾，即使文案不包含“晚安”。待投递期间禁止普通消息插队，发送后停止主动 Agent
到下次活动开始；抽样未命中或选择空内容不会提前收尾。收尾状态从流水推导，跨会话和重启保留，不另外保存静默状态。

对话工具在工作 query 中跨最近100个登记会话补读新增的已完成轮次，每次主人输入与其回复保持一组，排除工具和注入消息。
默认最多10轮、每条2000字符、总量20000字符；读取失败时不改变游标，游标写入完成后才响应取消。

`write_proactive_inbox(message)` 只保存带时间的 Markdown 消息，关注事项、提醒反馈和配置要求都写进 message，
由 Proactive 下次苏醒后理解，并在工作 query修改自己的配置。已排队或已接收仅表示消息已保存或读到，不代表要求已完成或配置已生效。
暂停提醒、处于非活动时段或配置无效时，inbox 请求保留到下次允许运行；禁用或无效配置需先从外部重新启用或修复。
`read_proactive_inbox()` 读取新增或修改的输入并推进内容哈希
游标，保留源文件；直接返回完整的未读 Markdown 正文，没有变化时返回空字符串，
排队消息显示为“时间 内容”，省略请求 ID 和来源标签，手写 Markdown 保留原文。
回执按请求 ID 和时间识别，不依赖展示的 Agent 名称，已有条目仍可正常读取。
无需再读取文件。这取代了原先返回路径的结构。
缺失的个人记忆在路径锁内初始化；Proactive 只读取，不修改个人记忆。

`manage_temp_proactive_agent` 支持列出、创建、修改、暂停、恢复和取消一次性任务。创建需提供明确的 `query`；
`run_at` 与 `delay_minutes` 二选一，延迟支持小数。后续操作使用稳定的 `task_id`，版本变更使旧 worker 失效。
后台 `temp_proactive` Job 负责等待和取消，到期后调用服务私有的 `temp_agent` Job；后者 fork 来源对话，只执行一次 query。完整最终结果先保存到
`{metadata_dir}/pa/temporary/{id}/result.md`，再立即投递，不增加发送 query，也不设展示字符上限。
重启后直接补投递已保存结果，不重复模型工作；query 中断且没有保存结果时明确失败，由 resume 创建新版本重试。
停机期间到期的任务补执行一次并标注延迟。提交、原生历史写入、桌面展示和用户已读是独立状态。

Proactive 使用 `get_chat_messages(limit=10, offset=0)` 按创建时间从新到旧读取每条提醒的最新状态，只读、不废弃历史。
工具直接返回完整消息内容和投递状态，可通过 `limit`、`offset` 分页。
后台结果已生成或 Agent 已消费，不代表主人已读或提醒已送达，不能据此判定已经分享。
JSONL 用相同 ID 追加 `pending`、`sent`、`discarded`、`failed` 状态，包含创建、更新、过期、发送时间和目标会话，
替代原来的 `is_draft` 字段。
对话 Agent 的提醒相关追问以当前对话和主人明确提供的上下文为准，不查询该历史。

`send_chat_message(content, kind="normal")` 按后端绑定的发送者执行投递，模型不能改模式。普通 Proactive 每轮接受一次决策，不设展示字符上限。`kind` 取值为 `normal`、`morning`、`midday`、`evening`。空内容使用 `normal`。空内容撤销绑定会话的待发送提醒并返回 `skipped`；
新内容成功落盘后替换旧待发送提醒并返回 `pending`，即使对话 Agent 已经空闲也不会直接注入。
原生 `reminder_delivery` 后台 Job 从提交后等待对话 Agent 连续空闲10秒再投递，期间再次对话或轮次变化则重新计时。
投递前检查当前会话、最新原生轮次、活动时间和有效期；有效期不超过一个唤醒间隔或当前活动结束时间。
无会话、会话切换、离开活动时间或过期时废弃。重启从 JSONL 重建待发送状态，并重新检查和等待。
重试复用原生消息 ID；`sent` 表示已注入，不代表主人已读。注入与 JSONL 状态落盘仍是两个操作。
`pa_events` 的快照和状态事件携带后端推导的 `notifications` 展示窗口（绝对起止时间、收尾消息 ID 和 enabled）。周期紧急消息和临时消息携带 `delivery: immediate`，越过免打扰、主动暂停、活动窗口和晚间收尾。
临时任务履行主人的明确约定，结果未必紧急；语音仍受显式的语音开关及音量控制，即时提醒不受暂时静音限制。
桌面气泡和主动系统通知在展示前复核窗口，收尾后只允许该收尾消息，丢弃旧队列，结束时停止展示；旧消息次日不补弹。
聊天历史保留。升级前停止服务并执行一次显式迁移，运行时不兼容缺少 `kind` 的旧记录或旧工具参数：

```bash
python scripts/migrate_pa_reminder_kinds.py /path/to/workspace/reme_metadata/pa/chat_inbox.jsonl
```

脚本校验全部记录、保留原始 `.bak`，依据已保存的 hint 轮次补充类型，其余旧记录标为 `normal`，原子写回。

主动 Agent 装备指令读取、任务结果读取、截图、消息历史、消息提交和 JSON 校验六个 Job 工具。
会话状态查询归通用会话操作，不再放在主动工具目录。相关性、语义去重和提醒价值由 Agent 判断。
长期个人记忆由对话 Agent 维护，旧对话和恢复输入不能恢复主人已要求遗忘的信息。

Desktop 的 Remy 侧栏可切换提醒列表和定时任务结果收件箱。每张卡片标注已读／未读；收件箱支持任务筛选与分页。
`proactive_inbox_view(action="list", kind="all", filter="all", offset=0, limit=20)` 只读查询，支持任务筛选、
分页和全局待处理数量。`kind` 可选 `all`、`task_result`、`instruction`，`filter` 可选 `all`、`pending`、`unread`；
`unread` 与 `task_id` 只适用于 `task_result` 类型，其中 `unread` 表示用户未查看。
`action="read", entry_id=...` 返回当前全文与回执，不推进消费游标；部分读取失败会提示且不显示确定的总待处理数。
`task_results(action="read", path=..., mark_read=true)` 独立记录用户已读时间，保留 Proactive 消费状态。
“已接收”不代表工作完成，编辑后的指令会重新处理。
结果列表跳过异常文档，通过 `complete=false` 和 `warnings` 报告问题，保留原文件。
历史摘要选择和 Proactive 消费也跳过异常结果；消费告警放在响应 metadata 中。
查看结果不创建聊天消息。提醒可以来自其他会话，先只读打开来源对话，对话 Agent 继续工作，
空闲后通过 `pa_select_session(session_id)` 选择该来源继续讨论。快照和即时事件携带从投递流水重建的有界 `inbox_messages`。
桌面展示、阅读及语音回执和原生会话 `sent` 状态独立。

### 定时任务 query

对话 Agent 维护 `{metadata_dir}/pa/cron.json`：JSON 列表，每项只允许四个必填字段：稳定 `id`、五段 `cron`、
非空 `query` 和布尔 `enabled`。任务每次执行后根据结果决定是否立即提醒，无需配置投递类型。
修改前重读最新文件，保留其他任务和身份并原子保存，写后读回并调用 `check_json`。
调度器校验任务字段并自动加载有效修改；JSON 解析通过不等于已加载。旧配置中的额外字段须显式移除，运行时不兼容转换。
截图任务使用普通截图工具比较当前画面；不再提供模型调用前的整轮预检查。
默认每五分钟观察屏幕活动，每小时整点获取主人可能感兴趣的财经新闻，并在每天08:00、12:00、18:00
搜索今天主人可能喜欢的新鲜事，三个时点共用 `daily_fresh_news` 任务身份和历史记录。默认值只初始化缺失文件；
已有 `cron.json` 需要明确编辑，修改 YAML 不会覆盖已有任务。

`cron_schedule` 后台 Job 每秒检查配置文件修改时间，为每个启用的任务管理一个 ReMe `CronJob`。
有效修改会新增、替换或停止调度；配置无效时报告错误，保留已有有效调度。替换时先关闭旧 Job 再启动新 Job，
同一任务不会重叠执行，所有 Job 和执行中的 Agent 随应用关闭。从下一个 cron 时刻开始执行；每个任务串行，错过的触发不补跑。
`proactive.json` 的 `tasks_follow_active_hours` 默认为 `true`：定时任务在执行阶段检查相同的 `start_time` / `end_time`，
时段外直接跳过，不消耗模型 token；改为 `false` 可全天按 cron 执行。修改即时生效，无需重启。
晚间收尾只停止主动交流，独立任务仍遵循所配置的活动时间。

每个任务跨轮次复用一个原生 Codex 会话。一次 query 执行任务、读取最新有效结果和提醒历史，
把有用的累计发现写入指定的暂存 Markdown；没有有价值的更新时不写文件，仅返回 `NO_UPDATE`。
摘要保留结论、必要背景、已核实来源及合并后的 24 小时时间线。紧急发现可在读取最新提醒历史后调用
`send_chat_message_immediate(content)`，代码先发布结果，再提交提醒；消息不设展示字符上限。

结果发布到 `{metadata_dir}/pa/cron/{task_id}/YYYY-MM-DD/HHMMSSffffff.md`，任务身份和开始时间标识一次执行。
`state.json` 保存原生会话、当前执行和持久化完成意图；恢复时继续发布和投递，不重复调用模型或发布结果。
过时提醒候选被丢弃，报告仍保留。任务修改使未完成工作失效，但保留该任务的会话；原生会话缺失时重建，
传输失败不丢弃历史。紧急提醒保留原目标对话，15 分钟后过期。Proactive 消费与 Desktop 用户已读互相独立。
运行时只读取当前格式，旧工作区须停机后通过离线工具显式迁移，保留原文件。

升级旧工作区前停止服务，执行离线迁移，再启动服务：

```bash
python -m scripts.migrate_pa_tasks /path/to/workspace
```

可先加 `--check` 只验证和预览；自定义目录传 `--metadata-dir` 和 `--daily-dir`。
脚本将旧 daily/cron 和工作区根目录下的结果复制到 PA，重命名三份流水，并更新结果和任务状态记录中的路径。
原文件保留，就地更新的引用文件另存备份；目标冲突在写入前报错。旧流水里的内联结果转成已消费的 Markdown 归档，
保留正文且不制造新的主动输入。最后发布 `chat_inbox.jsonl`；启动遇到未迁移布局会明确报错，不初始化空状态替代。
迁移后仅新路径为运行时事实源，保留的旧文件作为备份。

Proactive 的任务步骤调用一次私有工具 `read_cron_results(limit=3)`，再逐篇完整阅读返回路径对应的文档。
它只消费周期 cron 结果，一次性临时任务走独立的即时投递链路。
工具只返回**每个配置任务**各自最新的未读摘要路径，每项最多3篇，包括暂停任务和此前日期；正文从文件读取，
避免被工具输出限制截断。删除配置条目后不再扫描其任务目录，历史文件仍然保留。
完整输入先写入恢复上下文，再将实际返回的文档标为 `read`，被累计摘要覆盖的更旧未读版本标为 `superseded`。
已读只表示 proactive 已消费，不表示主人已读或已发送提醒；轮次失败时通过恢复上下文继续处理。
相关性与提醒价值仍由 Agent 判断，也可以主动调用 `screen_capture` 和可用的原生 `web_search` 补充调查。
`proactive` 配置开启原生实时搜索，实际可用性取决于配置的模型服务支持。

旧 `run_due_tasks` Job、Step 和依赖主动唤醒的调度逻辑已移除；`cron_agent` 改为接收完整 `task` 对象的私有 Job，
不保留兼容入口。原始资源 TTL 清理仍由独立 CronJob 执行。

## MCP

ReMe 0.4.2.2 使用 FastMCP 4 与 MCP Python SDK 2。升级时请同步更新 ReMe 和宿主侧 MCP bridge；
独立 bridge 环境需要 `fastmcp>=4.1.4,<5`。CLI 工具列表统一使用 SDK 2 的 snake_case 字段，
包括 `input_schema`、`output_schema` 和 annotation hints；读取旧驼峰字段的调用方需要同步更新。

默认 HTTP 服务会在 `/mcp` 挂载 streamable HTTP MCP：

```yaml
service:
  backend: http
  mcp_enabled: true
  mcp_path: /mcp
  mcp_stateless_http: false
```

也可以改用独立 MCP Service：

```bash
reme start service.backend=mcp service.transport=stdio
reme start service.backend=mcp service.transport=sse service.port=2333
reme start service.backend=mcp service.transport=streamable-http service.port=2333
```

MCP tool 来自 `enable_serve: true` 的非流式 Job。`service.jobs` 可以设置允许列表；`injected_job_kwargs` 可以注入服务端管理、调用方不能覆盖的参数。

```yaml
service:
  backend: http
  jobs: [search, read, traverse, auto_memory]
  injected_job_kwargs:
    tenant_id: local-user
  tool_error_on_failure: true
```

## ReMe Studio

安装 `reme-ai[web]` 或 `reme-ai[core]` 后，默认 HTTP 地址同时提供 ReMe Studio：

```text
http://127.0.0.1:2333/
```

可通过配置关闭或指定自定义构建：

```yaml
service:
  web_enabled: false
  # web_static_dir: /absolute/path/to/static
```

静态资源缺失不会阻止 Job API 启动。

## 一次性 Job

需要脚本式执行而不常驻服务时：

```bash
reme start job=search query="用户偏好" limit=5
```

它会切换到一次性 CLI Service，但仍经过正常的 Application、Component 和 Job 生命周期。

## 服务发现

```bash
reme find_reme
```

ReMe 会记录本机运行服务的启动参数。普通 `reme <action>` 优先使用实际运行服务的 backend、host、port 和 transport；找不到运行服务时才回退到本地配置。

## 安全边界

ReMe 默认定位为本地服务：

- 默认绑定 `127.0.0.1`；
- HTTP CORS 配置允许任意 origin；
- Job 可进行文件写入、移动和删除；
- 当前服务层不提供通用用户认证。

远程访问必须通过 `reme start service.host=0.0.0.0` 显式启用。不要直接把服务暴露到公网。请在受控网络或带
身份认证、TLS、访问控制和请求大小限制的反向代理后部署，并通过 `service.jobs` 只开放必要 Job。

## OpenAPI

HTTP 服务使用 FastAPI。启用 HTTP 服务时，可以通过 `/docs`、`/redoc` 和 `/openapi.json` 查看当前配置实际注册的端点；Studio 的 SPA fallback 不会覆盖这些保留路径。
