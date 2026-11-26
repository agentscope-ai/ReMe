# 改造计划

本文记录已采用的新方案及实现契约，后续继续按编号追加。
本版按最新讨论区分 Component 的生命周期与状态、Job 的执行流程，以及每次调用新建的 Step。
本文作为后续开发依据：第 1—10 点定义结构与配置，第 11—15 点明确接口、执行协议、持久化、请求边界和验收。
目标公共类、配置键和数据契约按本文落实；私有实现可以结合当前代码调整，涉及契约变化时先同步本文、schema、配置、客户端与测试。
旧名称仅用于迁移对照；实际代码、注册、配置和调用方统一使用新入口，不提供兼容别名。

当前实现参考：
[BaseComponent](../../reme/components/base.py)、
[Application](../../reme/core/application.py)、
[BaseJob](../../reme/components/job/base_job.py)、
[ProactiveManager](../../reme/components/proactive/manager/local.py) 和
[ProactiveRuntime](../../reme/components/proactive/runtime/task.py)。

## 1. Chat 与 Proactive 共用 Inbox 基类

```text
BaseComponent
└── BaseInbox
    ├── EventInbox           # Proactive / cron Runtime 使用
    └── NotificationInbox    # Remy Runtime 使用
```

| 类 | 职责 |
| --- | --- |
| `BaseInbox` | 定义收件、查看、领取、确认、释放和恢复接口，复用真正相同的存储机制。 |
| `EventInbox` | 保存统一 Event，管理后台任务输入的校验、容量、持久化和处理回执。 |
| `NotificationInbox` | 保存收到的提醒引用及处理状态，正文从 daily 源记录读取，支持按选定 ID 领取。 |

`claim` 只表示领取，成功处理后才 `ack`；失败和取消需要释放批次。
优先级、空闲唤醒、模型决策不进入 Inbox 基类。

每个 Runtime 持有独立的 Inbox 和对应数据目录。Inbox 由所属 Runtime 启动和关闭，
不必在 YAML 中再手工配置五套 Inbox。
Runtime 显式管理这些自有子组件的启动、失败回收和关闭；不假设现有 bind 的 default_factory
会在带 ApplicationContext 的模式下自动创建缺失依赖。
多个 EventInbox 不能分别占用整个 `reme_agents` 根目录的同一把单写锁，应按 Runtime 的实际存储范围隔离。

当前的 `AgentEventInbox -> LocalAgentInbox` 继承链按该目标拆分。
NotificationInbox 取代目前依赖任务 Outbox `read_at` 表达 Chat 处理状态的逻辑，避免两处重复维护是否已处理。

## 2. AgentRuntime 共用基类，分为两个组件实现

```text
BaseComponent
└── BaseAgentRuntime
    ├── ChatAgentRuntime         # Remy
    └── ProactiveAgentRuntime    # 主动任务与 cron 任务
```

Cron 是触发方式，不单独派生一种 Runtime。Memory、Notebook、金融、AI 早报是后台 Agent 的业务执行差异，
放到 Job / Agent Step 层区分，不再作为四种 Runtime 子类。

### 每个 Runtime 的组成

```text
ChatAgentRuntime
├── NotificationInbox
├── Agent 执行绑定：AgentWrapper + Chat Job + 原生会话
└── ChatOutbox → 用户会话

ProactiveAgentRuntime
├── EventInbox
├── Agent 执行绑定：AgentWrapper + 领域 Job + 任务会话
└── NotificationOutbox → Remy.NotificationInbox
```

这里的 Agent 执行绑定是长期保存的配置、模型适配器引用、会话引用和执行 Job 名称。
实际 Agent Step 仍由每次 Job 调用创建，Runtime 不缓存 Step 或 RuntimeContext 实例。
多个 Runtime 可以共享一个 AgentWrapper / Codex App Server，但会话、输入、发件状态和锁分别隔离。

### BaseAgentRuntime 定义

| 基类持有的能力 | 定义 |
| --- | --- |
| 身份与配置 | Runtime 名称、工作区目录、启停设置、执行 Job 引用和代理执行配置。 |
| Inbox / Outbox | 组合持有、初始化、恢复和关闭自己拥有的收发件资源。 |
| 会话与活动 | 原生会话引用、执行锁、正在进行的调用、错误和运行状态。 |
| 并发边界 | 提供单轮执行租约或上下文，防止同一 Runtime 被 Job 并发消费。 |
| 可靠提交 | 提供输入确认、产物引用、发件交接和中断恢复等原子操作或窄接口。 |
| 生命周期 | 启动恢复状态；关闭时停止接收新执行、取消或等待活动调用、释放自有资源。 |

组件可以长期存在，并提供有状态操作；“Component 管生命周期”不意味着它只能有 `start/close`。
一轮任务如何组合这些操作、何时调用模型、如何组织产物，由 Job / Step 执行。

| 实现 | 长期持有的专有状态与配置 |
| --- | --- |
| `ChatAgentRuntime` | 当前前台会话、个人记忆路径、前台输入版本、空闲时间、提醒活动批次及 ChatOutbox。 |
| `ProactiveAgentRuntime` | TASK.md 路径、事件批次、cron 计划、触发间隔、执行重试状态及 NotificationOutbox。 |

### Chat 独立边界

Chat 的代码集中在 `components/proactive/runtime/chat.py` 和 `steps/proactive/chat/`，
不再由 ProactiveManager 持有 Chat 锁、会话选择或提醒 worker。
现有 `ChatNotifications` 混合状态、收发件操作与 workflow，按职责拆开后分别迁移。
提醒处理的执行编排移到 Chat Job / Step，状态与原子操作分别放回 Runtime、Inbox 和 Outbox。

普通用户请求继续进入 Chat Job；NotificationInbox 管理后台投递来的提醒，不把普通聊天强制改成 Event。
提醒 Job 负责选批、读取源记录、创建临时 fork 并提交发送或静默处理决定。
Chat Runtime 提供会话版本和批次检查，防止旧 fork 把提醒写进已经变化的会话。

### 组件与 workflow 的划分

组件封装有明确所有者的状态、资源、生命周期和可靠操作；workflow 组合这些能力完成一次工作。
是否读写持久文件不是划分标准，例如 Connector 的进度落文件，但采集仍属于 workflow。
Runtime 内部的辅助对象也不必各自注册成组件。

| 逻辑 | 所属位置 | 边界 |
| --- | --- | --- |
| 会话引用的读写、锁和复用 / 切换操作 | components/proactive/runtime/_sessions.py | Runtime 拥有的内部能力；何时创建或切换会话由调用 workflow 决定。 |
| 本轮输出快照缓存、版本及变更通知 | components/proactive/runtime/_live_turn.py | Runtime 拥有的临时状态，可由原生会话重建；不放模型调用或订阅执行循环。 |
| 收件查询、批次容量、领取、确认和释放 | components/proactive/inbox/notification.py | NotificationInbox 的队列状态与可靠操作。 |
| 当前提醒轮次、前台输入版本、空闲时间与并发保护 | components/proactive/runtime/chat.py | Chat 专有状态，不抽出含义宽泛的 state.py。 |
| 写入用户会话、稳定消息 ID、发送确认和恢复 | components/proactive/outbox/chat.py | ChatOutbox 提供可靠提交，结合 Runtime 的会话与批次校验。 |
| 按优先级、空闲时间和重试条件发起提醒处理 | steps/proactive/chat/notifications.py | 提醒调度 Step 与单轮处理 Step 同模块，循环由独立 BackgroundJob 托管。 |
| 读取本批材料、构建提示、临时 fork、调用模型并提交决定 | steps/proactive/chat/notifications.py | 单轮提醒 workflow，通过组件接口处理收发件。 |
| 订阅等待、查询聚合与响应格式转换 | steps/proactive/api.py | 服务 workflow；Runtime 只提供快照和变更通知。 |

Runtime 的私有辅助模块就近平铺为 _sessions.py、_live_turn.py，分别保存会话操作和本轮输出缓存。
这些辅助对象由 Runtime 实例持有，不单独注册组件，也不为少量文件再建立 _internal 包。
角色专有状态直接放在对应 Runtime，不创建含义宽泛的 state.py 或混合职责的 notifications.py。
公开请求、响应和状态类型继续放 schema；角色专有的内部字段或小型数据结构就近放在所属 Runtime 中。
以后确有体量和复用需要，再按具体能力拆文件。

## 3. Outbox 分离输出语义

```text
BaseComponent
└── BaseOutbox
    ├── ChatOutbox
    └── NotificationOutbox
```

| 实现 | 输出与状态 |
| --- | --- |
| `ChatOutbox` | 面向用户会话，保存发送关联和可恢复的交付状态，维护原生消息 ID 与会话版本要求。 |
| `NotificationOutbox` | 面向其他 Runtime 的 NotificationInbox，保存已生成提醒的源记录引用、待投递目标和投递结果。 |

BaseOutbox 只抽象真正共有的发件、确认和恢复契约，不统一两种输出的具体载荷与发送策略。
NotificationOutbox 的“已投递”和 NotificationInbox 的“已处理”分别归属发送方与接收方。

提醒正文仍保存在 daily Markdown。记录写入、进入 Remy 收件箱、写入用户会话是三个不同阶段。
前台回复已由 AgentWrapper 写入原生会话时，ChatOutbox 只确认和关联结果，不能再追加一遍同样的回复。
跨输入、发件和原生会话的交接使用稳定身份、持久化回执和恢复流程，不假设它们天然属于同一个原子事务。

## 4. AgentManager 只管理后台 Runtime

新增 `components/proactive/manager/`，承接现有 `proactive_manager/` 中的后台任务管理职责。
私有的定义发现与实例索引辅助逻辑放在同目录 _registry.py，由 Manager 持有。

AgentManager 负责：

- 维护后台 Runtime 的名称、定义和实例索引，发现工作区中的动态任务。
- 管理任务创建、更新、暂停、恢复、删除和状态查询，保留历史文件。
- 将统一 Event 路由到一个或多个后台 Runtime，报告各接收方的投递结果。
- 向调度 Job 提供到期任务、待处理输入和重试信息。
- 管理自己创建的动态 Runtime 的生命周期。

Remy 也是 AgentRuntime 组件，但不进入 AgentManager 的后台任务管理集合。
统一 Agent 列表、会话查看和 Studio 订阅接口可以在服务 Job 中聚合 Remy 与后台 Runtime。

### 静态与动态实例的生命周期所有权

| 来源 | 启动与关闭的所有者 | AgentManager 的角色 |
| --- | --- | --- |
| YAML 中声明的 Remy 和四个预设 Runtime | Application | 引用和管理四个后台实例，不重复启动或关闭这些组件。 |
| 通过管理工具新建、从工作区恢复的其他后台 Runtime | AgentManager | 创建、启动、登记、停止和关闭，启动失败必须回收已创建资源。 |
| 每个 Runtime 内部的 Inbox / Outbox | 所属 Runtime | Manager 不单独操作其资源生命周期。 |
| 配置中共享的 AgentWrapper / App Server | Application | Runtime 使用依赖引用，不替其他所有者关闭。 |

当前 Application 支持配置实例化和替换已有组件，尚没有通用的动态添加 / 删除组件接口。
因此动态 Runtime 先作为 Manager 拥有的子组件管理；不能直接塞进
`app_context.components` 后假设 Application 会自动登记其关闭顺序。
Manager 的查找入口统一返回静态引用或自有动态实例。

## 5. Connector 全部作为 workflow，不新增组件

当前 [BaseConnectorStep](../../reme/steps/proactive/data_connectors/base_connector.py) 已有资源发布流程，
但 [CodexDialogueStep](../../reme/steps/proactive/data_connectors/codex_dialogue.py) 另行实现了投递与进度事务。
采集代码统一放到独立的 `steps/proactive/connector/`，由 Job / CronJob 编排执行。
进度需要持久化，并不意味着必须有一个长期存在的 Connector 组件。

```text
BaseStep
└── BaseConnectorStep
    ├── ScreenCaptureStep
    ├── FinanceCollectorStep
    │   ├── FinanceNewsStep
    │   └── QuoteCollectorStep
    │       ├── TencentQuotesStep
    │       └── BtcQuotesStep
    └── BaseHarnessConnectorStep
        └── CodexDialogueStep
```

- 来源和接收方配置放在 Job / Step 配置中，调用时创建新的 Step。
- BaseConnectorStep 统一资源路径检查、发布、稳定 Event 生成、逐接收方投递和恢复流程。
- 派生 Step 提供数据源读取、原生格式解析、附件导出和来源特有的增量、去重逻辑。
- 游标、baseline 和未完成投递批次保存在工作区文件中，每次调用加载和提交，不依赖 Step 跨轮次存活。
- 客户端、线程任务和临时资源由本次调用释放；现有 Job 生命周期负责停止调用和等待取消清理。
- 定时扫描由 CronJob 触发，不新增 Connector 组件或独立的常驻 watcher。

```text
CollectConnectorJob
  → 检查采集 workflow 是否启用
  → 重试未完成的投递
  → 读取新增内容并写入 resource
  → 校验、发布 metadata.json
  → 生成统一 Event
  → AgentManager 路由到多个后台 Runtime 的 EventInbox
  → 保存各接收方结果与采集进度
```

资源包完成发布后才发送 Event。冻结批次内容、ID 和接收方，部分成功时只重试剩余接收方。
资源已发布但投递未完成时保留可恢复状态，不生成另一个 ID 重新发送同一批。
Connector 产生观察，不决定用户提醒，也不直接修改 Agent 的业务产物。

### manage_connector 管理 workflow

enable/pause 的目标是已配置的采集 workflow / Job，而不是一个 Connector 组件。
启停控制状态持久化，由 workflow 入口在采集和自动投递前检查；调度 Job 可以继续等待，暂停期间跳过采集。
恢复后先处理保留的待投递批次，再继续增量读取。
不通过修改每次新建的 Step 字段控制后续调用，也不把单独修改 Job.enabled 布尔值当成已经实现完整的动态启停。

### 公共逻辑收拢到 base_connector.py

Connector 暂不建立 _internal/resource_delivery.py 和 _internal/workflow_state.py：

- 当前 [resource_delivery.py](../../reme/steps/proactive/data_connectors/resource_delivery.py) 的稳定 Event 生成、
  接收方冻结、待投递恢复和投递结果落盘，收拢为 BaseConnectorStep 的共用方法或同模块辅助函数。
  各来源通过继承复用，避免基类与反向持有基类对象的 ResourceDelivery 再拆成两套入口。
- workflow_state.py 尚无独立职责。启用 / 暂停状态的公共读写函数放在 base_connector.py，
  供采集入口和 manage_connector 工具共用；管理工具不为修改状态而执行一次采集。
- Harness 的 baseline、文件游标和 pending batch 归 Harness workflow；新闻游标与报价去重归各自来源。
  不把不同来源的进度集中到一个通用状态文件或状态类。

这是代码职责的合并，不合并持久化数据的含义：workflow 控制状态、每个资源包的投递状态和来源采集进度
仍按各自范围读写，并保留路径检查、锁、原子发布和中断恢复。

### Harness 基类的抽象边界

| 基类能力 | Codex 实现提供的差异 |
| --- | --- |
| 来源目录配置、只读访问、进度隔离 | Codex rollout 文件的发现规则。 |
| 首次 baseline 与恢复协议 | Codex 有效记录边界和原生游标的解释。 |
| 增量扫描预算、公平轮转、取消 | Codex 新增记录的读取、完整性检查和下一游标。 |
| 标准消息和附件描述 | Codex 会话、轮次、消息、角色、阶段与图片解析。 |
| 稳定消息身份与去重机制 | 原生消息 ID，自身及子代理会话的过滤。 |
| 资源发布、投递和进度提交 | 只提供来源内容，不重写公共事务。 |

游标由来源实现解释，基类不假设所有 Harness 都使用按字节追加的 JSONL。
保留 Codex 当前的半行等待、文件替换去重、图片复制、扫描限额和中断恢复能力。
当前只完善基类和 Codex，暂不实现 Claude Code、Hermes，也不建立空占位类。

Harness Connector 只读外部会话目录；用于执行模型的 AgentWrapper 是独立的扩展轴。

### Finance workflow 展开

金融采集放在 steps/proactive/connector/finance/，按现有实现中的实际复用关系拆分：

| 文件 | 类与职责 |
| --- | --- |
| base_finance.py | FinanceCollectorStep：共享 HTTP 超时、重试、取消、已发布资源查找和 JSONL 写入。 |
| base_quotes.py | QuoteCollectorStep：共享报价采集、最近报价比较、重复快照过滤和历史资源读取。 |
| finance_news.py | FinanceNewsStep：财联社新闻分页、增量游标、新闻 ID 去重，生成 news.jsonl 和可读 news.md。 |
| tencent_quotes.py | TencentQuotesStep：按 instrument 参数采集 gold、oil、nasdaq，复用报价基类。 |
| btc_quotes.py | BtcQuotesStep：采集 BTC/USDT 的滚动 24 小时报价，复用报价基类。 |

当前 _finance.py 同时包含 FinanceCollectorStep 和 QuoteCollectorStep，迁移时按上述职责拆到两个基类文件。
黄金、原油和纳指共用 TencentQuotesStep，通过三个独立采集 Job 配置 instrument，不为每个标的复制一个文件或类。
加上财经新闻和 BTC，共有五个现有金融采集 workflow，可以分别开启或暂停。

这些文件只采集、保存资源并发送 Event，不包含金融 Agent 的分析和提醒判断，也不需要空的提示 YAML。
金融任务的要求与记录、提醒方法位于 steps/proactive/task/finance.py 及其同名 finance.yaml，继承公共执行流程。

## 6. Event 契约与四个模型工具

### 统一 Event

继续以 [Event schema](../../reme/schema/event.py) 为唯一观察输入契约：

```json
{
  "id": "resource-<stable-package-hash>",
  "event_ts": "2026-11-26T10:00:00+08:00",
  "source": "codex_dialogue",
  "description": "新增项目对话，包含进展和待讨论问题。",
  "paths": [
    "resource/codex_dialogue/2026-11-26/100000_abcdef/metadata.json",
    "resource/codex_dialogue/2026-11-26/100000_abcdef/dialogue.md"
  ],
  "metadata": {}
}
```

- 重试保持相同的稳定 ID、内容和带时区时间；路径必须位于工作区内。
- 资源事件统一引用 manifest 和需要阅读的材料，大段正文及图片留在资源包中。
- `task_names` 属于投递请求，不进入 Event；不同 Runtime 独立处理同一个 Event。
- `source=cron` 保留给框架，可以没有资源路径；Notification 不强行转换成原始观察 Event。
- 保留当前触发约束：无 cron 的任务接收外部 Event；有 cron 的任务只接收框架定时 Event。创建后及同名恢复时触发类型固定；Notebook 仅接收事件。
- Cron 与事件任务共用 Runtime 类与执行协议，不表示同一个任务同时接受两种触发。

### 四个工具

| 原工具名 | 新工具名 | 建议 action | 职责 |
| --- | --- | --- | --- |
| `manage_proactive_agent` | `manage_proactive_runtime` | `list/create/status/update/delete/query` | 通过 AgentManager 管理后台 Runtime、任务要求、执行 Job、运行配置和整体启停。 |
| `append_task_events` | `manage_connector` | `send/enable/pause` | 保留显式投递标准 Event 的能力，增加采集 workflow / Job 的开启、暂停管理。 |
| `submit_chat_reminder` | `submit_chat_reply` | `send/skip` | Chat 提醒 fork 提交当前批次的 Chat 回复或静默决定。 |
| `todo`、`idea` | `manage_notebook` | 按 kind 区分记录操作，另有公共 list_projects | 统一提供 Todo / Idea 的全部现有操作，复用业务校验、权限和存储。 |

这里的原工具名是当前配置中对模型暴露的 Job / 工具名称。
实施时除文件名外，还要同步 Job 名、Step backend、注册导入、job_tools、提示 / skill 中的调用名称及相关测试和文档。
当前 submit_chat_reminder 的实现文件是 steps/proactive/tools/chat_reminders.py，迁移后对应 submit_chat_reply.py。

`manage_connector send` 接收 task_names 和非空 events，保留路径权限、Event 校验和逐接收方结果。
它不隐式采集或开启 workflow。enable/pause 控制采集 workflow 的自动采集和自动投递，保留已有资源、游标与待投递批次。
暂停 Connector 不清空 Runtime 的 Inbox，不停止已经排队的后台任务。

`submit_chat_reply` 替代 `submit_chat_reminder`，只作用于注入到当前 Chat fork 的批次。
send 要求非空正文，skip 明确表示静默处理，继续检查调用者、批次和会话版本。
Proactive Job 通过记录中的提醒字段产生 Notification，不调用这个 Chat 工具。

模型不直接获得 claim/ack/nack。四个工具按执行角色绑定，不要求每个 Agent 都获得全部工具。
文件等其他领域工具以及 Studio 查询与订阅接口不计入这里的四个工具。

### tools 与 Notebook 工具文件

管理工具目录统一叫 steps/proactive/tools/，不再使用 agent_tools/。
Notebook 不再建立子目录，工具与存储文件平铺到 tools/：原 manage.py 对应 manage_notebook.py，原 _store.py 对应 _notebook_store.py。
当前 steps/notebook/proactive_notebook.py 及其 YAML 属于自动跟进 workflow，迁到 steps/proactive/task/notebook.py 和 notebook.yaml，
不混入工具目录。

模型的统一入口是 manage_notebook，对应 ManageNotebookStep / manage_notebook_step；job_tools 不再同时绑定 todo 和 idea。
现有 Studio、CLI 文档和测试里的 todo / idea 调用全部迁移到 manage_notebook，不保留旧 Job、旧 Step 或兼容入口。
Chat、后台 Notebook workflow 和服务入口共用业务与存储能力；notebook_dir 工作区数据位置保持不变。
四个工具的 Job、Step 注册导入、角色绑定和提示 / skill 引用一起迁移，调用方不依赖旧 Python 模块路径。

_notebook_store.py 保留为独立的私有辅助模块，不合并进 manage_notebook.py：

- manage_notebook.py：统一工具入口、Todo / Idea 请求验证、业务操作、查询结果组织和自动修改权限检查。
- _notebook_store.py：RecordStore 的 JSON 读写、数据校验、文件版本检查、待提交日志恢复和归档分片。

保留两个职责清晰的文件即可，不为存储再建目录、组件或独立 workflow；工具每次调用按需创建存储辅助对象。

### manage_notebook 的操作范围

记录操作使用 kind=todo 或 kind=idea，再按 action 校验该类型支持的参数；list_projects 是公共操作，除 action 外仅接收 scope，不带 kind。
统一工具保留全部现有能力，不把 Todo 与 Idea 的不同状态变化混成一套宽松 CRUD。

| 对象 | action | 覆盖的现有能力 |
| --- | --- | --- |
| Todo | create / list / get / update | 创建、跨项目查询、分页、排序和筛选、读取完整进展、修改内容 / 项目 / 截止日期 / 来源 Idea，以及关闭和重开。 |
| Todo 进展 | update | 追加进展；用 progress_match 匹配已有时间和正文后修改或删除单条进展。 |
| Idea | create / list / get / update / archive / restore | 创建、跨项目查询、分页、读取、修改、归档与恢复。 |
| 全 Notebook 项目 | list_projects | 返回项目、todo_count、idea_count 和 total；默认包含活动与归档记录，scope 可选 active / archive / all。 |

Todo 通过 update 的 status=closed 关闭，通过 pending 或 in_progress 重开；Idea 继续使用显式 archive / restore。
create Todo 时可用 source_idea 引用已有 Idea 并沿用其内容与项目，不隐式归档原 Idea。
已有操作不包含整条记录的永久删除，工具整合不额外增加该行为。

ManageNotebookRequest 按 kind / action 选择请求分支，记录分支复用 TodoRequest、IdeaRequest 的字段和验证，公共项目查询使用单独的小请求分支。
路由字段与业务修改字段分开，保留按动作拒绝无关参数、显式提供字段判断、路径授权、工作区锁和存储恢复机制。

后台 Notebook workflow 改为绑定 manage_notebook，继续由服务端注入可信的 _automatic_notebook 上下文。
自动跟进可以读取 Todo / Idea 和项目，只能修改活动 Todo 的进展及将状态设为 in_progress；保留进展匹配、自检和去重。
创建、关闭、重开、修改内容 / 项目 / 截止日期 / 来源关联以及 Idea 写操作，仍交给 Remy 根据用户意图处理。
工具支持全部操作不代表后台 Agent 获得全部写权限，调用参数不能切换或清除服务端的自动执行限制。

### Skill 与工具一起迁移

三个操作技能统一放在 steps/proactive/skills/，统一输出 name、description、content 三个字符串字段；源 YAML 可用 description_zh、content_zh 提供中文版本，由 PromptHandler 选择。
name 使用稳定的连字符名称；文件名采用下划线，与工具文件一致。BaseAgentStep.get_skills(names=None) 统一加载与校验，
具体 Agent 通过 skill_names 或执行 Job 的 skills 名称列表选择；配置不再传只有 name 的外部技能引用。

| YAML 源文件 | 技能 name | 使用者 |
| --- | --- | --- |
| skills/manage_proactive_runtime.yaml | manage-proactive-runtime | Chat |
| skills/manage_connector.yaml | manage-connector | Chat |
| skills/manage_notebook.yaml | manage-notebook | Chat、Notebook |

加载函数使用 PromptHandler 读取三个字段，再转成完整 SkillSpec。发布时为每次调用分配唯一名称，避免 Chat 与 Notebook 并发使用同一模板时互相占用临时目录。
Chat 会话签名对生成技能比较内容和说明，不把临时名称视为绑定变化；修改技能正文仍会触发新绑定。
SKILL.md 只在 Wrapper 的技能作用域内生成，覆盖会话准备、推理和本轮修复，结束后清理。源文件不放在 reme/skills/。

Notebook 技能的调用映射如下，操作范围仍以本节工具契约为准：

| 原调用 | 新调用方式 |
| --- | --- |
| todo 的 create / list / get / update | manage_notebook，传 kind=todo 和对应 action，其他有效参数保留。 |
| idea 的 create / list / get / update / archive / restore | manage_notebook，传 kind=idea 和对应 action，其他有效参数保留。 |
| todo 的 list_projects | manage_notebook，传 action=list_projects 和可选 scope，不传 kind。 |
| todo get 后修改或撤回进展 | 先用 kind=todo、action=get 读取完整进展，再以 update 和原 progress_match 精确修改；删除单条进展继续省略 progress_text。 |
| 从 Idea 创建 Todo | 使用 kind=todo、action=create、source_idea；明确这一步不会自动归档 Idea。 |

Skill 保留先查已有条目、按 total 翻页、更新前读取完整进展、只记录新增事实和核对操作结果等指导。
原内置 Notebook skill 中的后台跟进流程迁入 task/notebook.yaml 的 notebook_skill / notebook_skill_zh；
内置 manage-notebook 负责工具操作指导，领域 notebook_skill 负责本轮证据、自动修改范围和提交给 Remy 的建议，避免重复维护完整参数说明。
Notebook workflow 同时引用内置 manage-notebook，并按本轮需要注入以 notebook-follow-up 为前缀的临时领域技能；
临时技能名带本轮唯一标识，复用现有 SkillSpec 的发布与清理，不与内置技能重名。

迁移时同步检查这些使用方：

- proactive.yaml 中的 Job 描述、agent_wrapper.skills 和各执行角色的 job_tools。
- Chat 与 Notebook 的中英文提示、SkillSpec 的 name，以及领域 YAML 中对技能和工具的引用。
- 管理技能中的 append_task_events / HTTP 路径示例，统一改为 manage_connector 的 send 请求；开启或暂停采集与 Runtime 启停分别说明。
- 提醒决策提示中的工具名，统一使用 submit_chat_reply 的 send / skip 契约。
- 技能加载、工具请求验证和 Notebook 自动修改权限的测试，保留实际行为验证。

manage-connector 的内容包括：从已配置名单选择 connector_job；区分采集暂停与 Runtime 暂停；有资料时先保存合法资源再发送引用；
使用稳定 ID、原始 event_ts 和规范相对路径，禁止手动 source=cron；按 deliveries / rejected / errors 核对各接收方，只重试可重试失败目标。
它还要说明 enable 不等于已有新资料、Event 入箱不等于任务执行完成、提醒提交不等于用户已读。
manage-proactive-runtime 保留“创建任务后接入来源”的衔接，不再重复完整的 Event 投递说明；两个技能都由 Remy 使用。

三个内置技能名是 manage-proactive-runtime、manage-connector 和 manage-notebook，模型工具名仍使用下划线形式。
submit_chat_reply 只承担受保护批次的 send / skip，操作指导放在 notifications.yaml，不为简单提交另建第四个内置技能。
技能中的 CLI 示例统一为先启动 reme start config=proactive，再从另一终端调用 reme manage_notebook kind=todo action=... 等新入口；公共 list_projects 示例省略 kind。
Notebook 的进展更新与项目统计示例按目标请求分支验证；Chat 和后台 Notebook 的技能选择分别核对，不能只替换正文里的旧工具名。
三个操作技能以 skills/ 下 YAML 为源；本轮 record_skill 和 notebook_skill 继续位于对应任务 YAML，由现有 SkillSpec 生成原生 Markdown。

## 7. Job / Agent Step 承载具体执行逻辑

本次子系统的执行代码统一归入 steps/proactive/，按执行职责划分为 chat/、task/、connector/ 和 tools/。
Runtime 名称留给 components/proactive/runtime/ 的长期组件；后台任务的一轮执行放在 task/，不再设置 steps 下的 agent_runtime/ 和 proactive_runtime/ 两个包。
task/ 保存后台任务执行代码及同名 YAML，默认 TASK.md 内容仍是 YAML 中的模板键。
Todo/Idea 工具与存储平铺到 tools/manage_notebook.py 和 _notebook_store.py，自动跟进执行逻辑放在 task/notebook.py 及其同名 YAML。

| steps/proactive/ 下的位置 | 执行职责与使用方 |
| --- | --- |
| base.py | BaseAgentStep 的组件访问、参数提取和路径授权供 Chat、后台任务、管理工具与服务 Step 复用；只读查询辅助方法供管理工具和服务入口复用。 |
| api.py | 面向 Studio 等客户端的现有服务 Job 入口，校验、路由并转换 Chat 与后台任务的结果。 |
| loop.py | 跨后台 Runtime 调度，按组件状态调用各自配置的执行 Job。 |
| chat/ | Remy 对话、单轮提醒处理与提醒调度。 |
| task/ | 后台任务初始化、公共单轮执行及 Memory、Notebook、Finance、News 四种领域执行。 |
| connector/ | 采集资源、生成标准 Event 并投递给目标后台 Runtime。 |
| tools/ | manage_proactive_runtime、manage_connector、submit_chat_reply、manage_notebook 四个工具及共用存储辅助模块。 |

根目录 base.py 只依赖组件和基础辅助能力；chat/、task/、tools/ 与 api.py 按需使用它。
Chat 专用的 stream_reply 放回 chat/chat.py；task/base.py 保留后台任务自己的单轮执行流程。
api.py 调用 Chat 或共用查询能力，其他 workflow 不反向调用服务入口。
下面列的是目标 workflow 划分；复用现有执行、会话和查询能力迁移，不表示这些新目录和入口已经实现。

### chat 可以执行的 workflow

| workflow | 文件 | 入口与结果 |
| --- | --- | --- |
| 前台对话 | chat.py / chat.yaml | 接收用户输入，确保或复用当前 Remy 会话，调用模型和工具，输出会话回复与流式状态。 |
| 单轮提醒处理 | notifications.py / notifications.yaml | 领取 NotificationInbox 的选定批次，读取源记录，在临时 fork 中判断发送或静默，提交到 ChatOutbox 并确认输入。 |
| 提醒调度 | notifications.py | 同模块的调度 Step 按优先级、空闲时间、可发送条件和重试状态，触发单轮提醒 Job；自身不执行提醒推理。 |

notifications.py 中定义两个执行 Step：ProcessChatNotificationsStep 处理一次提醒，ChatNotificationsLoopStep 调度处理时机。
分别由普通 Job 和 BackgroundJob 调用，不为调度再建立 loop.py。
notifications.yaml 保存单轮处理所需的提示；调度 Step 不使用提示，无须另建 loop.yaml。
循环的停止由 BackgroundJob 管理；空闲时间、重试时间、前台版本和活动批次由 ChatAgentRuntime 持有。

创建或切换 Chat 会话复用 chat.py 的会话准备逻辑，由服务入口调用，状态操作仍由 ChatAgentRuntime 提供。
Chat 的个人记忆维护属于前台对话中的任务能力，当前不增加独立的记忆整理 workflow。

### task 可以执行的 workflow

| workflow | 文件 | 入口与结果 |
| --- | --- | --- |
| 创建 / 初始化任务要求 | initialize.py | 创建任务或初始化预设时，渲染对应领域的 default_task，create-only 保存 TASK.md；显式任务文本优先，已有文件不覆盖，不调用模型。 |
| 项目与记忆跟进 | memory.py / memory.yaml | 消费事件或所属任务允许的定时触发，读取项目材料和 TASK.md，检索历史、判断增量，记录项目进展、难点与讨论事项。 |
| Notebook 跟进 | notebook.py / notebook.yaml | 消费观察，匹配 Todo/Idea，按工具权限更新有依据的进展，将需要用户决定的事项形成记录和提醒。 |
| 金融任务执行 | finance.py / finance.yaml | 阅读 Connector 产出的行情和新闻资源，按 TASK.md 核对标的、时间、涨跌基准及提醒条件，记录结果并决定是否产生提醒。 |
| AI 早报执行 | news.py / news.yaml | 到约定时间后按任务要求检索新闻、核对来源、去重并整理早报记录，提交对应提醒。 |

task/base.py 承载四种领域执行共用的一轮流程：领取输入、准备上下文和技能、执行模型、校验与有限修复、提交记录及提醒交接、确认或释放输入。
初始化只生成任务要求，不执行这一整套推理和写记录流程。
历史检索、增量判断、写后自检、格式修复和失败重试是共用执行流程的阶段，不各建立一个领域 Job。

### 共用查询与跨 Runtime 调度

这两项属于跨入口复用或跨任务调度，直接放在 steps/proactive/ 根目录：

| 能力 | 所属位置 | 职责与调用方 |
| --- | --- | --- |
| 任务只读查询 | steps/proactive/base.py | 复用当前 BaseAgentStep.query_reply 的执行能力，供管理工具与服务 API 调用；在任务最近会话的只读临时 fork 中回答，不消费事件、不替换正式会话，不单独建立 query.py 或查询 Job。 |
| 后台任务调度 | steps/proactive/loop.py | AgentManagerLoopStep 对多个 Runtime 做调度：枚举任务、生成到期 cron Event、检查输入和重试条件，再调用各 Runtime 配置的 Job。由 BackgroundJob 托管循环和停止。 |

根目录 base.py 只提供 workflow 的辅助方法，不注册为独立执行入口；会话租约、fork 生命周期和状态保护由 Runtime 组件提供。
AgentManager 组件提供任务索引、状态和管理操作，后台循环与 Job 分发仍属于调度 workflow，不搬进组件。
Remy 的提醒调度放在 steps/proactive/chat/notifications.py，与跨后台 Runtime 的 loop.py 分开。
调度源码文件命名为 loop.py，配置中的 Job 名 agent_manager_loop 和 Step 注册名 agent_manager_loop_step 保留。
AgentManager 继续只管理后台任务，不因合并提醒文件而接管 Remy。

初始化、查询和管理流程复用现有工具、服务调用和后台启动入口，不另建立任务初始化或查询工具。
以下相对模块路径以 steps/proactive/ 为根：任务 CRUD 留在 tools/manage_proactive_runtime.py，
采集与投递留在 connector/，统一列表和订阅留在根目录 api.py。

### api.py 是现有服务 Job 的薄适配入口

steps/proactive/api.py 统一承接现有跨 Chat 与后台 Runtime 的服务调用。
这里的 API 是 Job 调用入口，不是新增 HTTP 路由层；HTTP、MCP、CLI 和 Studio 仍通过既有 Job 机制使用这些能力。

| 现有入口 | api.py 保留的职责 | 实际执行归属 |
| --- | --- | --- |
| list_agents、read_agent_session | 参数校验、聚合 Runtime 状态与会话快照、转换响应。 | Runtime / Manager 提供状态和会话读取能力。 |
| watch_agent_session | 订阅、等待变更、发送快照 / 更新 / 心跳，结束时清理订阅。 | Runtime 提供快照缓存与变更通知，订阅循环属于本次服务调用。 |
| list_agent_inbox、list_task_reminders、read_agent_reminder | 根据 Runtime 身份读取收发件视图并返回。 | 对应 Inbox / Outbox 提供读取操作。 |
| ask_agent | 根据目标路由并转发结果，保留原有流式响应语义。 | Remy 进入 Chat workflow；后台任务使用共用 base.py 的只读查询辅助方法。 |
| create_chat_session | 校验请求，调用 Chat 的新会话逻辑并返回快照。 | Chat workflow 组织调用，ChatAgentRuntime 提供会话状态操作。 |
| task_document | 校验任务身份与输入，转发读取 / 保存操作并返回版本信息。 | 复用任务管理能力及现有文件 Step，保留路径授权、锁和 expected_mtime 校验。 |

api.py 不持有跨调用状态、不维护收发件队列、不另写模型执行和调度循环，也不复制对应 workflow 的业务流程。
管理工具需要只读查询时直接复用共用 base.py，不通过 api.py 的服务入口绕一层。
源码拆分保留这些服务 Job 的名称、请求和响应契约；模型工具的改名与整合另按第 6 点迁移。

### 领域执行的继承关系

之前“四个 ProactiveRuntime 派生”的方案调整为四个 Proactive Agent Step 派生：

```text
BaseStep
├── BaseAgentStep                 # 根目录 base.py：组件访问、参数、授权及只读查询辅助能力
│   ├── ChatAgentStep
│   ├── ProcessChatNotificationsStep
│   ├── ChatNotificationsLoopStep
│   └── BaseProactiveAgentStep     # task/base.py：后台任务共用的单轮执行
│       ├── MemoryAgentStep
│       ├── NotebookAgentStep
│       ├── FinanceAgentStep
│       └── NewsAgentStep
├── BaseConnectorStep             # 具体来源由派生 Step 实现
└── AgentManagerLoopStep
```

Job 可以复用这些 Step。Runtime 配置执行 Job 名称，不按任务名称硬编码分支。
同一套 Memory 执行逻辑可以被多个不同名称、不同 TASK.md 的 Runtime 复用。
所有后台实例都是 `ProactiveAgentRuntime`，不因为领域不同改变组件继承关系。
管理工具只能选择允许的领域执行 Job，不能通过任务配置指定任意 Python 类路径或其他服务 Job。

### 职责分工

| 长期组件提供 | Job / Step 编排 |
| --- | --- |
| 锁、执行租约、输入与会话状态 | 进入执行范围，选择并领取本轮输入。 |
| Inbox / Outbox 的可靠读写与恢复操作 | 读取 TASK.md 和材料，组合技能、工具、提示并调用模型。 |
| 文件、会话和发送能力 | 检查记录、有限修复、提交产物、确认输入和投递提醒。 |
| 到期时间与重试状态 | 循环检查可运行任务，调用其配置的 Job 并更新结果。 |

Runtime 启动时只初始化和恢复状态，不调用尚未启动的 Job。
Application 先启动组件，再启动普通、流式、后台和 cron Job，沿用现有生命周期顺序。
Job 本身也有生命周期，但每次调用的 RuntimeContext、Response 和 Step 都是独立的，
不得用 Step 实例字段保存跨轮次状态。

### 公共主动工作 skill 与四个派生

将现有 example.yaml 中的 record_skill_zh 迁入正式的 steps/proactive/task/base.yaml，作为公共 `proactive-record` 技能的模板，
补齐对应语言版本，由 BaseProactiveAgentStep 按本轮上下文渲染和注入；迁移完成后按第 8 点删除示例文件：

1. 阅读 TASK.md、本轮事件和源材料。
2. 检索历史，分别判断记录重复与提醒重复。
3. 按任务处理业务，只保留有价值的新增事实和来源。
4. 独立决定是否提醒、提醒等级与正文。
5. 写入本轮记录，重新读取并自检；沿用事件明确跳过与定时每轮留记录的契约。

技能描述模型工作方法；权限、路径、schema、版本和确认顺序仍由代码执行。
单轮时间、输出路径和检索预算由执行上下文渲染，临时技能由调用范围发布和清理。
现有本地搜索次数只有统计和超限警告，不将其描述为已强制执行的硬上限。

| 派生 Step | 自己的任务要求 | 公共 skill 之外的差异 |
| --- | --- | --- |
| `MemoryAgentStep` | 项目、进展、难点、讨论事项与记录范围。 | 当前可直接复用公共 skill，读取截图和 Harness 资源。 |
| `NotebookAgentStep` | Todo/Idea 跟进、证据、进展及需要用户决定的事项。 | 内置 manage-notebook、额外跟进 skill、manage_notebook 工具与受信任的自动修改权限。 |
| `FinanceAgentStep` | 标的、阈值、基准、行情有效性和新闻范围。 | 必要的领域阅读方法或程序校验，避免复制通用执行器。 |
| `NewsAgentStep` | AI 早报时间、过去 24 小时范围、来源和输出要求。 | 实际可用的联网检索能力。 |

四个预设 Runtime 都拥有自己的 TASK.md，包括 proactive_memory；默认 YAML 模板只在首次创建时生成任务文件。
派生 Step 修改领域技能、工具绑定和必要的领域校验，不复制领取、重试、记录提交或投递流程。
Notebook 的自动权限继续由工具层限制，不降为纯提示约定。
输入去重不等于领域副作用恰好执行一次，Notebook 进展写入后的不确定失败要有核对或操作去重恢复路径。

### 按 PromptHandler 组织模板，不另建模板目录

以当前 [PromptHandler](../../reme/steps/prompt.py) 和 [BaseStep](../../reme/steps/base_step.py) 的能力为准：

- BaseStep 按类继承链逆序加载模板，父类先加载，子类的同名键覆盖父类，最后叠加调用方的 prompt_dict。
- 每个类按其 Python 模块路径查找同目录、同名的 .yaml，找不到时再查 .yml；不会自动扫描 prompts、skills 或 tasks 子目录。
- 取模板时优先选择带语言后缀的键，例如 default_task_zh，缺少时回退到 default_task。
- YAML 顶层保存字符串模板键；PromptHandler 不会将嵌套的 skills/tasks 对象作为模板读取。
- prompt_format 支持格式参数和行首条件标记；布尔参数用于条件过滤，其余参数用于字符串格式化。

因此将系统提示、查询、默认任务和技能正文作为具名字符串放在对应 workflow 模块旁的同名 YAML 中。
steps/proactive/skills/ 仅存放三个可复用操作技能 YAML；角色系统提示、record_skill、notebook_skill 和 default_task 仍放在对应模块的同名 YAML。
不新增 prompts/、tasks/ 目录，也不复制 PromptHandler 的继承和语言规则。

| 所属模块与模板 | 主要模板键 | 继承和输出 |
| --- | --- | --- |
| chat/chat.py / chat.yaml | system_prompt、system_prompt_zh | 前台 Chat 的系统提示；提醒 Step 若继承 Chat 基类，可复用这些键。 |
| chat/notifications.py / notifications.yaml | hint、hint_zh | 提醒 workflow 自己的输入与决策提示。 |
| task/base.py / base.yaml | system_prompt、task_query、event_input、record_skill、repair_record_query 及其语言版本 | BaseProactiveAgentStep 共用的执行和记录方法，供四个派生继承。 |
| task/memory.py / memory.yaml | default_task、default_task_zh，以及确有需要的领域提示键 | Memory 的默认 TASK.md 内容。 |
| task/notebook.py / notebook.yaml | default_task、notebook_skill 及其语言版本 | Notebook 的默认任务要求和本轮自动跟进技能；工具参数指导复用内置 manage-notebook。 |
| task/finance.py / finance.yaml | default_task 及其语言版本，需要时追加领域提示键 | 金融任务的默认要求。 |
| task/news.py / news.yaml | default_task 及其语言版本，需要时追加领域提示键 | AI 早报的默认要求。 |

不同子类按实际需要补充或覆盖键，不复制父类整份流程。纯调度 Step 不使用提示，不为它创建空的 YAML。
例如所有主动 Agent 从 base.yaml 继承 record_skill_zh，Notebook 再从 notebook.yaml 取得 notebook_skill_zh。

PromptHandler 只加载和渲染字符串，不负责写 Markdown：

- SKILL.md：执行 workflow 用 prompt_format 渲染 record_skill、notebook_skill 等内容，交给现有 SkillSpec / 技能发布机制生成，调用结束后清理本轮临时技能。
- TASK.md：创建或初始化任务的 workflow 从对应 Step 的 PromptHandler 取得 default_task，交给 Manager / Runtime 以 create-only 方式保存；组件不另行扫描或解析模板目录。
- 显式提供 task_description 时以用户内容为准；重启和后续执行读取已有 TASK.md，不因模板更新或语言切换覆盖用户修改。

任务初始化不调用模型，也不在 Runtime 中长期保存 Step；静态预设在开始消费输入前完成默认任务初始化。
YAML 是初始模板，生成后的 TASK.md 是用户工作区中的持久要求；临时 SKILL.md 可由模板重新渲染。
运行时产物不作为仓库里的模板源文件提交。

## 8. 目标代码目录

以下只列本次计划调整的源代码目录，工作区数据布局沿用 resource_dir、reme_agent_dir、daily_dir 等配置。
现有 AgentWrapper / App Server 继续作为依赖复用，本次不调整其实现，因此不列入改造目录图。
schema 部分额外列出现有全部模块及候选新增模块，具体对象和复用判断见第 9 点。

组件与 workflow 都按 proactive 子系统归组：components/proactive/ 管长期资源与状态，steps/proactive/ 管执行流程。
组件下分 inbox/、outbox/、runtime/、manager/，文件名在包内使用 base.py、chat.py 等简短名称，类名沿用前文定义。
runtime/chat.py 对应 ChatAgentRuntime，runtime/task.py 对应 ProactiveAgentRuntime；Memory 等四种领域差异继续放在 Step。

```text
reme/
├── components/
│   └── proactive/
│       ├── inbox/
│       │   ├── __init__.py
│       │   ├── base.py  # BaseInbox：持久队列与处理回执
│       │   ├── event.py
│       │   └── notification.py
│       ├── manager/
│       │   ├── __init__.py
│       │   ├── _registry.py
│       │   ├── base.py
│       │   └── local.py
│       ├── outbox/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── chat.py
│       │   └── notification.py
│       ├── runtime/
│       │   ├── __init__.py
│       │   ├── _live_turn.py
│       │   ├── _sessions.py
│       │   ├── base.py  # BaseAgentRuntime：资源生命周期与执行保护
│       │   ├── chat.py  # ChatAgentRuntime：Remy
│       │   └── task.py  # ProactiveAgentRuntime：事件 / cron
│       └── __init__.py
├── steps/
│   └── proactive/
│       ├── chat/
│       │   ├── __init__.py
│       │   ├── chat.py
│       │   ├── chat.yaml
│       │   ├── notifications.py
│       │   └── notifications.yaml
│       ├── connector/  # 采集与发布 workflow，不是组件
│       │   ├── finance/
│       │   │   ├── __init__.py
│       │   │   ├── base_finance.py
│       │   │   ├── base_quotes.py
│       │   │   ├── btc_quotes.py
│       │   │   ├── finance_news.py
│       │   │   └── tencent_quotes.py
│       │   ├── harness/
│       │   │   ├── __init__.py
│       │   │   ├── _codex_dialogue.py
│       │   │   ├── base_harness_connector.py
│       │   │   └── codex_dialogue.py
│       │   ├── __init__.py
│       │   ├── base_connector.py
│       │   ├── screen_capture.py
│       │   └── screen_capture.yaml
│       ├── skills/  # 三个 YAML，各含 name、description、content
│       │   ├── manage_connector.yaml
│       │   ├── manage_notebook.yaml
│       │   └── manage_proactive_runtime.yaml
│       ├── task/
│       │   ├── __init__.py
│       │   ├── base.py  # BaseProactiveAgentStep：单轮执行与提交
│       │   ├── base.yaml
│       │   ├── finance.py
│       │   ├── finance.yaml
│       │   ├── initialize.py  # 从领域 YAML 创建缺失的 TASK.md
│       │   ├── memory.py
│       │   ├── memory.yaml
│       │   ├── news.py
│       │   ├── news.yaml
│       │   ├── notebook.py
│       │   └── notebook.yaml
│       ├── tools/  # 四个模型工具与 Notebook 存储辅助
│       │   ├── __init__.py
│       │   ├── _notebook_store.py
│       │   ├── manage_connector.py
│       │   ├── manage_notebook.py
│       │   ├── manage_proactive_runtime.py
│       │   └── submit_chat_reply.py
│       ├── __init__.py
│       ├── api.py  # 服务入口与请求拥有的订阅
│       ├── base.py  # BaseAgentStep：共用依赖、get_skills 与只读查询
│       └── loop.py  # 后台任务调度 Step
├── schema/  # 完整数据类与类型别名清单见第 9 点
│   ├── __init__.py
│   ├── agent_activity.py
│   ├── agent_inbox.py
│   ├── agent_message.py
│   ├── agent_outbox.py
│   ├── agent_reply.py
│   ├── agent_session.py
│   ├── agent_skill.py
│   ├── agent_view.py
│   ├── application_config.py
│   ├── compaction.py
│   ├── dream.py
│   ├── emb_node.py
│   ├── event.py
│   ├── file_chunk.py
│   ├── file_front_matter.py
│   ├── file_link.py
│   ├── file_node.py
│   ├── finance.py
│   ├── graph_snapshot.py
│   ├── idea.py
│   ├── job_execution.py
│   ├── notebook.py
│   ├── proactive.py
│   ├── proactive_agent.py
│   ├── proactive_runtime.py
│   ├── request.py
│   ├── resource_metadata.py
│   ├── response.py
│   ├── speech.py
│   ├── stream_chunk.py
│   ├── todo.py
│   ├── token_usage.py
│   └── traverse_graph.py
└── config/
    └── proactive.yaml
```

每个实际 Python 包都补齐 __init__.py 与注册导入。
components/__init__.py 导入 proactive，proactive/__init__.py 再导入四组组件实现，各子包导出对应基类和具体实现。
proactive 是源码包的归组名称；注册仍按各类声明的 component_type 和 backend，配置保持本方案的 agent_runtime、agent_manager 等类型键，不新增 components.proactive 配置层级。
实施时同步迁移 Python 导入、类型声明和注册入口；源码包路径调整不改变实例名称、依赖身份或工作区数据位置。
steps/proactive/__init__.py 统一导入本子系统的 workflow 和 Step 实现，steps/__init__.py 保留其注册入口。
通用文件和文档工具继续位于现有 utils 下，schema 不随组件移动。
Chat 与后台任务的长期组件统一位于 components/proactive/runtime/，不再平行增加角色包或四个领域 Runtime 包。
Connector 只出现在 steps/proactive/connector/，采集、模型调用和调度循环继续由 workflow 执行。

### 已删除的示例文件

以下两个文件已删除；技能渲染、修复复用与清理由正式任务流程及其测试接管：

| 计划删除的文件 | 先迁移的内容 | 正式归属 |
| --- | --- | --- |
| reme/steps/proactive/example.py | SkillProactiveAgentStep 中的技能渲染、按轮注入及修复期间复用同一调用范围。 | steps/proactive/task/base.py；通过现有 SkillSpec 和调用范围管理技能。 |
| reme/steps/proactive/example.yaml | record_skill_zh，以及仍适用的系统提示和本轮业务输入模板。 | steps/proactive/task/base.yaml；由公共基类和领域派生按 PromptHandler 继承。 |

example_config、独立命令行入口、示例常量和仅用于演示的工具参数裁剪随示例移除，不搬进正式公共工具配置。
旧示例类和专用测试已删除。
将该测试中记录 / 提醒提交、跳过、修复重试、输入确认和临时技能清理的覆盖迁入正式任务执行测试，
移除对 SkillProactiveAgentStep、example_config 和示例常量的依赖；仅验证演示配置的断言随入口删除。
正式模板、执行流程与迁移后的测试接管了这两份示例的能力；旧导入和命令入口已清理。

## 9. Schema 对象清单与复用

当前 reme/schema/ 下除 __init__.py 外有 33 个模块、92 个类；包含 TypedDict 和私有请求辅助类，重新导出不重复计数。
以下列出全部对象及职责；字段和验证规则以对应源码为准，不在文档复制一套定义。

### 当前数据类清单

| 当前模块 | 数据类 | 职责与本次处理 |
| --- | --- | --- |
| [agent_activity.py](../../reme/schema/agent_activity.py) | `AgentActivity` | 一轮 Agent 执行的临时活动状态；Chat 和后台任务共用。 |
| [agent_inbox.py](../../reme/schema/agent_inbox.py) | `InboxItem`、`InboxReceipt`、`NotificationRef` | 通用输入条目和已处理批次的持久回执；复用存储与处理提交契约。 |
| [agent_outbox.py](../../reme/schema/agent_outbox.py) | `NotificationDelivery`、`ChatDelivery` | 分别保存提醒交付与原生 Chat 消息交付，不重复维护接收方处理状态。 |
| [agent_message.py](../../reme/schema/agent_message.py) | `ClientModel`、`TextItem`、`SummaryItem`、`ToolAction`、`ToolItem`、`ArtifactItem`、`AgentMessage` | 客户端消息、显示块及公共模型约束；Chat、任务查询和订阅继续共用。 |
| [agent_reply.py](../../reme/schema/agent_reply.py) | `AgentReply` | 模型调用终态、结果、用量及完整事件轨迹；沿用现有 Wrapper 协议。 |
| [agent_session.py](../../reme/schema/agent_session.py) | `SessionStatus`、`SessionMessage`、`SessionSnapshot` | 后端中立的会话历史和快照；继续由服务投影为客户端视图。 |
| [agent_skill.py](../../reme/schema/agent_skill.py) | `SkillSpec` | 已有技能引用或本轮生成的技能正文；公共 record skill 与 notebook skill 复用。 |
| [agent_view.py](../../reme/schema/agent_view.py) | `AgentView`、`AgentRequest`、`AskAgentRequest`、`TaskDocumentRequest`、`ClientActivity`、`AgentSessionView`、`AgentSnapshot`、`AgentUpdate`、`AgentHeartbeat`、`ClientInfo`、`AgentPageRequest`、`AgentInboxRequest`、`ListTaskRemindersRequest`、`AgentReminderRequest` | Agent 列表、会话查询、订阅、分页及原始任务文档读写的公共契约；服务入口继续复用。 |
| [application_config.py](../../reme/schema/application_config.py) | `ComponentConfig`、`JobConfig`、`ApplicationConfig` | 应用装配、组件与 Job 配置；五个 Runtime 使用现有配置容器。 |
| [compaction.py](../../reme/schema/compaction.py) | `CompactionOptions`、`CompactionRequest`、`CallDecision`、`CompactionResult` | 上下文压缩请求、预算及替换建议；保持独立能力，不作为 Inbox 或任务执行状态。 |
| [dream.py](../../reme/schema/dream.py) | `DreamUnit`、`DreamExtractOutput`、`IntegrateOutcome`、`DreamState` | 跨文件记忆抽取、整合和 dream workflow 状态；按原功能保留。 |
| [emb_node.py](../../reme/schema/emb_node.py) | `EmbNode` | 文本、向量与检索元数据；继续供检索能力使用。 |
| [event.py](../../reme/schema/event.py) | `Event` | Connector 和框架定时触发的统一观察输入；直接复用。 |
| [file_chunk.py](../../reme/schema/file_chunk.py) | `FileChunk` | 带文件位置和检索评分的文本块；不另造 Agent 检索结果类型。 |
| [file_front_matter.py](../../reme/schema/file_front_matter.py) | `FileFrontMatter` | Markdown 的通用描述元数据；提醒决策继续使用专用校验。 |
| [file_link.py](../../reme/schema/file_link.py) | `FileLink` | 文件间链接与锚点；沿用现有文件能力。 |
| [file_node.py](../../reme/schema/file_node.py) | `FileNode` | 工作区文件、版本、链接和分块引用；沿用现有索引能力。 |
| [finance.py](../../reme/schema/finance.py) | `FinanceNewsItem`、`MarketQuote`、`CryptoQuote` | 财经新闻、以前收盘价为基准的行情、滚动 24 小时加密行情；金融 Connector 直接复用。 |
| [graph_snapshot.py](../../reme/schema/graph_snapshot.py) | `GraphSnapshotNode`、`GraphSnapshotEdge`、`GraphSnapshot` | 分类根图快照；保留原有展示语义。 |
| [idea.py](../../reme/schema/idea.py) | `Idea` | 项目想法记录；Notebook 工具直接复用。 |
| [job_execution.py](../../reme/schema/job_execution.py) | `JobExecution` | Job 执行历史和当前 Step 路径；沿用现有 Job 跟踪，不承担 Runtime 持久状态。 |
| [notebook.py](../../reme/schema/notebook.py) | `TodoProgress`、`_RecordRequest`、`TodoRequest`、`IdeaRequest`、`NotebookProjectsRequest` | Todo 进展、工具请求及私有请求基类；工具移到 tools/ 后继续复用。 |
| [proactive.py](../../reme/schema/proactive.py) | `ProactiveTopic`、`ProactiveState`、`ProactiveStateFile`、`ProactiveResult` | 现有 steps/evolve/proactive/ 的话题发现、链内状态及工作区文件；保留给该功能，不充当新 AgentRuntime 状态。 |
| [proactive_agent.py](../../reme/schema/proactive_agent.py) | `ManageConnectorRequest`、`NotificationWindow`、`ProactiveRuntimeSettings`、`ProactiveRuntimeConfig`、`CreateProactiveRuntimeConfig`、`ProactiveRuntimeDefinition`、`ProactiveRecordDecision`、`ManageProactiveRuntimeRequest` | 后台任务配置、定义、记录决策和工具请求；复用并按新 Job 选择及工具语义调整。 |
| [proactive_runtime.py](../../reme/schema/proactive_runtime.py) | `NotificationConfig`、`AgentSession`、`ChatAgentConfig`、`AgentReminder`、`ProcessChatNotificationsRequest`、`SubmitChatReplyRequest` | 会话引用、Remy 配置、提醒策略、聚合提醒及请求；复用基础对象，拆开提醒状态归属。 |
| [request.py](../../reme/schema/request.py) | `Request` | 通用服务请求信封；具体操作继续使用专用请求验证。 |
| [resource_metadata.py](../../reme/schema/resource_metadata.py) | `ResourceFile`、`ResourceMetadata` | 资源包文件引用与 manifest；所有 Connector 共用。 |
| [response.py](../../reme/schema/response.py) | `Response` | 通用 Job 响应信封；继续复用。 |
| [speech.py](../../reme/schema/speech.py) | `ASRResult`、`TTSResult` | 语音识别与合成结果；保持独立接口。 |
| [stream_chunk.py](../../reme/schema/stream_chunk.py) | `StreamChunk` | 增量输出、调用边界和终态事件；继续作为 Wrapper 和 Job 的流式基础契约。 |
| [todo.py](../../reme/schema/todo.py) | `Todo` | 待办记录及规划字段；Notebook 工具直接复用。 |
| [token_usage.py](../../reme/schema/token_usage.py) | `TokenUsage` | 已有用量统计对象；按现有转换使用，不因目录调整改写 AgentReply 的 usage/raw 协议。 |
| [traverse_graph.py](../../reme/schema/traverse_graph.py) | `TraverseGraphNode`、`TraverseGraphEdge`、`TraverseGraph` | 按根节点、深度与方向限制的链接遍历结果；保留独立查询契约。 |

### 全部现有类型别名

| 模块 | 类型别名 |
| --- | --- |
| agent_activity.py | `ActivitySource`、`ActivityPhase` |
| agent_message.py | `ItemStatus`、`TranscriptItem` |
| agent_reply.py | `AgentReplyStatus` |
| job_execution.py | `JobExecutionStatus` |
| notebook.py | `NotebookProject`、`TodoStatus`、`ManageNotebookRequest` |
| proactive_runtime.py | `AgentName`、`AgentType`、`ChatMessageId`、`InboxPriority` |
| speech.py | `AudioFormat` |
| stream_chunk.py | `StreamStatus` |

AgentName、ChatMessageId、InboxPriority 等身份与枚举约束继续共用。
AgentType 当前是 chat/event/time，描述客户端看到的类型；它不等于 Runtime 组件的 chat/proactive 两个 backend。
Cron 复用同一种后台 Runtime，并不要求把公共 AgentType 的值直接改成 proactive；当前客户端协议先保留。

### 直接复用与需要调整的边界

1. Connector 的输入输出复用 Event、ResourceFile、ResourceMetadata 和三个金融数据对象；不新增重复的 ConnectorEvent、ResourcePackage 或报价对象。
2. BaseInbox 复用 InboxItem 的通用存储约束和 InboxReceipt 的处理提交契约；EventInbox 仍按 Event 校验，NotificationInbox 按通知引用校验，不强制所有载荷继承 InboxItem。
3. 会话引用复用 AgentSession，活动复用 AgentActivity，模型输出复用 AgentReply / StreamChunk，客户端响应复用现有会话和 Agent 视图；不新增一整套 RuntimeSession、RuntimeActivity 或 RuntimeReply。
4. Runtime 配置复用 ProactiveRuntimeSettings / Config / Definition 的继承和验证。将受限 step_backend 选择迁移为允许的 run_job，补充提醒目标等必要配置，保留 Notebook 的触发限制和批次校验。
   CreateProactiveRuntimeConfig 与创建请求同时支持显式 task_description 或已选领域的默认 TASK.md 模板；已有 TASK.md 仍优先。
5. 管理和提醒请求按下面的映射改造，模型工具计数及配置名称仍以第 6 点为准。

| 现有请求对象 | 目标请求对象 | 复用与修改 |
| --- | --- | --- |
| ManageProactiveAgentRequest | ManageProactiveRuntimeRequest | 复用动作、身份和设置校验；补充受限执行 Job 选择及默认任务创建规则。 |
| AppendTaskEventsRequest | ManageConnectorRequest | send 分支复用 task_names、events、去重和禁止手动 source=cron 的校验；enable/pause 分支只接收已配置采集 Job 的身份，不要求事件参数。 |
| ProcessChatRemindersRequest | ProcessChatNotificationsRequest | 保留框架选定批次的等级、源记录引用及数量限制，属于内部处理请求。 |
| SubmitChatReminderRequest | SubmitChatReplyRequest | 在现有 content 约束上显式区分 send/skip；send 校验非空正文，skip 校验静默分支，不再靠空正文猜动作。 |
| TodoRequest、IdeaRequest | ManageNotebookRequest 的记录分支 | 用 kind 区分 Todo / Idea，复用原字段和动作校验；公共 list_projects 使用小请求分支，默认统计两类记录。 |

这些是现有请求的迁移，不在新旧模块各复制一份。具体请求继续就近放在现有 proactive_agent.py / proactive_runtime.py；
本轮不因组件目录改名而新增空的 schema/agent_runtime.py，也不为 Connector 另建只有重复载荷的 schema/connector.py。
如果实施时控制请求确有独立体量，再按请求职责拆分模块。
Notebook 的统一请求与公共项目查询分支就近放在 schema/notebook.py，Todo / Idea 数据模型及存储不重复定义。

### AgentReminder 拆分载荷与状态

当前 AgentReminder 同时包含 priority、message、源记录路径 id、delivered_at、read_at 和 chat_message_ids。
旧 ReminderState 已删除，发送与接收状态改由各自的所有者保存。
这两个对象不能整体作为新的 NotificationInbox / NotificationOutbox 存储模型。

| 目标对象或能力 | 归属 | 复用方式 |
| --- | --- | --- |
| NotificationRef（新增） | agent_inbox.py，双方共享的通知引用载荷 | 引用发送方、源 daily 记录和稳定条目身份；等级与正文从源记录解析，本轮快照单独校验版本。 |
| NotificationDelivery（新增） | agent_outbox.py，NotificationOutbox 持有 | 提醒引用、投递目标、发送确认及恢复所需状态；不保存接收方的处理状态。 |
| ChatDelivery（新增） | agent_outbox.py，ChatOutbox 持有 | 待提交回复、目标会话与版本约束、原生消息关联和交付结果。 |
| InboxReceipt 与接收方处理状态 | NotificationInbox 持有 | 按批次确认发送或静默处理；复用已有处理回执，不将其当作 Outbox 发送确认。 |
| AgentReminder | 服务聚合读取视图 | 从源文件及各状态所有者组合现有读取结果；read_at 表达 Chat 已处理，不能解释为用户已读。 |

AgentReminder.id 当前是源记录路径，InboxItem.id 则有 128 字符上限；不能直接将任意源路径塞进通用条目 ID。
内部队列身份与源路径引用分别处理，外部 AgentReminder 的路径 ID 语义保持明确。
这三个新增对象按第 11 点的字段和状态实现；不为每个组件添加一个空的 Base 数据模型。
现有 ReminderState 中的 read_at 迁入接收方处理状态，chat_message_ids 由 ChatOutbox 的交付关联提供，发送方不再重复维护。

### 相似但不合并的对象

- AgentSession 是持久会话引用；SessionSnapshot 是后端中立的历史快照；AgentSessionView 是客户端投影，三者继续分别承担各自契约。
- SessionMessage 保留原始 StreamChunk，AgentMessage 使用显示块 TranscriptItem；沿用现有投影关系，不因字段相似合成一个消息模型。
- AgentActivity 与 ClientActivity 分别表达执行状态与公开视图，沿用字段转换；JobExecution 跟踪 Job，不能代替 Agent 的活动状态。
- Event 是外部观察，NotificationRef 是任务产出的提醒引用；两者共享收件机制，保持各自载荷语义。
- MarketQuote 使用前收盘价，CryptoQuote 使用滚动 24 小时窗口；保持两套基准和时间校验。
- GraphSnapshot 与 TraverseGraph 分别服务分类快照和有界遍历，不在本轮抽成通用图结果。
- ProactiveState 与 DreamState 是各自 workflow 的执行上下文，不复用为应用生命周期状态，也不替代 Runtime 组件字段。

Schema 核查只影响本次计划的对象复用与迁移范围；其他功能现有模型不随 Agent 重构改写。

## 10. proactive.yaml 初始化与执行关系

本节只列与上述组件、Step、四个工具和技能直接相关的目标配置，不复制整个 proactive.yaml。
以下片段已落实到正式配置；完整参数以配置文件和 Schema 为准。模型服务、索引、文件 / 语音工具和 HTTP 服务等配置继续沿用当前定义。
下面省略各 Job 的 parameters JSON Schema：现有服务请求保持原契约，四个模型工具按第 6、9 点的请求分支同步填写，不能把这些片段当作完整可运行配置。

### 共享 Wrapper、五个 Runtime 与 Manager

Wrapper 的原生执行配置保持现有值，技能改由执行角色选择。只有 Runtime 和 Manager 显式列为这组应用组件；
Inbox / Outbox 由所属 Runtime 初始化，Connector 仍配置为 Job。

```yaml
components:
  agent_wrapper:
    default:
      backend: codex
      app_server: default
      project_path: ..
      cwd: .
      model: ${REME_MODEL_NAME:-qwen4-max}
      effort: xhigh
      approval_mode: deny_all
      sandbox: workspace-write
      skills: []  # 各角色在执行 Job 中选择技能

  agent_runtime:
    Remy:
      backend: chat
      agent_wrapper: default
      run_job: run_chat_agent
      enabled: true
      initial_memory: |
        主人关注 AI 相关的大新闻，以及当前项目的进度、进展、难点和待讨论问题。
      notification_config:
        p0_delay_seconds: 0
        p1_idle_seconds: 10
        p2_idle_seconds: 1800
    proactive_memory:
      backend: proactive
      agent_wrapper: default
      run_job: run_memory_agent
      notification_target: Remy
      enabled: true
      max_batch_events: 2
    proactive_note:
      backend: proactive
      agent_wrapper: default
      run_job: run_notebook_agent
      notification_target: Remy
      enabled: true
      max_batch_events: 2
    proactive_fin:
      backend: proactive
      agent_wrapper: default
      run_job: run_finance_agent
      notification_target: Remy
      enabled: true
      min_trigger_interval_seconds: 10
      max_batch_events: 20
    proactive_news:
      backend: proactive
      agent_wrapper: default
      run_job: run_news_agent
      notification_target: Remy
      enabled: true
      cron: "0 8 * * *"

  agent_manager:
    default:
      backend: local
      agent_wrapper: default
      notification_target: Remy
      runtimes: [proactive_memory, proactive_note, proactive_fin, proactive_news]
      allowed_run_jobs:
        - run_memory_agent
        - run_notebook_agent
        - run_finance_agent
        - run_news_agent
```

notification_config 属于 Remy 的接收策略，allowed_run_jobs 是 Manager 接受动态任务选择的执行 Job 白名单；这两个配置字段随组件契约一起实现。
Manager 的 agent_wrapper / notification_target 是创建动态 Runtime 时使用的非拥有依赖；共享资源仍由 Application 管理，Manager 不接管 Remy 状态。
每个后台 Runtime 的批次限额和其他默认参数一起迁移，首次通过对应领域 Step 的 default_task 模板生成自己的 TASK.md。
四份默认任务正文从当前配置迁入各领域 YAML，金融标的、涨跌基准与阈值、项目跟进范围和早报要求一并保留。
创建时提供显式 task_description 的任务以用户内容为准，不被默认模板覆盖。
组件名、持久化身份和工作区目录必须一致；同名静态实例与恢复定义不能各创建一个。

### 对话、提醒、四种任务与后台调度

以下八个 Job 是内部执行入口；对外对话和订阅仍由现有服务入口承接。
内部 run_chat_agent 使用普通 Job 返回 Response，并转发本轮流式回调 / 队列；外部 ask_agent 继续使用 StreamJob，负责服务流的终止。

```yaml
jobs:
  run_chat_agent:
    backend: base
    enable_serve: false
    steps:
      - backend: chat_agent_step
        agent_runtime: Remy
        job_tools: [manage_proactive_runtime, manage_connector, manage_notebook]
        skills:
          - manage-proactive-runtime
          - manage-connector
          - manage-notebook

  process_chat_notifications:
    backend: base
    enable_serve: false
    steps:
      - backend: process_chat_notifications_step
        agent_runtime: Remy
        agent_manager: default
        job_tools: [submit_chat_reply]
        skills: []
        timeout_seconds: 120

  chat_notifications_loop:
    backend: background
    enable_serve: false
    steps:
      - backend: chat_notifications_loop_step
        agent_runtime: Remy
        agent_manager: default
        process_job: process_chat_notifications

  run_memory_agent:
    backend: base
    enable_serve: false
    track_execution: true
    steps:
      - &task_execution
        backend: memory_agent_step
        agent_manager: default
        effort: low
        timeout_seconds: 1200
        max_local_search_calls: 10
        recent_history_hours: 24
        sandbox: null
        job_tools: []
        skills: []

  run_notebook_agent:
    backend: base
    enable_serve: false
    track_execution: true
    steps:
      - <<: *task_execution
        backend: notebook_agent_step
        job_tools: [manage_notebook]
        skills:
          - manage-notebook

  run_finance_agent:
    backend: base
    enable_serve: false
    track_execution: true
    steps:
      - <<: *task_execution
        backend: finance_agent_step

  run_news_agent:
    backend: base
    enable_serve: false
    track_execution: true
    steps:
      - <<: *task_execution
        backend: news_agent_step

  agent_manager_loop:
    backend: background
    enable_serve: false
    steps:
      - backend: agent_manager_loop_step
        agent_manager: default
```

四个任务执行 Step 都由 BaseProactiveAgentStep 注入本轮 proactive-record 技能，Notebook 再注入本轮跟进技能；生成内容不静态写进配置。
job_tools 只列通过 ReMe Job 暴露给模型的工具。文件读取、写入和联网检索继续使用所选 Wrapper 已配置的原生能力，不引用本配置中尚不存在的文件工具 Job。
Step 中的 agent_runtime / agent_manager 是组件引用，执行请求中的 agent_name 则选择当前处理的 Runtime。

提醒 workflow 冷启动 Remy 会话时，应使用正式 Chat Job 的工具和技能绑定准备会话，再为临时提醒 fork 选择 submit_chat_reply。
提醒调度与单轮处理 Step 通过 Manager 读取发送方当前的 enabled / notify_window 等资格，Remy Runtime 自身不反向依赖 Manager。
不能用提醒 fork 的工具集合覆盖前台会话配置。ask_agent 调用内部 Chat Job 时显式传递输出回调 / 流式上下文，
保留客户端投影和终止语义；当前 BaseStep.run_job 不会自动共享调用者的 RuntimeContext。

### 四个模型工具 Job

对应文件均位于 steps/proactive/tools/。这里只列注册与组件绑定；完整参数按第 6、9 点的请求契约补齐到各 Job.parameters。

```yaml
jobs:
  manage_proactive_runtime:
    backend: base
    description: 管理后台 Runtime、任务要求、运行设置和只读查询。
    steps:
      - backend: manage_proactive_runtime_step
        agent_manager: default

  manage_connector:
    backend: base
    description: 发送标准 Event，或开启、暂停已配置的采集 Job。
    steps:
      - backend: manage_connector_step
        agent_manager: default
        connector_jobs:
          - screen_capture_cron
          - codex_dialogue_cron
          - finance_news_cron
          - gold_quotes_cron
          - oil_quotes_cron
          - nasdaq_quotes_cron
          - btc_quotes_cron

  submit_chat_reply:
    backend: base
    enable_serve: false
    description: 提交当前提醒批次的 Chat 回复或静默决定。
    steps:
      - backend: submit_chat_reply_step
        agent_runtime: Remy

  manage_notebook:
    backend: base
    description: 按 kind 和 action 操作 Todo、Idea，或汇总 Notebook 项目。
    steps:
      - backend: manage_notebook_step
```

manage_proactive_runtime 的 run_job 参数只能选择 Manager 的 allowed_run_jobs；不能指定任意服务 Job。
manage_connector 的 enable / pause 请求用 connector_job 指定 connector_jobs 中的采集 Job；send 仍使用 task_names 和 events，不要求 connector_job。
四个工具的参数分支如下，服务端继续拒绝不适用当前动作的字段：

| 工具 | 请求对象 | 主要参数分支 |
| --- | --- | --- |
| manage_proactive_runtime | ManageProactiveRuntimeRequest | action、agent_name，以及动作允许的 task_description / run_job / 运行设置或 question。 |
| manage_connector | ManageConnectorRequest | send 使用 task_names / events；enable / pause 使用 connector_job。 |
| submit_chat_reply | SubmitChatReplyRequest | send 使用非空 content；skip 提交静默决定，批次与会话身份由框架注入。 |
| manage_notebook | ManageNotebookRequest | Todo / Idea 操作使用 kind 与对应 action；list_projects 只接受可选 scope。 |

### 七个 Connector 采集 Job

采集调度、目标任务和现有采集参数保留。connector_job 是计划新增的 Step 配置，
将采集调用与 manage_connector 的持久启停状态关联；它使用配置中的 Job 名，不另建一套 Connector 组件身份。
这些 Job 保持启动，暂停状态由基础采集 Step 在自动采集和投递前检查，恢复时沿用游标和待投递批次。

```yaml
jobs:
  screen_capture_cron:
    backend: cron
    enabled: true
    cron: "*/5 * * * *"
    steps:
      - backend: screen_capture_step
        agent_manager: default
        connector_job: screen_capture_cron
        similarity_threshold: 0.99
        task_names: [proactive_memory]

  codex_dialogue_cron:
    backend: cron
    enabled: true
    cron: "*/5 * * * *"
    init_steps:
      - backend: codex_dialogue_init_step
        agent_manager: default
        connector_job: codex_dialogue_cron
        codex_home: &codex_home ${CODEX_HOME:-~/.codex}
    steps:
      - backend: codex_dialogue_step
        agent_manager: default
        connector_job: codex_dialogue_cron
        codex_home: *codex_home
        task_names: [proactive_memory, proactive_note]

  finance_news_cron:
    backend: cron
    enabled: true
    cron: "*/5 * * * *"
    steps:
      - backend: finance_news_step
        agent_manager: default
        connector_job: finance_news_cron
        task_names: [proactive_fin]

  gold_quotes_cron:
    backend: cron
    enabled: true
    cron: "*/5 * * * *"
    steps:
      - backend: tencent_quotes_step
        agent_manager: default
        connector_job: gold_quotes_cron
        instrument: gold
        task_names: [proactive_fin]

  oil_quotes_cron:
    backend: cron
    enabled: true
    cron: "*/5 * * * *"
    steps:
      - backend: tencent_quotes_step
        agent_manager: default
        connector_job: oil_quotes_cron
        instrument: oil
        task_names: [proactive_fin]

  nasdaq_quotes_cron:
    backend: cron
    enabled: true
    cron: "*/5 * * * *"
    steps:
      - backend: tencent_quotes_step
        agent_manager: default
        connector_job: nasdaq_quotes_cron
        instrument: nasdaq
        task_names: [proactive_fin]

  btc_quotes_cron:
    backend: cron
    enabled: true
    cron: "*/5 * * * *"
    steps:
      - backend: btc_quotes_step
        agent_manager: default
        connector_job: btc_quotes_cron
        task_names: [proactive_fin]
```

Codex 的 init_steps 沿用建立首次基线的语义，恢复已有进度时不得重建基线而覆盖游标，且遵守同一采集启停状态。
开启 / 暂停只操作已配置且启动的采集 Job 的持久执行开关，不把它等同于修改 Job.enabled 或动态创建 / 删除 Job。

### 新增绑定字段的实现归属

| 配置字段 | 实现归属与含义 |
| --- | --- |
| Runtime.notification_config | ChatAgentRuntime 使用 NotificationConfig 验证接收方的唤醒策略。 |
| Manager.allowed_run_jobs | AgentManager 校验动态任务允许选择的领域执行 Job。 |
| Step.agent_runtime / agent_manager | 对应 Step 通过 Ref / bind 获取应用组件，不能按源码包名猜组件身份。 |
| chat_notifications_loop Step.process_job | 提醒调度 Step 调用的单轮处理 Job 名。 |
| manage_connector Step.connector_jobs | 工具允许控制的已配置采集 Job 名单。 |
| Connector Step.connector_job | 基础采集流程及 Harness 初始化关联的持久启停状态键，与实际采集 Job 名保持一致。 |
| 执行 Step.skills | 各角色将配置的原生技能选择转交 Wrapper，再叠加本轮生成的技能；不修改 AgentWrapper 协议。 |

这些绑定随相应组件和 Step 实现，YAML 能解析不代表当前代码已支持目标配置。

### 保留的相关服务 Job

以下入口的请求、响应和服务可见性沿用当前配置，Step 实现迁到 steps/proactive/api.py，组件访问改为引用 Remy Runtime 和 AgentManager：

| Job 名 | backend | Step backend | 执行归属 |
| --- | --- | --- | --- |
| list_agents | base | agent_list_step | 聚合 Remy 与后台 Runtime 视图。 |
| read_agent_session | base | agent_session_step | 读取目标 Runtime 会话。 |
| watch_agent_session | stream | watch_agent_session_step | 订阅 Runtime 快照与变更；保留 stream_queue_size=1。 |
| ask_agent | stream | ask_agent_step | Remy 进入其 run_chat_agent；后台任务走共用只读查询。 |
| create_chat_session | base | agent_new_session_step | 使用 Remy 的正式 Chat 绑定准备新会话。 |
| list_agent_inbox | base | agent_inbox_step | 查询对应收件箱的处理视图。 |
| list_task_reminders | base | list_task_reminders_step | 查询后台任务提醒视图。 |
| read_agent_reminder | base | agent_reminder_step | 聚合源记录、投递与 Chat 处理关联。 |
| task_document | base | task_document_step | 读取 / 保存原始 TASK.md，保留版本与路径校验。 |

todo / idea 两个旧服务 Job 及 todo_step / idea_step 注册名删除，Studio 与其他调用方使用 manage_notebook 的 kind / action。

### 当前配置中替换和移出的部分

| 当前配置 | 目标处理 |
| --- | --- |
| components.agent_inbox.default | 移除全局实例配置，由各 Runtime 持有自己的 Inbox / Outbox。 |
| components.proactive_manager.default | 拆为五个 agent_runtime 组件及一个 agent_manager 组件。 |
| proactive_manager.chat | initial_memory 等 Chat 配置移入 agent_runtime.Remy。 |
| proactive_manager.runtimes | 改为四个具名 Runtime 组件；原任务正文迁入各领域 default_task 模板。 |
| run_proactive_agent 与 step_backend 分发 | 改为四个领域执行 Job，由 Runtime.run_job 选择，领域 Step 继承共用执行流程。 |
| proactive_loop | 改为 agent_manager_loop；源码文件为 steps/proactive/loop.py。 |
| process_chat_reminders | 改为 process_chat_notifications，并由独立 chat_notifications_loop 调度。 |
| manage_proactive_agent / append_task_events / submit_chat_reminder | 分别迁移为 manage_proactive_runtime / manage_connector / submit_chat_reply。 |
| todo / idea Job 与模型绑定 | 合并为 manage_notebook，调用方同步迁移，删除旧入口。 |
| Wrapper 全局的两份管理技能 | 改名后移到 Chat 执行绑定，并增加 manage-connector；后台 Notebook 单独选择 manage-notebook。 |

执行请求带 Runtime 名称，由 Job 获取对应实例的状态与能力。
普通用户请求通过 ask_agent 路由到 Chat Job；chat_notifications_loop 调度 process_chat_notifications 处理 Remy 收件箱；
agent_manager_loop 通过 AgentManager
找到可运行的后台 Runtime，再调用其配置的 run_job。
只读查询由管理工具和服务 API 复用 steps/proactive/base.py 的查询辅助方法。
Connector 由自己的 cron / 显式采集 Job 调用。

### 启动、依赖与关闭

静态依赖方向为 AgentWrapper → Remy Runtime → 后台 Runtime → AgentManager。
后台 Runtime 可以直接声明 Remy 为提醒目标，Remy 不反向持有 AgentManager；
Chat 管理工具在 Step 调用时获取 Manager，避免组件依赖环。

Manager 配置中的多个 Runtime 名称要转成框架能识别的显式依赖，不能把 Dependency 对象藏在列表里，
然后假设现有 bind 会自动递归解析。

Application 启动共享资源和各组件，再启动 Job。后台循环在 Job 启动后才分发执行。
关闭时先停止 Job 和新调用，再按所有权关闭动态 Runtime、静态 Runtime 及其收发件资源，
最后释放共享 Wrapper / App Server。启动失败按相同所有权回收已启动资源。

### 改造顺序与验收

1. 实现 Inbox / Outbox 的基类和两种语义，保留输入确认与发件交接恢复。
2. 建立 BaseAgentRuntime 及 Chat / Proactive 两个长期组件，隔离状态和文件锁。
3. 将现有 Manager 的职责拆到 Runtime、Outbox 和新的 AgentManager，明确静态与动态实例所有权。
   将 ChatNotifications 的状态、收发件操作与执行编排按第 2 点拆开，不机械搬移整个文件。
4. 保留并整理 Job / Agent Step 的执行流程，接入公共记录 skill 和四个领域派生；迁移示例能力与测试后，删除第 8 点列出的两个 example 文件。
5. 将 Connector workflow 统一到 steps/proactive/connector，复用发布与恢复流程，完成 Harness 基类和 Codex 接入。
6. 更新五个预设组件、管理工具、Job 绑定、schema、注册导入、客户端、中英文提示与文档，并按第 6 点同步迁移内置 skill 和引用。
7. 验证依赖拓扑、重复启动关闭、动态实例回收、任务隔离、Notebook 权限、技能选择与清理，以及各提交阶段的中断恢复。

## 11. 组件接口、状态与新增数据模型

本节固定需要跨模块调用的最小接口。具体私有辅助函数可以调整，不再把完整业务流程塞入组件，也不为每个动作新增一个组件。

### 单实例 Inbox 契约

一个 Inbox 对象只服务所属 Runtime，构造时接收已解析的工作区相对 directory；方法不再重复接收 name。
原 create / mount / forget 等多队列管理操作由 Runtime 初始化和关闭承担，任务枚举由 Manager 承担。

| 方法 | 返回与约束 |
| --- | --- |
| append(items, *, wait=False) | 返回已持久接收的 ID 列表。先完整验证该目标的批次，再写入；同 ID 同内容幂等，同 ID 不同内容拒绝。 |
| claim(limit=None, *, item_ids=None) | 返回 token 和 items，或无可领取输入时返回 None；每个 Inbox 同时最多一个有效领取。 |
| ack(token, *, metadata=None, retain=False) | 持久化 InboxReceipt 后才移除待处理输入、释放容量，返回 consume_ts 的 ISO 字符串。 |
| nack(token) | 释放尚未提交的领取，不删除输入。旧 token 不能确认后来重新领取的输入。 |
| snapshot() | 返回 receipt、pending_count、in_flight_count；pending_count 沿用当前语义，包含已领取但未确认的输入。 |
| read(*, state, offset=0, limit=20) | 只读分页，不创建会话、不领取、不确认。支持 all / pending / processing / processed。 |
| pending_ids() / receipts() | 返回本 Inbox 的待处理 ID 集合 / 保留回执列表；跨 Runtime 的聚合由调用方组织。 |
| retain(item_ids, *, release=False) | 保护结果尚未交接的输入和回执，交接完成后释放保护。 |
| wait(timeout=None) | 等待可领取输入或关闭；关闭时唤醒等待者。 |

存储复用当前 JSONL、原子替换、目录同步和单写者锁机制。EventInbox 保留 event_ts / id 排序及按 ID 重试；
NotificationInbox 由提醒 workflow 按等级选定 ID 后领取，不把优先级塞进 BaseInbox 的 FIFO 规则。
重启后未确认的领取重新变为 pending，旧 token 失效；落盘回执是是否已处理的判断依据。

| 默认容量 / 预算 | 目标值与范围 |
| --- | --- |
| max_pending_items | 10000，每个 Runtime 独立计数。 |
| max_pending_bytes | 16 MiB，每个 Runtime 独立计数。 |
| max_item_bytes | 32 KiB；大材料放资源文件。 |
| claim_max_items / claim_max_bytes | 20 / 64 KiB，再受领域 Runtime 的 max_batch_events 限制。 |
| append_timeout_seconds | 5 秒，仅 wait=True 的容量等待使用；管理投递和跨组件交接使用 wait=False。 |
| max_read_items | EventInbox 沿用 1000 条历史窗口；未完成交接的回执不能被裁剪。 |

NotificationInbox 的处理回执是接收方的持久事实，必须保留已处理 ID、时间、send / skip 和交付关联。
它的显示窗口可以限制为 1000 条，但不能沿用 EventInbox 的物理历史裁剪而丢掉通知的处理事实，否则重建后会重发旧提醒。
缓存和分页视图可以重建，通知处理回执本身不能当缓存删除。EventInbox 的去重保证仍限于保留的输入与历史，不宣称无限期去重。

损坏的完整 JSONL 记录应报告文件和错误，不能静默跳过；只恢复没有换行结束的尾部写入。
存储写入结果不确定时标记该 Inbox 不可继续写入，恢复或重启核对回执后再开放，不能用内存状态猜测成功。

### Runtime 与 Manager 的最小接口

| 所属类 | 接口与拥有的状态 |
| --- | --- |
| BaseAgentRuntime | start / close；execution() 执行租约；current_session / ensure_session / query_session；snapshot；stop_work。持有 Wrapper 的非拥有引用、Inbox / Outbox、会话辅助对象、活动调用与关闭状态。 |
| ChatAgentRuntime | 前台 turn 租约与 chat_revision；提醒 turn 的注册、失效与提交校验；前台活动 / 空闲时间；ChatOutbox 与 NotificationInbox。 |
| ProactiveAgentRuntime | 当前定义、事件批次、失败批次 ID、retry_at、调度缓存；commit_round 完成受保护提交；不包含组装提示、模型调用或领域判断。 |
| BaseAgentManager | resolve / list / definition / create / update / delete / status / deliver_events。静态引用和动态子组件分开登记；只管理后台 Runtime。 |
| LocalAgentManager | 通过 _registry.py 发现 / 保存定义，负责动态实例的创建失败回收、停止与恢复。 |

execution() 在排队前登记调用，关闭或删除可以取消正在等待锁的调用；离开调用范围时注销。
同一个后台 Runtime 的正式执行串行，不同 Runtime 可以并行；同一 Remy 的前台输入按现有顺序串行。
Runtime 只提供状态与可靠操作，Manager 不提供替代 agent_manager_loop 的常驻 run 循环。

Manager 创建动态 Runtime 时使用配置的 agent_wrapper / notification_target 非拥有引用，不从某个静态预设实例猜默认值。
静态与动态 Runtime 都使用同一组件实现，动态实例不能永久保留 Step、RuntimeContext 或模型调用回调。
配置中的 runtimes 列表逐项转换为显式 Dependency；传给子 Runtime 的共享对象是借用引用，不注册为其自有关闭资源。

### Outbox 的最小接口

BaseOutbox 提供 prepare(delivery)、pending()、deliver(delivery_id)、recover()、read(...) 及 start / close。
prepare 先持久化稳定身份和不可变发送内容，相同身份的冲突内容必须报错；deliver 是一次可恢复发送操作，组件里不启动后台轮询。
NotificationOutbox.deliver 调用接收方 Inbox.append(wait=False)；ChatOutbox.deliver 使用现有 Wrapper 的原生会话追加与版本检查。
恢复时先核对已持久化的交付事实，不重新调用模型生成另一份内容。

### 三个新增模型的确定字段

三个模型都使用 extra=forbid、带时区时间和合法工作区相对路径校验；schema_version 初始为 1。

| 模型 | 字段 |
| --- | --- |
| NotificationRef | schema_version、id、source_agent、record_path、created_at。id 使用源记录规范相对路径的稳定哈希生成，满足 InboxItem 的 128 字符限制。 |
| NotificationDelivery | schema_version、id、notification: NotificationRef、target_agent、status、created_at、delivered_at、last_error。status 为 prepared / delivered；id 由通知身份与目标身份稳定生成。 |
| ChatDelivery | schema_version、id、notification_ids、record_versions、session_id、chat_revision、session_revision、session_turn_id、content、message_id、status、created_at、delivered_at、acknowledged_at、last_error。status 为 prepared / sent / acknowledged / aborted。 |

record_versions 是源记录路径到内容摘要的映射，session_revision / session_turn_id 可为空，按空闲或运行中发送路径使用。
content 是本次 Chat 决策的非空回复，允许与多个源记录的提醒正文不同；这份发送意图是恢复依据，不是第二份日记源记录。
message_id 使用现有 ChatMessageId 约束，在 prepare 时确定并持久化，重试不能换 ID。

NotificationRef 不复制 priority 和 message：调度和本轮材料从 daily 解析 ProactiveRecordDecision，并记录读取时的文件版本。
因此待处理记录编辑后可以重新判断，不必把同 ID、不同正文伪装成第二条 Event。
已经处理的记录不会因为正文后来被修改而自动再次提醒；新的提醒应产生新的源记录。

AgentReminder 继续作对外聚合视图：id 是源路径，message / priority 来自源记录，read_at 来自接收方处理回执，chat_message_ids 来自 ChatDelivery 关联。
新增队列内部 ID 不替换现有客户端的路径 ID。内部模型和新工具结果不直接混入 AgentView / AgentMessage 的既有序列化字段。

## 12. 执行、交接与失败恢复协议

### 后台一轮任务

1. AgentManagerLoop 检查定义、触发、重试时间和未完成交接，选择 Runtime 的 run_job。输入为空时不调用模型。
2. 领域 Step 进入 Runtime 执行租约，冻结当前定义和本轮输入 ID；cron 一次领取一条，事件任务按 max_batch_events 领取。
3. 准备未占用的 daily 路径，读取 TASK.md 与材料，渲染本轮技能并创建本轮原生执行会话。会话引用登记在所属 Runtime。
4. 调用 Wrapper，保留终态、事件轨迹和统计。只有 completed 且没有未解决的流式错误才进入产物校验。
5. 校验记录路径、frontmatter、正文要求、提醒字段和文件版本；至多追加一次修复，修复复用同一会话、技能发布范围和总 deadline。
6. 持有源记录路径锁，重新核对 RoundRecord 版本。提交 InboxReceipt，metadata 保留 session_id、reminder_path 和生成交接所需的稳定引用。
7. 有提醒时 ack 使用 retain=True。回执成功后由 NotificationOutbox.prepare 登记，再投递到 Remy.NotificationInbox；没有提醒则直接完成。
8. 接收方已持久接收且发送方已记录 delivered 后，释放受保护输入。交接失败只重试交接，不重新执行已确认批次。

事件轮次没有新增内容且未写记录时，可以返回 NO_KEY_INFORMATION；代码确认本轮路径未被另行创建后再 ack。
cron 轮次必须有符合 TASK.md 和记录协议的产物；reminder_priority=skip 只表示不提醒，不表示 cron 可以不写记录。
文件已写但 ack 未提交时允许重试模型，原文件保留并参与路径避让；不能覆盖、删除用户已写文件来让重试通过。

每个后台轮次沿用当前新建正式原生执行会话的语义；最近一次会话供用户后续只读查询。
不因引入 Runtime 组件而自动变成跨轮续聊，临时查询 fork 不替换正式会话。

### 提醒处理与 Chat 交付

1. ChatNotificationsLoop 先恢复未完成的 ChatDelivery，再从接收方队列选择可处理通知。发送方 enabled / notify_window 由 workflow 通过 Manager 核对。
2. 选择同等级、有边界的 ID 批次。P0 使用 p0_delay_seconds；P1 / P2 分别等待 Remy 空闲达到 10 / 1800 秒，前台忙碌时不处理 P1 / P2。
3. ProcessChatNotifications 在 120 秒总超时内准备会话与临时 fork，读取本批源记录并冻结内容版本、目标会话和前台 chat_revision。
4. 临时 fork 只获得 submit_chat_reply。服务端绑定 _agent_name、_inbox_turn 及受信任的批次上下文，模型不能指定别的批次、目标会话或权限模式。
5. skip 在原生会话版本保护和源记录版本校验下直接提交接收方处理回执，不创建用户消息。
6. send 先持久化 ChatDelivery，再用固定 message_id 写入原始目标会话；成功后记录 sent，再确认这一批通知，最后记录 acknowledged。
7. 发送后返回的原生消息与源通知 ID 保持关联；多个通知合成一条回复也能逐条追溯到 daily 原记录。

前台用户输入一进入等待队列就递增 chat_revision，不等模型完成才使旧提醒失效；前台完成 / 中断会更新空闲计时。
前台存在等待输入时，P0 也不得越过该输入发送；版本在异步读取前捕获，并在提交前完成所有异步校验后再次核对。
消息尚未写入而会话、前台版本或源文件变化时，旧决策失效，释放输入并在新上下文重评，不把旧回复改投到新会话。
消息已写入而回执未完成时，按固定 message_id 查询 / 确认原生消息并补齐回执，不重新生成回复或换会话重发。
已有用户可见消息不会因为源文件后来变化而撤回；处理回执记录这次交付使用的原始版本。

P0 运行中 fork 和发送使用 Wrapper 已有的活动会话能力与 expected_turn_id；空闲路径使用 expected_revision。
不支持所需能力或 SDK 接口不兼容时保留输入并显示能力错误，不能伪造已发送。
120 秒涵盖 session_exists、状态读取、fork、读取上下文和模型决策；取消必须撤销本轮工具能力，迟到调用不得确认输入。

### 必须处理的中断位置

| 中断位置 | 恢复行为 |
| --- | --- |
| 模型输出之后、后台 ack 之前 | 输入仍 pending，保留源文件；重新执行前换未占用路径并核对历史。 |
| InboxReceipt 已落盘，但 ack 返回异常 | 读取持久回执确定是否已提交；已提交的输入不重跑，无法判定时暂停该 Inbox 写入并报告错误。 |
| 后台 ack 之后、NotificationDelivery 登记之前 | 用受保护回执重建同一稳定交接，不再次调用模型。 |
| Remy Inbox 已接收，发送方 delivered 尚未写入 | 复用同一 NotificationRef 重新 append，接收方去重后补写发送方状态。 |
| ChatDelivery 已 prepare，尚未写入原生会话 | 核对原生消息是否已存在；存在则恢复接收回执，不存在则 aborted 并释放通知，由新一轮重新判断。 |
| 原生消息已写入、接收方尚未 ack | 找回固定 message_id，补齐通知处理回执和 acknowledged。 |
| 原生交付核对失败，或 sent 消息暂时不可见 | 保留原发送意图并预留其通知 ID，不允许子批次或新模型回复重复发送；其余通知可继续处理。 |
| skip 的 ack 失败 | 不产生用户消息，按处理回执判断是否完成；未完成且版本变化时重新评估。 |
| 源记录缺失、变为目录、被替换或校验失败 | 不凭缓存正文伪造源记录，保留交接和回执，报告到对应 Runtime / 记录；其他 Runtime 继续工作。 |

这些协议保证已提交事实可恢复，不承诺整个模型执行或所有领域工具副作用恰好一次。
例如 Notebook 自动进展要沿用匹配与重复事实核对，不能把输入 Event 去重当作业务写入去重。

### 锁、取消与公平性

- Manager 的索引 / 定义锁仅用于解析、保存和发布索引，不跨模型调用持有；删除先阻止新调用，再等待或取消已登记调用。
- Runtime 执行租约与 Inbox / Outbox 的短存储锁分开。不要同时持有发送方状态锁和等待接收方容量；跨箱追加使用 wait=False。
- 源记录提交复用文件 Job 的同一路径锁，校验、ack 与交接登记期间保持版本保护，避免文件工具和 Outbox 各用一套不相干的锁。
- 单轮超时和取消覆盖初始化与模型调用；进入必要的落盘提交后，用现有 finish_cleanup 完成可判定的短提交，不能用 shield 延长模型工作。
- 可选的原生工具统计读取受同一总时限约束，取消或到期后不再开始读取；不可重试的流式错误即使随后收到 completed 也不得提交。
- 后台 Job 创建的 asyncio.Task 必须登记到对应 Runtime，并在 Job 停止 / Runtime 关闭时取消并等待完成；不引入未跟踪的 worker。
- 恢复和投递不要求先申请新的收件容量或发起新的模型轮次。满收件箱中的既有提醒仍能被处理并释放容量。
- 一个 Runtime、源文件或采集目标失败不能阻止其余 Runtime 的调度；全局列表把错误限制在对应项。

## 13. 持久化布局、定义优先级与迁移边界

### 工作区布局

以下目录均相对 workspace_dir，使用配置中的目录名，不硬编码仓库的 .reme：

```text
{reme_agent_dir}/
├── Remy/
│   ├── MEMORY.md
│   ├── sessions.jsonl
│   ├── inbox/
│   │   ├── .lock
│   │   ├── inbox_unread.jsonl
│   │   └── inbox_read.jsonl
│   └── outbox/chat/<delivery_id>.json
└── <task_name>/
    ├── config.json
    ├── TASK.md
    ├── sessions.jsonl
    ├── inbox/
    │   ├── .lock
    │   ├── inbox_unread.jsonl
    │   └── inbox_read.jsonl
    └── outbox/notifications/<delivery_id>.json

{metadata_dir}/proactive/connectors/<connector_job>.json
{resource_dir}/<source>/progress.jsonl       # Harness 已有进度
{resource_dir}/<source>/pending.json        # Harness 已有冻结批次
{resource_dir}/<source>/<date>/<package>/
├── metadata.json
├── .delivery.json
└── <payload files>
{daily_dir}/<date>/<task_name>_<HHMMSS>[_N].md
```

MEMORY.md、TASK.md、daily 正文、资源、会话引用和处理 / 交付回执是需要保留的工作区文件。
临时模型活动、worker、领取 token、当前订阅者和 live_turn 缓存不持久化；丢失缓存不能导致输入被确认或提醒重发。
Notebook 继续使用 notebook_dir 下现有 Todo / Idea 文件与归档格式。

config.json 保存任务身份、run_job、运行设置、批次限额、created_at 和 deleted_at；agent_wrapper 与 notification_target 来自应用组件依赖。
YAML 是静态实例的首次默认值和装配来源，已有合法 config.json 是运行设置的持久来源；重启不覆盖管理工具已保存的 enabled / cron 等设置。
静态实例名必须与持久身份一致。已有定义与 YAML 的执行 Job 或触发类型冲突时明确报错，不能创建第二个同名 Runtime。
TASK.md 已存在时始终读取原文件，默认任务仅 create-only 创建；即使模板或应用语言改变，也不覆盖用户内容。

删除静态任务只停用其执行身份并写入 tombstone，Application 仍拥有组件；同名恢复复用该组件。
删除动态任务停止其调用并关闭 Manager 自有实例，恢复时由同一工厂重新创建；两者都保留工作区历史。
暂停 Runtime 阻止下一轮开始，允许已经取得执行租约的轮次完成；删除会取消正在运行和等待的调用。

### Connector 状态

控制文件只保存 schema_version、connector_job、enabled 和 updated_at，缺省 enabled=true；工作流与管理工具使用同一路径锁及原子写入。
pause 返回表示控制状态已持久化。正在发布的资源包可以完成到合法 manifest，投递前再次检查开关；已经持久接收的 Event 不回滚。
恢复先重试冻结批次。来源配置或接收方列表后来改变，不重写旧批次的 Event 和目标。
接收方暂时断连属于可重试交付失败，必须先恢复旧交付再采集新资料；不能把它作为损坏资源跳过。取得来源锁后再次核对 pause，避免排队中的采集在暂停后开始。
.delivery.json 保存冻结 Event、原目标名单、剩余目标及永久拒绝结果；临时错误继续待重试，全部目标成功或明确拒绝后才推进对应进度。
外部响应丢失时复用旧 Event ID，不重新下载成第二个资源包来重试投递。
同一个来源进度目录只能由一个配置身份拥有；重复绑定或来源目录变更必须显式处理，不能复用旧游标猜测新来源。

### 旧代码与旧数据

代码按新方案迁移，不保留旧模块转发、旧类别名、旧工具名称或双格式常驻读取。Studio、配置、技能、测试和文档同时更新。
已有工作区数据不自动删除或重写。开发验证使用 tmp_path；部署现有工作区前先备份，再按明确映射转换需要变化的状态文件。
旧 step_backend 转为选定 run_job，Notebook 映射 run_notebook_agent；其余定义根据预设配置或显式选择映射，不能靠任务名写运行时特殊分支。
旧提醒状态中的 delivered_at 迁为发送方交付记录，read_at 迁为接收方处理回执，chat_message_ids 迁为 Chat 交付关联。
TASK.md、MEMORY.md、daily、原生会话和资源正文不参与格式改写；无法识别的数据报出具体路径，不能用空状态替代。

## 14. 工具请求、调度和领域约束

### 管理请求的动作边界

| 工具 / action | 必填 | 可选或具体约束 |
| --- | --- | --- |
| manage_proactive_runtime list | action | 不接收单任务身份和更新字段；返回后台任务 items / total。 |
| create | agent_name | 新任务 run_job 默认 run_memory_agent，也可从白名单选择；task_description 可省略并使用所选领域默认模板；允许初始运行设置。 |
| status / delete | agent_name | 拒绝无关设置；删除保留历史。 |
| update | agent_name、至少一个运行设置 | enabled、cron、min_trigger_interval_seconds、notify_window；不能修改身份、执行 Job、触发类型或用 task_description 覆盖文件。 |
| query | agent_name、非空 question | 在最近会话的只读临时 fork 查询；没有会话时明确失败，不创建空正式会话。 |
| manage_connector send | 非空 task_names、非空 events | 任务名忽略大小写不得重复，Event 结构和所有路径权限在投递前验证；cron 来源禁止手工发送。 |
| enable / pause | connector_job | 只能操作配置白名单中的采集 Job；不接收 events 或 task_names。 |
| submit_chat_reply send | 非空 content | 只接收 action / content；批次、会话、调用角色从服务端注入。 |
| skip | action | 不接收正文，不以空字符串暗示动作。 |
| manage_notebook | action；记录操作还需 kind | 复用 TodoRequest / IdeaRequest 的动作字段边界，list_projects 为公共分支。 |

已有被删除任务恢复时保持原 run_job、触发类型、TASK.md 与历史；请求若显式选择冲突 Job 或类型则失败。
创建前验证允许 Job、可用默认任务、合法 cron 下一次触发和名称冲突，再创建文件与实例；实例启动失败要关闭已启动子资源。
发现或重新解析已有任务时，同样核对 run_job 和触发类型与 Runtime 固定接线一致；冲突定义不得接收事件或进入调度。
配置更新通过完整 Pydantic 校验，不能用未经验证的 model_copy(update=...) 绕过字段和跨字段约束。

send 继续返回 deliveries / rejected / errors，成功与失败按目标分别报告；Response.success 仅在 rejected 和 errors 都为空时为 true。
队列满、临时文件或投递错误归 errors；不存在 / 已删除、时间触发目标和永久 ID 冲突归 rejected。一个目标失败不回滚其他已持久接收的目标。
启停工具返回 connector_job、enabled、changed。失败请求返回 success=false 和明确错误，不把错误文本当成功回答。

四个工具的 Job.parameters 仍使用框架现有 JSON Schema 机制，不新增隐式 request_model 配置键或外部引用加载器。
JSON Schema 展示字段、枚举和基本形状，Pydantic 模型执行动作级互斥验证；两者使用同一组成功 / 拒绝样例核对。
服务端注入字段与公开请求分开，用户输入不能覆盖 _automatic_notebook、_inbox_turn、_agent_name 或权限路径。

### 调度与重试

- cron 表达式使用应用时区与现有 next_cron_time，保留 DST 两次同名本地时刻的处理；计划 Event ID 包含任务身份和真实计划时间戳。
- 启动按当前时间计算下一次触发，不新增停机期间全量补跑；已经入箱的计划 Event 保留并按顺序处理，改日程不清空旧输入。
- 时间任务每轮一条计划事件；事件任务的 min_trigger_interval_seconds 从上次成功消费时间计算，间隔内继续累积输入。
- 失败轮次冻结原输入 ID，后续新输入不并入失败批次。初始重试间隔 5 秒，指数增长，上限 300 秒；成功清除失败状态。
- 暂停不清空输入或重试批次；重启恢复未确认输入，retry_at 等临时计时可以重新计算。已确认回执优先于临时失败标记。
- 未完成交接在普通轮次之外恢复；禁用来源的提醒不发送给用户，notify_window 在调度和真正提交时都核对。
- 调度循环可继续等待事件与停止信号；不把一次 tick 的失败变成后台 Job 无限忙循环。

### 四个领域能改什么

| 领域 | 派生 Step / YAML 的差异 | 共用执行不改变 |
| --- | --- | --- |
| Memory | 默认任务聚焦项目进展、难点与讨论；阅读截图 / Harness 材料。 | 输入领取、会话范围、技能发布、记录验证、ack 与交接。 |
| Notebook | 默认任务、自动跟进技能、manage_notebook 绑定、受信任的自动写权限。 | 不复制模型重试或收发件流程；不使用任务名判断是否走 Notebook。 |
| Finance | 默认标的、阈值、时间基准和新闻范围；按已有金融 schema 阅读资源。 | 不把默认 3% 阈值写死成不受 TASK.md 控制的新交易引擎。 |
| News | 默认早报范围、来源与时间要求，使用实际可用的联网检索能力。 | 没有检索能力时明确失败，不编造已检索结果。 |

硬规则由代码执行：权限、路径、队列状态、文件版本、结构化产物、Notebook 自动修改范围、角色和批次能力。
技能指导模型执行语义检索、去重、事实判断、提醒必要性和领域要求；max_local_search_calls 沿用统计 / 告警语义，不伪称硬性拦截。
常规 Todo 创建等操作不新增“自动重试一定幂等”的承诺；不确定失败后先读取结果核对。

## 15. 开发落点与验收

### 代码迁移清单

| 当前实现 | 目标位置 / 处理 |
| --- | --- |
| components/agent_inbox/ | components/proactive/inbox/，共享存储收进 BaseInbox，删除 LocalAgentInbox / AgentEventInbox 旧导出与注册链。 |
| components/proactive_runtime/runtime.py | components/proactive/runtime/task.py，并接入 BaseAgentRuntime 生命周期。 |
| proactive_manager 的 sessions / live_turn / notifications / outbox | 按第 2、3、11、12 点分配到 Runtime、Inbox、Outbox；旧混合模块删除。 |
| proactive_manager 的 registry 与管理操作 | components/proactive/manager/；去掉常驻调度和 Remy 状态。 |
| steps/proactive/agent/ | 拆到根 base.py / api.py / loop.py、chat/ 和 task/；旧包删除。 |
| steps/proactive/data_connectors/ | connector/，合并公共交付；保留各来源增量、去重、附件和取消行为。 |
| steps/notebook/ | 工具移到 tools/manage_notebook.py 和 _notebook_store.py，自动执行移到 task/notebook.py；旧目录删除。 |
| example.py / example.yaml | 按第 8 点迁入正式能力与测试后删除。 |
| reme/skills/ 旧目录 | 删除；三个操作技能改为 steps/proactive/skills/ 下 YAML，由 BaseAgentStep.get_skills 按角色加载。 |
| components/__init__.py、steps/__init__.py 与 ComponentEnum | 导入新包并注册新类型 / backend，删除旧导出和旧类型，核对 Application 本地注册表。 |
| reme_studio/client/ 的 Notebook 请求路由 | 改用 manage_notebook，使用 kind 区分对象；项目列表去掉 kind。UI 的 Todo / Idea 概念保留。 |
| proactive.yaml、demo 配置、双语服务与资源文档 | 对齐新组件和 Job，删除旧调用示例；不改生成站点输出。 |

移植的是明确保留的能力，不要求保留旧类结构或旧测试夹具。旧入口不加转发层；有价值的行为测试迁移到新公共入口。
AgentWrapper / App Server 执行协议、模型目录、原生历史与流式协议保持独立，不将本次目录重构扩成 Provider 重写。

### 验收矩阵

| 场景 | 必须证明的结果 | 现有证据或新增覆盖位置 |
| --- | --- | --- |
| 注册与装配 | 新包导入后可发现所有组件 / Step，旧类型不可发现；五个 Runtime 只有一个 Remy。 | Application / registry 与配置单测。 |
| 组件生命周期 | 静态组件只启动关闭一次；动态启动失败反向回收；关闭等待任务和线程退出。 | 新 Runtime / Manager 生命周期测试。 |
| 队列隔离 | 两个 Runtime 的容量、进程锁、领取与回执互不影响；关闭唤醒等待者。 | 迁移 test_agent_inbox.py。 |
| 去重和背压 | 同 ID 同内容重复接收、冲突拒绝；满队列为可重试错误；其他目标仍成功。 | test_proactive_delivery_fixes.py。 |
| 处理提交 | receipt 写入前后故障均能区分未提交和已提交，模型不重跑已确认批次。 | test_agent_recovery.py。 |
| 提醒跨箱交接 | 接收后发送方落盘失败、发送方登记失败与源缺失可恢复；源文件和回执不被清理。 | 新 NotificationOutbox / NotificationInbox 故障注入测试。 |
| 通知历史 | 超过显示窗口、重启或索引重建后，已处理 send / skip 不重发。 | 新通知处理回执持久化测试。 |
| 原生消息提交 | 发送结果丢失后按固定 ID 找回消息，补 ack，保留多源关联。 | test_agent_recovery.py / test_proactive_delivery_fixes.py。 |
| 前台竞态 | 输入排队、会话切换、源记录修改会使未发送的旧决策失效；过期工具无权提交。 | test_proactive_reliability.py / 提醒 fork 测试。 |
| 单轮超时 | 会话准备、模型与修复都受总时限约束；超时不消费输入，临时技能清理。 | test_proactive_reliability.py / 正式任务执行测试。 |
| 调度 | 固定触发类型、Notebook 禁用 cron、DST、计划耗尽、队列满及失败批次顺序。 | test_agent_cron.py / test_proactive_delivery_fixes.py。 |
| Collector 控制 | pause 跨调用 / 重启生效，保留游标与 pending；恢复只重试失败目标。 | 新 manage_connector 与基类测试。 |
| Harness | 首次基线、半行 UTF-8、替换去重、轮转预算、来源损坏隔离、图片与取消全部保留。 | test_codex_dialogue.py；新增基类契约测试。 |
| Finance | 五个采集 Job、报价基准 / 时间验证、新闻去重和关闭客户端。 | 现有金融 Collector 测试。 |
| Notebook | 所有现有动作走新工具，自动模式无法越权，进展更正 / 撤回及来源关联保持。 | test_notebook.py / test_proactive_notebook.py。 |
| Skill | 三个内置技能可加载；角色选择正确；修复共享调用范围，结束后清理临时内容。 | Wrapper skill 测试与迁移后的 example 覆盖。 |
| 服务和 Studio | 查询、订阅、终止、错误、任务编辑不回归；Todo / Idea 客户端只调用新工具。 | API / Runtime / Studio 类型检查与测试。 |
| 统计 | 模型终态、用量、工具计数、结果与耗时仍按本轮归集，修复不重复结算。 | test_proactive_statistics.py。 |

先完成每个阶段的离线单测，再运行 tests/unit 全量与仓库格式 / lint；Studio 变更运行类型检查、测试和格式检查。
文档执行 github-pages 的 npm test 与 npm run build。真实模型、宿主采集和设备测试与离线结果分别记录，不用模拟测试冒充线上验证。
提交前核对旧模块、旧注册、旧工具名、旧技能名在运行代码、配置、客户端和使用文档中没有遗留；历史说明可以明确标记为迁移来源。

### 明确不扩展的范围

本次不实现 Claude Code / Hermes Harness，不新增 Connector 组件、第四种领域 Runtime、通用 Job 动态增删服务或跨进程 Notebook 事务系统。
不修改 AgentWrapper 公共协议，不增加用户已读回执，不新增金融交易功能，不自动删除工作区历史，也不新增持续运行的旧格式兼容层。
这些边界不影响完成本文列出的组件、workflow、工具、技能、客户端、测试和文档迁移。

## 16. 实施落点与验证

新组件和 workflow 已按上述结构接线，旧包、旧工具 / Job 注册、todo / idea Job 和示例入口已移除。
Studio Notebook 请求使用 manage_notebook，记录动作补 kind，公共项目查询不传 kind；任务请求改用 manage_proactive_runtime。
技能统一为 steps/proactive/skills/ 下的三个 YAML：manage-proactive-runtime、manage-connector、manage-notebook，由 get_skills 按角色加载并选择语言。配套统一 Wrapper 的 SkillSpec 列表输入和临时技能作用域，覆盖会话创建、执行与修复，并清理本轮创建的文件。

普通采集器将冻结投递批次写到每个资源包的 .delivery.json；Harness 的 pending.json 同时保存不透明源游标和相同投递字段，
不再为同一批次复制一份 .delivery.json。只有全部目标已交付或明确拒绝后才推进源游标。

回归测试按新所有权重建测试装配，覆盖原有业务语义；旧示例专属测试由正式领域 Job 的技能、修复、跳过与清理测试接管。
离线验证包含项目 reme 环境的完整单元测试、pre-commit run --all-files、Studio 测试 / 类型 / 格式检查、客户端 schema 检查，以及 github-pages 测试与构建。
原生 Codex 技能协议另用本地 Responses stub 验证发现、续轮读取与退出清理；不使用真实模型或凭据。真实模型、外部采集设备和带凭据的集成测试未运行。
