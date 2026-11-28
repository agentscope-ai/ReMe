# 主动智能运行总览

实现日期：2026-11-28。Web、桌面和移动端共用 Studio React 界面；后端契约以公共模型为准。

## 界面与交互

主动智能使用全宽图，保留应用导航和标题栏，任务列表直接融合到 Runtime 节点。

```text
独立数据源 → 事件收件箱 → Proactive Runtime → Remy 收件箱 → 提醒
                               ↕ 读取 / 写回
                         横向工作区：资源 / 任务 / 记录
```

来源逐个显示 emoji、名称及 event/time。事件收件箱、Remy 收件箱、提醒均显示最近三条列表。
四个默认 Runtime 聚合在同一节点内，各自独立执行、排队和保存产物。工作区文件标明任务归属。
标题栏仅保留连接状态和视图菜单；添加来源、创建任务、搜索、展开成员均放在对应节点。
选择来源、事件或任务会突出相关路径，清除聚焦后恢复整图。

| 点击对象 | 展示内容 | 操作与内容边界 |
| --- | --- | --- |
| 事件来源 | 名称、开关、接收任务、显示图标；有值时显示投递积压 | 来源 ID、地址、采集及入队时间折叠到原始信息 |
| 时间来源 | 易读的计划、时区、下次触发 | 编辑计划进入所属任务设置，不登记第二份时间源 |
| 输入事件 | 正文、来源、处理状态和时间、相关文件 | 输入与执行 ID 折叠；查看不领取或确认事件 |
| 任务 | 概览、进展、收件箱、发件箱、设置五个页签 | 概览只显示当前状态、最近一轮和产物；不重复 TASK.md 或投递历史 |
| 任务设置 | 任务说明、执行配置分别保存 | 文件保存保留 mtime 检查；配置只提交有变化的字段 |
| Remy 收件箱 | 筛选、数量；单条显示提交正文和相关记录 | 复用列表与详情接口，不叠加第二个大标题 |
| 提醒 | 完整消息、发送时间、来源、去重后的文件引用 | 来源尾注归入引用；查看不标记已读 |
| 工作区 | 资源、任务文件、每日记录横向排列 | 文件名为主信息，目录为次要信息；沿用文件查看器 |

所有详情只出现一次标题、正文和操作。未知事实不显示占位行，技术身份按需展开。
详情使用原生 dialog：打开滑入并淡入、关闭退场、背景渐变，宽度切换有过渡；支持 Escape 和背景关闭。
减少动态效果偏好关闭动画。窄屏将图纵向排列，来源使用横向滚动列表。

图标属于设备显示偏好，以后端地址、工作区和 source_id 关联；不进入后端业务模型。
来源改名不改变身份；历史事件保留入队时的来源快照。

## 后端接口与归属

只新增 `watch_proactive_overview` 一个公开 StreamJob，其他能力复用现有入口。

| 入口 | 当前能力 | 数据归属 |
| --- | --- | --- |
| `watch_proactive_overview` | 来源、任务状态、最近输入、Remy 收件箱、提醒、工作区；快照、增量、心跳和短暂动作 | 只读组合各组件，不调用模型、不创建会话、不消费输入 |
| `manage_connector` | `list/create/update/send`；预设与 SSE 来源统一目录及 source_id | ConnectorManager；采集控制与订阅定义仍保存在各自文件 |
| `manage_proactive_runtime` | 创建及设置 event/time 任务；计划自动形成 time 来源 | 原 Runtime 及任务文件；投递使用现有自动恢复 |
| `list_agent_inbox` | 状态筛选、单任务分页、真实入队时间、来源和执行关联 | Inbox 锁内统一投影 |
| `list_task_reminders` / `read_agent_reminder` | 完整正文、投递和接收状态、原生消息及完整来源追踪 | 发送方 Outbox、接收方 Inbox、原生会话 |
| 文件接口 | 原相对路径、正文读取和编辑 | 原权限、大小限制及版本检查 |

预设来源为 `preset:<job-name>`，自定义订阅为 `source_<uuid>`。时间来源为 `time:<agent-name>`。
按 Cron 采集的屏幕、对话和行情仍以 event 输入；仅 Runtime 自身计划产生 time 输入。

`manage_connector` 参数：

| action | 参数与行为 |
| --- | --- |
| `list` | 返回统一 `items/total` |
| `create` | `name`、非空 `runtime_names`、`binding.url`；固定 SSE/CloudEvents |
| `update` | `source_id`，以及 `name/runtime_names/enabled` 的非空子集；缺省不修改，显式 null 拒绝 |
| `send` | `runtime_names` 和非空 `events`，按接收方分别返回成功、拒绝和错误 |

`runtime_names` 完整替换；空数组继续采集但不投递新批次。`enabled` 只控制来源，与任务开关独立。
保存的批次冻结目标、事件及来源，改名、改目标和重启不改变重试内容。
状态区分配置开关、Job 可用性、连接、采集时间、入队时间和投递积压；损坏状态隔离到对应来源。
只保留当前参数与入箱存储格式，不保留 enable/pause、connector_job 请求别名或未封装的 Inbox 读取分支。
实现不自动迁移或删除用户工作区内容。

## 执行事实与状态

Inbox 持久化事件、真实 received_at 和来源快照；幂等重试保持首次入队时间。
一条输入可以进入多个任务，以 `(agent_name, event_id)` 区分。每轮领取批次拥有独立 run_id，
关联全部 event_ids、source_refs、原生会话/turn、阶段和已确认 record_paths。
提交回执保留有产物但无需提醒的轮次；恢复从这些文件重建视图。

| 字段或事实 | 含义 |
| --- | --- |
| `event_ts` | 来源事件发生时间 |
| `received_at` | 接收方首次成功持久入队的时间 |
| `waiting_count / processing_count` | 互不重叠；原 pending_count 仍包含两者 |
| `started_at / completed_at` | 当前执行尝试的起止时间 |
| `waiting_reason / waiting_until` | 与调度器共用判断：暂停、等输入、等计划、冷却、退避、执行等待或停止 |
| `retry_at / next_trigger_at` | 执行重试时间与计划时间分别表示 |
| notification `delivered` | 已进入目标 Inbox；不代表 Remy 已处理或用户已读 |
| 提醒 `created_at / delivered_at` | 创建和确认投递分开；未确认时 delivered_at 为空 |
| `remy_processing / read_at` | 接收方处理状态与用户阅读状态分别表示 |

## 订阅与动效

连接首先返回 `snapshot`。后续 `update` 带 epoch、revision、previous_revision、实体替换/删除及 transitions；
空闲返回 heartbeat，沿用 StreamChunk error/done。版本不连续或慢读者超过缓冲范围时重新取快照。
共享投影最多八组，内部变更及时刷新，外部文件修改约五秒校验一次；投影和动作日志均可丢弃重建。

| 已确认事实 | 界面反馈 |
| --- | --- |
| 首次持久入队 | 来源 → 事件收件箱流动一次 |
| 成功领取批次 | 收件箱 → Runtime 流动一次 |
| 明确读取任务文件 | 工作区 → Runtime 流动一次 |
| 记录提交 | Runtime → 工作区流动一次 |
| 通知进入 Remy Inbox | Runtime → Remy 收件箱流动一次 |
| 提醒写入原生会话 | Remy 收件箱 → 提醒流动一次 |

事件采用蓝色实线，时间采用紫色虚线，使用短拖尾、柔和光晕和一次性接收高亮。
执行活动可轻微呼吸，失败保留原因和自动恢复状态。工具名称不能作为文件读取证据。
首次快照、重连、重新进入页面、后台恢复及重新打开动画均不重播历史动作。
隐藏页面、减少动态效果偏好或高流量时省略动画，计数与状态照常更新。离线保留最后快照并标注更新时间。

总览仅保留最近窗口；完整历史通过已有单任务分页查询。打开的详情移出窗口后仍保留当前条目。
工作区只保留配置入口、最近投递记录、当前/最近执行产物及最近列表引用；窗口移出不会删除原文件。

## 实现入口

- [总览公共模型](../../reme/schema/proactive_overview.py)、[来源模型](../../reme/schema/proactive_connector.py)
- [总览订阅](../../reme/steps/proactive/overview.py)、[有界动作日志](../../reme/components/proactive/overview_events.py)
- [来源管理](../../reme/components/proactive/connector/manager.py)、[Inbox](../../reme/components/proactive/inbox/base.py)
- [共享客户端](../../reme_studio/client/src/proactive-overview.ts)、[共享界面](../../reme_studio/workbench/src/proactive/index.tsx)

独立 HTML 模拟原型已由正式共享界面替代。验证使用隔离工作区和真实队列、投递组件；模型边界使用测试替身。
