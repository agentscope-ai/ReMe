---
title: ReMe5 前瞻 · Proactive Runtime
description: ReMe 如何将文件化记忆、事件驱动的 Agent 和定时任务整合到一套本地优先的运行时中。
---

# ReMe5 前瞻 · Proactive Runtime

ReMe 从面向 Agent 的 Memory Kit 起步。**ReMe5** 是即将发布的 `0.5` 版，ReMe 将由此发展为一套**以文件化长期记忆为基础的主动智能系统**。ReMe 的 Proactive Runtime 用工作区记录重建任务上下文，调用 Codex 等执行后端，并保存结果和提醒。
你可以把一件需要持续跟进的事交给 Agent。收到新信息，或者到了约定时间，它就会运行一轮，记住你的要求，接着上次的进展往下做。
遇到重要变化时，它会把消息带回对话，你也可以随时追问发生了什么。

我们计划在 **2026 年底前**发布 ReMe5。Proactive Runtime 和 ReMe Desktop 的早期原型已合入 `main`，Chat Agent 已经能与
事件驱动和定时 Agent 协作。目前我们正在用真实工作流测试原型，继续改进 Agent 记什么、什么时候提醒，以及任务能否稳定地跨多轮推进。
[原型 PR](https://github.com/agentscope-ai/ReMe/pull/743) ·
[当前配置](https://github.com/agentscope-ai/ReMe/blob/main/reme/config/proactive.yaml) · [English](../en/reme5_preview.md)

## 为什么需要 Proactive Runtime

很多工作从一段对话开始，要在对话结束后继续跟进。你处理其他事情时，coding agent 可能跑完了一次实验。一个项目决策可能散落在
好几个会话里。一篇新论文可能改变你接下来要研究的问题。除了能回忆之前聊过什么，Agent 还需要知道你让它关注什么、哪些变化有意义、
什么已经记过，以及什么时候该来告诉你。

ReMe 现有的记忆流程是这个能力的基础。Auto Memory 把对话提炼为带有来源链接的每日记录，Auto Dream 将有长期价值的内容沉淀成知识。
在 ReMe5 中，Agent 可以用同样的文件化记忆持续跟进任务，每次收到事件或按日程运行时都能接着已有进展处理。
[文件即记忆](./memory_as_file.md) · [Auto Memory](./auto_memory.md) · [Auto Dream](./auto_dream.md)

例如，你可以让 Chat Agent 跟进多个工具里的同一个项目，并在遇到需要你决定的阻塞时提醒你。一次会话可能记录了测试失败和初步判断，
之后另一段会话证实测试已经通过。Work Agent 可以把两次进展和来源都记下来，更新项目状态。如果还有需要你决定的事，它就会提醒你。
你可以沿着链接核对原始记录，也可以随时调整之后的提醒条件。

## ReMe5 要做什么

你可以通过 **Chat Agent** 交办需要持续跟进的事，也可以随时调整关注范围和提醒条件。
**Proactive Runtime** 在收到新事件或到达约定时间时执行任务，并把进展和证据保存在工作区。Chat Agent 可以根据这些记录回答追问，
也会把需要关注的提醒带回对话。

![ReMe5 方案总览，展示 Chat Agent 如何管理事件驱动、定时和自定义 Agent，以及运行它们的后端和计划接入的宿主](../figure/reme5-proactive-runtime-overview.svg)

常见任务可以从预设开始。你也可以在同一个对话里创建任务，为它接入观察来源并按需添加 Cron 日程。
这套运行时会在 ReMe 自己的桌面预览中提供，也计划接入已有的 Agent 工具。

## 运行时如何工作

运行时按名称管理后台任务。**观察事件** 可以来自一轮对话、一条日志或订阅源发布的论文。**可选日程** 会在约定时间向同一收件箱追加触发事件，例如每天启动简报任务，或在夜间触发记忆整理。等待下一轮时无需持续调用模型。

![Proactive Runtime 如何把新观察交给 Work Agent，并将进展、来源和提醒决定写入同一记录](../figure/proactive-runtime.svg)

图中是现有的 Work 原型。原型配置内置了 Codex 对话和截图采集。图中的 Claude Code 执行轨迹演示了其他来源如何通过事件接口接入。
[运行时指南](./proactive_runtime.md)

1. **采集与排队。** 采集器提交事件，并附上来源和相关文件。到了约定时间，Cron 也会提交一条定时事件。每个任务有独立的收件箱，
   一次处理一批输入。
2. **带着上下文执行。** Agent 读取任务说明和原始资料，再查找近期记录，把收到的信息和已有进展对照，判断这次有什么实质变化。
3. **保存结果并按需提醒。** Agent 把新的发现、来源链接和提醒决定写在同一份每日记录里。运行时校验并保存结果，再确认输入已处理。
   执行失败的输入可以重试。已经保存的提醒可以独立投递。

任务说明和每日记录都保存在用户自己的工作区中。任务每轮读取最新要求，用户也可以直接打开文件检查和修正后续执行要用的背景。
配置决定哪些来源会接入，记录里的链接可以追溯到原始对话、文档或观察结果。
[执行和存储细节](./proactive_runtime.md)

## ReMe5 的记忆工作流如何变化

![ReMe5 的记忆计划，已配置的 Harness 事件触发 Work，定时任务分别整理 Personal、Procedural 和 Wiki](../figure/reme5-memory-workflows.svg)

### Auto Memory 成为 Work 的一部分

事件驱动的 **Work** Agent 会承担 Auto Memory 从对话中提取长期有用信息的工作。我们计划让 Work 监听你配置并接入的每个 Agent Harness
的新增日志和会话记录，汇集多个来源的更新，持续跟进项目进展、决定、阻塞和下一步。后续会话仍可利用这些记录，有进展符合你的提醒要求时，
Work 也可以主动汇报。现在的原型已经用新增 Codex 对话和截图测试这套流程。

### Auto Dream 拆成三个定时 Agent

现在的 Auto Dream 会把每日记录沉淀为个人、流程经验和 Wiki 知识。我们计划把这三类整理工作交给三个独立的 Agent，
每个 Agent 采用自己的 Cron 日程和资料整理规则。

现有 `daily/` 和 `digest/` 中的笔记是过渡时的重要背景。迁移里程碑会说明如何继续使用已有工作区和来源链接，
以及如何把现有的 Auto Memory 和 Auto Dream 配置迁移到新任务。
[当前 Auto Dream 的输入输出](./auto_dream.md#输入和输出)

### 预设 Agent 和工作流

ReMe5 计划提供一组预设，用于跟进持续进行的工作、定时整理长期记忆，以及管理日常待办和总结。

| 预设 | 触发方式 | 负责什么 |
| --- | --- | --- |
| **Work** | 事件驱动 | 包含 Auto Memory。跟进已接入 Agent Harness 的日志和会话，记录项目进展、决定和阻塞。 |
| **Finance** | 事件驱动 | 跟进用户选定的话题或资产相关的新闻和行情。 |
| **Research** | 事件驱动 | 跟进与当前研究问题有关的新论文和项目结果。 |
| **Consolidate Personal Memory** | Cron 定时 | Auto Dream 的个人记忆预设。整理稳定偏好、长期目标和个人或项目背景。 |
| **Consolidate Procedural Memory** | Cron 定时 | Auto Dream 的流程经验预设。整理可复用的工作流程、排查方法和实践经验。 |
| **Consolidate Wiki Knowledge** | Cron 定时 | Auto Dream 的 Wiki 预设。维护值得长期查阅的主题知识，并保留来源链接。 |
| **Todo** | 事件驱动 | 从有关事件中提取明确的待办，并链接回事件或相关决定。 |
| **Daily Summary** | 事件驱动 | 有实质变化时更新当天的总结，并保留仍需跟进的事项。 |

原型已有事件驱动的 Finance 示例和定时的科技简报。我们用这些示例测试哪些内容值得记下来，哪些需要及时告诉用户。
现有桌面原型里的 Todo 界面和后台任务总结也可以作为这些工作流的基础。
[原型 Agent 配置](https://github.com/agentscope-ai/ReMe/blob/main/reme/config/proactive.yaml) ·
[桌面原型](https://github.com/agentscope-ai/ReMe/tree/main/reme_desktop)

## 在对话中创建和管理 Agent

Chat Agent 通过 `manage-proactive-agent` skill 创建自定义 Agent，并管理预定义和自定义任务。你可以在对话中查看任务状态、追问进展，
修改任务要求和运行设置，也可以暂停或恢复任务。
[任务管理 skill](https://github.com/agentscope-ai/ReMe/blob/main/skills/manage-proactive-agent/SKILL.md)

要创建定时任务，你可以说「每周五总结这周的实验结果，附上报告链接」。要创建事件驱动任务，你可以让 Chat Agent 跟进实验日志，
完成或失败时告诉你。事件驱动任务需要接入能提供相应信息的来源。

后台 Agent 根据任务要求提出提醒。Chat Agent 结合当前对话决定是否投递提醒。[对话与任务管理指南](./proactive_runtime.md)

## 桌面预览、执行后端和宿主集成

- **ReMe Desktop** 计划发布 **macOS 和 Windows** 的概念验证预览。你可以体验从聊天交办任务到查看输入、进展、知识和提醒的完整流程。
  `main` 上的 Electron 原型已能连接 ReMe 服务，展示主对话和各个后台任务。
  [桌面源码和启动说明](https://github.com/agentscope-ai/ReMe/tree/main/reme_desktop)
- **执行后端** 计划支持 **AgentScope、Codex 和 Claude Code**。目前默认原型使用 Codex，也有独立的 Claude Code 配置。
  我们会验证每个后端的任务执行、会话处理和提醒行为，再在文档中列出各自支持的预览能力。
  [Codex 配置](https://github.com/agentscope-ai/ReMe/blob/main/reme/config/proactive.yaml) ·
  [Claude Code 配置](https://github.com/agentscope-ai/ReMe/blob/main/reme/config/proactive_cc.yaml)
- **宿主集成** 计划覆盖 **QwenPaw、Codex、Claude Code、Hermes Agent 和 OpenClaw**。宿主可以向 ReMe 提供事件，也可以在现有工作流中
  展示结果和提醒。我们会在已有记忆集成上继续推进，按各宿主的能力和开发进度逐步接入。
  [现有 Agent 集成](./integrations.md)

Codex 和 Claude Code 可以同时出现在两种角色中。作为执行后端，它们负责运行 ReMe Agent。作为宿主，ReMe 可以跟进用户在现有会话中
开展的工作，并把结果带回原有工作流。

## 里程碑

| 里程碑 | 状态 | 内容 |
| --- | --- | --- |
| **运行时和桌面原型** | 已合入 `main` | Chat Agent、事件驱动和定时任务、在工作区保存任务状态、提醒投递。 |
| **真实工作流测试** | 进行中 | 测试 Work、Finance 和定时任务，检查能否记录实质变化、来源链接是否有效，以及重试或重复事件下的提醒表现。 |
| **记忆工作流和预定义任务** | 计划纳入 ReMe5 | 将 Auto Memory 纳入 Work，实现三个定时 Auto Dream Agent，说明如何迁移现有工作区和配置，并开发首批 Research、Todo 和 Daily Summary 工作流。 |
| **ReMe5 和桌面预览** | 目标是 2026 年底前 | 发布运行时和 macOS、Windows 概念验证，提供安装指南和已支持的后端、工作流说明。 |
| **宿主集成** | 分阶段推进 | 接入各宿主可提供的事件来源，验证任务跟进和结果投递。 |

## 试用原型和分享反馈

源码中已经有桌面原型及配套服务配置。桌面 README 说明了如何在一个终端启动 ReMe 服务，再在另一个终端启动客户端。
[从源码运行原型](https://github.com/agentscope-ai/ReMe/tree/main/reme_desktop#start)

欢迎告诉我们，你希望 ReMe 跟进哪个 Harness，哪些变化值得记录，什么情况值得发一条提醒。如果你已经在用 Auto Memory 或 Auto Dream，
也欢迎分享对你最重要的数据来源、记忆文件或任务日程。
