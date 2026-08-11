# ReMe：让你的个人知识库，在每一次对话后继续生长

我们每天都在和 AI 对话。

它帮我们分析项目、阅读论文、排查问题，也听我们讲过偏好、计划和那些还没完全想清楚的念头。

但大多数时候，对话结束，价值也被关进了历史记录。下一次打开新窗口，AI 可能还记得一句结论，却不知道结论从哪里来；可能搜到一段旧对话，却无法把它和后来读过的资料、做过的决定连起来。

真正有用的长期记忆，不应该只是“存过”，而应该能够持续整理、建立联系，并在未来需要时重新参与思考。

这正是 ReMe 想解决的事情。

> **ReMe 是一个面向 AI Agent 的、local-first 的自进化个人知识库。它让对话与资料持续沉淀为可读、可编辑、可检索、相互链接的
Markdown 记忆，并从中提炼值得继续关注的线索。**

项目地址：[https://github.com/agentscope-ai/ReMe](https://github.com/agentscope-ai/ReMe)

项目文档：[https://docs.agentscope.io/reme](https://docs.agentscope.io/reme)

<p align="center">
  <img src="../figure/reme-blog/reme-blog-cover-benchmark.png" alt="ReMe 自进化个人知识库与公开基准结果" width="100%">
</p>

## 一个会持续生长的记忆循环

<p align="center">
  <img src="../figure/reme-blog/reme-blog-architecture.svg" alt="ReMe 自进化记忆循环" width="100%">
</p>

ReMe 不是另一个聊天机器人，也不试图替代你正在使用的 Agent。它更像一个可以被 QwenPaw、OpenClaw、Hermes、Claude Code 等 Agent
共享的本地记忆层。

它围绕一套普通文件，完成四件事：

- Auto Memory 从对话中提炼值得保留的信息；
- Auto Resource 将外部资料整理为可追溯的记忆；
- Auto Dream 把每日记忆进一步整合为长期知识；
- Index、Search 和 Proactive 让旧记忆重新回到新的任务中。

这形成了一个 `capture → index → consolidate → recall` 的循环：

- 对话和外部资料先被保留下来；
- 有价值的信息被整理为每日记忆；
- 零散事件进一步合并为长期知识节点；
- 搜索、知识链接和兴趣发现，让旧记忆重新参与未来的思考。

更重要的是，这个循环的中心不是一个用户看不见的黑盒数据库，而是用户自己拥有的文件。索引、图谱和缓存都只是可重建的派生状态。

## Memory as File：记忆首先是你的文件

<p align="center">
  <img src="../figure/reme-blog/reme-blog-memory-as-file.svg" alt="ReMe Memory as File" width="100%">
</p>

ReMe 的核心设计称为 **Memory as File, File as Memory。**

“Memory as File”意味着，长期记忆不是藏在产品内部，而是落在 workspace 里的 Markdown、JSONL、YAML 和原始资源文件中。你可以用 VS
Code、Typora 或 Obsidian 直接打开，也可以用 Git、网盘或自己的同步方案备份和迁移。

“File as Memory”意味着，每个文件又不只是普通文本。它可以带有 YAML frontmatter、章节结构、行号范围和
Wikilink，成为一个可索引、可连接、可继续演化的记忆节点。

例如，一条关于写作偏好的长期记忆可以是：

```markdown
---
name: 用户偏好：技术文章风格
description: 喜欢先讲问题和结果，再解释技术细节，并用例子帮助理解。
kind: preference
---

用户希望技术文章有清晰主线，避免堆砌术语。

撰写文章时可参考 [[digest/procedure/技术内容写作流程.md]]。

## Sources

该偏好观察自 [[daily/2026-08-07/content-discussion.md]]，其中记录了用户对写作方式的要求。
```

几个月后，即使你已经忘了这次对话，Agent 仍然能读到偏好、找到关联流程，并顺着 `Sources` 回到当时的上下文。

这也是 ReMe 与“黑盒记忆”的关键区别：Agent 可以整理记忆，但用户永远保留查看、修正、移动和删除它的权利。

## Auto Memory：把聊过的事，写成每天的日记

<p align="center">
  <img src="../figure/reme-blog/reme-blog-auto-memory.svg" alt="ReMe Auto Memory 将对话整理为 daily 记忆" width="100%">
</p>

很多有价值的信息并不是以“请记住”开头的。

比如你在一次对话里说：

> “这周先不要重构登录模块，客户演示之后再做。上次直接升级依赖导致兼容问题，这次先补回归测试。”

这段话里同时包含了项目状态、时间约束、一次失败经验和后续行动。Auto Memory 会把它从聊天流水中提炼出来，写成当天的一张 daily
记忆卡片；可追溯的对话来源记录则保存在 `session/dialog/` 中。

```text
session/dialog/project-a.jsonl                 对话来源记录
daily/2026-08-07/login-refactor-decision.md    按内容命名的记忆卡片
daily/2026-08-07.md                            当天索引，负责总览
```

`session_id` 仍保留在卡片 frontmatter 中，用于稳定定位和追溯；文件名来自 Agent 生成的主题/事件 `name`，不必与 session ID
相同。

以后再讨论登录模块，Agent 不必翻遍聊天记录，就能先看到：当前为什么没有重构、曾经踩过什么坑、下一步应该先做什么。

它像一位一直在场的记录者，但不是机械地抄写逐字稿，而是把“以后还会用到什么”整理出来。

## Auto Resource：让外部资料进入同一套记忆系统

<p align="center">
  <img src="../figure/reme-blog/reme-blog-auto-resource.svg" alt="ReMe Auto Resource 将外部资料整理为可追溯的个人记忆" width="100%">
</p>

并不是所有有价值的信息都来自对话。研究资料、项目文档、会议纪要、网页存档和结构化数据，同样可能成为个人知识库的一部分。

Auto Resource 提供了一条更通用的外部资料入口。资料进入 `resource/` 后，ReMe 保留原文，再把主题、关键事实和可行动信息整理为带有
`source_resource` 链接的 daily 卡片。当前可以处理 Markdown、纯文本、JSON、JSONL、CSV、YAML 和 HTML 等文本类资料。

这意味着，Auto Memory 负责从对话建立个人知识，Auto Resource 负责从非对话资料建立个人知识。两条输入最终进入同一个 daily
记忆层，再由 ReMe 统一索引、整合和检索。

### Daily Paper：外部资料工作流的一个例子

Daily Paper 是建立在这套文件化记忆之上的可选 Cookbook。它会从 Hugging Face Papers 的周榜和月榜收集论文，去除近期已经推荐过的内容，排序后精选三篇，保存
PDF，并生成中文论文笔记与一份约五分钟可读完的简报。

想象一下，你持续关注 Agent Memory：每天早上收到的不只是三个论文链接，而是三篇已经保存到本地的详细笔记。简报通过 Wikilink
指向原始笔记，原始笔记又能回到 PDF。一个月后再问“最近有哪些方法在做长期记忆压缩”，这些材料已经进入同一套检索系统，不需要重新从浏览器历史里寻找。

Daily Paper 展示了 Auto Resource 可以怎样被组合成具体工作流，但外部资料入口并不局限于论文。

## Auto Dream：让日记长成相互连接的长期知识

<p align="center">
  <img src="../figure/reme-blog/reme-blog-auto-dream.svg" alt="ReMe Auto Dream 从 daily 抽取、分类并整合长期知识，同时写入 Wikilink" width="100%">
</p>

日记多了，新的问题也会出现：信息虽然都在，却仍然散落在不同日期里。

假设你先后从对话和外部资料中得到三条关于同一个问题的信息：

- 第一次排查构建卡死，清缓存无效；
- 第二次在项目文档中确认根因是 Node 内存不足；
- 第三次又补充了大型 TypeScript 项目下更容易触发这个问题。

Auto Dream 默认查看以目标日期结尾的最近两天，只把相对上次运行发生变化的 daily 文件一起交给抽取器。它合并指向同一抽象的跨文件
证据，并在默认最多五个 unit 的额度内只保留最值得复用的记忆，再按内容写入三类长期记忆：

- `Personal`：用户、团队或项目特定的偏好、约定和约束；
- `Procedure`：可以再次执行的流程、方法和排查手册；
- `Wiki`：通用的定义、原则、观察和知识。

例如，上面的信息会被整理为 `digest/procedure/前端构建卡死排查.md`，包含触发条件、排查顺序、无效尝试、解决方法和适用边界，而不是几篇日记的简单拼接。

整合每个记忆单元时，Auto Dream 会先跨 `personal`、`procedure` 和 `wiki` 搜索已有节点，区分“同一抽象”和“相关知识”。同一抽象决定如何演化目标节点：

- `CREATE`：没有相同记忆，创建新节点；
- `CORROBORATE`：同一结论再次出现，补充来源并增强可信度；
- `REFINE`：新材料补全条件、步骤或细节；
- `CORRECT`：新信息修正旧结论。

相关知识则在同一次整合中以 Wikilink 写进正文，这就是 Auto Link。比如“前端构建卡死排查”可以同时连接通用知识、团队偏好和原始证据：

```markdown
这个问题常见于 [[digest/wiki/大型 TypeScript 项目.md]]，处理时遵循
[[digest/personal/团队变更偏好.md]] 中“先补回归测试”的约定。

## Sources

根因与适用场景记录在 [[daily/2026-08-07/build-debug.md|构建排查记录]] 中。
```

知识的演化与链接发生在同一条流程里。关系不是藏在图数据库里的不可见边，而是正文中可读、可改的内容；文件可以重建图，图不会反过来绑架文件。

## Memory Index：普通文件，如何变成可搜索的记忆网络？

<p align="center">
  <img src="../figure/reme-blog/reme-blog-memory-index.svg" alt="ReMe Memory Index 构建过程" width="100%">
</p>

Markdown 适合人读，但如果只是把文件堆进目录，Agent 仍然很难快速找到它们。默认实时索引持续监听 `daily/` 与 `digest/` 中的
Markdown；`resource/` 由独立资源流程监听，转成 daily 卡片后进入同一索引。需要从现有文件完整重建时，`reme reindex` 还会扫描
`resource/` 与 JSONL。

一份 Markdown 会被解析为：

- 一个文件节点：包含路径、frontmatter 等文件级信息；
- 多个语义 chunk：尽量沿标题、段落、列表和代码块边界切分，并保留章节骨架与行号；
- 多条 Wikilink 边：记录它指向谁，以及谁又指向它。

在检索侧，ReMe 可以组合三类信号：

| 检索信号       | 解决的问题                         | 例子                               |
|----------------|------------------------------------|------------------------------------|
| BM25 关键词    | 精确名称、术语和编号不能丢         | “宁德时代”“issue #184”             |
| Embedding 向量 | 用户换了一种说法，也要理解语义接近 | “构建卡死”与“打包阶段没有响应”     |
| Wikilink 图谱  | 命中一个节点后，看到它的上下游关系 | 从“钴”找到“三元正极”和相关调研记录 |

默认配置开箱启用 BM25 与 Wikilink 展开，Embedding 是可选能力，开启后才会参与向量召回。索引、图谱和缓存都写在 `metadata/`，
即使删除也可以根据用户的源文件重新构建。

## Memory Search：先找到答案，再沿着关系渐进展开

<p align="center">
  <img src="../figure/reme-blog/reme-blog-memory-search.svg" alt="ReMe 混合搜索与渐进式展开" width="100%">
</p>

很多 RAG 系统会一次性把 Top-K 文本全部塞进上下文。这样做简单，却容易带来两个问题：孤立切片缺少上下文，而把邻居正文全部展开又会迅速消耗
token。

ReMe 的混合搜索先让 BM25 与可选的向量检索各自召回候选，再使用 RRF 按排名融合。RRF 不强行比较 BM25
分数与余弦相似度这两种不同量纲，而是综合一个结果在两张榜单中的位置。

召回之后，信息按三层渐进式展开：

1. **先看命中片段**：返回最相关的 chunk、文件路径和行号；
2. **再看关系目录**：展示该文件的出链与入链，只给邻居的路径、名称、描述和锚点，不急着加载全文；
3. **最后按需深入**：Agent 判断哪条关系真正相关，再读取原文或沿图谱继续遍历。

例如，你问：“上次 Alice 推荐的那本讲注意力的书叫什么？”

第一步可能命中一张聚餐日记，其中只写着“标题里有‘深度’两个字”；结果同时显示，这张日记链接到了 Alice 的个人节点，也被《深度工作》的阅读笔记反向引用。

Agent 不需要把 Alice 的全部档案、所有读书笔记和整个月的日记都塞进上下文。它只要沿着最相关的链接再读一次，就能回答：

> 是《深度工作》。Alice 在那次聚餐时推荐了它，你后来还读了第三章并留下了笔记。

这更像人的联想过程：先想起一个片段，再顺藤摸瓜找到完整上下文。

## Proactive：从你的记忆里，发现那些尚未说出口的需要

<p align="center">
  <img src="../figure/reme-blog/reme-blog-proactive.svg" alt="ReMe Proactive 内外双向记忆循环" width="100%">
</p>

到这里，ReMe 已经有了两条不断丰富知识库的输入流：

- Auto Memory 从持续发生的对话中沉淀个人上下文；
- Auto Resource 从外部资料中补充新的知识。

Proactive 则把方向反过来：它从已经积累的对话和资料中，发现你仍未解决或可能希望继续推进的主题，以及你尚未关注、但与近期工作紧密相关的信息。这些发现会反哺输入流，为补充外部知识指明新的路径。

例如，你最近一周分别聊过：

- 搜索结果里缺少来源；
- 长文档切片后容易失去章节上下文；
- 想比较几种 Agent Memory 评测方法。

尽管你从未明确说“帮我系统研究记忆检索的可解释性”，但 Auto Dream 可以从这些 daily 记忆中提炼出一个兴趣主题：

```yaml
title: 记忆检索的可解释性评估
reason: 用户近期持续关注来源追溯、结构化切片和记忆评测。
evidence: daily/2026-08-07/search-discussion.md
keywords:
  - memory search
  - source attribution
  - benchmark
```

在未来的 beta 版本中，上层 Agent 通过 Proactive
读取这个主题后，可以选择在合适的时机追问：“要不要把最近讨论过的检索问题整理成一份评测方案？”也可以据此启动一个经过用户授权的资料收集流程。用户不需要预先梳理并显式指定自己的兴趣关键词和范围；与对话中潜在需求相关的外部资料，也可以持续进入知识库。

这里有一个重要边界：**ReMe 的 Proactive 本身只读取并暴露兴趣主题，不会擅自联网、推送或改写知识库。**
它不是凭空猜测你的兴趣，而是让那些已经出现在行为和对话中、却还没有被明确表达的线索浮出水面。

## Performance：它真的能从很长的历史里找回信息吗？

ReMe 使用 LongMemEval 和 BEAM 验证多会话与超长对话中的记忆能力。评测时，Agent 可以用 ReAct 方式进行多轮搜索和读取，生成答案后再由
LLM-as-judge 评分。

| 基准                      | 设置        |            样本量 | Agentic 得分 | 主要检验内容                   |
|---------------------------|-------------|------------------:|-------------:|--------------------------------|
| **LongMemEval cleaned-s** | **整体**    |        **500 题** |    **89.4%** | 跨会话检索、知识更新与时间推理 |
| BEAM                      | 100K 上下文 | 20 cases / 400 题 |        66.1% | 十类长上下文记忆任务           |
| BEAM                      | 1M 上下文   | 35 cases / 700 题 |        65.0% | 更大规模的超长对话设置         |

LongMemEval cleaned-s 包含单会话事实、偏好、多会话推理、知识更新和时间推理等题型。ReMe 在 500 道问题上取得 89.4% 的整体
Agentic 得分。完整流程和分项结果见 [LongMemEval 评测说明](../../benchmark/longmemeval/README_ZH.md)。

BEAM 覆盖矛盾消解、事件排序、信息抽取、知识更新、多会话推理、偏好遵循、摘要和时间推理等十类任务。ReMe 在 100K 设置下的 20
cases / 400 题上取得 66.1%，在 1M 设置下的 35 cases / 700 题上取得
65.0%。完整设置见 [BEAM 评测说明](../../benchmark/beam/README_ZH.md)。

此外，ReMe 还使用 $\pi$-Bench 验证了基于多会话推理提升 Agent 主动性的潜力。$\pi$-Bench 中的 PROC 分数旨在评估 Agent
在隐藏意图直接完成、针对性澄清引导、跨会话偏好恢复、跨会话规范复用、跨任务依赖推断以及欠规格请求推进等方面的主动性能力。ReMe
Agent 在 5 种用户角色（User Persona）上平均取得 0.580 的 PROC 分数，超出相同测试模型配置的 NanoBot
2.4%。关于该基准的详细介绍见 [$\pi$-Bench 论文](https://arxiv.org/abs/2605.14678)。

## ReMe 能帮谁？

### 直接使用 Agent 的人

如果你希望 AI 在长期协作中持续了解你，ReMe 可以让个人助理不再每次都从零开始。你的偏好、项目背景、重要资料和过去做过的决定，会在持续对话中沉淀下来，并在真正相关的时候被重新找到。

研究者、工程师、分析师和其他知识工作者都属于这一类直接用户。研究者可以让论文、讨论和阅读笔记彼此连接；工程师可以保留项目决定与跨会话排障经验；分析师可以持续积累事件、观点与来源。职业不同，共同需求都是让
AI 能够理解过去、积累经验，并在下一次任务中找回依据。

### 构建 Agent 的开发者

如果你正在构建 Agent、Harness 或 AI 产品，ReMe 提供了一层可以独立接入的长期记忆基础设施。你可以通过 CLI、HTTP API、MCP Server
或 Python API，让不同 Agent 共享同一个文件化 workspace，而不必为每个应用重新实现记忆抽取、知识整理、混合检索和关系展开。

文件是事实来源，索引与缓存可以随时重建，也更容易定位一次错误召回究竟来自原始资料、记忆整理还是检索链路。

归根结底，ReMe 适合那些希望 AI 不只“回答这一次”，还能够理解过去、积累经验，并在长期协作中越来越懂自己的用户和开发者。我们希望
Agent 越用越懂你，但“懂”不应该建立在一个无法查看、无法修正、无法带走的黑盒上。

ReMe 给出的答案很朴素：

- 记忆是用户拥有的文件；
- 原始信息保留现场，长期知识保留抽象；
- 新对话和新资料持续进入，旧知识也持续被补充和修正；
- 每个结论都可以通过 Wikilink 找到关系和来源；
- 索引与缓存服务于文件，而不是取代文件；
- Agent 可以记住、整理、搜索和发现，但最终控制权始终属于用户。

当这些机制连接起来，个人知识库就不再是一座需要你手工维护的仓库。

它会在每一次对话后多记住一点，在每一份资料到来后多理解一点，在夜晚把零散经验重新整理，在未来的某个问题出现时，再沿着知识之间的联系，把真正需要的那段记忆带回来。

这就是 ReMe 想做的事：**让记忆不只被保存，也能持续进化。**

## 接入你正在使用的 Agent

ReMe 既可以作为本地记忆服务，通过 CLI、HTTP API 或 MCP Server 接入，也可以通过 Python API 嵌入宿主进程。默认 HTTP
服务还可在同一地址提供 ReMe Studio，用于浏览、编辑、搜索 workspace 和查看 digest Wikilink 图。不同 Agent 可以选择适合自身
运行环境的路径，并按需共享同一个本地 memory workspace。

| Agent                                  | 推荐接入方式                                                                                     | 接入后能力                                                                                                   |
|----------------------------------------|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| **QwenPaw**                            | 通过 Python API 在进程内嵌入 ReMe。                                                              | 复用宿主应用的生命周期和模型配置，同时保持记忆本地、文件化。                                                 |
| **Claude Code**                        | 启动 streamable HTTP MCP Service，并安装 [`plugins/claude_code/reme`](../../plugins/claude_code/reme)。 | MCP 记忆召回工具、`reme-memory` skill，以及自动记录会话的 Stop hook。                                        |
| **Hermes**                             | 启动 HTTP Service，并安装 [`plugins/hermes_agent`](../../plugins/hermes_agent)。                  | 在模型调用前自动召回相关记忆，并在每轮对话完成后异步调用 `auto_memory`。                                     |
| **OpenClaw、Codex 等支持 CLI 的 Agent** | 复制或安装 [`skills/reme_memory/SKILL.md`](../../skills/reme_memory/SKILL.md)。                    | 通过 CLI 搜索、读取和写入记忆；自动记录需要宿主 Agent 显式接入会话生命周期。                                 |

安装、配置与集成演示可查看 [README 中文版](../../README_ZH.md)。

## 欢迎贡献

ReMe 已经开源，我们也欢迎社区一起把这套自进化记忆系统做得更完整：

- 接入更多 Agent 与 Harness，让不同运行环境都能使用同一套用户拥有的长期记忆；
- 贡献新的 Auto Resource 数据源和工作流，让论文、新闻及其他公开资料能够持续进入知识库；
- 改进 Auto Memory、Auto Dream、Auto Link、混合搜索与 Proactive，让记忆整理得更准确、关系更清晰、召回更可靠；
- 补充新的应用案例、评测任务和诊断报告，帮助我们看见真实长期使用中的成功与失败；
- 完善文档、测试，或通过 Issue 分享你对个人 AI 记忆的需求与想法。

无论是一段代码、一份使用案例、一次问题反馈，还是一个新的记忆工作流，都可能帮助 ReMe 更接近真正可读、可控、可持续进化的个人知识库。

贡献指南：[https://docs.agentscope.io/reme/latest/en/contribution](https://docs.agentscope.io/reme/latest/en/contribution)
