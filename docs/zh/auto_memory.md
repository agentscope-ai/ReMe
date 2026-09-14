# Auto Memory

Auto Memory 是 ReMe 的对话记忆入口：在目标日期内，它用 `session_id` 定位或更新最多一张 daily 记忆卡片，文件名由 Agent
根据内容生成简洁的主题或事件名，再由当天的 `YYYY-MM-DD.md` 统一索引。它负责把“聊过”变成“记住”，并保留可追溯的对话记录。

<p align="center">
  <img src="../figure/auto-memory-resource.svg" alt="ReMe Auto Memory 与 Auto Resource 写入 daily 记忆卡片的流程" width="92%">
</p>

关于 `daily/`、`session/`、frontmatter 和 wikilink 的通用文件语义，见 [Memory as File](./memory_as_file.md)。

```text
Conversation
  ├─ step 1: daily/YYYY-MM-DD/<generated_name>.md # 每个 session 一张主题卡片
  ├─ step 2: daily/YYYY-MM-DD.md                  # 当天索引再串起来
  └─ source: session/dialog/<session_id>.jsonl    # 对话来源记录
```

## 它记录什么

它不记录聊天流水账，只记录以后可能还会用到的内容：

- 用户偏好：喜欢什么风格、习惯怎么协作、长期要求是什么。
- 关键事实：项目背景、重要数字、明确结论、限制条件。
- 过程决定：发生了什么，为什么这么选，哪些方案被放弃。
- 当前状态：做到哪一步，卡在哪里，下一步是什么。
- 可复用经验：命令、流程、排查方法、解决方案。

## 写入位置

Auto Memory 会把整理后的记忆放进 `daily/`。当天发生的对话会先被整理成一张张小卡片：

示例目录：

```text
workspace/
  daily/
    2026-06-20.md
    2026-06-20/
      login-refactor-decision.md
      retrieval-regression.md
```

日期目录下的两个文件是不同对话整理出的主题卡片，`daily/2026-06-20.md` 是当天索引页。资源文件也会进入
同一个 daily 记忆层，见 [Auto Resource](./auto_resource.md)。

当调用时带上 `session_id`，Auto Memory 会通过 frontmatter 用它定位卡片，Agent 则通过 `name` 决定可读文件名：

```yaml
name: login-refactor-decision
session_id: session-a
source_conversation: "[[session/dialog/session-a.jsonl]]"
```

这样既能分开不同对话，又不必把不透明的 ID 当文件名。更新时会按 `session_id` 或 `source_conversation` 找到旧卡片；如果 Agent
提供了更好的 frontmatter `name`，系统可重命名并重定向入链。查看某天内容时从 `YYYY-MM-DD.md` 开始。

## 同时保存原始信息

整理后的 daily note 负责“好读”，过滤后的对话来源记录负责“可信”。

Auto Memory 在生成记忆卡片的同时，也会保存对话来源消息：

```text
session/
  dialog/
    session-a.jsonl
    session-b.jsonl
```

daily note 会指向对应的对话记录。持久化时会排除 tool-result block 和 base64 data block，避免召回记忆或二进制负载在后续流程中被误当成
用户提供的证据。

## 可选的 Session 图像

默认不开启图像。`include_images` 和 `supports_vision` 是两个独立的布尔选项，默认都为 `false`。
发送图像前，既要开启功能，也要明确声明所选模型支持视觉：

```bash
reme auto_memory session_id=session-a include_images=true supports_vision=true messages='[...]'
```

单次 CLI 调用可使用 `reme start job=auto_memory`，其余参数相同。
不需要修改 YAML 或重新编译。`include_images` 控制是否考虑图像；`supports_vision` 声明 AgentScope wrapper 绑定的 `as_llm`
是否支持图像，不会自动探测能力或切换模型。仅当 `include_images=true supports_vision=true` 时才发送图像；
请选用支持这些图像的模型与 AgentScope formatter。

`include_images=false` 时，无论 `supports_vision` 如何设置，各种 wrapper 都保持原有纯文本行为。
纯文本仍通过已有的 `agent_wrapper.reply()` 字符串输入路径执行，无需为其他 wrapper 改造多模态接口。
`include_images=true` 时必须使用 AgentScope wrapper；其他后端会抛出 `NotImplementedError`，不会降级为纯文本。
`image_mode` 默认为 `direct`，目前只支持此模式；开启图像时选择其他模式会抛出 `ValueError`。

如果输入含图像但 `supports_vision=false`，Auto Memory 不读取图像，记录 warning 后沿用原始纯文本输入，
并在 `auto_memory_images` metadata 中报告 `status=fallback`、`reason=supports_vision=false`。
开启功能但输入中没有图像块时报告 `status=skipped`。

`messages` 使用 AgentScope 格式，处理顶层图像 `DataBlock`（`source.media_type` 以 `image/` 开头）。
支持 Base64、HTTP(S) URL 和 workspace 内的 `file://` URI；本地读取遵守 `_allowed_paths`。
图像输入使用有界读取与 provider 预处理，包括已有的最长边 2048 像素限制、方向纠正和必要的格式转换。
调用方原始消息与文件保持不变。

### 直接多模态输入

使用默认的会话渲染时，Auto Memory 构建 AgentScope `UserMsg`，在会话历史中按原顺序交错放置文本与图像 `DataBlock`，
保留每条消息的说话人和时间边界，也支持只有图像的消息及重复 block ID。
这里描述的是 ReMe 传给 wrapper 的输入；实际 provider 请求中的表示方式由所选 AgentScope formatter 决定。
记忆 Agent 同时读取文本与图像，不单独调用 caption 模型。这是一次 Agent 工作流，不保证只有一次 API 请求：
调用写笔记等工具后，后续模型轮次仍可能携带图像。
未显式设置图像数量上限时，直接模式会为全部输入图像保留 Agent 上下文容量。
若显式设置的 `context_config.max_image_num` 小于输入图像数，会整次显式回退为纯文本，不静默移除较早图像。
模型和 provider 自身的上下文限制仍然适用。

纯文本与图文 Auto Memory 均使用 AgentScope wrapper 通过 `as_llm` 绑定的同一个模型。
`supports_vision=true` 应当描述这个模型的能力。图像资源处理使用的可选 `components.as_llm.vision`
不会改变 Auto Memory 的模型。

Auto Memory 不创建 resource 图像文件或独立 caption 卡片；
相关图像事实仍可被提取进普通 daily 记忆笔记。AgentScope wrapper 仍按原有行为在 `mem_session/agentscope` 保存
内部 Agent 状态，直接模式的状态中包含图像输入。

**开启与关闭图像时的源 JSONL 保存行为完全不变。** 不补充 caption 或图像 metadata，仍执行上文的过滤规则。
重放已保存的 JSONL 无法恢复被过滤掉的 Base64 图像；再次处理这些图像需要重新提交原始带图消息。

图像读取或预处理失败时，Auto Memory 会记录 warning，并在响应的 `auto_memory_images` metadata 中说明降级。
本次请求的所有临时图像增强都会被丢弃，然后使用原始纯文本输入继续提取记忆，不会混入部分成功的结果。
只要普通 memory 操作成功，CLI 仍可成功；需查看 metadata 区分纯文本降级与图像处理成功。
取消操作、开启图像时的无效模式配置、不支持的 wrapper，以及图像阶段以外的错误，不会被这个降级逻辑吞掉。
记忆 Agent 开始运行后，错误按原有方式传播，不会再以纯文本重跑，以免重复执行已经写入记忆的工具。
直接模式在调用 Agent 前记录 `prepared`，正常返回后记录 `completed`，`captioned_images` 始终为零。
`completed` 表示图像处理完成，不保证一定生成了笔记。

若需持久默认值，在应用配置（或 `reme start` 对应覆盖项）中设置 `jobs.auto_memory.include_images` 和
`jobs.auto_memory.supports_vision`。两个选项各自遵守相同优先级：调用时参数高于 Job 默认值，Job 默认值高于 Step 配置；
两个选项默认均为 `false`。无需编译。本适配针对 AgentScope 消息，不扩展 Claude Code 原始 transcript 的解析。

## 消息时间

Auto Memory 会在 prompt 和对话来源 JSONL 中保留每条已保留消息的 `created_at`。导入历史对话或 benchmark 数据时，建议为每条
message 提供真实发生时间，避免模型把事件时间误解为运行时间：

```bash
reme auto_memory \
  session_id=locomo-session \
  messages='[
    {"role":"user","content":"Jon lost his job today.","created_at":"2023-01-19T08:00:00"},
    {"role":"assistant","content":"I am sorry to hear that.","created_at":"2023-01-19T08:01:00"}
  ]'
```

为了兼容常见数据集字段，`auto_memory` 也会在缺少 `created_at` 时读取 `time_created`、`timestamp`、`createdAt`、
`timeCreated` 或 `created_time`。这些字段可以放在 message 顶层，也可以放在 `metadata` 中。

当调用没有显式传入 `date` 时，Auto Memory 会使用消息中最晚的有效 `created_at` 日期作为 daily note 日期；如果消息没有有效时间，
则回退到当前日期。历史导入也可以显式指定目标日期：

```bash
reme auto_memory \
  session_id=locomo-session \
  date=2023-01-19 \
  messages='[{"role":"user","content":"Jon lost his job today."}]'
```

## 后续流向

默认的 `auto_memory` 和 `auto_memory_cc` Job 会在记录记忆后执行 `auto_tag_step`，只为实际新增或修改的 daily 笔记打标，
并使用重命名后的最终路径。Claude Code 调用方仍只需传入 `session_id`；重复 Stop 没有新增消息时，记忆生成和打标都会跳过。

标签描述文档的核心实体，写入配置的 frontmatter 字段，默认为 `memory_tags`。单文件打标失败记录在 `metadata.auto_tag`，
保留原有记忆响应；没有笔记变化的调用不会自动重试失败的打标。标签索引通过现有文件 watcher 异步更新。

Auto Memory 只生成 daily 层记忆。要把这些材料进一步沉淀为长期 `digest/` 节点，使用 [Auto Dream](./auto_dream.md)；要搜索
daily 和 digest，使用 [Memory Search](./memory_search.md)。
