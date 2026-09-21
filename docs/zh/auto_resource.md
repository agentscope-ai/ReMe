# Auto Resource `Beta`

Auto Resource 是 ReMe 的资源解读入口，目前处于 **Beta**。资源文件先进入 `resource/`（推荐按日期放置），再被解读成 daily
资源卡片；卡片文件名由 LLM 生成的 frontmatter `name` 决定，并通过 `source_resource`
追溯原始文件。

<p align="center">
  <img src="../figure/auto-memory-resource.svg" alt="ReMe Auto Memory 与 Auto Resource 写入 daily 记忆卡片的流程" width="92%">
</p>

关于 workspace 分层、`resource/` 和 `daily/` 的通用文件语义，见 [Memory as File](./memory_as_file.md)。对话进入 daily 的流程见
[Auto Memory](./auto_memory.md)。

```text
resource/[YYYY-MM-DD/]<resource_file>
  ├─ step 1: daily/YYYY-MM-DD/<generated_name>.md # 资源解读卡片
  ├─ step 2: source_resource 指回原始资源
  └─ step 3: daily/YYYY-MM-DD.md                  # 当天索引再串起来
```

## 它记录什么

它不只是搬运文件内容，而是把资料里以后方便检索和理解的信息提炼出来：

- 核心内容：这份资料主要讲什么。
- 结构脉络：章节、表格、字段、数据组织方式。
- 关键细节：重要数字、名称、日期、结论。
- 背景用途：这份资料为什么存在，和当前工作有什么关系。
- 可行动项：任务、截止时间、后续跟进。

简单说，它负责把“文件存档”变成“资料可用”。

## 原始资料入口

Auto Resource 以 `resource/` 作为原始资料入口。推荐按日期放置，目录日期会决定它进入哪一天的 daily 记忆层；也支持直接放在
`resource/` 根目录，首次处理时使用应用时区中的今天。之后即使跨天更新或删除，也会通过精确匹配
`source_resource` 继续操作原 daily 卡片，不会重复新建卡片或留下孤立链接。

示例目录：

```text
workspace/
  resource/
    quick-note.txt             # 进入今天的 daily
    2026-06-20/
      market-report.md
      meeting-notes.csv
```

当前 Beta 版本以文本类资源为主，例如 `md`、`txt`、`json`、`jsonl`、`csv`、`yaml`、`html`；图像资源（`png`、`jpg`、`jpeg`、`webp`、`gif`、`bmp`、`tiff`、`heic`）会生成 caption 卡片，见下文[图像资源](#图像资源)一节。

内部由统一的 `AutoResourceStep` 接收每批变更，并将每一项交给配置中第一个匹配它的 processor。
`AutoImageResourceStep` 声明图像后缀匹配规则，`AutoTextResourceStep` 作为最后的 fallback。后续新增模态时，
只需注册新的 processor、提供独立 prompt 并在 `dispatch_steps` 中增加一项，无需修改 router。

## 图像资源

文本和图像共用 agent-wrapper 与笔记写入工具；图片输入只是在理解提示词旁加入原生 AgentScope 图像块。
Agent 写入一张 caption 卡片并链接原图。卡片正文以 `![[resource/...]]` 嵌入链接开头，frontmatter 携带
`kind: image` 与 `media_type`，文本检索因此可以通过 caption 命中图像内容。

图像处理默认开启（`include_images=true`），需要绑定兼容模型与 formatter 的 AgentScope wrapper。
通过 `components.agent_wrapper.<name>.as_llm` 配置模型，在资源 Step 上用 `agent_wrapper` 选择对应 wrapper。
原图片 Step 的 `as_llm` 覆盖项和自动选择 `as_llm.vision` 的逻辑由此绑定方式替代。
不再单独调用 caption 模型或执行额外的 schema 提取；Agent 失败后也不以纯文本重跑。
这是一次 Agent 工作流，工具调用可能带来多轮模型请求。

每次解读图片都会新建会话，处理记录可通过返回的 `agent_session_id` 查找。更新同一张图片时，仍然修改原来的卡片。
卡片正文应包含原图引用和 `## Caption` 下的描述或文字转录，不能留空或直接写入 JSON。
`status` 留给后续流程填写，更新卡片时保留原值。

自定义图片提示词使用 `prompt_dict.resource_instructions`，中文使用 `resource_instructions_zh`；旧配置中的
`user_message` / `user_message_zh` 需相应改名。公共创建或更新模板通过
`{resource_instructions}` 插入图片要求；旧模板没有该占位符时，图片要求会追加到末尾。

在 `auto_resource` 调用或 Job 默认值中设置 `include_images=false`，会跳过图片的**全部事件，包括删除**。
调用参数优先于 Job 默认值，两者都未设置时默认开启。监听任务可设置 `jobs.resource_watch_loop.include_images=false`，
手动任务默认值可设置 `jobs.auto_resource.include_images=false`。图片子类通过现有逐资源结果和 warning 日志说明跳过原因，
不影响文本处理。已有图片卡片保持不变，即使原图被删除也不清理。重新开启不会自动补处理旧事件，需要显式把相关路径再次提交给
`auto_resource`。尊重 wrapper 配置的图片数量上限，每次资源调用至少需要容纳一张图片，不会自动提高上限。

在启动常驻服务时配置 wrapper，例如将每个 Agent 上下文的图片上限设为一张：

```bash
reme start components.agent_wrapper.default.context_config.max_image_num=1
```

监听任务会自动处理资源变更。如需显式重新处理已存在的 `resource/photo.png`，在另一个终端使用同一 workspace 调用客户端：

```bash
reme auto_resource include_images=true changes='[{"path":"resource/photo.png","change":"modified"}]'
```

宽或高超过 2048px 的图像会降采样，格式不被模型接受的图像会转码；这些处理只发生在请求前的内存副本中，
`resource/` 下的原图文件不会被修改。图像处理开启时，图像变更会原地更新卡片，图像删除会清理关联卡片。

在完整解码前，系统会检查图像尺寸，默认上限为 40,000,000 像素；超限图像或 Pillow
decompression-bomb 警告只会导致当前资源失败。缩放或转码前，会按 EXIF orientation 校正仅用于请求的内存副本。
尺寸过大的 JPEG 会先使用 decoder-level downsampling，并在需要时再完成最终缩放。
VLM 请求的 MIME 和卡片 frontmatter 中的 `media_type` 都使用 Pillow 根据实际图像字节识别的格式，
而不是直接信任文件扩展名。

图像预处理使用 `core` extra 中的 Pillow。HEIC 资源还需要可选的 `image-heif` extra：
`pip install "reme-ai[image-heif]"`。其他受支持图像格式不会加载或依赖 HEIF 插件。

## 资源卡片

每个资源文件会生成一张 daily 资源卡片。创建时先使用资源文件 stem 作为临时路径，对应 processor 写入后，系统会根据 frontmatter `name`
重命名文件：

```text
resource/2026-06-20/market-report.md
        ↓
daily/2026-06-20/市场报告要点.md
```

资源卡片通过 frontmatter 关联原始文件：

```yaml
source_resource: "[[resource/2026-06-20/market-report.md]]"
```

对于已启用处理的资源，文件更新时 Auto Resource 只会通过精确匹配的 `source_resource` 找到对应卡片并更新；文件删除时也只会清理显式关联的
daily note。缺少该来源标记的同 stem 笔记会被视为用户笔记并保留，新资源卡片则会使用无冲突路径。

处理失败时，`modified` 会标明卡片文件有没有变化。Agent 写完文件后再报错或被取消，已写内容仍然保留；
系统会尝试补齐通过 `source_resource` 关联的卡片元数据，并更新当天索引，调用仍按原来的错误或取消结束。
图片笔记未通过格式检查时也会报错，已写内容同样保留。失败的调用不会自动重试。

## 当天索引

资源卡片会进入和 Auto Memory 相同的 daily 记忆层。当天的 `YYYY-MM-DD.md` 会作为索引页，把这些资源卡片组织起来：

```text
daily/
  2026-06-20.md
  2026-06-20/
    市场报告要点.md
    会议纪要整理.md
```

以后想回看这一天处理过哪些资料，先看 `YYYY-MM-DD.md`；想看某份资料沉淀了什么，再进入对应的资源卡片。

## 同时保留原始资料

解读后的 daily note 负责“好读”，原始资源负责“可信”。

Auto Resource 不会把原始文件挪走：它仍然留在 `resource/` 下的原路径。这样，文本与图像资料会进入 daily 记忆流，原始文件也始终保留在它来时的位置。

## 后续流向

Auto Resource 只生成 daily 层的资源解读。要把资源中的长期知识沉淀进 `digest/`，使用 [Auto Dream](./auto_dream.md)；默认实时检索会
索引 daily 卡片和 digest 节点。手动 `reindex` 只基于摄取流程已经接受的 chunks 重建检索索引，不会把原始资源文件加入检索范围。
详见 [Memory Search](./memory_search.md)。
