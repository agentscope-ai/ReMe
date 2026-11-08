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

## 资源保留与 TTL

默认配置不自动删除原始资源。`config=proactive` 使用一个 catalog 记录**已确认处理的文件状态**，并显式启用以下保留协议：

- 临时资源统一为 `resource/YYYY-MM-DD/{uuid_hex}/` 资源包，每包通过 `metadata.json` 发现。
- 永久资料放在 workspace 中 `resource/` 以外的目录，例如 `archive/`。
- `resource/` 根目录文件、非日期目录、无效日期和日期目录符号链接不参与物理清理。

目录、文件名筛选和 TTL 只在 catalog 中配置一次：
原来的 `jobs.resource_cleanup_cron.ttl_days` 已迁移到 `components.file_catalog.resource.ttl_days`。

```yaml
components:
  file_catalog:
    resource:
      backend: local
      watch_dirs: [resource_dir]
      watch_suffixes: []
      watch_filenames: [metadata.json]
      ttl_days: 7
```

`watch_dirs` 支持已有的应用目录字段或 workspace 内的目录，例如 `resource_dir`、`resource_dir/inbox`、`inbox`。
清理及扫描根目录必须位于 workspace 内，且路径中不能含符号链接。`watch_filenames` 精确匹配文件名，
与 `watch_suffixes` 同时配置时必须同时满足；两者为空时接受全部文件。
物理清理不按文件筛选条件过滤，会删除过期日期目录的全部内容，包括未登记的图片和空目录。

`resource_watch_loop` 使用 `file_catalog: resource` 选择上述配置，只监听并记录实时变更，不推进 catalog，也不调用 LLM 或生成 daily 卡片。
`resource_changes` 每次直接扫描文件，与 catalog 比较后返回仍存在、未过期且新增或修改后尚未确认的 `metadata.json` 路径。
即使 watcher 未运行也能查询；查询不推进 catalog，未确认的资源下次继续返回，已删除文件不返回。

在一个终端启动服务：

```bash
reme start config=proactive components.file_catalog.resource.ttl_days=7
```

在另一个终端查询待消化资源：

```bash
reme resource_changes
reme resource_changes limit=40
```

`answer` 直接返回按路径排序的字符串列表，路径相对于 workspace，最多返回 `limit` 条（正整数，默认 `20`）。
未返回的资源仍保留待处理；确认一批后再次调用 `resource_changes` 获取下一批。
先读取 manifest 的 `description` 和 `files`，整包消化成功后，将对应的 manifest 路径作为 `resource_processed` 的 `paths` 参数：

```bash
reme resource_processed paths='["resource/2026-11-05/7bf0379a0c0d4bc6a436508052727dae/metadata.json"]'
```

`answer` 和 `success` 均为布尔值：整批确认成功为 `true`；路径越界、不存在、不是普通文件、已过期或 I/O 失败时为 `false`，
不更新这批确认状态。`paths` 必填，空列表成功，重复确认允许；只传成功消化的子集，其他资源保留待处理。
两个 Job 在 Step 配置中绑定 `file_catalog: resource`，不接受 catalog 选择参数。

proactive 的 watcher 和 `resource_changes` 只发现精确名为 `metadata.json` 的文件，图片、其他 JSON 和裸文件不单独发现。
资源包发布后按不可变快照使用，更新内容应发布到新 UUID 目录。路径确认记录文件当前 mtime，不校验消化前后的版本；
因此不能在处理期间原地修改资源包。直接修改或删除包内数据文件也不会使资源重新待处理。
消费方仍需确认清单中的文件存在且路径位于当前资源包内。catalog 的确认状态不写回 metadata，Auto Resource 的默认处理流程不变。

消化状态保存在 `metadata/file_catalog/resource.jsonl.zst`。如需重新消化全部已有资源，
先停止服务，备份并移走这个派生 catalog，再启动；保留原始资源文件。

TTL 优先于消化：按应用 `timezone` 中的日历日期计算，目录日期距今天达到 `ttl_days` 即过期，与文件 mtime 无关。
例如 2026-11-05、`ttl_days: 7` 时，10-29 及更早日期目录已过期，10-30 及之后保留。
过期资源立即退出 `resource_changes` 查询范围，`resource_processed` 也不再确认；每小时的 `resource_cleanup_cron`
调用 catalog 删除源目录及对应条目，不等待消化。因此清理前发起的处理也不能在清理后通过 `resource_processed`
恢复过期条目。`ttl_days` 必须为正整数。

Cron 在启动后的下一个整点执行，服务停止期间不会物理删除。
`jobs.resource_cleanup_cron.enabled=false` 只停止物理清理，`resource_changes` 和 `resource_processed` 仍遵守 catalog 中的 TTL。
删除失败会保留仍存在目录的 catalog 条目并标记任务失败，后续调度重试。
清理要求平台支持基于目录描述符的安全删除（Linux、macOS）；不支持的平台会报错并保留文件。
实现见 [`LocalFileCatalog`](https://github.com/agentscope-ai/ReMe/blob/main/reme/components/file_catalog/local_file_catalog.py)。
现有 `file_catalog.delete()` 仍然只删除 catalog 条目，不删除源文件。

清理流程不改写其他目录或引用。若在其他配置中同时启用 Auto Resource，删除事件仍会按其规则清理关联 daily 卡片。

## 定时屏幕采集

`config=proactive` 的 `screen_capture_cron` 在应用时区的每小时 00、15、30、45 分触发，不在启动时立即截图，
也不补拍服务停止期间的画面。安装 `pip install "reme-ai[core]"`，
再用前述 `reme start config=proactive` 启动。可用 `jobs.screen_capture_cron.enabled=false` 关闭截图任务。

截图来自运行 ReMe 的宿主机桌面。macOS 需要为实际运行进程授予屏幕录制权限；无图形会话、无显示器、
缺少权限或依赖时该轮任务失败，不发布资源包，等待下次定时触发。支持情况由本机 MSS 后端决定。

`reme/steps/proactive/data_connectors/` 中的 `BaseConnectorStep` 负责暂存、校验和原子发布，
`ScreenCaptureStep` 逐屏保存 MSS 返回的原尺寸 PNG，不合并或缩放不同分辨率的显示器：

```text
resource/YYYY-MM-DD/{uuid_hex}/
  1.png
  2.png
  metadata.json
```

`metadata.json` 使用 `ResourceMetadata` schema：

```json
{
  "schema_version": 1,
  "dataclass": "ScreenCaptureStep",
  "source": "screen",
  "created_at": "2026-11-05T15:00:00+08:00",
  "modified_at": "2026-11-05T15:00:01+08:00",
  "description": "同一轮采集的全部显示器截图，请结合全部图片理解当时的用户活动。",
  "files": [{"path": "1.png"}, {"path": "2.png"}]
}
```

`dataclass` 为实际 Step 类名，`source` 为来源渠道；时间均带时区，日期目录取采集开始时间。
`files` 是非空对象列表，每项仅要求 `path`，可增加 `description` 或来源专属字段；路径相对于包目录，不能逃逸或引用符号链接。
图片编号是该次采集的显示器枚举编号，不承诺跨重启稳定。多屏依次采集，不保证完全同时。
任何屏幕失败都会丢弃整轮暂存数据，所有图片及 manifest 完成后才在 `resource/` 中发布目录。
取消时等待正在执行的同步采集退出并清理暂存文件；每日产生 96 次计划采集，失败轮次不保存。
资源包沿用上述七天 TTL。该任务只采集，不直接调用 LLM，也不生成 daily 卡片。

## 带凭证的远程采集器

`reme start config=proactive_remote` 在 `proactive` 工作区之上追加 `mail_capture_cron` 和 `github_capture_cron` 两个
cron Job，其余配置完全继承。它们产出的是普通资源包，因此既有的 watcher、`resource_changes` / `resource_processed`
确认机制、PA 采集轮次和 7 天 TTL 都照常生效。`reme/config/proactive_remote.yaml` 中的任何取值都可以在命令行覆盖：

```bash
reme start config=proactive_remote jobs.mail_capture_cron.cron="*/30 * * * *"
```

凭证来自环境变量或工作目录下的 `.env`（ReMe 在解析配置前加载它）。缺少凭证的采集器只记录一条告警且不发布资源包，
Job 本身仍然成功，因此在配置任何密钥之前启动该配置是安全的。采集器严格只读：邮件以 `BODY.PEEK` 在只读 `SELECT`
下读取，不标记已读、不回信；GitHub 通知也不会被标记已读。采集内容属于隐私数据：邮件正文和 issue 文本写入
`resource/`，并随其 TTL 一起删除。

密钥在每次轮询时从环境读取，故意不写成 `proactive_remote.yaml` 里的 `${VAR}` 占位符：配置解析会对形似数字的取值
做类型转换，例如 `1e5` 会被变成 `100000.0` 再交给服务商。非密钥配置仍留在 YAML 中，可在命令行覆盖。`.env`
有三个容易踩的点：搜索从工作目录开始、最多向上 5 层、且只加载第一个命中的 `.env`，因此要在持有该文件的目录
或其子目录里启动 ReMe；`.env` 中的值会覆盖已经 export 的同名 shell 变量，临时用 shell 覆盖会看起来"不生效"；
每行必须是纯 `KEY=VALUE`——`export ` 前缀会成为键名的一部分，`${}` 不做插值，没有 `=` 的行被静默忽略。

缺少凭证时该轮跳过并记一条告警、Job 仍然成功；凭证被拒绝时该轮直接失败报错，所以配置完要主动验证一次，
不要以"日志安静"当作成功。`mailboxes` 必须使用服务商自己的文件夹名（以 IMAP `LIST` 结果为准）；文件夹不存在
会让该轮失败而不是被跳过，因此一期建议只配 `INBOX`。SASL 认证要求服务器在 capability 中声明 `AUTH=`；只提供
`IMAP4rev1 ID` 的服务器必须走 `auth_mode=password`，配合应用专用密码或服务商授权码。同一数据源要并行接入两个
账号时，需要为每个 Step 配置不同的 `state_key`；更换账号应删除对应的 `metadata/step/{state_key}.json`——
`UIDVALIDITY` 变化会自动重置邮箱游标，但若两者恰好相同，就会沿用上一个账号的位置。

`ImapMailStep`（`imap_mail_step`）需要 `IMAP_HOST`、`IMAP_USER`，以及 `IMAP_PASSWORD`（应用专用密码或服务商授权码）
或 `IMAP_ACCESS_TOKEN` 之一。`IMAP_AUTH_MODE=oauthbearer` 或 `xoauth2` 走 SASL 认证；配置 `IMAP_REFRESH_TOKEN`、
`IMAP_CLIENT_ID`、`IMAP_CLIENT_SECRET` 时，访问令牌会通过 `IMAP_TOKEN_URL`（默认 Google 令牌端点）自动刷新。
`IMAP_MAILBOXES` 为逗号分隔列表，每个邮箱各自保存 `UIDVALIDITY` 与已消费的最大 UID 游标；`UIDVALIDITY` 变化说明
服务端重新编号，该邮箱回退到 `initial_window_days` 限定的时间窗重新开始。消息按 UID 从旧到新消费，单轮上限
`max_messages`，积压会在后续轮次继续排空而不会被跳过；`unseen_only` 只搜索未读邮件。超过 `max_message_bytes`
的邮件会被整封跳过，正文按 `max_body_bytes` 截断，附件只记录名称、不下载。每封邮件产出一个 Markdown 文件：

```text
resource/YYYY-MM-DD/{uuid_hex}/
  messages/001-INBOX-4242.md
  metadata.json
```

`GithubNotificationsStep`（`github_notifications_step`）和 `GithubActivityStep`（`github_activity_step`）需要
`GITHUB_TOKEN`（或 `GH_TOKEN`）：通知接口要求带 `notifications` 权限的 classic token，活动接口要求对 issue 和
pull request 的读权限。`GITHUB_API_BASE` 用于 GitHub Enterprise 的 API 根地址。通知以 `since` 轮询，并用响应的
`Last-Modified` 作为下一次游标（GitHub 文档保证该方式不漏数据）；`include_read`、`participating`、`repos`、
`exclude_repos`、`reasons` 控制保留范围，被过滤掉的通知同样会关闭其时间窗。活动接口必须配置
`GITHUB_ACTIVITY_REPOS`（逗号分隔或 JSON 列表，数量上限 `max_repos`），每个仓库各自维护 `since` 游标并推进到本轮
见到的最新 `updated_at`；`exclude_labels`、`include_issues`、`include_pulls`、`only_involving_you` 用于过滤，
当前登录账号通过 `/user` 解析一次后缓存。额度耗尽时只停止本轮翻页而不会丢数据，因为两个游标都来自服务端时间戳。
每个采集器发布一份摘要：

```text
resource/YYYY-MM-DD/{uuid_hex}/notifications.md      # 按仓库分组，含原因与可点击链接
resource/YYYY-MM-DD/{uuid_hex}/activity.md           # 按仓库列出状态、作者、标签、是否与本人相关
```

两个远程采集器都继承 `BaseRemoteConnectorStep`，它在 `BaseConnectorStep` 之上补齐带凭证数据源需要的采集侧能力。
配置项依次从 Step kwargs、`env_defaults` 中映射的环境变量、默认值解析；`timeout_seconds` 约束每次网络调用，
`max_retries` 与 `retry_backoff_seconds` 约束指数退避，服务端的 `Retry-After` 或限流重置时间会被遵守，上限为
`max_backoff_seconds`。`min_interval_minutes` 在发出任何请求之前就拒绝过密的轮询，避免手工调用重复打扰服务商。
`max_pages` 与 `per_page` 约束翻页。通过 `stage_state` 暂存的游标只有在资源包原子发布成功后才写入
`metadata/step/{state_key}.json`，因此失败或被取消的轮询会重读同一时间窗；没有新内容的轮询不发布资源包，只刷新
`last_poll_at` 与 `polls`。`state_key` 默认取 Step 的 `backend`，所以同一数据源要并行接入多个账号时，需要为每个
Step 配置不同的 `state_key`。删除状态文件即可让该采集器从初始时间窗重新开始。载荷辅助方法保证服务商文本不会逃出
资源包：`write_payload` 拒绝空载荷和不安全路径，`safe_name` 把不可信文本压缩成一个文件名成分，`truncate` 在 UTF-8
字符边界处截断。

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
