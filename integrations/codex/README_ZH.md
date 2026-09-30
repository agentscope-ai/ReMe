# Codex 的 ReMe 长期记忆插件

[English](./README.md)

先启动自己的 ReMe 服务，再安装 ReMe 插件，通过原生 MCP 设置界面配置，即可让 Codex 跨会话使用记忆。
插件在提交消息前召回相关历史，在完成对话后自动记录。持久记忆保存在你掌控的 ReMe 工作区文件中。

## 开始前准备

- **Codex：**宿主需支持本地插件、命令 Hook 和 Hook 信任审核。原生配置表单还需支持
  `openai/settings` MCP 扩展。CLI 0.145.0 支持 Hook 和工具，但不渲染该表单。
- **插件运行环境：**宿主的 `python3` 为 Python 3.11+，已安装 FastMCP >= 3.4.2。
  插件内的 MCP Server 和 Hook 都使用此环境，即使 ReMe 服务运行在其他环境或机器上。
- **ReMe：**HTTP 服务已启动并启用 MCP，开放 `search`、`auto_memory` 和 `health_check`。
  按 [ReMe 配置指南](../../docs/zh/configuration.md) 配置记忆提取模型。默认 BM25 召回不需要 Embedding 模型。
- **源码：**本地有 ReMe 仓库副本。插件及 marketplace 位于 `integrations/codex/`，从该目录安装。

在启动 Codex 的 Python 环境中检查依赖：

```bash
python3 -c "import sys, fastmcp; print(sys.version); print(fastmcp.__version__)"
```

如果宿主环境缺少 FastMCP，在该环境执行 `python3 -m pip install "fastmcp>=3.4.2"`。

## 1. 启动 ReMe 服务

```bash
pip install "reme-ai[core]"
reme start \
  workspace_dir=/absolute/path/to/reme-workspace \
  service.backend=http \
  service.host=127.0.0.1 \
  service.port=2333
```

保持服务运行，MCP 地址为 `http://127.0.0.1:2333/mcp`。如果使用其他端口，在第 3 步填写对应地址。
地址必须能**从运行 Codex 的机器访问**；`127.0.0.1` 指向的是这台机器。

插件不会启动 ReMe。HTTP 服务没有内建鉴权，请保持 loopback 监听或置于受保护的代理之后。
服务的 `workspace_dir` 决定读写哪一份记忆文件。

## 2. 安装插件并信任 Hook

使用仓库的绝对路径添加 marketplace，再安装插件：

```bash
codex plugin marketplace add /absolute/path/to/ReMe/integrations/codex
codex plugin add reme@reme-codex
```

> **截图待补 — `plugin-installed.png`:** 截取真实 Codex 插件详情，保留 ReMe 名称、启用状态和 `reme` MCP Server 卡片。

<!--
![Codex 中已安装并启用的 ReMe 插件](./figures/plugin-installed.png)
-->

重启桌面客户端，或启动新的 Codex CLI 会话。打开宿主的 Hook 设置（CLI 中使用 `/hooks`），
找到 **ReMe / `reme@reme-codex`**，审核以下四个 Hook：

| Hook | 作用 | 所需状态 |
| --- | --- | --- |
| `UserPromptSubmit` | 在提交消息前召回相关记忆 | 已启用、已信任 |
| `Stop` | 捕获已完成的对话轮次，提交达到条件的记忆批次 | 已启用、已信任 |
| `SessionStart` | 重试之前会话中尚未提交的批次 | 已启用、已信任 |
| `SessionEnd` | 在短时间预算内尝试最后一次刷新 | 已启用、已信任 |

> **截图待补 — `hooks-trusted.png`:** 截取四个 ReMe Hook 及其已启用、已信任状态；桌面设置页或 CLI `/hooks` 均可。

<!--
![Codex 中已启用并信任的四个 ReMe Hook](./figures/hooks-trusted.png)
-->

安装插件不会自动授予 Hook 信任。未完成审核时，MCP 连接可能正常，但自动召回和记录不会执行。
宿主规则见 [Codex Hook 信任说明](https://learn.chatgpt.com/docs/hooks#review-and-trust-hooks)。

更新仓库后，再执行一次 `codex plugin add reme@reme-codex`，重启宿主；如提示 Hook 定义有变化，
重新审核后新建会话。用户配置保存在插件缓存之外，升级会保留。

## 3. 配置 ReMe MCP 连接

在桌面客户端的插件目录中打开**已安装的 ReMe 插件**，找到 `reme` MCP Server 卡片，
点击标为 **Open MCP settings** 的设置入口，再打开该连接的 ReMe 配置表单。
不同宿主版本的文字和位置可能不同；这里调整的是 MCP 连接的配置，插件启用开关和 Hook 信任位于各自的设置入口。

首次验证时：

1. 将 **ReMe MCP address**（`mcp_url`）设为完整的 ReMe MCP 地址，包括 `/mcp`。
2. 开启 **Automatic recall**（`auto_recall`）和 **Automatic memory**（`auto_memory`）。
3. 将 **Turns per memory batch**（`memory_interval`）设为 `1`，使一轮完成的对话即可触发写入。
4. 保存后点击 **Check connection**。已验证的桌面客户端通过按钮上的提示浮层显示结果；
   将鼠标停在按钮上，可看到 ReMe 版本和 `healthy` 状态。

> **截图待补 — `mcp-settings.png`:** 截取原生设置表单，保留 MCP 地址、两个自动开关、批次大小和 Check connection 的健康结果浮层。

<!--
![ReMe 原生 MCP 设置与连接健康结果](./figures/mcp-settings.png)
-->

**连接健康仅代表服务可达，不代表 Hook 已加载、已信任或记忆提取成功。**
继续执行下方的双会话验证。连接检查不会向 ReMe 发送对话内容。

找不到表单时，先确认进入的是已安装插件的 MCP 连接，并确认 MCP 进程启动成功。
不支持 `openai/settings` 的客户端没有原生表单；用户配置文件的位置见[配置项参考](#配置项参考)。

## 4. 在两个独立会话中验证记录与召回

保持两个自动开关开启，并设 `memory_interval=1`。每次验证使用一个新的测试项目名，
避免旧测试的记忆让本次召回看似成功。

**会话 A：记录一条事实。**

```text
请记住 ReMe-Hook-Local-6502 的已确认决定：
发布评审在每周五 16:45 UTC，校验短语为 amber-lynx-6502。
不要调用工具、使用 Skill 或自行写入文件，只需确认这些事实。
```

等待最终回复和 Stop Hook 执行完成，查看本机插件日志：

```bash
tail -n 10 "${CODEX_HOME:-$HOME/.codex}/reme/hooks.log"
```

`memory_saved` 表示 ReMe 已确认收到该批次，同时检查 ReMe 工作区 `daily/` 下的笔记。
成功提交后，插件的 `reme/queue/` 目录会留下 `.done` 收据；未确认的批次仍为 `.json` 文件。
不要通过删除待提交文件来消除错误。

> **截图待补 — `memory-recorded.png`:** 截取会话 A 的测试事实及最终回复。此图展示对话；是否交付成功仍需以上述 `memory_saved` 和收据为准。

<!--
![Codex 测试会话 A 中的事实输入及最终确认](./figures/memory-recorded.png)
-->

**会话 B：新建独立会话后召回。**

```text
ReMe-Hook-Local-6502 的评审时间和校验短语是什么？
只引用已提供的记忆，并附来源文件路径。
不要调用工具或使用 Skill；没有提供记忆就说不知道。
```

预期回答包含**每周五 16:45 UTC**、**amber-lynx-6502** 和 ReMe 来源路径。
必须使用新会话，避免直接从会话 A 的上下文得到答案。

> **截图待补 — `memory-recalled.png`:** 截取独立会话 B 的提问和回答，保留评审时间、校验短语及 ReMe 来源路径。

<!--
![Codex 在新会话中自动召回测试事实并引用 ReMe 文件](./figures/memory-recalled.png)
-->

验证后，可保留 `memory_interval=1` 逐轮提交，或恢复默认值 `5` 批量提交。
需要主动查询时，可询问“之前关于这个项目做了什么决定？请附上 ReMe 来源”。
内置 `reme-memory` Skill 和 MCP 工具也可用于显式检索。

## 配置项参考

在 MCP 表单中修改并保存。下一次 Hook 或 MCP 工具调用即使用新配置，无需重新连接；
正在执行的调用沿用原配置。上游 ReMe 服务离线时，表单仍可用，可以修正连接地址。

| 配置项 | 默认值 | 作用 |
| --- | --- | --- |
| `mcp_url` | `http://127.0.0.1:2333/mcp` | 显式工具和 Hook 共用的 MCP 地址，支持自定义路径 |
| `auto_recall` | `true` | 用户提交消息前自动检索 |
| `auto_memory` | `true` | 自动捕获和提交已完成轮次；关闭后也暂停队列重试 |
| `memory_interval` | `5` | 每批完成的对话轮数；设为 `1` 则逐轮提交 |
| `recall_limit` | `5` | 检索结果上限 |
| `recall_min_score` | `0` | 自动召回最低分数 |
| `context_max_chars` | `8000` | 召回正文字符上限 |
| `recall_timeout` | `5` | 召回超时秒数，大于 0、最大 10 |
| `request_timeout` | `600` | 记忆写入和工具调用超时秒数，大于 0、最大 600 |
| `shutdown_timeout` | `2` | 退出刷新预算秒数，大于 0、最大 2 |
| `timezone` | `Asia/Shanghai` | 批次按此时区分日，应与 ReMe 工作区一致 |

只需召回时，关闭 `auto_memory` 并保持 `auto_recall` 开启。暂停所有自动行为时，关闭两个开关。
显式 MCP 工具仍然可用。

配置原子保存到 `${CODEX_HOME:-$HOME/.codex}/reme/config.json`。该文件由用户掌控，也可直接编辑，
只需保存要覆盖的选项，完整示例见 [config.example.json](plugins/reme/config.example.json)。
移除配置项即可恢复默认值。无效值和未知字段会被拒绝，界面保存失败时保留原文件。
请通过此配置调整插件，不要修改安装缓存。

修改 `mcp_url` 不会把旧队列中的对话发往新服务；切回原地址后可以继续重试。
旧 `api_url` 字段仍可读取，但不再使用，下次界面保存时会移除。标准 `/mcp` 的队列标识保持兼容，
自定义旧 HTTP 地址下的队列原样保留。

## 常见问题

按**服务连接 → Hook 状态 → 记录 → 新会话召回**的顺序排查。

| 现象 | 检查或处理 |
| --- | --- |
| 找不到 ReMe 设置 | 打开已安装插件的 `reme` MCP 卡片 → **Open MCP settings**，确认宿主支持原生表单且 MCP 进程启动成功。 |
| Check connection 转圈后没有聊天回复 | 将鼠标停在按钮上查看提示浮层；ReMe 版本与 `healthy` 表示服务检查成功。 |
| 连接 healthy，但记录和召回都没发生 | 检查四个 ReMe Hook 是否均已加载、启用且信任。安装插件和连接健康不会自动授予 Hook 信任。 |
| Hook 列表中没有 ReMe | 更新仓库，重新安装插件并重启宿主；兼容性说明见下文。 |
| Hook 已列出但等待审核 | 审核并信任当前定义，再新建会话。 |
| `hook_failed` 或 MCP 启动失败 | 检查宿主的 Python 3.11+/FastMCP 环境、保存的配置、服务连通性和 ReMe 服务端日志。 |
| 对话一轮后没有笔记 | 设 `memory_interval=1`，开启 `auto_memory`，等待最终回复及 Stop Hook。ReMe 还需要可用的记忆提取模型。 |
| `capture_skipped` | 升级到当前插件。记录需要本地 transcript 和最终助手消息；当前支持新旧 Codex 对话事件格式。 |
| `retry_failed` 或存在待提交队列 | 恢复原服务及模型连接，新建会话触发重试，保留待提交文件。 |
| 已有笔记，但新会话没有召回 | 开启 `auto_recall`，核对地址和工作区，等待索引更新，检查召回数量及分数过滤。可显式调用 ReMe 搜索，区分检索问题和 Hook 问题。 |

日志只记录状态和异常类型，不记录对话内容。成功召回或空闲 Hook 不一定写日志；
仅凭 `hooks.log` 不存在，不能判断 Hook 从未执行。队列文件则包含来源对话文本。

## 记忆行为与兼容性

- 召回内容有长度上限，以 `<reme-context>` 包裹并标为不可信历史证据。
  只记录完成轮次的用户与最终助手文本；排除工具输出、推理、子 Agent、中间回复和注入的记忆。
  ReMe 不需要访问 Codex 的 transcript 文件。
- 稳定的会话／消息 ID、会话锁和收据避免重复 Stop 再次交付同一条来源消息。
  写入失败时，队列按服务地址及会话保留。
- 为兼容会跳过异步 Hook 的 Codex 0.145.0，插件采用同步 Hook。
  达到批次条件或启动重试时，可能等待至 `request_timeout`。退出时仅尽力刷新，剩余批次留待下一会话。
- 交付是尽力而为，记忆提取采用至少一次语义。宿主在捕获前被强制结束可能丢失该轮；
  确认响应丢失可能导致再次提取。
- 持久记忆和每日整理由 ReMe 服务管理，插件不启动额外守护进程或调度器。

插件采用 `.codex-plugin/plugin.json` 与 `.mcp.json` 布局。原生验证发现，Codex 0.159.2 和
桌面运行时 0.158.0-alpha.2.1 在存在根目录 portable `plugin.json` 时会遗漏本插件的 Hook，
显式声明也无效，因此保留当前 Codex 布局。解析器同时支持旧版 `user_message` / 最终 `agent_message`
和新版 `item_completed` 对话事件，排除 `response_item`。实现见 [Hook 适配器](plugins/reme/hooks/auto_memory.py)。

CLI 0.145.0 无人值守会话无法收集显式工具调用审批时，可能返回 `user cancelled MCP tool call`。
请在交互会话中验证这类调用；自动 Hook 使用前文所述的信任审核机制。

## 维护者验证

五个真实截图槽位及替换方式见[截图清单](./figures/README.md)。中英文说明共用同一组 PNG；
补图前保留文字槽位，不显示缺失图片。

```bash
pytest tests/unit/test_coding_agent_plugins.py tests/unit/test_codex_plugin_mcp.py -v
REME_CODEX_BIN=/path/to/codex pytest tests/integration/test_codex_plugin_install.py -v
```

原生安装测试需要运行时支持 `hooks/list`，使用隔离配置目录，验证信任前的四个 Hook 发现行为，
不调用模型或执行 Hook。文档修改后，在 `github-pages/` 运行 `npm test` 和 `npm run build`。

参考：[Codex Hook 文档](https://learn.chatgpt.com/docs/hooks)、
[OpenAI MCP 设置协议](https://github.com/openai/mcp-extensions/blob/node-v0.1.0/docs/spec.md#structured-settings)。
ReMe 在 [agentscope-ai/ReMe](https://github.com/agentscope-ai/ReMe) 开发，采用 Apache-2.0 许可证。
