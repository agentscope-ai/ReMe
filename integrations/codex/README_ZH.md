# 在 Codex 中使用 ReMe

[English](./README.md)

将 Codex 接入自己的 ReMe 服务，即可跨会话记住事实、偏好和决策：提问前自动召回相关记忆，
对话完成后自动记录，并按日整理记忆。长期记忆保存在你自己的 ReMe 工作区 Markdown 文件中。

## 开始前

- 使用最新版 Codex，支持安装插件和审核 Hook；原生 MCP 设置使用桌面版打开。
- 启动 Codex 的环境中，`python3` 需要是 Python 3.11+，并安装 `fastmcp>=3.4.2`。
  即使 ReMe 在远程运行，Codex 所在机器也需要这个环境。检查命令：

  ```bash
  python3 -c "import sys, fastmcp; print(sys.version); print(fastmcp.__version__)"
  ```

  如未安装，执行 `python3 -m pip install "fastmcp>=3.4.2"`。
- 按[配置指南](../../docs/zh/configuration.md)配置 ReMe 用于提取记忆的模型。默认 BM25 检索不需要 embedding 模型。
- 准备本仓库的本地副本，用于安装。

## 1. 启动 ReMe 服务

```bash
pip install "reme-ai[core]"
reme start \
  workspace_dir=/absolute/path/to/reme-workspace \
  service.backend=http \
  service.host=127.0.0.1 \
  service.port=2333 \
  service.tool_error_on_failure=true \
  jobs.dream_cron.backend=base \
  jobs.dream_cron.enable_serve=false
```

保持服务运行。MCP 地址为 `http://127.0.0.1:2333/mcp`，必须能从 **Codex 所在机器**访问。
远程部署时填写实际可达的服务地址。`workspace_dir` 决定记忆文件存放位置。
未启用认证的服务应仅监听本机，或放在受保护的代理之后。

保留 `service.tool_error_on_failure=true`：记忆任务失败时，MCP 调用也必须报告失败，
尚未确认写入的对话才能留在队列中重试。连接健康不能证明这个选项已开启。

最后两项配置关闭 ReMe 自带的 Dream 定时任务，让 Codex 负责定时触发；使用现有服务配置即可，
无需修改 ReMe 代码。如果服务已有整理计划且希望继续使用，省略这两项，并在 Codex 中关闭
**Daily memory consolidation**（`autoDreamEnabled`）。同一个工作区保留一个调度来源。

## 2. 安装 ReMe 并信任 Hook

将路径替换为仓库所在的绝对路径：

```bash
codex plugin marketplace add /absolute/path/to/ReMe/integrations/codex
codex plugin add reme@reme-codex
```

重启桌面客户端，或打开新的 CLI 会话。在已安装的 ReMe 详情中确认已启用。

> **截图槽位 — `plugin-installed.png`：** 真实 Codex 中的 ReMe 详情页，保留启用状态和 `reme` MCP 卡片。

<!--
![Codex 中已安装并启用 ReMe](./figures/plugin-installed.png)
-->

在 Codex 的 Hooks 设置中，审核并信任 **所有 ReMe 条目**；CLI 可使用 `/hooks`。
共七个处理条目，分属五类事件：

| 事件 | 用途 |
| --- | --- |
| `UserPromptSubmit` | 提问前召回相关记忆。 |
| `Stop` | 先在本地保存已完成轮次，再后台提交到期批次；共两个条目。 |
| `SubagentStop` | 子代理的两个对应条目，仅在 `rootAgentsOnly=false` 时使用。 |
| `SessionStart` | 后台重试待提交的记录。 |
| `SessionEnd` | 在有限退出时间内尝试提交，保留未完成批次。 |

未信任的 Hook 不会执行。安装成功或 MCP 连接正常，都不等于 Hook 已获信任。
升级后如条目发生变化，需要重新审核。参见 [Codex Hook 审核与信任说明](https://learn.chatgpt.com/docs/hooks#review-and-trust-hooks)。

> **截图槽位 — `hooks-trusted.png`：** Codex Hooks 设置或 `/hooks` 中，七个 ReMe 条目均已启用、已信任。

<!--
![ReMe Hook 已启用并信任](./figures/hooks-trusted.png)
-->

## 3. 在 MCP 设置中连接服务

打开已安装的 ReMe 详情，找到 **reme MCP server** 卡片，点击 **Open MCP settings**。
服务地址、自动记忆开关等均在这个原生表单中调整。

首次验证建议保存以下配置：

| 字段 | 值 |
| --- | --- |
| ReMe MCP address（`mcpUrl`） | `http://127.0.0.1:2333/mcp`，或实际可达地址 |
| Automatic recall（`autoRecall`） | 开启 |
| Automatic memory capture（`autoMemoryEnabled`） | 开启 |
| Capture batch size（`autoMemoryInterval`） | `1` |
| Memory guidance language（`language`） | `zh` 或 `en` |

先保存，再点击 **ReMe status** 打开状态面板。顶部的服务连接区域会检查已保存的地址，
服务可达时显示 ReMe 版本及 `healthy`。这说明服务可访问；记录和召回还需要通过下一步的对话验证。

> **截图槽位 — `mcp-settings.png`：** 真实原生表单，保留服务地址、自动开关、批次大小及唯一的 ReMe status 入口。

<!--
![ReMe MCP 设置与连接结果](./figures/mcp-settings.png)
-->

## 4. 用两个独立会话验证记忆

新建会话 A，使用一组独立的项目名和校验词，例如：

```text
请记住这个已确认的项目决策：ReMe-Example-7319 每周五 16:45 UTC 进行评审，
校验词是 amber-lynx-7319。
不要调用工具或自己写文件，只需确认这些事实。
```

保持会话打开，等待自动记录完成。在 MCP 设置中点击 **ReMe status**，应看到 `memory_saved`，
待提交轮数为零；同时检查 ReMe 工作区的 `daily/` Markdown 文件中出现了这条事实。
Codex 的确认回复本身不能证明记忆已写入。

> **截图槽位 — `memory-recorded.png`：** 会话 A 中的测试事实和确认回复，不包含显式工具调用。

<!--
![在 Codex 会话中提供需要记住的事实](./figures/memory-recorded.png)
-->

另建会话 B：

```text
ReMe-Example-7319 的评审时间和校验词是什么？
请根据已提供的 ReMe 记忆回答并引用来源路径。不要调用工具；
如果没有提供相关记忆，就说不知道。
```

预期答复包含正确时间、校验词及 `daily/...md` 等来源路径；**ReMe status** 中应有 `recall_found`。
这样可以单独确认自动召回生效。日常使用时，也可以让 Codex 通过 `reme_search` 主动检索过去的决策。

> **截图槽位 — `memory-recalled.png`：** 独立会话 B 中的正确事实和 ReMe 来源路径。

<!--
![Codex 在独立会话中召回记忆](./figures/memory-recalled.png)
-->

验证后可调整批次大小，默认每五个完成轮次提交一次；不足一批时，也会在会话边界尝试提交。
失败或中断的批次保留在本地，供后续会话重试。短暂运行的 `codex exec` 可能在后台写入完成前退出；
验证自动记录时，应保持会话运行。

## 5. 设置定时整理或立即整理

在 **Auto Dream** 分组中设置 **Daily memory consolidation**、**Auto Dream schedule**、
**Workspace timezone**，按需填写 **Auto Dream hint**，然后保存。默认 `0 23 * * *`、
`Asia/Shanghai`，即每天上海时间 23:00。仅支持 `分钟 小时 * * *` 的每日计划，例如每天 02:30 为
`30 2 * * *`。Hint 用于指导 ReMe 将现有日常记录整理为长期记忆。

**ReMe status** 会展示计划、时区、下次运行时间和最近结果。只有 Codex 保持 ReMe MCP 连接时，
定时任务才会触发；关闭 Codex 后停止计时，重启时不补跑错过的时点。同一 `CODEX_HOME` 下的多个会话
共用一个调度器；不同配置目录或机器需自行协调，同一个 ReMe 工作区仅启用一处定时整理。

需要立即整理时，先等待待提交记录完成，再点击 **Consolidate now (updates memory files)**，
或明确让 Codex 调用 `reme_run_dream`。该操作会更新已有记忆文件，关闭定时开关后仍可手动执行。
关闭定时开关不会取消已经发给 ReMe 的请求；超时也不代表服务端已停止处理。

## 6. 查看状态和调整配置

点击 **ReMe status**，从 MCP 设置打开独立状态面板。连接检查和记忆状态共用这一个入口，
面板分为 **总览、自动记忆、记忆整理、组件** 四个页签，分别展示连接健康与记忆流程，
记录开关、批次和队列、召回及最近活动，Dream 计划和结果，以及组件内存估算与进程 RSS。
点击 **刷新** 可重新检查，组件页的服务详情默认折叠。首次展示直接使用打开时返回的结果，不重复请求；面板适配宿主主题及保存的 `language`。

也可以在对话中让 Codex 调用 `reme_status`；不展示 App 的宿主仍会收到文字结果。
打开或刷新面板不会记录或整理记忆。服务不可达时仍可查看本地队列与活动。
当前 ReMe MCP 响应不包含组件级健康和索引文档数量，面板仅展示可获取的内存估算，不用其他计数替代。
Hook 是否可信，需另到 Codex 的 Hooks 设置中检查。

> **截图槽位 — `plugin-status.png`：** 真实 ReMe status 面板或 `reme_status` 输出，保留连接健康、待提交轮数、最近活动和下次 Dream 时间。

<!--
![Codex 中的 ReMe 写入和整理状态](./figures/plugin-status.png)
-->

| 配置项 | 默认值 | 含义 |
| --- | --- | --- |
| `mcpUrl` | `http://127.0.0.1:2333/mcp` | 完整 ReMe MCP 地址。 |
| `autoRecall` | `true` | 每次提问前自动召回。 |
| `autoMemoryEnabled` | `true` | 自动记录完成的用户/助手轮次。 |
| `autoMemoryInterval` | `5` | 每批完成轮数，范围 1–1000。 |
| `rootAgentsOnly` | `true` | 排除子代理的召回与记录。 |
| `searchLimit` | `5` | 最多返回的检索结果数，范围 1–50。 |
| `recallMinScore` | `0` | 检索最低得分，不小于零。 |
| `language` | `en` | 记忆指引和本地结果使用 `en` 或 `zh`；设置表单标签仍为英文。 |
| `autoDreamEnabled` | `true` | MCP 连接存活期间启用每日整理。 |
| `dreamCron` | `0 23 * * *` | 每日整理时间。 |
| `dreamHint` | 空 | 定时和手动整理使用的指导文字。 |
| `timezone` | `Asia/Shanghai` | 记忆日期及每日计划采用的 IANA 时区。 |
| `requestTimeoutMs` | `10000` | 前台请求超时，范围 1000–120000 毫秒。 |
| `backgroundTimeoutMs` | `3600000` | 记忆提取与整理超时，范围 1000–3600000 毫秒。 |
| `shutdownTimeoutMs` | `5000` | 退出等待时间，范围 100–60000 毫秒；受 Codex `SessionEnd` 限制，退出提交最多等待 2 秒。 |

保存后对后续调用生效，无需重新安装。配置保存在插件缓存之外：
`${CODEX_HOME:-~/.codex}/reme/config.json`。完整字段见[配置示例](plugins/reme/config.example.json)。
修改 `mcpUrl` 不会将旧队列中的对话发给新服务；切回原地址后才能重试这些记录。
整理进行期间修改的配置用于后续运行。

## 排查与更新

| 现象 | 检查方式 |
| --- | --- |
| 找不到 MCP 设置 | 在最新版桌面端打开已安装的 ReMe 详情及 `reme` MCP 卡片；确认启用后重启。 |
| 状态卡片空白，随后提示“插件功能未成功加载” | 检查 Codex 沙箱页面的网络访问；见下方说明。 |
| 连接健康，但没有记录或召回 | 检查全部 ReMe Hook 的信任状态、自动开关及最近活动；连接健康不代表 Hook 已执行。 |
| 记录一直在队列中 | 检查 ReMe 模型配置和服务日志，保持会话运行，并确认 `service.tool_error_on_failure=true`；后续 Hook 会尝试重试。 |
| 没有召回结果 | 先确认事实已在 `daily/` 中，使用独立项目名查询，再检查 `searchLimit` 和 `recallMinScore`。 |
| Dream 未触发 | 查看保存的时区、下次运行时间、开关及 MCP 连接；离线期间错过的时点不会补跑。 |
| 升级后提示未知配置字段 | 备份 `reme/config.json`，按当前配置示例替换旧字段，再打开 MCP 设置；不支持旧字段名。 |
| Python 或 FastMCP 启动报错 | 检查 Codex 实际继承的 `python3` 环境，尤其是桌面端的启动环境。 |

状态面板通过本地 MCP 提供 HTML 和状态数据，但 Codex 桌面端还需要加载自己的沙箱页面。
如果卡片空白，且 Codex 客户端日志出现 `guest_load_failed`、`ERR_UNEXPECTED` 或
`MCP sandbox RPC timed out`，可在运行 Codex 的机器上检查：

```bash
curl -i --max-time 20 https://web-sandbox.oaiusercontent.com/mcp-app.html
```

若响应为公司网络或安全软件的拦截提示，请按其正规流程放行 `web-sandbox.oaiusercontent.com`，
包括该域名下页面所需的静态资源，然后重启 Codex 并重新打开面板。这个加载错误本身不能说明
ReMe 服务或 Hook 是否正常，也不需要因此修改 `mcpUrl`、清空记忆或重装插件。
暂时可以在对话中要求“调用 `reme_status`，用文字展示返回的状态”，查看服务连接、队列和整理计划。

诊断日志位于 `${CODEX_HOME:-~/.codex}/reme/hooks.log`，仅包含事件名、计数和错误类型，不记录对话正文。
尚未确认写入的对话数据另存于该目录。记录采用尽力交付：本地捕获前强制终止可能丢失轮次，
服务端处理成功但确认丢失时，重试也可能导致重复提取。

更新仓库后重新安装：

```bash
codex plugin remove reme@reme-codex
codex plugin add reme@reme-codex
```

重启 Codex 并重新审核发生变化的 Hook。设置和待提交记录位于安装缓存之外，会继续保留。
[截图清单](./figures/README.md)列出了上面的六处真实 Codex 截图槽位，中英文共用图片。
