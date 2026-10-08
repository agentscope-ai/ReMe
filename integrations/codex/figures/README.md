# Codex manual screenshot workflow / 手动截图流程

仅使用真实 Codex 界面截图。按下面的顺序完成六张必需截图，中英文使用说明共用同一组 PNG。
截图槽位在收到对应图片前保留；无需为两种语言各拍一套。

Use real Codex screenshots only. Follow the capture order below; the English and Chinese guides share
the same six PNGs. Keep each placeholder until its corresponding screenshot is available.

## 开始前 / Before capture

1. 按[中文使用指南](../README_ZH.md)或[English guide](../README.md)完成服务启动和安装。
   已安装时无需重新安装；先在 Codex 的已安装插件中确认 ReMe 已启用。
   Keep your ReMe service running and confirm the installed ReMe plugin is enabled.
2. 使用 **Codex CLI >= 0.159.2** 和 **Codex 桌面端 >= 26.924.22138**。
   原生 MCP 设置和状态面板需在桌面端拍摄；Hook 页面可使用桌面端或 CLI `/hooks`。
   Use the desktop app for native MCP settings and the status panel; CLI `/hooks` is also valid for the Hook image.
3. 在启动 Codex 的同一环境中检查 Python。需要 Python 3.11+、FastMCP >= 3.4.2；
   若终端默认 Python 版本较低，先切换到已安装这些依赖的环境。
   Check Python in the environment used to launch Codex:

   ```bash
   codex --version
   python3 -c "import sys, fastmcp; print(sys.version); print(fastmcp.__version__)"
   ```

4. 为两次测试对话选择一个空目录作为 Codex 项目，例如：
   Use an empty Codex project for both test conversations:

   ```bash
   mkdir -p "$HOME/reme-codex-demo"
   ```

   在 Codex 中打开该目录作为项目；若这个目录已有内容，使用另一个新目录。
   Open that directory as a Codex project; choose a new directory name if it already contains files.
   两个会话连接**同一个 ReMe 服务和记忆工作区**，但会话 B 必须是新建会话，不能接续或分叉会话 A。
   空目录用于避免 Codex 从仓库里的示例文档读取答案；它与 ReMe 的记忆工作区是不同目录。
   Both conversations use the same ReMe endpoint and memory workspace. Start B as a new conversation,
   without continuing or branching A. An empty project keeps repository examples out of the answer.
5. 调整窗口宽度和字体，让截图中的文字清晰可读，保留足够的 Codex 界面以辨认页面。
   macOS 可用 `Shift + Command + 4` 手动框选区域，再将 PNG 重命名并移到本目录。
   裁去无关对话、账号信息和凭据；使用适合公开展示的服务地址。
   Keep text readable and enough Codex UI to identify the page; crop unrelated conversations and account details.

## 六张必需截图 / Six required captures

| 顺序 / Order | 文件名 / Filename | 页面 / Page | 拍摄前确认 / Ready when |
| --- | --- | --- | --- |
| 1 | `plugin-installed.png` | 已安装的 ReMe 插件详情 / Installed ReMe details | ReMe 图标、名称、已启用状态及 `reme` MCP 卡片可见 / Icon, name, enabled state, and MCP card visible |
| 2 | `hooks-trusted.png` | Hooks 设置或 CLI `/hooks` / Hook review | 七个 ReMe 条目均已启用、已信任 / All seven ReMe entries enabled and trusted |
| 3 | `mcp-settings.png` | 原生 **Open MCP settings** 表单 / Native settings | 已保存地址、自动开关、批次大小；**ReMe status** 入口可见 / Saved address, switches, batch size, and status action visible |
| 4 | `memory-recorded.png` | 独立会话 A / Conversation A | 测试事实和回复可见，后台已出现 `memory_saved` / Fact and reply visible, recording confirmed |
| 5 | `memory-recalled.png` | 新建会话 B / New conversation B | 问题、正确时间、校验词和来源路径可见，已出现 `recall_found` / Answer, source path, and automatic recall confirmed |
| 6 | `plugin-status.png` | **ReMe status → 自动记忆 / Auto Memory** | 最近活动已展开，可见保存和召回记录，待提交轮数为零 / Activity expanded, saved/recalled events visible, queue zero |

### 1. 已安装并启用 / Installed and enabled

打开 Codex 的已安装插件列表，进入 **ReMe** 详情。确认已启用后，截取名称、图标、启用状态和
`reme` MCP 卡片，保存为 `plugin-installed.png`。若缺少 MCP 卡片或设置入口，先检查版本、启用状态，
再重启客户端；页面未加载完成时先不要截图。

Open the installed ReMe details. Capture the icon, name, enabled state, and `reme` MCP card.
If the card or settings action is missing, check the version and enablement, then restart the app.

### 2. 审核并信任 Hook / Review and trust Hooks

进入 Hooks 设置，或在新 CLI 会话输入 `/hooks`。逐一审核 ReMe 条目，并由你在 Codex 中启用、信任。
当前共七个条目：`UserPromptSubmit`、`SessionStart`、`SessionEnd` 各一个，`Stop` 和 `SubagentStop` 各两个。
保存为 `hooks-trusted.png`，保留事件名、ReMe 来源及信任状态。
若单页放不下，可另拍 `hooks-trusted-more.png`，按补图步骤在对应位置追加第二张图片。

Review and trust the seven ReMe entries yourself. Capture their event names, plugin source, and trust state.
Use a second image if the list does not fit legibly; do not shrink the text just to fit everything.

### 3. 保存 MCP 设置并检查连接 / Save settings and check the connection

在 ReMe 详情的 **reme MCP server** 卡片中点击 **Open MCP settings**，保存以下值：
Open the native MCP form and save these values:

| 字段 / Field | 测试值 / Test value |
| --- | --- |
| ReMe MCP address | 当前服务的真实可达地址 / Your actual reachable endpoint |
| Automatic recall | 开启 / On |
| Automatic memory capture | 开启 / On |
| Capture batch size | `1` |
| Memory guidance language | `zh` 或 / or `en` |

点击 **ReMe status**，等 **总览 / Overview** 显示 **健康 / Healthy** 和 ReMe 版本。
返回设置表单，截取地址、两项自动开关、批次大小及 **ReMe status** 入口，保存为 `mcp-settings.png`。
若字段分布在不同滚动位置，可补拍 `mcp-settings-more.png`，保留可读字号。

Wait for Healthy and the ReMe version in Overview, then return to the settings form for this capture.
Connection health confirms connectivity; the following steps separately verify recording and recall.

### 4. 在会话 A 记录事实 / Record a fact in conversation A

在空项目中新建会话 A，发送下面的中文或英文提示。两次对话使用同一个项目名和校验词。
若已经测试过 `7319`，先将两个提示里的后缀统一改成一个未使用过的值。
不要额外请求调用 `auto_memory`，让自动记录完成本次验证。

Start A in the empty project. Choose one prompt below; change the suffix in both prompts if it was used before.
Let automatic capture save the turn rather than requesting an explicit `auto_memory` call.

```text
请记住这个项目决策：ReMe-Example-7319 每周五 16:45 UTC 进行评审，
项目关键词是 amber-lynx-7319。
```

```text
Remember this project decision: ReMe-Example-7319 reviews happen Friday
at 16:45 UTC. The project keyword is amber-lynx-7319.
```

回复完成后保持 Codex 和 ReMe 运行，打开 **ReMe status → 自动记忆 / Auto Memory → 最近活动 / Recent activity**。
点击 **刷新 / Refresh**，等待 **记忆已保存 / Memory saved**（`memory_saved`），并确认 **待提交轮数 / Queued turns** 为 `0`。
然后回到会话 A，截取提示和完整回复，保存为 `memory-recorded.png`。

Keep Codex and ReMe running until Memory saved appears and queued turns reach zero. Then capture A's prompt and reply.
An acknowledgement alone does not establish that recording succeeded.

### 5. 在新会话 B 召回 / Recall in a new conversation B

在同一个空项目中另建会话 B，不复制 A 的内容，不在问题中泄露时间和校验词。发送：
Start a separate conversation in the same empty project. Do not copy A's answer or include the fact in the question:

```text
ReMe-Example-7319 的评审时间和项目关键词是什么？
请根据 ReMe 记忆回答，并附上来源路径。
```

```text
For ReMe-Example-7319, when are the reviews and what is the project keyword?
Use ReMe memory and include the source path in your answer.
```

确认回答包含 **每周五 16:45 UTC**、**amber-lynx-7319** 和真实的 ReMe 来源路径，例如 `daily/...md`。
再检查最近活动中出现 **已召回相关记忆 / Relevant memory recalled**（`recall_found`）。
若不在最近五条中，点击 **查看全部 … 条活动 / Show all … events**；点击单条活动可查看事件码和完整时间。
答复正确且自动召回活动已确认后，截取 B 的问题、回答和来源路径，保存为 `memory-recalled.png`。

Expect Friday at 16:45 UTC, `amber-lynx-7319`, and a real ReMe source path. Also confirm Relevant memory recalled
in Recent activity. Use Show all … events if it is outside the latest five; expand an event for its code and full timestamp.
A correct answer alone does not establish that automatic recall ran.

### 6. 展示状态面板 / Capture the status panel

再次从原生设置打开 **ReMe status**，选择 **自动记忆 / Auto Memory**，点击刷新并展开 **最近活动 / Recent activity**。
等待会话 B 的记录也保存完成、待提交轮数回到 `0`，再截取标题、页签、自动记录与召回状态及本次保存和召回活动，
保存为 `plugin-status.png`。
在最近活动仍保留这两条记录时拍摄，避免继续发送无关测试消息。
若需要展示的记录不在最近五条中，先点击 **查看全部 … 条活动 / Show all … events**。

Open the real status panel, select Auto Memory, refresh, and expand Recent activity. Capture the tabs,
automatic switches, zero pending turns after B is also saved, and the saved/recalled events before unrelated activity replaces them.
Use Show all … events if the needed entries are outside the latest five.

状态面板的总览、自动记忆和记忆整理是不同页签，不需要把所有内容挤到一张图里。
可另拍以下真实页面；没有开启每日计划时，显示暂停或没有计划是有效状态，无需为截图改动计划或执行整理。
The tabs show different information. Optional captures below can show connection health and Dream separately;
a paused schedule is valid, so there is no need to enable or run consolidation just for a screenshot.

| 可选文件 / Optional file | 内容 / Content |
| --- | --- |
| `plugin-status-overview.png` | **总览 / Overview** 中的健康状态、ReMe 版本和待提交轮数 / Health, version, and queue |
| `plugin-status-dream.png` | **记忆整理 / Consolidation** 中的 cron、时区、下次运行和最近结果 / Schedule, timezone, next run, and last result |

## 遇到失败时 / If a step fails

| 现象 / Symptom | 下一步 / Next step |
| --- | --- |
| 找不到设置入口 / Missing settings action | 检查版本、ReMe 已启用状态，再重启 Codex / Check version and enablement, then restart |
| 状态卡片为空或加载失败 / Empty or failed status card | 先按使用指南的网络排查步骤检查宿主沙箱连接；不要用浏览器仿制页面代替 / Follow the guide's network troubleshooting |
| 回复确认但没有 `memory_saved` / Acknowledged but not saved | 检查全部 Hook 信任、批次为 `1`、模型配置及服务日志；未保存前不进行 B / Check Hook trust, batch size, model setup, and logs before B |
| 答对但没有 `recall_found` / Correct answer without recall event | 检查 Automatic recall 和 Hook 信任，确认 B 是独立会话，再用新后缀重试 / Check recall, trust, and fresh conversation |

## 保存图片并接入文档 / Save and add the images

从仓库根目录看，所有 PNG 放在 `integrations/codex/figures/`；六个必需文件为：
Save the PNGs in this directory with these exact names:

```text
integrations/codex/figures/
├── plugin-installed.png
├── hooks-trusted.png
├── mcp-settings.png
├── memory-recorded.png
├── memory-recalled.png
└── plugin-status.png
```

图片保存好后即可交给维护者接入文档；若自行完成，对每张已存在的图片依次执行：
Once the files are ready, hand them off for integration, or make these changes yourself for each supplied image:

1. 在 `../README.md` 和 `../README_ZH.md` 中找到同名截图槽位，删除对应的可见占位引用块。
   Remove its visible screenshot-placeholder blockquote in both guides.
2. 移除对应图片行外的 `<!--` / `-->`，保留相对路径。例如：
   Uncomment the matching image line, keeping its relative path:

   ```markdown
   ![ReMe MCP settings in Codex](./figures/mcp-settings.png)
   ```

3. 缺少图片的槽位继续保留。可选补图放在相关必需图片后，在中英文说明中分别加上对应图片行。
   Leave missing-image slots intact. Add optional images after their related required image, with captions in each language.
4. 从仓库根目录执行以下检查，再打开预览中的中英文 Codex 页面，确认图片加载、文字清晰且排版正常：
   Run these checks from the repository root, then inspect both Codex guide pages in the preview:

   ```bash
   npm --prefix github-pages test
   npm --prefix github-pages run build
   npm --prefix github-pages run preview
   ```

不要提交服务日志、对话导出文件、配置凭据或临时记忆工作区。只提交已审核的 PNG 和对应文档修改。
Commit the reviewed PNGs and documentation changes; keep logs, transcript exports, credentials, and demo memory out of the commit.
