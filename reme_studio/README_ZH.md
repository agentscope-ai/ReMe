# ReMe Studio

[English](https://github.com/agentscope-ai/ReMe/blob/main/reme_studio/README.md) | 简体中文

ReMe Studio 是 [ReMe](https://github.com/agentscope-ai/ReMe) 的本地 Web 工作区。你可以在一个界面中浏览和编辑用户拥有的记忆文件、查看文件之间的关系、与 ReMe Agent 对话，并了解本地服务的运行状态。

Studio 遵循 ReMe 的本地优先理念：Markdown 和其他工作区文件始终是持久数据的唯一事实来源；搜索索引、目录、图谱、缓存和运行时元数据都是可重建的派生数据。

![ReMe Studio 概览](https://raw.githubusercontent.com/agentscope-ai/ReMe/main/reme_studio/figures/studio-overview.png)

> 本 README 中的所有截图均来自真实运行的本地 ReMe 服务，Studio 界面统一设置为英文。截图使用隔离的虚构 `Project Aurora` 工作区，不包含用户数据、凭证或私有服务地址。

## Studio 提供什么

| 区域       | 能力                                                                                                  |
| ---------- | ----------------------------------------------------------------------------------------------------- |
| 工作区     | 浏览完整工作区、聚焦日记或长期知识；磁盘文件变化后自动刷新                                            |
| Markdown   | 预览 front matter 和 GitHub Flavored Markdown，使用 Monaco 编辑、安全保存、下载，并同时打开多个标签页 |
| 记忆图谱   | 按知识分类查看已索引的 wikilink、识别链接方向、缩放和适应画布，并打开源笔记                           |
| Agent 对话 | 流式展示回答、推理、工具、审批、结构化数据和用量；可把工作区文件拖入对话作为引用                      |
| 服务中心   | 查看进程和组件健康状态、重建派生索引、检查脱敏后的生效配置，并核对版本                                |
| 个性化     | 切换中英文，以及浅色、深色或跟随系统的外观                                                            |

## 工作方式

```text
用户拥有的本地工作区文件
            │
            ▼
      ReMe HTTP 服务
       ├── 文件操作
       ├── 派生索引
       ├── wikilink 图谱
       └── 只读 Agent 对话
            │
            ▼
        ReMe Studio
```

Studio 是 ReMe HTTP 服务的客户端。它不会用独立文档数据库替代工作区，也不会把浏览器中的状态当作持久数据来源。标签页和界面偏好只是使用便利；配置的 ReMe 工作区中的文件始终具有权威性。

## 环境要求

- Python 3.11 或更高版本，并已安装 ReMe。
- 正在运行的 ReMe HTTP 服务。
- 仅在使用 Agent 对话时需要可用的 Agent 和模型配置。
- 仅在从源码开发或构建 Studio 时需要 Node.js 22.13 或更高版本。

后端安装、工作区和模型配置请参阅 [ReMe 仓库中文 README](https://github.com/agentscope-ai/ReMe/blob/main/README_ZH.md)。

## 安装

安装 Studio 及 ReMe 的可选集成功能：

```bash
pip install "reme-ai[core]"
```

如果不需要其他可选集成，只安装 Web extra：

```bash
pip install "reme-ai[web]"
```

基础 `reme-ai` 包以无界面模式分发，不包含前端资源。

Node.js 应用也可以安装同一份预构建静态工作区：

```bash
npm install @agentscope-ai/reme_studio
```

静态入口安装在 `@agentscope-ai/reme_studio/dist-static/index.html`。

## 启动 Studio

启动 ReMe HTTP 服务：

```bash
reme start
```

然后打开 <http://127.0.0.1:2333>。默认 HTTP 服务会同时提供 API 和打包后的 Studio 前端。

如需使用其他工作区或端口，可以传入正常的 ReMe 配置覆盖：

```bash
reme start workspace_dir=/absolute/path/to/workspace service.port=8000
```

Studio 会在导航器底部显示当前连接的服务地址；绿色指示点表示浏览器能够访问该服务。如果 Studio 与后端分开托管，请按[前端配置](#前端配置)设置 API 地址。

## 界面与能力详解

### 1. 工作区导航器

左侧导航器提供四个主要入口：

- **Files** 显示整个 ReMe 工作区中受支持的文件。
- **Daily** 聚焦配置的日记目录。
- **Knowledge** 聚焦长期摘要记忆下的 `wiki`、`personal` 和 `procedure` 目录。
- **Chat** 新建 Agent 对话，同时保留已经打开的文件和图谱。

目录可以独立展开。导航器会隐藏点文件和点目录，优先显示最新文件，应用有界的结果数量，并在磁盘文件变化后刷新。可以拖动分隔线调整导航器宽度，也可以使用顶部菜单按钮将其收起。

![工作区导航器与 Markdown 预览](https://raw.githubusercontent.com/agentscope-ai/ReMe/main/reme_studio/figures/files-workspace.png)

标签栏会把文件、图谱和对话放在同一个工作空间中。关闭标签页不会删除文件。文件存在未保存修改时，Studio 会阻止误关闭；标签页上下文菜单还支持关闭当前标签页或关闭其他标签页。

### 2. 日记

Daily 视图会汇总日期索引页面和按日期组织的笔记目录，但不会改变它们在磁盘上的路径。它只是同一本地工作区的聚焦视图，因此在这里打开的笔记也可以出现在 Files 中，并继续参与 ReMe 的索引和 Agent 工作流。

![日记视图](https://raw.githubusercontent.com/agentscope-ai/ReMe/main/reme_studio/figures/daily-notes.png)

可以用 Daily 检查时间顺序记忆、打开资源生成的笔记，并在无需浏览完整工作区树的情况下查看近期工作。

### 3. Markdown 预览

预览模式支持：

- 在独立摘要区展示 YAML front matter。
- 渲染标题、列表、链接、代码、表格、任务列表和其他 GitHub Flavored Markdown。
- 在文档上方显示原始的工作区相对路径。

下载按钮会通过浏览器保存文件副本；预览模式不会修改源文件。

![Markdown 预览](https://raw.githubusercontent.com/agentscope-ai/ReMe/main/reme_studio/figures/files-workspace.png)

### 4. Markdown 编辑与安全保存

编辑模式使用 Monaco，提供 Markdown 语法高亮、行号、键盘导航和占满可用高度的编辑区域。可以随时在 **Preview** 和 **Edit** 之间切换。

![Markdown 编辑器](https://raw.githubusercontent.com/agentscope-ai/ReMe/main/reme_studio/figures/markdown-editor.png)

Studio 会在加载文档时记录文件修改时间，并在保存时把该值发送给后端。如果其他进程已经修改文件，ReMe 服务会拒绝过期写入，而不是静默覆盖更新后的内容。保存成功后，Studio 会刷新文件状态并清除未保存标记。

### 5. 知识与记忆图谱

Knowledge 将长期记忆分为三个约定目录：

- `wiki`：实体、项目、主题和稳定知识。
- `personal`：用户偏好和长期个人上下文。
- `procedure`：可重复执行的流程和操作知识。

每个分类都有 **Graph** 操作。图谱由 ReMe 已索引的 wikilink 生成，并保留原始的源到目标方向。

![记忆图谱](https://raw.githubusercontent.com/agentscope-ai/ReMe/main/reme_studio/figures/memory-graph.png)

图谱视图提供：

- 已索引文件节点、分类根节点和有向 wikilink 边。
- 当前分类的节点数与链接数。
- 放大、缩小和适应画布控制。
- 文件节点和链接方向图例。
- 选择文件节点后直接打开相应 Markdown 源文件。

图谱属于派生状态。如果图谱为空，请先让 ReMe 摄取相关文件；Settings 中的搜索索引重建不会重新扫描文件，也不会重新构建 wikilink 图谱。

### 6. Agent 对话

Chat 会在同一个标签工作区中打开，因此对话进行时仍可保留文件和图谱。

![Agent 对话](https://raw.githubusercontent.com/agentscope-ai/ReMe/main/reme_studio/figures/agent-chat.png)

对话界面支持：

- ReMe Agent 逐 token 流式输出。
- 分别展示回答文本、推理、工具调用、结构化数据、审批和用量信息。
- 自动折叠已经完成的推理和工具块，让长对话更易阅读。
- 多个独立对话标签页和可恢复的后端 session id。
- 常见记忆问题的起始提示。
- 将工作区文件拖入输入框以插入路径引用。
- 按 `Enter` 发送，按 `Shift+Enter` 换行。

内置 chat job 对工作区只读：它可以搜索和读取上下文，但不应替代明确保存长期知识的操作。Agent 对话需要有效的 Agent/模型配置；文件浏览、编辑、图谱和服务检查不需要模型。

### 7. 服务状态

从顶部栏打开 **Settings**，可以检查 Studio 实际连接的后端。

![服务状态](https://raw.githubusercontent.com/agentscope-ai/ReMe/main/reme_studio/figures/settings-status.png)

Status 会显示：

- ReMe 服务是否可达及当前使用的服务地址。
- 当前 ReMe 后端版本。
- 进程常驻内存及有状态组件估算的总内存。
- 健康组件数量。
- 已配置的文件图谱、文件存储、关键词索引和 embedding 存储的运行信息，包括节点、边、chunk、文档、词表、维度、缓存大小和组件内存等可用字段。

在文件完成索引或服务配置变化后，可以用 **Refresh** 获取新快照。

### 8. 索引管理

Index 页面为派生搜索状态提供明确的维护入口。

![索引管理](https://raw.githubusercontent.com/agentscope-ai/ReMe/main/reme_studio/figures/settings-index.png)

**Rebuild index** 会基于 ReMe 已经摄取的 chunk 重建 BM25、embedding 和 tag 索引。Studio 会在执行前要求确认。该操作不会扫描工作区、重新分块、修改源记忆，也不会重建 wikilink 图谱。这些边界保证 ReMe 的文件仍是唯一事实来源。

### 9. 生效配置

Configuration 显示当前后端返回的完整解析后应用配置。

![生效配置](https://raw.githubusercontent.com/agentscope-ai/ReMe/main/reme_studio/figures/settings-configuration.png)

在文件配置与命令行覆盖完成合并后，可以在这里确认实际工作区、目录布局、服务选项、jobs 和组件后端。ReMe 会在配置到达浏览器之前对敏感字段进行脱敏。

### 10. 版本与服务地址

Version 会显示 ReMe 后端版本和服务地址，并与左上角品牌区域显示的 Studio 版本明确区分。

![版本与服务地址](https://raw.githubusercontent.com/agentscope-ai/ReMe/main/reme_studio/figures/settings-version.png)

这有助于定位前后端版本不一致的问题：顶部栏标识当前安装的 Studio 构建，Settings 标识当前连接的 ReMe 服务。

### 11. 语言与外观

顶部栏可以把整个界面切换为英文或中文。外观支持 **Light**、**Dark** 和 **System**；System 会跟随操作系统的颜色偏好。

![ReMe Studio 深色外观](https://raw.githubusercontent.com/agentscope-ai/ReMe/main/reme_studio/figures/appearance-dark.png)

语言和外观选择会保存在浏览器中。两套主题都复用 ReMe 文档站的品牌色，并为健康状态、选择状态、焦点和图谱保留足够对比度。

## 前端配置

如需持久保存本地覆盖，可以将 `.env.example` 复制为 `.env.local`。vinext/Sites 构建读取 `NEXT_PUBLIC_*`，FastAPI/static 构建读取对应的 `VITE_*` 变量。

| 配置项                                  | 构建类型     | 作用                                             |
| --------------------------------------- | ------------ | ------------------------------------------------ |
| `NEXT_PUBLIC_REME_API_URL`              | vinext/Sites | ReMe HTTP 服务地址，默认 `http://127.0.0.1:2333` |
| `NEXT_PUBLIC_REME_WORKSPACE_EXTENSIONS` | vinext/Sites | 工作区中显示的扩展名，使用逗号分隔               |
| `VITE_REME_API_URL`                     | static       | ReMe HTTP 服务地址；FastAPI 同源托管时使用 `/`   |
| `VITE_REME_WORKSPACE_EXTENSIONS`        | static       | static 构建使用的工作区扩展名列表                |

默认显示 Markdown 和文本文件。如需增加其他文本格式：

```bash
NEXT_PUBLIC_REME_WORKSPACE_EXTENSIONS=md,txt,mdx
VITE_REME_WORKSPACE_EXTENSIONS=md,txt,mdx
```

只应启用浏览器能够安全视为文本的格式。在 API 边界，ReMe 仍会执行工作区范围限制、允许路径、编码检查、大小限制和乐观修改时间检查。

## 从源码开发

先在仓库根目录启动 ReMe，然后在另一个终端运行前端：

```bash
# 终端 1：仓库根目录
reme start

# 终端 2
cd reme_studio
npm install
npm run dev
```

打开 <http://localhost:3000>。开发前端默认连接 `http://127.0.0.1:2333`，需要时可覆盖：

```bash
NEXT_PUBLIC_REME_API_URL=http://127.0.0.1:8000 npm run dev
```

需要从受控网络中的另一台机器进行开发时，应显式开放两个服务，并让浏览器连接 ReMe 主机的可达地址（请按
实际情况替换 `192.168.1.10`）：

```bash
# 在运行 ReMe 的主机上，从仓库根目录执行
reme start service.host=0.0.0.0

# 在同一主机的 reme_studio/ 目录执行
NEXT_PUBLIC_REME_API_URL=http://192.168.1.10:2333 npm run dev:remote
```

ReMe 服务当前不提供通用身份认证。请勿在不可信网络中使用远程开发命令，也不要把任一端点直接暴露到公网。

## 构建由 ReMe 托管的静态前端

ReMe 可以使用提供 HTTP API 的同一个 FastAPI 进程托管 Studio。构建静态版本并重启 ReMe：

```bash
cd reme_studio
npm ci
npm run build:static
cd ..
reme start
```

打开 <http://127.0.0.1:2333>。静态构建默认使用同源请求。进行独立静态开发时：

```bash
VITE_REME_API_URL=http://127.0.0.1:2333 npm run dev:static
```

需要显式进行远程静态开发时，请使用 `dev:static:remote`，并将 `VITE_REME_API_URL` 设置为可达地址。

`npm run build` 仍用于 vinext/Sites 部署构建；`npm run build:static` 仅为 FastAPI 以及 Python/npm 包分发生成 `dist-static/`。应修改前端源文件，而不是提交生成的分发文件。

## 常见问题

### Studio 提示服务不可用

- 确认 `reme start` 仍在运行。
- 检查导航器底部显示的服务地址。
- 前后端分开托管时，请在启动或构建前设置正确的 `NEXT_PUBLIC_REME_API_URL` 或 `VITE_REME_API_URL`。
- 修正地址后，打开 Settings → Status 并点击 Refresh。

### 导航器中缺少文件

- 确认 ReMe 使用预期的 `workspace_dir` 启动。
- 点文件和点目录会被刻意隐藏。
- 如果不是 Markdown 或纯文本，请检查 `*_REME_WORKSPACE_EXTENSIONS`。
- 使用 Files，而不是仅显示部分目录的 Daily 或 Knowledge。

### 图谱为空或不完整

- 确认笔记位于 `digest/wiki`、`digest/personal` 或 `digest/procedure`。
- 确认源文件包含 wikilink，并已被 ReMe 摄取。
- Settings → Rebuild index 只从现有 chunk 重建搜索索引，不会扫描文件或重建图谱。

### Markdown 保存被拒绝

文件在 Studio 加载后被其他进程修改。请保留磁盘上的新版本，重新打开或刷新文件，再应用目标修改并保存。这是防止静默数据丢失的预期冲突保护。

### Chat 无法启动

文件功能可在 Chat 不可用时正常工作。请检查 ReMe 后端配置的 Agent wrapper、模型、API key 和提供商地址，再查看 Settings → Status。不要把凭证放进前端环境变量。

## 验证

在 `reme_studio/` 中运行与改动范围相称的检查：

```bash
npm run format:check
npm run lint
npm run build
npm run build:static
npm test
```

以上截图也构成受支持界面的可视化导览，但前端行为仍以自动化检查为准。
