# Auto Fin 插件

[English](README.md)

Auto Fin 自动拉取一个滚动时间窗口内的财联社电报（默认 24 小时），按配置 topics 归类相关新闻，逐主题搜索 ReMe 中有回顾价值的历史材料并为每个主题写一份笔记，
最后把所有主题笔记合并成一份带校验 wikilink、且回链到各主题笔记的当日总览。当前新闻和筛选结果只存在于本次运行内存中，只有主题笔记和总览成为持久记忆。本目录是一个独立 Python
distribution：单个 `reme.plugins` entry point 暴露 `plugin.yaml`，其中声明四个 Step backend，并在
`application_defaults` 下提供 Job 配置；通过 `plugins=["auto-fin"]` 显式启用这个已安装插件。

> Auto Fin 没有可靠行情数据，不计算收益、目标价或买卖点，也不提供投资建议。

## 快速开始

### 1. 安装 ReMe 和 Auto Fin

```bash
python -m pip install "reme-ai[core]>=0.4.1.12"
reme plugins install plugins/auto-fin
```

### 2. 配置模型环境变量

按照 ReMe README 的[可选模型配置说明](../../README_ZH.md#可选模型配置)配置 LLM 环境变量，也可以使用其他兼容的模型和服务商。

### 3. 带插件启动 ReMe

```bash
reme start plugins='["auto-fin"]'
```

未显式传入 `config` 时，ReMe 会加载 `default.yaml`，并将插件叠加到该服务上。

在另一个终端中，通过 ReMe CLI client 调用正在运行的 HTTP 服务：

```bash
reme auto_fin topics="黄金,AI,存储芯片"
```

也可以直接调用 HTTP endpoint：

```bash
curl -s http://127.0.0.1:2333/auto_fin \
  -H 'Content-Type: application/json' \
  -d '{"topics":"黄金,AI,存储芯片"}'
```

HTTP service 也会在 `/mcp` 中将同一个 Job 暴露为 `auto_fin` MCP tool。默认 topics 是 `黄金,机器人,半导体`，
传入空值也会使用默认值。

如果需要同时通过 JSON 和 MCP 访问同一个应用：

```bash
reme start plugins='["auto-fin"]' \
  service.backend=http
```

自定义应用配置需要提供 `agent_wrapper.default`、启用 tag index 的 `file_store.default`，以及
Auto Fin 和自动标签使用的 `search`、`list_tags`、`frontmatter_read` 和
`frontmatter_update` Jobs。

## 流程

```text
财联社公开电报接口（滚动24小时）
        ↓
在 RuntimeContext 中规范化和去重
        ↓
Topic Agent 按提示词长度分批输出“主题 → news_id”
        ↓
每个有新闻的主题由独立 Research Agent 研究最新 20 篇，最多搜索 3 次，
并写出一份带校验历史 wikilink 的主题笔记
        ↓
Digest Agent 把各主题笔记合并成当日总览，代码在末尾追加回链到每一份笔记
        ↓
生成记忆标签，由后台文件 watcher 刷新索引
```

`auto_fin_data_step` 使用财联社网页同源接口的签名和分页方式，从分析时刻开始向前翻页，直到完整覆盖严格的最近 24
小时。请求带有限速和重试；损坏记录及窗口外记录会被丢弃。

`auto_fin_topic_step` 按完整提示词的 10 万字符上限分批接收当前新闻，返回每个主题的相关 `news_id`。代码会忽略未知 ID、去除重复 ID，并保持源新闻顺序；一条新闻可属于多个主题。如果没有相关新闻，Job
会成功跳过，不写任何文件。

`auto_fin_research_step` 为每个有新闻的主题研究最新 20 篇，并各写一份主题笔记；只向 Agent 开放 `search`，按主题限制最多 3 次搜索。当前新闻以 CLS ID、时间和标题作为普通证据。
Prompt 要求 Agent 只链接实际使用过的历史 Markdown；代码边界则独立保证只保留真实存在、相对
workspace 的 Markdown 目标。不存在、绝对路径、越界、带反斜杠和自引用的目标都会降级为可读 alias。

`auto_fin_digest_step` 把各主题笔记合并成当日总览，自身不开放任何工具——研究结论已经在笔记里。代码在正文末尾追加 `## 主题详解`
章节，回链到每一份主题笔记，然后刷新当天索引并把总览路径交给下游步骤。两步在工作流已被判定跳过时都会提前返回。

同日重跑会按 frontmatter 中的 `topic` 找回该主题已有的笔记，把它的正文作为上下文重新研究并原地覆盖；总览同理。所有写入都是原子的，随后通过 `auto_tag_step`
更新这些文件的记忆标签 frontmatter；常规后台文件 watcher 会观察源文件变化并刷新派生索引。流程不会写入 JSONL、
中间 Markdown 或 Agent 结构化输出。

## 参数

| 参数               |                 默认值 | 作用                                         |
|--------------------|-----------------------:|----------------------------------------------|
| `date`             |                   `""` | 空值使用上海时区当天；显式日期必须等于当天   |
| `now`              |                   `""` | 测试或回放使用的 ISO 8601 分析时间           |
| `topics`           | `"黄金,机器人,半导体"` | 逗号分隔的主题；空值也使用这些默认值         |
| `window_hours`     |                   `24` | 向前抓取财联社电报的滚动小时数，必须大于 0   |
| `request_interval` |                   `10` | 每次财联社请求尝试后的最小等待秒数，可设为 0 |
| `max_retries`      |                    `3` | 每页财联社请求的最大尝试次数，至少为 1       |

插件的 cron Job 随应用启动，并按应用配置的时区在每天 09:00 运行，默认时区为 `Asia/Shanghai`。滚动窗口按时间戳计算，允许跨自然日；09:00 是启动时间，报告完成时间取决于新闻量与模型耗时。

## 产物

```text
.reme/daily/YYYY-MM-DD/<主题>.md                    # 每个有相关新闻的主题一份
.reme/daily/YYYY-MM-DD/主题新闻观察（YYYY-MM-DD）.md  # 合并总览，回链到每一份主题笔记
```

文件名取自配置的主题（`topics`）和运行日期，而不是 Agent 生成的标题——Agent 标题是自由文本，可能长到超出文件名长度上限。文件名经过
非法字符净化、字节数截断和重名消歧；Agent 标题保留在 frontmatter 的 `title` 中。每份文件都带 `kind` frontmatter（`auto-fin-topic`
或 `auto-fin-digest`），因此同日重跑会找回并覆盖自己产出的笔记，而不是重复生成。文件包含标题、说明、当前 CLS 证据、历史分析、上下文
wikilink 和固定非投资建议声明；总览末尾另有一个 `## 主题详解` 列表，链接到各主题笔记。抓取失败与全部主题研究失败
会明确失败；单个主题研究失败只记 warning 并继续其余主题；没有相关当前新闻则成功跳过。

## 验证

```bash
python -m pytest plugins/auto-fin -v
```

单元测试 mock CLS 与 Agent 边界，不访问外部服务。
