# Auto Fin 插件

[English](README.md)

Auto Fin 自动拉取一个滚动时间窗口内的财联社电报（默认 24 小时），按配置 topics 筛选相关新闻，搜索 ReMe 中有回顾价值的历史材料，最后写入一份带校验
wikilink 的中文 Markdown 报告。当前新闻和筛选结果只存在于本次运行内存中，只有最终报告成为持久记忆。本目录是一个独立 Python
distribution：单个 `reme.plugins` entry point 暴露 `plugin.yaml`，其中声明三个 Step backend，并在
`application_defaults` 下提供 Job 配置；通过 `plugins=["auto-fin"]` 显式启用这个已安装插件。

> Auto Fin 没有可靠行情数据，不计算收益、目标价或买卖点，也不提供投资建议。

## 快速开始

### 1. 安装 ReMe 和 Auto Fin

```bash
python -m pip install "reme-ai[core]>=0.4.1.9"
reme plugins install reme-auto-fin
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

自定义应用配置需要提供 `agent_wrapper.default`，以及 Auto Fin 使用的 `search` 和 `read` Jobs。

## 流程

```text
财联社公开电报接口（滚动24小时）
        ↓
在 RuntimeContext 中规范化和去重
        ↓
Topic Agent 分批选择真实 news_id
        ↓
Research Agent 使用 search + read 检索历史记忆
        ↓
代码校验历史 wikilink
        ↓
daily/YYYY-MM-DD/auto_fin.md
```

`auto_fin_data_step` 使用财联社网页同源接口的签名和分页方式，从分析时刻开始向前翻页，直到完整覆盖严格的最近 24
小时。请求带有限速和重试；损坏记录及窗口外记录会被丢弃。

`auto_fin_topic_step` 分批接收当前新闻，只返回相关的 `news_id`。代码会忽略未知 ID、去除重复 ID，并保持源新闻顺序。如果没有相关新闻，Job
会成功跳过，不写报告也不发送通知。

`auto_fin_merge_step` 只接收筛选后的当前新闻，并向 Agent 开放 `search` 和 `read`。当前新闻以 CLS ID、时间和标题作为普通证据。
Prompt 要求 Agent 只链接实际使用过的历史 Markdown；代码边界则独立保证只保留真实存在、相对
workspace 的 Markdown 目标。不存在、绝对路径、越界、带反斜杠和自引用的目标都会降级为可读 alias。

同日重跑会参考当天已有报告并覆盖为修订结果。最终写入使用原子替换并刷新当天索引；流程不会写入 JSONL、中间 Markdown 或 Agent
结构化输出。

## 参数

| 参数               |                 默认值 | 作用                                         |
|--------------------|-----------------------:|----------------------------------------------|
| `date`             |                   `""` | 空值使用上海时区当天；显式日期必须等于当天   |
| `now`              |                   `""` | 测试或回放使用的 ISO 8601 分析时间           |
| `topics`           | `"黄金,机器人,半导体"` | 逗号分隔的主题；空值也使用这些默认值         |
| `window_hours`     |                   `24` | 向前抓取财联社电报的滚动小时数，必须大于 0   |
| `request_interval` |                   `10` | 每次财联社请求尝试后的最小等待秒数，可设为 0 |
| `max_retries`      |                    `3` | 每页财联社请求的最大尝试次数，至少为 1       |

插件的 cron Job 随应用启动，并按应用配置的时区在每天 18:00 运行。

## 产物

```text
.reme/daily/YYYY-MM-DD/auto_fin.md
```

报告包含标题、说明、当前 CLS 证据、历史分析、上下文 wikilink 和固定非投资建议声明。网络错误与无效 Agent 输出
会明确失败；没有相关当前新闻则成功跳过。

## 验证

```bash
python -m pytest plugins/auto-fin -v
```

单元测试 mock CLS 与 Agent 边界，不访问外部服务。
