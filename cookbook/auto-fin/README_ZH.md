# Auto Fin Cookbook

[English](README.md)

Auto Fin 自动拉取最近 24 小时的财联社电报，按配置 topics 筛选相关新闻，搜索 ReMe 中有回顾价值的历史材料，
最后写入一份带校验 wikilink 的中文 Markdown 报告。当前新闻和筛选结果只存在于本次运行内存中，只有最终报告成为
持久记忆。

> Auto Fin 没有可靠行情数据，不计算收益、目标价或买卖点，也不提供投资建议。

## 快速开始

```bash
python -m pip install -e ".[core]"
export LLM_API_KEY="your-api-key"
reme start config=daily_cookbook job=auto_fin
```

默认 topics 是 `黄金,机器人,半导体`。可在运行时覆盖：

```bash
reme start config=daily_cookbook job=auto_fin topics="黄金,AI,存储芯片"
```

传入空值也会使用默认 topics。

## 流程

```text
财联社公开电报接口（滚动24小时）
        ↓
在 RuntimeContext 中规范化和去重
        ↓
Topic Agent 分批选择真实 news_id
        ↓
Research Agent 使用 memory_search + read 检索历史记忆
        ↓
代码校验历史 wikilink
        ↓
daily/YYYY-MM-DD/auto_fin.md
```

`auto_fin_data_step` 使用财联社网页同源接口的签名和分页方式，从分析时刻开始向前翻页，直到完整覆盖严格的最近
24 小时。请求带有限速和重试；损坏记录及窗口外记录会被丢弃。

`auto_fin_topic_step` 分批接收当前新闻，只返回相关的 `news_id`。代码拒绝未知 ID 并自动去重。如果没有相关
新闻，Job 会成功跳过，不写报告也不发送通知。

`auto_fin_merge_step` 只接收筛选后的当前新闻，并向 Agent 开放 `memory_search` 和 `read`。历史检索截止到昨天；
当前新闻以 CLS ID、时间和标题作为普通证据，只有历史 workspace Markdown 才能成为 wikilink。代码拒绝不存在、
绝对路径、越界、反斜杠和自引用目标，无效链接会降级为可读 alias。

同日重跑会参考当天已有报告并覆盖为修订结果。最终写入使用原子替换并刷新当天索引；流程不会写入 JSONL、
中间 Markdown 或 Agent 结构化输出。

## 参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `date` | `""` | 空值使用上海时区当天；显式日期必须等于当天 |
| `now` | `""` | 测试或回放使用的 ISO 8601 分析时间 |
| `topics` | `"黄金,机器人,半导体"` | 逗号分隔的主题；空值也使用这些默认值 |
| `window_hours` | `24` | 向前抓取财联社电报的滚动小时数，必须大于 0 |
| `request_interval` | `10` | 每次财联社请求尝试后的最小等待秒数，可设为 0 |
| `max_retries` | `3` | 每页财联社请求的最大尝试次数，至少为 1 |

内置定时任务每天按 `Asia/Shanghai` 在 09:30、11:30 和 18:00 运行。

## 产物

```text
reme_workspace/daily/YYYY-MM-DD/auto_fin.md
```

报告包含标题、说明、当前 CLS 证据、历史分析、上下文 wikilink 和固定非投资建议声明。网络错误与无效 Agent 输出
会明确失败；没有相关当前新闻则成功跳过。

## 验证

```bash
pytest tests/unit/test_auto_fin.py -v
```

单元测试 mock CLS 与 Agent 边界，不访问外部服务。
