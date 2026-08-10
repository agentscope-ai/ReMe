# Auto Fin Cookbook

[English](README.md)

Auto Fin 是一个 local-first、file-native 的主题新闻研究流程。它读取免费的财联社电报 JSONL，按用户提供的
topics 解读当天新闻，并允许最终 Agent 使用 ReMe 搜索和阅读以前的新闻、分析文章。最终报告以经过代码校验的
workspace-relative wikilink 指回当前材料和历史材料。

> Auto Fin 没有可靠行情数据，不计算收益、目标价或买卖点，也不提供投资建议。

## 快速开始

先准备本地财联社电报 JSONL。每行是一个 JSON 对象，包含 `id`、`ctime`（Unix 时间戳）、`title` 和
`content`：

```json
{"id": 2448247, "ctime": 1786323600, "title": "示例标题", "content": "新闻正文"}
```

请将新闻源保留在版本控制之外。Auto Fin 只读取该文件，不会主动从网络抓取新闻。

然后启动主题研究：

```bash
python -m pip install -e ".[core]"
export LLM_API_KEY="your-api-key"
reme start config=daily_cookbook job=auto_fin topics="黄金,机器人,半导体"
```

默认输入是 `datasets/cls_news_last_7_days.jsonl`。也可使用参数或环境变量指定其他文件：

```bash
export AUTO_FIN_NEWS_FILE=/data/cls_news.jsonl
```

## 流程

```text
免费 CLS JSONL
      ↓
daily/YYYY-MM-DD/auto_fin_news.md
      ↓
更新 ReMe 索引
      ↓
研究 Agent（memory_search + read）
      ↓
在正文中构建 contextual wikilink，代码校验路径
      ↓
daily/YYYY-MM-DD/auto_fin.md
```

`auto_fin_data_step` 只处理本地文件，不请求付费 API。它按 `Asia/Shanghai` 时间把 JSONL 分成每日 Markdown，
当天文件只包含 00:00 到分析时刻的新闻。默认回看七个自然日；已有的往日文件会复用，当天文件会刷新。

`auto_fin_merge_step` 是唯一使用模型的业务 Step。它收到当天新闻、topics、上一份往日报告和同日已有报告，并且只
开放两个 ReMe Job tool：

- `memory_search`：召回以前的新闻和 `auto_fin.md`；
- `read`：读取真正可能相似的完整 Markdown。

Agent 需要围绕不同 topic 和重要事件主动搜索，将候选分为 similar、related、unrelated，只引用实际读取且支持正文
判断的文件。同日重跑会在当天已有报告基础上修订；往日文章是历史记录，不会被修改。

## 结构化输出

最终 Agent 只返回一个 Markdown 型契约，三个字段均为必填：

```json
{
  "title": "黄金与半导体主题新闻观察",
  "description": "关注避险需求和存储芯片供需变化。",
  "body": "## 今日判断\n\n……"
}
```

Agent 不重复输出 `sources` 或引用理由列表。代码负责清理 title 中多余的 Markdown 标题符号、写入最终文件、补充固定
免责声明并刷新索引。结构化结果经 wikilink 校验后保存到 `resource/YYYY-MM-DD/auto_fin_merge_output.json`。

## Wikilink 契约

最终 Agent 的结构化输出只有 `title`、`description` 和 `body`。历史 wikilink 直接写在 `body` 的相关句子中，
不再通过 `sources` 重复输出。代码仅保留满足以下条件的链接：

- workspace 内实际存在的 Markdown；
- 完整 workspace-relative path，且以 `.md` 结尾；
- 不是当天正在生成的报告；
- 不使用绝对路径、反斜杠或越界路径。

链接必须像 Dream Integrate 一样出现在解释关系的自然句子中，例如：

```markdown
本次避险需求与 [[daily/2026-08-01/auto_fin.md|8 月 1 日的黄金观察]] 背景相似，
但当前美元和利率信号并不一致。
```

不存在、越界或指向当前报告自身的链接会降级为 alias 普通文本，不会删除周围分析。若正文没有引用当天新闻文件，
代码会固定补充 `## 来源` 链接。报告写入后会刷新当天索引，因此后续 `memory_search` 可以召回报告并沿 wikilink
扩展到证据材料。

## 参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `date` | `""` | 空值使用上海时区当天；显式日期必须等于当天 |
| `now` | `""` | 测试或回放使用的 ISO 8601 当前时间 |
| `topics` | `""` | 逗号分隔的关注主题；空值表示不限制主题 |
| `news_file` | `""` | 覆盖本地 CLS JSONL 路径；空值使用 Step 配置或环境变量 |

`news_file` 的运行参数优先于配置；内置配置读取 `AUTO_FIN_NEWS_FILE`，未设置时使用
`datasets/cls_news_last_7_days.jsonl`。`topics` 只限制研究范围：仅出现关键词但没有真实主题关系的新闻应被 Agent 忽略。

内置定时任务仍在每天 09:30、11:30 和 18:00 运行。新闻研究不再依赖交易日历，因此周末或节假日也可以生成报告。

## 产物

```text
reme_workspace/
├── daily/
│   ├── YYYY-MM-DD.md
│   └── YYYY-MM-DD/
│       ├── auto_fin_news.md
│       └── auto_fin.md
└── resource/
    └── YYYY-MM-DD/
        └── auto_fin_merge_output.json
```

JSONL 和每日 Markdown 是事实材料；`auto_fin.md` 是可回顾的研究判断；索引和图谱都可由这些文件重建。

缺失或损坏的 JSONL、Agent 调用失败、无效的模型结构化输出会明确终止 Job。不存在或不安全的历史 wikilink 不会
终止报告生成，而会降级为可读的普通文本。当前新闻来源链接由代码保证存在。

## 验证

```bash
pytest tests/unit/test_auto_fin.py -v
```

单元测试使用本地 JSONL 和 mock Agent，不访问新闻、模型或其他外部服务。
