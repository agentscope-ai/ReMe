# Auto Fin Cookbook

[English](README.md)

Auto Fin 是一个 local-first、file-native 的 ETF 事件研究工作流。它通过 Tushare 获取财联社新闻和行情数据，从固定
ETF 列表中识别与当日新闻相关的标的，利用 ReMe 本地记忆检索可比历史事件，计算事件后的实际收益，并生成中文研究报告。

> Auto Fin 只提供事件研究和持有时间参考，不构成投资建议，不连接券商，也不会执行或模拟交易。

工作流由 [`daily_cookbook.yaml`](../../reme/config/daily_cookbook.yaml) 装配；公开 schema 位于
[`reme/schema/auto_fin.py`](../../reme/schema/auto_fin.py)，四个 Step 位于
[`reme/steps/cookbook/auto_fin/`](../../reme/steps/cookbook/auto_fin/)。

## 快速开始

要求 Python 3.11 或更高版本、`core` 依赖、Tushare token 和可用的 AgentScope LLM。

```bash
python -m pip install -e ".[core]"
export TUSHARE_TOKEN="your-tushare-token"
export LLM_API_KEY="your-api-key"
reme start config=daily_cookbook job=auto_fin
```

内置 LLM 组件默认使用 `qwen3.7-plus`。`LLM_BASE_URL` 没有内置默认值；如果服务商要求自定义 OpenAI 兼容
endpoint，需要显式设置。可通过 `LLM_MODEL_NAME` 和 `LLM_BASE_URL` 覆盖模型与 endpoint。

默认 workspace 是进程启动目录下的 `reme_workspace/`。可通过 `DAILY_PAPER_WORKSPACE_DIR` 覆盖；Auto Fin 与
Daily Paper 共用该设置。

日期和时间使用 `Asia/Shanghai`。显式传入的 `date` 必须是当天：

```bash
reme start config=daily_cookbook job=auto_fin date=2026-08-07
```

Auto Fin 首先检查上交所交易日历；休市日会跳过整个工作流。

## 工作流

```text
Tushare 交易日历
        │
        ├─ 休市 ──► 跳过
        ▼
采集财联社新闻 + 固定 ETF 行情历史
        ▼
更新 ReMe 索引
        ▼
Agent 筛选 ETF/当日新闻关系
        ▼
从本地记忆检索可比历史新闻
        ▼
Agent 选择 same/opposite 事件 + 代码计算 D1/D2/D3/D5 收益
        ▼
Agent 生成报告 ──► 刷新当日索引 ──► 钉钉（可选）
```

| Step | 职责 | Agent |
|---|---|---|
| `auto_fin_data_step` | 检查交易日、维护新闻并缓存固定 ETF 的完整行情历史 | 否 |
| `auto_fin_topic_step` | 筛选当日新闻与固定 ETF 的直接关系 | 是 |
| `auto_fin_history_step` | 检索可比新闻、校验选择并计算实际收益 | 是 |
| `auto_fin_merge_step` | 汇总证据和上一份报告，生成最终 Markdown | 是 |

三个模型 Step 都使用 Pydantic 结构化输出。Agent 负责语义判断；标识校验、来源解析、行情计算和文件写入由代码负责。

## 数据与筛选边界

### 新闻

`auto_fin_data_step` 调用 Tushare `major_news`，并固定传入 `src="财联社"`。默认回看 60 个自然日（包含当天）。
更早日期已有的文件会复用；当天文件始终覆盖为 00:00 至当前决策时刻的新闻。单次请求返回至少 400 条时，时间区间会递归拆分。

每条新闻写入 `daily/YYYY-MM-DD/auto_fin_news.md`，其稳定 ID 由发布时间和短内容哈希组成。Topic Step 使用当天
完整文件，不是从上一次运行到本次运行之间的增量。

### 固定 ETF

内置配置当前启用：

- `518880.SH`
- `159530.SZ`
- `512760.SH`

`daily_cookbook.yaml` 中还保留了其他被注释的示例。Data Step 通过 `etf_basic` 解析每个启用代码的名称，然后对
`fund_daily` 和 `fund_adj` 向前分页，并覆盖写入完整本地 JSONL 行情历史。任一 ETF 无法解析名称都会终止运行。

Topic Agent 只接收固定 ETF 的 code/name 和当天本地新闻。每只 ETF 默认最多保留
`current_news_limit_per_etf=10` 条有效且唯一的新闻引用。未知 ETF、未知 news ID、空理由和重复项会被代码移除；
没有有效事件的 ETF 不进入后续步骤。

## 历史比较与收益

对每个有效的 ETF/当日新闻组合，`auto_fin_history_step` 会在 60 日新闻窗口内调用配置中的 `memory_search`，
结束日期为昨天。`historical_search_limit` 控制每个当前事件最多请求多少条检索结果。只有路径名为
`auto_fin_news.md` 的命中才会贡献候选 ID；Step 会重新读取源 Markdown 并解析 ID，再调用 History Agent。

History Agent 默认最多选择五条候选，并将关系标记为 `same` 或 `opposite`。代码会移除未知或重复 ID 以及空理由，
随后计算 D1、D2、D3、D5 的复权累计收益：

- 交易日 15:00 前发生的事件，以当日复权收盘价为入场价，D1 是下一交易日收盘价；
- 15:00 或之后发生的事件，以下一交易日复权开盘价为入场价，D1 是该日收盘价；
- 如果无法从有效正价格和复权因子计算入场价或某个期限，该值为 `null`。

最终 Agent 接收固定 ETF 列表、所有当前/历史证据、`same`/`opposite` 方向、代码计算的收益，以及此前最近一份
`auto_fin.md`。它自行判断证据是否支持推荐或应明确观望；代码不会计算评分、期望收益，也不强制给出持有期限。

## 产物

```text
reme_workspace/
├── daily/
│   ├── YYYY-MM-DD.md
│   └── YYYY-MM-DD/
│       ├── auto_fin_news.md
│       └── auto_fin.md
└── resource/
    ├── fin/
    │   ├── etfs.json
    │   ├── 518880.SH.jsonl
    │   └── <其他固定 ETF>.jsonl
    └── YYYY-MM-DD/
        ├── auto_fin_topic_output.json
        ├── auto_fin_history_001_output.json
        ├── ...
        ├── auto_fin_analysis.jsonl
        └── auto_fin_merge_output.json
```

每日新闻和报告是用户拥有的 Markdown。`resource/fin/` 是确定性收益计算所用的行情缓存；日期目录下的 JSON/JSONL
保留结构化 Agent 回复和整理后的分析。写入通过同目录临时文件原子替换；报告写完后会刷新当日索引。

## 参数与默认值

公开 Job 参数：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `date` | `""` | 空值使用 `Asia/Shanghai` 当天；非空值必须是严格 `YYYY-MM-DD` 且等于当天 |
| `historical_search_limit` | `10` | 每个当前事件请求的 `memory_search` 结果上限；最小值为 1 |

`daily_cookbook.yaml` 中相关的 Job 级配置：

| 配置 | 默认值 | 作用 |
|---|---:|---|
| `etf_codes` | 上述三个启用代码 | 固定 ETF 研究范围 |
| `news_lookback_days` | `60` | 本地新闻及历史检索窗口 |
| `current_news_limit_per_etf` | `10` | 每只 ETF 最多保留的当前事件数 |
| `historical_news_limit` | `5` | 每个当前事件最多保留的可比历史事件数 |

当前没有公开 `force` 参数。更早的新闻文件会复用；当天新闻和所有固定 ETF 行情文件会刷新；同一天再次成功运行会覆盖
当天报告和 resource 产物。

## 环境变量与定时任务

| 变量 | 必需 | 作用 |
|---|---|---|
| `TUSHARE_TOKEN` | 是 | 交易日历、财联社新闻、ETF 元数据、价格与复权因子 |
| `LLM_API_KEY` | 取决于服务商 | 共享 AgentScope LLM 凭据；配置默认值为空 |
| `LLM_MODEL_NAME` | 否 | 默认 `qwen3.7-plus` |
| `LLM_BASE_URL` | 取决于服务商 | OpenAI 兼容 endpoint；无内置默认值 |
| `TUSHARE_MIRROR_URL` | 否 | 去掉末尾 `/` 后替换 Tushare SDK HTTP URL |
| `DAILY_PAPER_WORKSPACE_DIR` | 否 | standalone cookbook 的共享 workspace |
| `DINGTALK_*` | 否 | 可选的钉钉应用、机器人和群设置 |

镜像可按需配置，例如：

```bash
export TUSHARE_MIRROR_URL="http://112.124.63.173:4000/tushare"
```

`auto_fin_0930_cron`、`auto_fin_1130_cron` 和 `auto_fin_1800_cron` 按 `Asia/Shanghai` 时区每天 09:30、11:30 和
18:00 触发。Cron 在周末和节假日仍会启动，但如果 Tushare 返回当天不是上交所交易日，Data Step 会跳过后续工作流；
同一天的后续运行会在已有报告基础上继续完善。

要发送完成的报告，需要配置 `DINGTALK_APP_KEY`、`DINGTALK_APP_SECRET`、`DINGTALK_ROBOT_CODE` 和逗号分隔的
`DINGTALK_CONVERSATION_IDS`。没有会话 ID 时发送步骤无副作用。

## Agent 与失败边界

Auto Fin 和 Daily Paper 共用无工具的 `default` AgentScope wrapper，其模型调用不会暴露内置工具或配置型 Job
工具。Auto Fin 由确定性的 Step 代码主动调用 `memory_search`，这不是 Agent 工具调用。独立的交互式
`dingtalk_wait` Step 才有自己的 `bash` 和 ReMe Job tool allowlist。

standalone 配置默认未启用 embedding store，因此 `memory_search` 使用可用的 BM25 路径；只有启用被注释的
embedding 组件后才有向量/BM25 融合。

非法日期、缺少凭据或服务、模型结构化输出无效、固定 ETF 未知、行情文件缺失、记忆检索失败都会终止 Job；休市日是
成功跳过。工作流没有同日期全局执行锁或跨文件事务；重复成功运行也可能重复发送钉钉通知。

## 测试

聚焦单元测试会 mock 模型和行情数据边界：

```bash
python -m pip install -e ".[dev,core]"
pytest tests/unit/test_auto_fin.py -v
```

需要真实 Tushare、LLM 或钉钉凭据的测试应单独运行，且需要显式授权。
