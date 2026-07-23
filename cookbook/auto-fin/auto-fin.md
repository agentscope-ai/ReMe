# Auto Finance 模拟组合设计

## 1. 文档状态

本文档定义 ReMe Auto Finance（以下简称 Auto Fin）的产品边界、交易时间口径、收益率账本、四 Agent
协作流程、文件格式、失败处理和验收要求。

Auto Fin 当前只用于模拟组合研究，不连接券商、不提交真实委托，也不构成自动交易系统。系统在固定时间生成研究结果、
模拟操作建议和组合表现，并把最终报告发送到钉钉。

本文中的“成交”“买入”“卖出”均指模拟成交。

## 2. 设计目标

Auto Fin 需要实现：

1. 在每个 A 股交易日的 `09:00`、`11:45` 和 `14:45` 生成一次决策；
2. 由事件分析、回测分析、美股关联分析和组合分析四个 Agent 分工协作；
3. 最多持有 10 个股票或 ETF，使用 10 个等额仓位槽；
4. 遵守适用标的的 T+1 卖出限制；
5. 只用涨跌幅和归一化净值核算模拟收益，不在组合账本中记录实际股数和实际价格；
6. 将 Markdown 文件作为用户可读、可恢复的事实来源；
7. 保存数据来源、时间窗口、数据截止时间和分析版本，避免未来数据泄漏；
8. 支持安全重跑、失败降级和钉钉通知去重；
9. 优先从行业和行业 ETF 开始研究，再根据证据决定是否下钻到个股。

## 3. 非目标

第一阶段不实现：

- 真实账户、真实下单或券商接口；
- 毫秒级或秒级交易；
- 融资融券、杠杆、做空、期权和期货；
- 根据实际成交量模拟市场冲击；
- 精确的股数、手数、手续费、印花税和滑点；
- 因价格涨跌持续自动再平衡已有持仓；
- 未经验证就让 Agent 自动修改稳定回测代码；
- 在缺少数据权限或数据不可用时伪造结果。

## 4. 核心术语

### 4.1 交易日

“交易日”以 A 股交易日历为准。“上一日”和“昨天”在交易逻辑中都必须解释为“上一交易日”，不能简单使用自然日减一。

周末和 A 股休市日不运行常规决策。美股交易但 A 股休市时，美股数据留到下一个 A 股交易日的 `09:00` 分析。

### 4.2 决策时间

`decision_at` 是系统允许 Agent 使用截止数据并产生新操作建议的时间：

- `09:00`
- `11:45`
- `14:45`

时区固定为 `Asia/Shanghai`。

### 4.3 数据截止时间

`data_cutoff` 是某次分析允许使用的最新数据时间。任何发布时间、成交时间或修订时间晚于
`data_cutoff` 的信息都不能参与该次决策。

事件数据和行情数据可以有不同的截止时间。例如 `11:45` 决策可使用截至 `11:45` 的新闻事件，但上午行情只使用截至
`11:30` 的已完成交易数据。

### 4.4 模拟成交时间

`scheduled_fill_at` 是某次操作建议计划使用的模拟成交时点：

- `09:00` 产生的操作，在当日 `09:30` 开盘执行；
- `11:45` 产生的操作，在当日 `13:00` 午后开盘执行；
- `14:45` 产生的操作，在当日 `15:00` 收盘执行。

模拟操作必须在后续 checkpoint 中完成结算，不能在产生建议时直接修改持仓。

### 4.5 估值时间

`marked_at` 是计算持仓阶段表现的时间。成交基准和估值基准必须分开：

- `fill basis` 用于确定新持仓从哪个市场时点开始承担涨跌；
- `mark basis` 用于评价截至当前 checkpoint 的实际表现。

组合账本只保存这些时点之间的涨跌幅，不保存真实价格。

## 5. 十槽均仓模型

### 5.1 仓位槽

组合最多有 10 个仓位槽，每个槽的目标初始仓位是模拟组合当时净值的 `10%`：

- 一个标的最多占用一个槽；
- 最多同时持有 10 个不同标的；
- 未使用的槽位保持现金；
- 不允许负现金、做空或超过 10 个持仓；
- 新开仓时按当时组合净值的 `10%` 分配归一化名义本金；
- 多个新标的在同一成交时点开仓时，每个标的使用相同的 `10%` 目标权重；
- 同一时点先执行合法卖出，再执行买入，以便释放模拟现金。

“均仓”表示新开仓时使用相同目标权重，不表示系统因为持仓涨跌而持续自动再平衡。不同标的产生收益后，其模拟市值和实际
权重可以自然漂移。

### 5.2 归一化净值

组合初始状态：

```yaml
portfolio_nav: 1.0
cash_nav: 1.0
positions: []
```

每个持仓保存归一化名义本金和累计收益因子：

```text
position_value = entry_notional × cumulative_return_factor
portfolio_nav = cash_nav + Σ position_value
```

其中：

```text
cumulative_return_factor(t)
  = cumulative_return_factor(t-1) × (1 + interval_return)
```

例如，一个槽位开仓时：

```text
portfolio_nav = 1.0000
entry_notional = 1.0000 × 10% = 0.1000
```

若该标的随后上涨 `2%`：

```text
position_value = 0.1000 × 1.02 = 0.1020
portfolio contribution = +0.0020
```

账本记录 `+2%`、`0.1000` 和 `0.1020` 等归一化值，不记录成交价、股数或手数。

### 5.3 收益率来源

数据工具可以在内部读取行情价格以计算涨跌幅，但 `portfolio.md` 只保存：

- 涨跌幅；
- 涨跌幅对应的起止时点；
- 数据来源；
- 归一化持仓价值；
- 组合收益贡献。

需要保证同一区间的起点和终点使用一致的行情口径。原始行情可以作为可重建的数据快照保存，但不是组合账本的一部分。

### 5.4 现金和卖出

买入时：

```text
cash_nav -= entry_notional
```

卖出时：

```text
cash_nav += position_value_at_fill
realized_return += position_value_at_fill - entry_notional
```

第一阶段只支持整槽买入和整槽卖出，不支持单个持仓的部分加仓、部分减仓或一个标的占用多个槽。

## 6. T+1 规则

### 6.1 持仓批次

即使一个标的只占一个槽，也必须记录开仓交易日：

```yaml
code: 510300.SH
instrument_type: etf
buy_trade_date: 2026-07-23
eligible_sell_date: 2026-07-24
```

卖出校验使用交易日，而不是自然日。

### 6.2 卖出限制

- 当日 `09:30` 买入的标的，不能在当日 `13:00` 或 `15:00` 卖出；
- 当日 `13:00` 买入的标的，不能在当日 `15:00` 卖出；
- 当日 `15:00` 买入的标的，从下一交易日起可卖；
- 非交易日不会使持仓提前变为可卖；
- Agent 可以提出非法卖出建议，但执行器必须拒绝，并在报告中明确记录原因。

第一阶段的默认标的范围只包括按 T+1 处理的 A 股股票、境内宽基 ETF 和境内行业 ETF。跨境、债券、商品、货币等可能支持
T+0 的 ETF 默认排除；未来如需纳入，必须为标的增加明确的 `settlement_cycle`，不能统一套用 T+1。

## 7. 操作模型

### 7.1 操作类型

第一阶段使用离散操作：

```yaml
action: BUY | SELL | HOLD
code: 510300.SH
slot_count: 1
reason: "..."
confidence: 0.0
```

不再使用含义不清楚的 `+x%/-x%`。每个 `BUY` 或 `SELL` 对应一个完整的 `10%` 目标槽。

### 7.2 操作生命周期

每个操作按以下状态流转：

```text
PROPOSED
  ├── FILLED
  ├── REJECTED
  ├── PENDING_DATA
  └── MISSED_CUTOFF
```

- `PROPOSED`：组合 Agent 已提出，尚未到模拟成交时点；
- `FILLED`：后续 checkpoint 已使用约定行情基准完成模拟成交；
- `REJECTED`：违反 T+1、持仓上限、现金约束、重复持仓或标的范围；
- `PENDING_DATA`：应成交但缺少可靠行情数据；
- `MISSED_CUTOFF`：决策结果在计划成交时间之后才完成，不能追认成交。

### 7.3 执行器优先于 Agent

Agent 只提出建议。确定性执行器负责：

- 校验交易日；
- 校验标的范围；
- 校验 T+1；
- 校验持仓数量和重复标的；
- 校验模拟现金；
- 按 SELL 后 BUY 的顺序执行；
- 计算并写入收益率账本；
- 生成最终合法操作列表。

portfolio Agent 不得直接修改已结算持仓。

## 8. 三个决策时间点

### 8.1 时间总览

| 决策点 | 先结算的操作 | 模拟成交基准 | 当前估值基准 | 新操作计划成交 |
|---|---|---|---|---|
| `09:00` | 上一交易日 `14:45` 操作 | 上一交易日 `15:00 close` | 上一交易日 `15:00 close` | 当日 `09:30 open` |
| `11:45` | 当日 `09:00` 操作 | 当日 `09:30 open` | 当日 `11:30 mark` | 当日 `13:00 open` |
| `14:45` | 当日 `11:45` 操作 | 当日 `13:00 open` | 当日 `14:45 mark` | 当日 `15:00 close` |

### 8.2 09:00 开盘前

执行顺序：

1. 读取上一交易日 `portfolio.md` 的 `14:45` 操作；
2. 使用上一交易日 `15:00 close` 作为模拟成交基准完成结算；
3. 对上一交易日 `14:45 mark` 到 `15:00 close` 的持仓表现进行补记；
4. 生成 `09:00` 事件分析；
5. 生成 `09:00` 回测分析；
6. 生成当日唯一一次美股关联分析；
7. portfolio Agent 汇总全部结果；
8. 生成新的 `09:00` 操作，计划在当日 `09:30 open` 执行；
9. 验证并原子写入四份 Markdown；
10. 发送钉钉报告。

此时尚未观察到当日 A 股开盘，不能使用当日 `09:30` 或之后的数据。

### 8.3 11:45 午间

`11:45` 的执行和评价口径必须明确区分：

- `09:00` 操作使用当日 `09:30 open` 执行；
- 持仓效果使用截至 `11:30` 的涨跌幅评价；
- 对 `09:30` 新买入的标的，区间效果是 `09:30 open → 11:30 mark`；
- 对开盘前已持有的标的，区间效果是“上一估值基准 → 11:30 mark”；
- `11:45` 后产生的新操作计划在 `13:00 open` 执行。

执行顺序：

1. 结算 `09:00` 操作；
2. 更新所有持仓截至 `11:30` 的区间涨跌幅、累计涨跌幅和组合贡献；
3. 生成 `09:00 → 11:45` 的事件分析；
4. 使用不晚于 `11:30` 的行情生成回测分析；
5. 复用当日 `09:00` 美股关联分析，并标记其 `as_of`；
6. portfolio Agent 生成午间判断和 `11:45` 操作；
7. 验证并写入文件；
8. 发送钉钉报告。

### 8.4 14:45 尾盘

执行顺序：

1. 使用当日 `13:00 open` 结算 `11:45` 操作；
2. 更新所有持仓截至 `14:45` 的区间涨跌幅、累计涨跌幅和组合贡献；
3. 生成 `11:45 → 14:45` 的事件分析；
4. 使用不晚于 `14:45` 的行情生成回测分析；
5. 复用当日 `09:00` 美股关联分析；
6. portfolio Agent 生成尾盘判断和 `14:45` 操作；
7. 只有在 `15:00` 前完成并通过校验的操作，才能计划按当日收盘基准模拟成交；
8. 验证并写入文件；
9. 发送钉钉报告。

`14:45 → 15:00` 只有 15 分钟。若 Agent 或数据源导致结果在 `15:00` 后才完成，该操作必须标记为
`MISSED_CUTOFF`，不能事后使用已知收盘数据追认操作。报告仍可生成，但不改变模拟持仓。

## 9. 事件时间窗口

事件分析采用连续 cursor，而不是只依赖模糊自然语言窗口：

| checkpoint | 逻辑窗口 |
|---|---|
| `09:00` | 上一交易日已处理 cursor 之后，直到当日 `09:00` |
| `11:45` | 当日 `09:00` 之后，直到当日 `11:45` |
| `14:45` | 当日 `11:45` 之后，直到当日 `14:45` |

每条事件至少保存：

- 来源；
- 原始发布时间；
- 数据获取时间；
- 标题或事件标识；
- 涉及行业和标的；
- 去重键；
- 是否可能在当前决策时点之前被市场知道；
- 对组合的方向、置信度和时间跨度；
- 原始数据或可重建查询的引用。

“分析所有事件”应解释为“分析配置的数据源在该窗口内可获得并通过去重的事件”，不能承诺覆盖全市场全部信息。

## 10. 四 Agent 设计

### 10.1 事件分析 Agent

输出文件：`event_analysis.md`

职责：

1. 获取上一个 cursor 到当前 `data_cutoff` 的事件；
2. 记录使用的工具、接口、查询参数和时间范围；
3. 去重并区分发布时间、采集时间和修订时间；
4. 结合上一交易日和当日已有事件分析；
5. 必要时使用 ReMe search 检索历史记忆；
6. 优先归纳行业影响，再下钻到 ETF 或个股；
7. 输出结构化事件、方向、置信度、影响周期和待验证条件。

事件 Agent 不直接产生模拟成交。

### 10.2 回测分析 Agent

输出文件：`backtest_analysis.md`

职责：

1. 获取满足当前 `data_cutoff` 的日线和分钟数据；
2. 使用 Tushare 获取数据，使用 Polars 完成清洗、聚合和统计；
3. 结合上一交易日、当日已有回测结果和当前持仓；
4. 对常规信号做历史统计；
5. 在大盘大涨、大跌或波动异常时运行极端情景分析；
6. 分析“下跌后是否有统计上的反弹优势”和“上涨后是否有统计上的回撤风险”；
7. 明确样本数量、时间范围、基准、命中率、收益分布和最大回撤；
8. 输出证据，不直接把相关性描述成因果。

所有回测必须满足：

- 不能使用当前决策时点之后的数据；
- 股票池和 ETF 池必须使用当时可获得的信息构建；
- 记录复权口径；
- 记录样本选择和缺失数据处理；
- 保存代码版本、参数哈希和数据快照引用；
- 样本不足时明确返回 `INSUFFICIENT_DATA`。

### 10.3 美股关联分析 Agent

输出文件：`us_correlation_analysis.md`

只在每个 A 股交易日 `09:00` 运行一次。`11:45` 和 `14:45` 复用同一份结果，不重新调用。

职责：

1. 获取已完成的美股交易时段数据；
2. 处理中美节假日和美国夏令时；
3. 将美股收盘信息与下一个 A 股交易日对齐；
4. 分析代表性股票、行业 ETF 和杠杆 ETF，例如 NVIDIA、存储产业链代表、SOXX、SOXL；
5. 对 A 股宽基 ETF 和行业 ETF 建立映射；
6. A 股候选池按近期平均成交额筛选，而不是按单日成交量直接排序；
7. 默认取流动性前 50 的宽基和行业 ETF 作为研究候选；
8. 输出 1 日、5 日和 30 日表现、相关性、领先/滞后关系和稳定性；
9. 记录汇率、交易日错位和样本数等限制。

SOXL 等每日杠杆 ETF 的多日累计收益不能直接除以 3。允许把单日收益除以目标杠杆倍数作为粗略敏感度参考，但 5 日和
30 日分析必须保留每日再平衡、复利和波动损耗的影响。

### 10.4 组合分析 Agent

输出文件：`portfolio.md`

职责：

1. 读取已结算的组合快照；
2. 读取当前 checkpoint 的事件、回测和美股关联结果；
3. 查看已有持仓的区间表现、累计表现和 T+1 可卖状态；
4. 在最多 10 个槽位和 long-only 约束下提出 `BUY`、`SELL` 或 `HOLD`；
5. 优先考虑行业和行业 ETF，再在证据充分时选择个股；
6. 对每个操作提供理由、反例、置信度和失效条件；
7. 把建议交给确定性执行器校验；
8. 生成适合钉钉阅读的最终摘要。

组合 Agent 的输入必须标明每个上游分析的：

- `run_id`
- `status`
- `data_cutoff`
- `generated_at`
- `stale`

## 11. 编排流程

每个 checkpoint 使用一个统一的 pipeline，不能将四个独立 cron job 设置在同一分钟后依赖文件碰运气：

```text
交易日校验
  → checkpoint 加锁
  → 读取上一合法组合快照
  → 结算上一批 PROPOSED 操作
  → 更新区间涨跌幅和组合净值
  → 严格串行运行当前需要的分析 Agent
      → event
      → backtest
      → us correlation（仅 09:00；11:45 和 14:45 复用 09:00 结果）
  → 校验分析结构化输出
  → portfolio Agent 合并分析
  → 确定性交易规则校验
  → 原子写入四份 Markdown
  → 刷新日索引
  → 发送钉钉
  → 记录 checkpoint 完成状态
```

同一 `trade_date + checkpoint` 同时只能有一个运行实例。

分析 Agent 不并行执行。每个 Agent 完成并通过结构化输出校验后，才启动下一个 Agent。固定顺序为：

1. event；
2. backtest；
3. us correlation（仅 `09:00`）；
4. portfolio。

`11:45` 和 `14:45` 跳过 us correlation 的新调用，直接读取并校验当日 `09:00` 的结果，然后继续执行
portfolio。

## 12. 文件布局

建议使用独立 Auto Fin workspace：

```text
.reme/
├── daily/
│   ├── YYYY-MM-DD.md
│   └── YYYY-MM-DD/
│       ├── event_analysis.md
│       ├── backtest_analysis.md
│       ├── us_correlation_analysis.md
│       └── portfolio.md
├── resource/
│   └── auto-fin/
│       └── YYYY-MM-DD/
│           ├── events/
│           ├── market/
│           ├── backtest/
│           └── manifests/
├── digest/
│   └── ...
└── metadata/
    └── auto-fin/
        ├── locks/
        ├── checkpoints/
        ├── notification-state/
        └── derived-cache/

<project-root>/
└── auto-fin/
    ├── backtest/
    ├── data/
    ├── schemas/
    └── tests/
```

目录职责：

- `daily/`：用户可读的研究结论、组合事实和操作记录，是主要事实来源；
- `resource/auto-fin/`：数据快照、查询 manifest 和实验结果，可根据来源重新获取；
- `metadata/auto-fin/`：锁、checkpoint 状态、通知状态和派生缓存，可重建；
- `<project-root>/auto-fin/`：经过测试和人工审查的稳定代码。

不使用含义模糊的 `auto-resource/{topic}` 作为统一目录。ReMe 已经使用 `resource` 作为外部资源目录，并使用
`auto_resource` 表示另一类“资源解释成记忆”的 Job。

四份 Markdown 直接位于 `daily/YYYY-MM-DD/` 下，不能再按 `0900/`、`1145/`、`1445/` 嵌套子目录，否则现有
daily index 默认不会发现它们。

## 13. Markdown 和 frontmatter 契约

### 13.1 通用规则

每份文件必须包含 YAML frontmatter 和可读 Markdown 正文。

frontmatter 保存可由 Pydantic 校验的结构化结果；正文保存适合人和检索系统阅读的内容。仓位、操作、重要事件和核心结论
不能只存在于 frontmatter，正文中必须有对应表格或摘要。

通用字段：

```yaml
---
schema_version: auto-fin/v1
document_type: event_analysis
trade_date: 2026-07-23
timezone: Asia/Shanghai
updated_at: 2026-07-23T11:46:10+08:00
runs:
  - run_id: 2026-07-23T1145+08:00
    checkpoint: "1145"
    status: COMPLETE
    decision_at: 2026-07-23T11:45:00+08:00
    data_cutoff: 2026-07-23T11:45:00+08:00
    generated_at: 2026-07-23T11:46:10+08:00
---
```

checkpoint 统一使用字符串：

```yaml
checkpoint: "0900"  # enum: "0900", "1145", "1445"
```

每次写入都使用稳定 `run_id`。相同 `run_id` 重跑时替换原 run，不得重复追加。文件必须通过临时文件加原子替换写入，
避免出现半写状态。

### 13.2 正文格式

每份文件按 checkpoint 分段：

```markdown
# 2026-07-23 Event Analysis

## 09:00

...

## 11:45

...

## 14:45

...
```

`us_correlation_analysis.md` 只包含 `09:00` 主分析；午间和尾盘只在其他报告中引用它，不复制生成新的分析。

### 13.3 event_analysis.md

每个 run 的结构化信息至少包含：

```yaml
window:
  start_exclusive: 2026-07-23T09:00:00+08:00
  end_inclusive: 2026-07-23T11:45:00+08:00
sources:
  - tool: tushare
    endpoint: news
    fetched_at: 2026-07-23T11:44:30+08:00
    query_hash: "..."
events:
  - event_id: "..."
    published_at: "..."
    industries: [半导体]
    codes: [512480.SH]
    direction: POSITIVE
    confidence: 0.72
    horizon: 1D
    summary: "..."
cursor:
  last_event_time: "..."
  last_event_id: "..."
```

### 13.4 backtest_analysis.md

每个 run 至少包含：

```yaml
market_cutoff: 2026-07-23T11:30:00+08:00
data_manifest: resource/auto-fin/2026-07-23/manifests/backtest-1145.json
code_version: "..."
parameter_hash: "..."
adjustment: raw_with_explicit_return_adjustment
experiments:
  - experiment_id: broad-market-drop-rebound
    sample_start: 2015-01-01
    sample_end: 2026-07-22
    sample_count: 84
    status: COMPLETE
    summary: "..."
signals:
  - scope: industry
    code: 512480.SH
    direction: POSITIVE
    confidence: 0.64
    horizon: 5D
limitations:
  - "..."
```

### 13.5 us_correlation_analysis.md

结构化信息至少包含：

```yaml
as_of: 2026-07-23T09:00:00+08:00
us_session_date: 2026-07-22
a_share_trade_date: 2026-07-23
universe_method: top50_by_recent_average_amount
lookbacks: [1D, 5D, 30D]
mappings:
  - us_code: NVDA
    a_share_industries: [半导体, 算力]
    a_share_codes: [512480.SH]
    correlation_method: aligned_log_return
    sample_count: 120
    conclusion: "..."
limitations:
  - "..."
```

### 13.6 portfolio.md

组合文件是模拟组合的主要事实来源。每个 checkpoint 必须保存：

```yaml
portfolio_before:
  nav: 1.0124
  cash_nav: 0.4124
  position_count: 6
settlements:
  - action_id: "..."
    status: FILLED
    fill_basis: 0930_OPEN
    filled_at: 2026-07-23T09:30:00+08:00
positions:
  - code: 510300.SH
    name: 沪深300ETF
    instrument_type: etf
    buy_trade_date: 2026-07-22
    eligible_sell_date: 2026-07-23
    entry_notional: 0.1000
    interval_return: 0.0062
    cumulative_return: 0.0180
    normalized_value: 0.1018
    portfolio_contribution: 0.0018
portfolio_after_mark:
  nav: 1.0161
  cash_nav: 0.4124
  position_count: 6
  interval_return: 0.0037
proposed_actions:
  - action_id: "..."
    action: BUY
    code: 512480.SH
    slot_count: 1
    proposed_at: 2026-07-23T11:45:00+08:00
    scheduled_fill_at: 2026-07-23T13:00:00+08:00
    status: PROPOSED
    confidence: 0.68
    reason: "..."
```

正文中必须用表格展示：

- 当前现金；
- 当前净值；
- 持仓数；
- 每个标的本区间涨跌幅；
- 每个标的累计涨跌幅；
- T+1 可卖状态；
- 对组合的收益贡献；
- 已结算操作；
- 新操作及计划成交时间；
- 被拒绝操作及原因。

## 14. 数据来源和可追溯性

每次获取数据必须记录：

- 工具和接口；
- 查询参数；
- 请求开始和完成时间；
- 数据覆盖时间；
- 返回行数；
- 数据最大时间戳；
- 数据快照路径或内容哈希；
- 权限或缺失字段警告；
- 是否使用缓存；
- 数据是否晚于当前决策截止时间。

如果 Tushare 权限不足、接口不可用或数据尚未更新，必须明确记录，不得以旧数据冒充新数据。

原始价格只用于计算可复现的区间涨跌幅。组合报告不展示实际价格，但查询 manifest 应足以重新计算对应涨跌幅。

## 15. 复权和收益率口径

第一阶段不在组合账本中保存真实价格，但仍需统一收益率计算口径：

- 模拟成交时点使用当时可观察的未复权行情基准；
- 跨除权除息区间使用明确的复权因子计算可比收益；
- 不使用会因未来公司行为发生变化的前复权历史价格直接充当模拟成交价；
- 日线和分钟线必须采用一致的标的、交易日和时区；
- 缺少必要复权数据时，该区间返回 `PENDING_DATA`，不能猜测收益。

## 16. 回测代码和 Agent 权限

稳定回测代码位于 `<project-root>/auto-fin/`，必须：

- 纳入版本控制；
- 有单元测试；
- 通过格式和 lint 检查；
- 由人工审查后更新；
- 在结果中记录代码版本。

定时运行中的 Agent：

- 可以读取稳定代码；
- 可以在当次 `resource/auto-fin/YYYY-MM-DD/backtest/` 下生成临时分析脚本、数据和结果；
- 不得自动覆盖稳定策略、公共 schema 或测试；
- 不得自动安装未声明依赖；
- 不得把 Tushare token、钉钉密钥或模型密钥写入文件和日志。

Polars 是 Auto Fin 的明确运行依赖，实施时应加入对应的项目依赖配置。

## 17. 失败和降级策略

### 17.1 强制失败

以下情况不能产生新操作：

- 找不到上一合法组合快照；
- 组合账本 schema 校验失败；
- 交易日无法确认；
- 当前净值、现金或持仓无法一致核算；
- 同一 checkpoint 出现无法安全合并的并发运行；
- 模拟成交所需收益率数据缺失。

系统可以生成故障报告，但必须保持上一合法持仓不变。

### 17.2 分析降级

如果事件、回测或美股分析中的某一项失败：

- portfolio 报告标记 `DEGRADED`；
- 禁止新增风险敞口，即不允许新的 `BUY`；
- 可以 `HOLD`；
- 只有在持仓数据完整、T+1 合法且理由明确时，才允许风险降低型 `SELL`；
- 钉钉消息必须显示缺失的分析和降级原因。

`11:45` 和 `14:45` 不重新运行美股分析，因此复用 `09:00` 结果不属于降级，但必须显示其 `as_of`。

### 17.3 通知失败

钉钉失败不能回滚已经写入的组合和分析文件。通知状态单独记录，并允许只重试通知。

## 18. 幂等、重跑和恢复

幂等键：

```text
run_id = trade_date + checkpoint + timezone
```

例如：

```text
2026-07-23T1145+08:00
```

相同 `run_id` 重跑时：

- 已结算操作不能重复结算；
- 已应用的区间涨跌幅不能重复累乘；
- 相同 Markdown run 和正文 checkpoint 段落必须替换；
- 原始数据快照可复用；
- 默认不重复发送钉钉；
- 显式 `force_notify=true` 才允许再次通知。

服务重启不会自动假设错过的 checkpoint 已完成。启动恢复逻辑检查 checkpoint 状态：

- 若计划成交时间尚未到，可继续本次运行；
- 若计划成交时间已经过去，操作标记 `MISSED_CUTOFF`；
- 不允许利用已知未来行情补做过去的决策；
- 允许以 `research_only=true` 重建研究报告，但不得改变历史组合。

## 19. 钉钉报告

每个 checkpoint 最多发送一条组合报告。消息至少包含：

1. 交易日、checkpoint、生成时间和数据截止时间；
2. 当前运行状态：`COMPLETE`、`DEGRADED` 或 `FAILED`；
3. 组合净值、现金、持仓数和本区间组合涨跌幅；
4. 当前持仓及其本区间/累计涨跌幅；
5. 已完成的模拟成交；
6. 新建议和计划成交时间；
7. T+1 或其他规则拒绝的建议；
8. 三个分析模块的核心结论和数据新鲜度；
9. 风险、限制和失效条件；
10. 明确说明“纯模拟盘，不会执行真实交易”。

钉钉通知使用 `run_id` 去重。通知成功状态属于派生运行状态，不写入组合事实字段。

## 20. 配置建议

建议提供独立配置，例如 `reme/config/auto_fin.yaml`：

```yaml
app_name: ReMe Auto Finance
workspace_dir: ${AUTO_FIN_WORKSPACE_DIR:-.reme}
timezone: Asia/Shanghai
language: zh
```

建议公开参数：

```yaml
max_positions: 10
slot_weight: 0.10
long_only: true
allow_partial_position: false
allowed_instrument_types: [stock, domestic_equity_etf]
checkpoints: ["0900", "1145", "1445"]
force_notify: false
research_only: false
```

建议使用三个 cron 入口，但三个入口都调用同一个带 `checkpoint` 参数的 pipeline：

```yaml
auto_fin_0900_cron:
  backend: cron
  cron: "0 9 * * 1-5"

auto_fin_1145_cron:
  backend: cron
  cron: "45 11 * * 1-5"

auto_fin_1445_cron:
  backend: cron
  cron: "45 14 * * 1-5"
```

工作日 cron 仍不能替代 A 股交易日历校验。

## 21. Prompt 共同要求

四个 Agent 的 prompt 都必须包含：

1. 当前时间、时区、交易日、checkpoint 和 `data_cutoff`；
2. 禁止使用晚于 `data_cutoff` 的信息；
3. 优先写代码完成数据下载、清洗和统计，而不是凭印象编造数值；
4. 缺少权限或数据时明确失败；
5. 优先分析行业，再选择行业 ETF，证据充分时才下钻个股；
6. 区分事实、统计关系、推断和建议；
7. 提供反例、限制和结论失效条件；
8. 不直接修改组合账本；
9. 不输出或保存任何凭据；
10. 只输出符合当前 Agent schema 的结构化结果。

Agent 会话和模型内部记忆不是事实来源。每次运行必须显式读取 Markdown、数据 manifest 和当前组合快照。

## 22. 验收和测试

实施至少覆盖以下单元测试：

### 22.1 组合账本

- 初始 `NAV=1.0`、`cash=1.0`；
- 买入一个槽后现金减少当前 NAV 的 `10%`；
- 最多 10 个不同持仓；
- 重复买入同一标的被拒绝；
- 涨跌幅按复利累乘；
- 卖出后归一化持仓价值回到现金；
- 组合净值等于现金加全部持仓价值；
- 不记录真实价格和股数。

### 22.2 T+1

- `09:30` 买入不能在当日 `13:00` 或 `15:00` 卖出；
- `13:00` 买入不能在当日 `15:00` 卖出；
- 上一交易日买入可在下一交易日卖出；
- 周五买入在周一可卖；
- 节假日前买入在下一实际交易日可卖。

### 22.3 三时点

- `09:00` 操作按 `09:30 open` 执行；
- `11:45` 使用 `09:30 open → 11:30 mark` 评价新持仓效果；
- `11:45` 操作按 `13:00 open` 执行；
- `14:45` 使用截至 `14:45` 的涨跌幅；
- `14:45` 操作只有在 `15:00` 前完成才可按 close 执行；
- 迟到结果标记 `MISSED_CUTOFF`。

### 22.4 幂等和恢复

- 同一 run 重跑不重复成交；
- 同一收益区间不重复计入；
- 同一 checkpoint 不重复追加 Markdown；
- 默认不重复发送钉钉；
- 并发运行只有一个实例获得锁；
- 部分写入失败后仍能读取上一合法快照。

### 22.5 数据和 Agent

- 所有分析拒绝晚于 `data_cutoff` 的数据；
- schema 不合法时不写组合；
- 上游分析失败时禁止 BUY；
- `11:45` 和 `14:45` 复用 `09:00` 美股分析；
- SOXL 多日收益不会简单除以 3；
- 所有数值都能追溯到 manifest、数据快照和代码版本。

### 22.6 外部边界

- 单元测试 mock Tushare、模型和钉钉；
- 默认测试不使用真实 token；
- 真实服务测试必须显式授权；
- 钉钉失败不会回滚 Markdown；
- 日索引可以发现四份当日 Markdown。

## 23. 第一阶段完成标准

第一阶段完成必须同时满足：

1. 三个 checkpoint 能在交易日独立、幂等地运行；
2. 10 槽组合账本通过全部不变量校验；
3. T+1、收益率累乘和三种成交基准都有测试；
4. 四类 Markdown 有公开 Pydantic schema；
5. 所有分析都有 `data_cutoff` 和来源 manifest；
6. 上游分析失败不会产生新的 BUY；
7. 重跑不会重复成交、重复计算收益或重复通知；
8. 钉钉报告明确区分模拟成交、当前效果和下一步建议；
9. 原始数据、稳定代码、用户事实和派生状态分别存放；
10. 不需要真实交易账户，也不会调用任何真实下单接口。
