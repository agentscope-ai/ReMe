# 每日论文 Cookbook

[English](README.md)

每日论文工作流从 Hugging Face Papers 的周榜和月榜中筛选三篇论文，下载 arXiv PDF，生成中文论文解读和一篇约五分钟可读完的中文简报。当前实现位于
[`reme/steps/cookbook/daily_paper/`](../../reme/steps/cookbook/daily_paper/)，由
[`daily_cookbook.yaml`](../../reme/config/daily_cookbook.yaml) 装配。

## 快速开始

要求 Python 3.11 或更高版本、`core` 依赖、可用的 AgentScope LLM，以及能访问 Hugging Face Papers 和 arXiv 的网络。

```bash
python -m pip install -e ".[core]"
export LLM_API_KEY="your-api-key"
reme start config=daily_cookbook job=daily_paper
```

内置 LLM 组件默认配置为：

- 模型：`qwen3.7-plus`
- endpoint：无内置 `LLM_BASE_URL`；请设置服务商要求的 OpenAI 兼容 endpoint
- 环境变量：`LLM_API_KEY`、`LLM_MODEL_NAME`、`LLM_BASE_URL`

Auto Fin 和 Daily Paper 共用这一个 `default` LLM，以及唯一的无工具 `default` AgentScope wrapper。
只有交互式 `dingtalk_wait` Step 会在调用时覆盖默认值，启用 AgentScope `bash` 和明确的 ReMe Job allowlist；Daily
Paper 的模型调用仍然无工具。

默认 workspace 是启动目录下的 `reme_workspace/`，可通过 `DAILY_PAPER_WORKSPACE_DIR` 覆盖。

## 工作流

```text
Hugging Face 周榜/月榜
          │
          ▼
Collect ──► Rank ──► Select 3 篇 ──► Analyze PDF ──► Digest ──► DingTalk（可选）
                                       │                │
                                       ├─ PDF           ├─ 每日简报
                                       └─ 论文解读      └─ 当日索引
```

### 1. Collect

`daily_paper_collect_step` 根据运行日期并发读取：

- 该日期所在 ISO week 的 Hugging Face 周榜；
- 该日期所在自然月的 Hugging Face 月榜；
- 严格前一个自然日的 Hugging Face Daily Papers。

周榜和月榜按 arXiv ID 合并，并保留各自排名。随后排除：

- 昨日 Daily Papers 中的论文；
- `history_days` 窗口内，已出现在 `daily/<date>/*.md` frontmatter `arxiv_id` 中的论文。

如果当天已经存在 `kind: daily-paper-brief` 的 Markdown 且 `force=false`，整个生成流程会跳过；已有简报仍可进入钉钉发送步骤。没有剩余候选论文时，Job 直接失败。

### 2. Rank

`daily_paper_rank_step` 使用 reciprocal-rank fusion：

```text
score = 1 / (rrf_k + monthly_rank)
      + weekly_weight / (rrf_k + weekly_rank)
```

缺失的榜单排名贡献为零。论文按融合分、upvotes、arXiv ID 排序，候选池最多保留 `candidate_limit` 篇。Rank 阶段不应用任何主题倾向。

### 3. Select

`daily_paper_select_step` 将候选元数据交给无工具的 AgentScope Agent，并要求返回恰好三项：

```json
{"papers": [{"arxiv_id": "2601.01234", "reasoning": "具体且可核验的选择理由"}]}
```

三个 ID 必须唯一且都属于候选池，理由不能为空。校验失败后，错误信息会反馈给 Agent 并重试一次。只有非空 `topics` 会向精选提示注入个性化主题，且不会改变固定的三篇数量。

### 4. Analyze

`daily_paper_analyze_step` 按精选顺序逐篇处理：

1. 校验新版 arXiv ID 格式 `YYYY.NNNN` 或 `YYYY.NNNNN`；
2. 下载 PDF 到 `resource/papers/<arxiv-id>.pdf`；
3. 如果目标文件已存在且以 `%PDF-` 开头，直接复用；
4. 用 `pypdf` 提取分页文本，受 `max_pdf_pages` 和 `max_pdf_chars` 限制；
5. 将论文元数据、选择理由和 PDF 文本交给无工具 Agent；
6. 将中文解读写入 `daily/<date>/<中文标题>.md`。

下载采用临时文件并在校验 PDF 文件头后原子替换，同时限制 `max_pdf_bytes`。当前没有 OCR；扫描版或无文本层 PDF 会失败。提取被截断时，笔记 frontmatter 中的 `pdf_text_truncated` 会记录为 `true`。

### 5. Digest

`daily_paper_digest_step` 直接使用内存中的三篇解读生成中文简报，不会重新读取或搜索其他资料。输出必须包含 `title`、`desc` 和 `body`。代码会：

- 去掉模型可能生成的 YAML frontmatter；
- 规范化中文标题并用作文件名；
- 确定性追加三篇源笔记的 wikilink；
- 写入 `daily/<date>/<中文简报标题>.md`；
- 重建 `daily/<date>.md` 当日索引。

最终响应 metadata 包含日期、周/月范围、入选 arXiv ID、选择理由、笔记/PDF/简报路径、源榜单数量和排重数量。

### 6. DingTalk

最后的 `dingtalk_markdown_send_step` 是可选步骤。未设置群会话 ID 时无副作用跳过；配置后会去掉 frontmatter，并把简报正文依次发送给所有群：

```dotenv
DINGTALK_APP_KEY=your-app-key
DINGTALK_APP_SECRET=your-app-secret
DINGTALK_ROBOT_CODE=your-robot-code
DINGTALK_CONVERSATION_IDS=cid-group-one,cid-group-two
```

任一群发送失败不会阻止继续尝试后续群，全部尝试结束后统一报告失败。

## 产物

```text
reme_workspace/
├── daily/
│   ├── YYYY-MM-DD.md
│   └── YYYY-MM-DD/
│       ├── <中文论文标题>.md       # 三篇，kind: daily-paper-analysis
│       └── <中文简报标题>.md       # 一篇，kind: daily-paper-brief
└── resource/
    └── papers/
        └── <arxiv-id>.pdf
```

文件名来自 Agent 返回的中文标题。代码会清理路径不安全字符，并处理同名文件。Markdown 和 PDF 都通过同目录临时文件写入后原子替换。

## 参数与默认值

可在调用时传入的 Job 参数：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `date` | `""` | 运行日期；空值使用应用时区当天，非空值必须为 `YYYY-MM-DD` |
| `force` | `false` | 已有当日简报时仍重新生成 |
| `use_hf_mirror` | `false` | 是否使用 Hugging Face 镜像站；优先读取 `HF_MIRROR_URL`，未配置时使用 `https://hf-mirror.com` |
| `topics` | `""` | 精选论文时优先考虑的主题 |
| `weekly_weight` | `0.7` | RRF 中周榜权重 |
| `history_days` | `30` | 历史推荐排重窗口 |

`daily_paper` Job 的步骤级配置：

| 配置 | 默认值 | 作用 |
|---|---:|---|
| `candidate_limit` | `20` | 送入 Select 的最大候选数 |
| `rrf_k` | `60` | RRF 常数 |
| `hf_timeout` | `600` 秒 | Hugging Face 单次请求超时 |
| `hf_max_retries` | `3` | Hugging Face 最大尝试次数 |
| `pdf_timeout` | `600` 秒 | arXiv PDF 下载超时 |
| `max_pdf_bytes` | `52428800` | PDF 上限，50 MiB |
| `max_pdf_pages` | `35` | 最多提取页数 |
| `max_pdf_chars` | `300000` | 最多送入 Agent 的 PDF 字符数 |

## 镜像站

数据客户端使用 httpx 默认的环境处理，因此存在 `HTTP_PROXY`、`HTTPS_PROXY` 或 `NO_PROXY` 时会自动生效。两个数据源启用镜像的方式不同：Hugging Face 由 `use_hf_mirror` 任务参数控制，arXiv 仅由环境变量驱动。

```dotenv
# 为内置 daily_paper_cron 定时任务启用镜像站
DAILY_PAPER_USE_HF_MIRROR=true

# 仅在手动任务或定时任务启用镜像时读取；未配置时使用 https://hf-mirror.com
HF_MIRROR_URL=https://hf-mirror.com

# 未设置时使用 https://arxiv.org
ARXIV_MIRROR_URL=https://export.arxiv.org

# 也支持带路径前缀的中转地址
# HF_MIRROR_URL=http://relay-host:18080/hf
# ARXIV_MIRROR_URL=http://relay-host:18080/arxiv
```

`HF_MIRROR_URL` 必须提供当前代码使用的 `/papers/...`、`/api/daily_papers` 和 `/api/papers/...` 路径。`ARXIV_MIRROR_URL` 必须支持 `/pdf/<arxiv-id>`。两种 base URL 都会保留路径前缀，末尾 `/` 可有可无。不存在备用地址回退：客户端选定哪个 base URL，就只访问该地址。

> **行为变更：** 以往只要设置 `HF_MIRROR_URL` 就会改变 Hugging Face 的访问地址；现在该变量仅在任务启用镜像时才会读取，否则直接访问官方站点，并输出一条“已忽略该变量”的告警日志。手动调用需传入 `use_hf_mirror=true`，`daily_paper_cron` 定时任务需设置 `DAILY_PAPER_USE_HF_MIRROR=true`，才能继续走镜像。

## 运行方式

生成指定日期的简报：

```bash
reme start \
  config=daily_cookbook \
  job=daily_paper \
  date=2026-08-06 \
  topics="Agent memory" \
  history_days=30
```

强制重跑；有效的本地 PDF 仍会复用：

```bash
reme start config=daily_cookbook job=daily_paper date=2026-08-06 force=true
```

启动 HTTP 服务和定时任务：

```bash
reme start config=daily_cookbook
```

内置服务监听 `127.0.0.1:8001`，`daily_paper_cron` 按 `Asia/Shanghai` 时区每天 08:00 运行。设置 `DAILY_PAPER_USE_HF_MIRROR=true` 可让该定时任务使用 Hugging Face 镜像站。可通过 `DAILY_PAPER_HOST`、`DAILY_PAPER_PORT` 或启动参数覆盖监听地址和端口。

```bash
curl -s http://127.0.0.1:8001/daily_paper \
  -H 'Content-Type: application/json' \
  -d '{"date":"2026-08-06","force":false,"topics":"Agent memory"}'
```

## 失败与重跑

- Hugging Face 请求失败会指数退避重试，最多尝试 `hf_max_retries` 次。
- 候选少于三篇、Agent 精选不合法、PDF 无效/过大/无文本或 Agent 输出为空都会终止 Job。
- 三篇论文按顺序处理；中途失败时，之前已完成的 PDF 和笔记会保留。
- `force=true` 会重新生成笔记和简报，但会复用有效 PDF。
- 多文件流程不是事务，也没有同一日期的全局运行锁。

## 测试

单元测试会 mock Hugging Face、arXiv、AgentScope 和 DingTalk 边界，不访问真实服务：

```bash
python -m pip install -e ".[dev,core]"
pytest tests/unit/test_daily_paper.py -v
```
