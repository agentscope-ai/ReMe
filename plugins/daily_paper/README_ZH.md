# 每日论文插件

[English](README.md)

每日论文从 Hugging Face Papers 的周榜和月榜中筛选三篇论文，下载 arXiv PDF，生成中文论文解读和一篇约五分钟可读完的
中文简报。本目录是一个独立 Python distribution：单个 `reme.plugins` entry point 暴露 `plugin.yaml`，其中声明五个
Step backend，并在 `application_defaults` 下提供 Job 配置；通过 `plugins=["daily-paper"]` 显式启用这个已安装插件。

## 快速开始

### 1. 安装 ReMe 和每日论文插件

```bash
python -m pip install "reme-ai[core]>=0.4.1.8"
reme plugins install reme-daily-paper
```

### 2. 配置模型环境变量

按照 ReMe README 的[可选模型配置说明](../../README_ZH.md#可选模型配置)配置 LLM 环境变量，也可以使用其他兼容的模型和
服务商。工作流还需要能够访问 Hugging Face Papers 和 arXiv。

### 3. 带插件启动 ReMe

```bash
reme start plugins='["daily-paper"]'
```

未显式传入 `config` 时，ReMe 会加载 `default.yaml`，并将插件叠加到该服务上。插件随应用启动每天 08:00 运行的
`daily_paper_cron`；在另一个终端中，也可以通过 ReMe CLI client 手动生成简报：

```bash
reme daily_paper topics="Agent memory"
```

也可以直接调用 HTTP endpoint：

```bash
curl -s http://127.0.0.1:2333/daily_paper \
  -H 'Content-Type: application/json' \
  -d '{"topics":"Agent memory"}'
```

如果只需运行一次 Job，无需启动长期服务：

```bash
reme start plugins='["daily-paper"]' job=daily_paper topics="Agent memory"
```

## 流程

```text
Hugging Face 周榜/月榜
          ↓
合并排名并排除昨日及近期已推荐论文
          ↓
RRF 排序后由 Agent 精选三篇
          ↓
下载并解析 arXiv PDF，生成三篇中文解读
          ↓
使用 memory_search + read 关联历史记忆并生成简报
          ↓
写入当日索引，并按需发送到钉钉
```

`daily_paper_collect_step` 并发读取运行日期所在周和所在月的榜单，以及严格前一日的 Daily Papers。候选按 arXiv ID
合并，并排除昨日榜单和 `history_days` 窗口内已经推荐的论文。

`daily_paper_rank_step` 使用 reciprocal-rank fusion 合并周榜和月榜排名，最多保留 `candidate_limit` 篇；
`daily_paper_select_step` 再让无工具 Agent 精选三个唯一的候选 ID。非空 `topics` 只影响精选偏好，不改变固定数量。

`daily_paper_analyze_step` 下载 PDF 到 `resource/papers/`，复用已有的有效文件，并在页数、字符数和文件大小限制内提取
文本。三篇中文解读按精选顺序写入当天目录；扫描版或没有文本层的 PDF 会明确失败。

`daily_paper_digest_step` 以本次生成的三篇解读为事实来源，只开放只读的 `memory_search` 和 `read` 来关联较早记忆。
代码会校验历史 wikilink、追加三篇源笔记链接，并重建当日索引。可选的 `dingtalk_markdown_send_step` 在配置群会话后
发送最终简报；未配置时无副作用跳过。

## 参数

| 参数            |  默认值 | 作用                                                                                         |
|-----------------|--------:|----------------------------------------------------------------------------------------------|
| `date`          |    `""` | 运行日期；空值使用应用时区当天，非空值必须为 `YYYY-MM-DD`                                    |
| `force`         | `false` | 已有当日简报时仍重新生成                                                                     |
| `use_hf_mirror` | `false` | 使用 `HF_MIRROR_URL`；未配置时使用 `https://hf-mirror.com`                                   |
| `topics`        |    `""` | 精选论文时优先考虑的主题                                                                     |
| `weekly_weight` |   `0.7` | RRF 中周榜权重                                                                               |
| `history_days`  |    `30` | 历史推荐排重窗口                                                                             |

步骤级默认值包括：`candidate_limit=20`、`rrf_k=60`、`hf_timeout=600`、`hf_max_retries=3`、
`pdf_timeout=600`、`max_pdf_bytes=52428800`、`max_pdf_pages=35` 和 `max_pdf_chars=300000`。

数据客户端自动使用 `HTTP_PROXY`、`HTTPS_PROXY` 和 `NO_PROXY`。手动任务通过 `use_hf_mirror=true` 启用 Hugging Face
镜像；定时任务默认启用，可设置 `DAILY_PAPER_USE_HF_MIRROR=false` 改用官方服务。以下环境变量可覆盖数据源和钉钉配置：

```dotenv
HF_MIRROR_URL=https://hf-mirror.com
ARXIV_MIRROR_URL=https://export.arxiv.org
DINGTALK_APP_KEY=your-app-key
DINGTALK_APP_SECRET=your-app-secret
DINGTALK_ROBOT_CODE=your-robot-code
DINGTALK_CONVERSATION_IDS=cid-group-one,cid-group-two
```

## 产物

```text
.reme/
├── daily/
│   ├── YYYY-MM-DD.md
│   └── YYYY-MM-DD/
│       ├── <中文论文标题>.md       # 三篇，kind: daily-paper-analysis
│       └── <中文简报标题>.md       # 一篇，kind: daily-paper-brief
└── resource/papers/
    └── <arxiv-id>.pdf
```

Markdown 和 PDF 都通过同目录临时文件原子写入。`force=true` 会重新生成本次入选论文的解读和简报，并复用有效 PDF；
不会删除当天已有的其他笔记。网络错误、候选不足、无效 Agent 输出和无法解析的 PDF 都会明确失败。

## 验证

```bash
python -m pytest plugins/daily_paper -v
```

单元测试 mock Hugging Face、arXiv、AgentScope 和钉钉边界，不访问外部服务。
