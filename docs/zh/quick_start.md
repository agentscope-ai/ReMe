---
title: 快速开始
description: 安装并启动 ReMe，完成文件写入、检索和自动记忆的第一个闭环。
---

# 快速开始

本页用于完成第一次可运行闭环。需要完整配置字段时查看[基础配置](./configuration.md)；接入 HTTP 或 MCP 时查看[服务与部署](./services.md)。

## 安装

ReMe 要求 Python 3.11+。

从 pip 安装：

```bash
pip install "reme-ai[core]"
```

从源码安装：

```bash
git clone https://github.com/agentscope-ai/ReMe.git
cd ReMe
pip install -e reme_studio -e ".[core]"
cd reme_studio
npm ci
npm run build:static
cd ..
```

静态构建步骤需要 Node.js 22.13 或更高版本，用于在从源码运行 ReMe 时提供 Studio。

`core` extra 建议安装：当前代码会导入 AgentScope wrapper，自进化记忆也依赖它。

如果要使用 `auto_memory`、`auto_resource`、`auto_dream` 这类 Agent 流程，再配置 LLM：

```bash
cat > .env <<'EOF'
LLM_BACKEND=openai
LLM_MODEL_NAME=qwen3.7-plus
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EOF
```

只跑基础文件读写和 BM25 检索，可以先不配。

---

## 启动

```bash
reme start
```

默认服务地址是 `127.0.0.1:2333`。如果端口被占用：

```bash
reme start service.port=8181
```

```bash
reme version
reme health_check
reme help
```

`reme help` 会列出服务端 action。普通命令会通过 HTTP 调用服务端 Job。

基础 `reme-ai` 包不包含前端资源。安装 `reme-ai[web]` 或 `reme-ai[core]` 后，浏览器打开
<http://127.0.0.1:2333/> 即可进入 ReMe Studio，在同一服务中浏览、编辑和搜索
workspace，并查看 digest Wikilink 图。可用 `service.web_enabled=false` 关闭，或通过 `service.web_static_dir` /
`REME_WEB_STATIC_DIR`
指定自定义静态目录；找不到构建产物时，Job API 仍会正常启动。

---

## Workspace 目录

默认 workspace 是当前目录下的 `.reme/`，启动时会自动创建：

```text
.reme/
├── metadata/   # 索引、图谱、catalog 等持久状态
├── session/    # 对话来源记录
├── mem_session/ # Agent wrapper 生成的 session/配置
├── resource/        # 外部资料
├── daily/           # daily note
└── digest/          # 长期记忆
```

目录分层、Markdown frontmatter 和 wikilink 语义见 [Memory as File](./memory_as_file.md)。

也可以启动时指定：

```bash
reme start workspace_dir=/tmp/reme-demo service.port=8181
```

---

## 写入、索引、检索

```bash
reme write \
  path=digest/wiki/quick-start-demo \
  name="Quick Start Demo" \
  description="快速开始示例记忆" \
  content="# Quick Start Demo

默认实时 watcher 会索引 daily 和 digest 目录中的 Markdown。

相关链接：[[digest/wiki/search-demo.md]]"
```

`path` 是 workspace 内路径；没有后缀时会自动补 `.md`；Markdown 文件会写入 `name` 和 `description` front matter。

后台 watcher 会自动摄取 workspace 文件。也可以基于它已经摄取的 chunks，手动重建派生的 BM25 和 Embedding 索引：

```bash
reme reindex
```

该命令不会扫描 workspace 文件、重新分块或重建 wikilink 图谱。

搜索：

```bash
reme search query="快速开始 示例 记忆" limit=5
```

读取：

```bash
reme read path=digest/wiki/quick-start-demo start_line=1 end_line=20
```

默认配置下，检索主要是 BM25 + wikilink 图谱扩展；向量检索能力在代码中支持，但默认未启用 embedding store。完整检索流程见
[Memory Search](./memory_search.md)。

---

## 文件与 Daily Note

```bash
reme stat path=digest/wiki/quick-start-demo
reme edit path=digest/wiki/quick-start-demo old="会索引" new="会持续索引"
reme frontmatter_read path=digest/wiki/quick-start-demo
reme frontmatter_update path=digest/wiki/quick-start-demo metadata='{"tags":["demo"]}'
```

文件列表 Job 可以直接通过 CLI 调用：

```bash
reme list path=digest recursive=true limit=50
```

等价的 HTTP 调用是：

```bash
curl -s http://127.0.0.1:2333/list \
  -H 'Content-Type: application/json' \
  -d '{"path":"digest","recursive":true,"limit":50}'
```

Daily note：

```bash
reme write path=daily/2026-06-20/demo-session.md name=demo-session description="Demo session" content="记录内容"
reme daily_list
reme daily_reindex
```

`write` 可直接创建 daily note；需要刷新当天索引时运行 `daily_reindex`。

---

## 自动记忆

```bash
reme auto_memory \
  session_id=chat-demo \
  messages='[{"role":"user","content":"我偏好把项目经验沉淀成 Markdown。"},{"role":"assistant","content":"已记录。"}]' \
  memory_hint="记录用户偏好"
```

外部资料放入 `resource/YYYY-MM-DD/` 或直接放在 `resource/` 下后，默认后台会监听 `md/txt/json/jsonl/csv/yaml/html`。
也可以手动触发：

```bash
reme auto_resource changes='[{"path":"resource/2026-06-20/report.md","change":"added"}]'
```

把 daily 整理到长期 digest：

```bash
reme auto_dream date=2026-06-20
reme proactive_read date=2026-06-20
```

这些流程需要可用 LLM；未配置 LLM 时请先使用 `write/read/search` 这类基础能力。

更多细节见 [Auto Memory](./auto_memory.md)、[Auto Resource](./auto_resource.md)、[Auto Dream](./auto_dream.md) 和
[Proactive](./proactive.md)。

---

## HTTP 与配置

每个可服务 Job 都暴露为 `POST /<job>`：

```bash
curl -s http://127.0.0.1:2333/version \
  -H 'Content-Type: application/json' \
  -d '{}'

curl -s http://127.0.0.1:2333/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"快速开始","limit":5}'
```

默认配置来自 `reme/config/default.yaml`。启动时可以用 dot notation 覆盖：

```bash
reme start \
  workspace_dir=/tmp/reme-demo \
  service.host=127.0.0.1 \
  service.port=8181 \
  enable_logo=false
```

也可以指定 YAML/JSON 配置文件：

```bash
reme start config=/path/to/custom.yaml
```

下一步可以查看 [CLI 参考](./reference/cli.md)、[Job API 参考](./reference/jobs.md)和[诊断、备份与恢复](./operations.md)。
