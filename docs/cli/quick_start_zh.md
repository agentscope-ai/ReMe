# ReMe CLI 快速开始

## 记忆管理：AI 为什么需要这个？

用过大模型的人都知道，上下文窗口是有限的。聊着聊着就超长了，然后：

- 对话直接断掉，没法继续
- 回答质量明显变差，前面说的东西它不记得了
- 开个新对话？之前聊的全忘了，从头来过

更烦的是，**就算上下文没满，新对话也是一张白纸**。上次定好的技术方案、你的个人偏好、干到一半的活——全没了。

ReMe 干了两件事来解决这个问题：

| 能力        | 干嘛用的                      |
|-----------|---------------------------|
| **上下文压缩** | 对话太长时，把旧内容自动浓缩成摘要，给新内容腾地方 |
| **长期记忆**  | 重要信息落盘保存，下次对话自动搜出来用       |

---

## 基于文件的记忆设计

ReMe 的长期记忆不依赖外部数据库——**Markdown 文件就是记忆本身**。你随时可以打开看、直接改。

> 记忆设计受 [OpenClaw](https://github.com/openclaw/openclaw) 记忆架构启发。

### 文件结构

```
.reme/
├── MEMORY.md
└── memory/
    ├── 2025-02-12.md
    ├── 2025-02-13.md
    └── ...
```

### MEMORY.md — 长期记忆

放那些不太会变的关键信息，相当于你的"个人档案"：

- **位置**：`{working_dir}/MEMORY.md`
- **内容举例**：项目用 Python 3.12、偏好 pytest、数据库选了 PostgreSQL
- **谁来写**：Agent 通过 `write` / `edit` 工具自动维护

### memory/YYYY-MM-DD.md — 每日日志

一天一个文件，追加写入，记今天干了啥：

- **位置**：`{working_dir}/memory/YYYY-MM-DD.md`
- **内容举例**：修了登录 Bug、部署了 v2.1、讨论了缓存方案
- **谁来写**：Agent 工具写入 + 压缩时自动触发

---

## ReMeCli Demo

<video src="https://github.com/user-attachments/assets/befa7e40-63ba-4db2-8251-516024616e00" width="80%" controls></video>

---

## 安装

### PyPI（推荐）

```bash
pip install reme-ai==0.3.0.0b1
```

### 从源码装

```bash
git clone https://github.com/agentscope-ai/ReMe.git
cd ReMe
pip install -e .
```

> Python >= 3.10

---

## 配置

### 环境变量

除了 yaml 配置文件，API 密钥通过环境变量设置，可以写在项目根目录的 `.env` 里：

| 环境变量                      | 说明                   | 示例                                                  |
|---------------------------|----------------------|-----------------------------------------------------|
| `REME_LLM_API_KEY`        | LLM 的 API Key        | `sk-xxx`                                            |
| `REME_LLM_BASE_URL`       | LLM 的 Base URL       | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `REME_EMBEDDING_API_KEY`  | Embedding 的 API Key  | `sk-xxx`                                            |
| `REME_EMBEDDING_BASE_URL` | Embedding 的 Base URL | `https://dashscope.aliyuncs.com/compatible-mode/v1` |

> 没有 embedding 服务的话搜索效果会打折扣，记得同时设 `vector_enabled=false`。

### 联网搜索（可选）

| 环境变量                | 说明                 |
|---------------------|--------------------|
| `TAVILY_API_KEY`    | Tavily 搜索 API Key  |
| `DASHSCOPE_API_KEY` | 百炼 LLM（带搜索）API Key |

> 二选一就行，有 Tavily 优先用 Tavily。

---

### 配置文件 cli.yaml

`remecli` 启动时加载 [cli.yaml](https://github.com/agentscope-ai/ReMe/blob/main/reme/config/cli.yaml)（
`config_path="cli"`），所有核心参数都在这一个文件里管。

#### 参数说明

**基础配置**

| 参数            | 值       | 说明               |
|---------------|---------|------------------|
| `backend`     | `cmd`   | 运行模式，CLI 用 `cmd` |
| `working_dir` | `.reme` | 工作空间目录，记忆文件存这里   |

**metadata — 上下文窗口与检索参数**

控制上下文空间怎么分配、记忆怎么搜：

| 参数                      | 默认值      | 说明                           |
|-------------------------|----------|------------------------------|
| `context_window_tokens` | `100000` | 上下文窗口总大小（token）              |
| `reserve_tokens`        | `30000`  | 给输出和系统开销预留的空间                |
| `keep_recent_tokens`    | `10000`  | 压缩后保留多少最近的对话                 |
| `vector_weight`         | `0.7`    | 向量搜索权重（BM25 = 1 - 0.7 = 0.3） |
| `candidate_multiplier`  | `2`      | 检索候选池倍数，越大召回越全、越慢            |

> 自动压缩的触发点：消息总 token ≥ `context_window_tokens - reserve_tokens`，即默认 70000 token。

**llms — LLM 模型**

| 参数                 | 说明                 |
|--------------------|--------------------|
| `backend`          | 后端类型，走 OpenAI 兼容接口 |
| `model_name`       | 模型名，默认通义千问         |
| `request_interval` | 请求间隔（秒），控速用        |

**embedding_models — Embedding 模型**

| 参数           | 说明                         |
|--------------|----------------------------|
| `backend`    | Embedding 后端类型             |
| `model_name` | 模型名，默认 `text-embedding-v4` |
| `dimensions` | 向量维度，`1024`                |

**memory_stores — 记忆存储**

| 参数                | 说明                         |
|-------------------|----------------------------|
| `backend`         | 存储后端，默认 `chroma`（ChromaDB） |
| `db_name`         | 数据库文件名                     |
| `store_name`      | 集合名                        |
| `embedding_model` | 用哪个 Embedding 模型           |
| `fts_enabled`     | 开不开 BM25 全文检索              |
| `vector_enabled`  | 开不开向量语义搜索                  |

> 建议 `fts_enabled` 和 `vector_enabled` 都开，混合检索效果最好。

**file_watchers — 文件监控**

| 参数               | 说明                 |
|------------------|--------------------|
| `backend`        | 监控模式，`full` = 全量扫描 |
| `memory_store`   | 对应的记忆存储配置          |
| `watch_paths`    | 要监控的目录/文件          |
| `suffix_filters` | 只关心哪些后缀（`.md`）     |
| `recursive`      | 是否递归子目录            |
| `scan_on_start`  | 启动时先全量扫一遍          |

**token_counters — Token 计数器**

| 参数        | 说明                     |
|-----------|------------------------|
| `backend` | 计数方式，`base` 用 tiktoken |

## 启动

```bash
remecli config=cli
```

启动后自动加载 [cli.yaml](https://github.com/agentscope-ai/ReMe/blob/main/reme/config/cli.yaml)，然后就可以直接跟 Remy
聊了。ReMe 在后台自动处理压缩和记忆。

---

## 系统命令

对话里输入 `/` 开头的命令控制状态：

| 命令         | 说明                  | 需要等 |
|------------|---------------------|-----|
| `/compact` | 手动压缩当前对话，同时后台存到长期记忆 | 是   |
| `/new`     | 开始新对话，历史后台保存到长期记忆   | 否   |
| `/clear`   | 清空一切，**不保存**        | 否   |
| `/history` | 看当前对话里未压缩的消息        | 否   |
| `/help`    | 看命令列表               | 否   |
| `/exit`    | 退出                  | 否   |

### 三个命令的区别

| 命令         | 压缩摘要  | 长期记忆 | 消息历史  |
|------------|-------|------|-------|
| `/compact` | 生成新摘要 | 保存   | 保留最近的 |
| `/new`     | 清空    | 保存   | 清空    |
| `/clear`   | 清空    | 不保存  | 清空    |

> `/clear` 是真删，删了就没了，不会存到任何地方。

---

## ReMeCli 的能力

### 什么时候会写记忆？

| 场景               | 写到哪                    | 怎么触发                 |
|------------------|------------------------|----------------------|
| 上下文超长自动压缩        | `memory/YYYY-MM-DD.md` | 后台自动                 |
| 用户执行 `/compact`  | `memory/YYYY-MM-DD.md` | 手动压缩 + 后台保存          |
| 用户执行 `/new`      | `memory/YYYY-MM-DD.md` | 新对话 + 后台保存           |
| 用户说"记住这个"        | `MEMORY.md` 或日志        | Agent 用 `write` 工具写入 |
| Agent 发现了重要决策/偏好 | `MEMORY.md`            | Agent 主动写            |

### 记忆检索

两种方式找回之前的东西：

| 方式   | 工具              | 什么时候用      | 举例                       |
|------|-----------------|------------|--------------------------|
| 语义搜索 | `memory_search` | 不确定记在哪，模糊找 | "之前关于部署的讨论"              |
| 直接读  | `read`          | 知道是哪天、哪个文件 | 读 `memory/2025-02-13.md` |

搜索用的是**向量 + BM25 混合检索**（向量权重 0.7，BM25 权重 0.3），自然语言和精确关键词都能搜到。

### 内置工具

| 工具              | 干什么      | 细节                                     |
|-----------------|----------|----------------------------------------|
| `memory_search` | 搜记忆      | MEMORY.md 和 memory/*.md 里做向量+BM25 混合检索 |
| `bash`          | 跑命令      | 执行 bash 命令，有超时和输出截断                    |
| `ls`            | 看目录      | 列目录结构                                  |
| `read`          | 读文件      | 文本和图片都行，支持分段读                          |
| `edit`          | 改文件      | 精确匹配文本后替换                              |
| `write`         | 写文件      | 创建或覆盖，自动建目录                            |
| `execute_code`  | 跑 Python | 运行代码片段                                 |
| `web_search`    | 联网搜索     | 通过 Tavily 或 DashScope 搜                |

---

## 上下文压缩怎么工作的

简单说就是把长对话浓缩成摘要，最近的对话保持原样。两种触发方式：

### 自动压缩

每轮对话前 ReMe 会检查当前 token 用量。超过阈值（`context_window_tokens - reserve_tokens`）就自动压缩旧消息：

```mermaid
graph TB
    subgraph 压缩前
        A1[消息1: 你好]
        A2[消息2: 帮我写代码]
        A3[消息3: 工具调用结果...很长]
        A4[消息4: 修改一下]
        A5[消息5: 新需求]
    end

    subgraph 压缩后
        B1[压缩摘要: 之前帮用户写了代码并完成调整]
        B2[消息5: 新需求]
    end

    A1 --> B1
    A2 --> B1
    A3 --> B1
    A4 --> B1
    A5 --> B2
```

### 手动压缩

随时输入 `/compact`，强制压缩所有当前消息，不看阈值。

### 摘要里会留什么？

| 内容    | 说的是啥       | 例子                      |
|-------|------------|-------------------------|
| 目标    | 用户想干什么     | "搞一个登录系统"               |
| 约束和偏好 | 用户提的要求     | "用 TypeScript，不要框架"     |
| 进展    | 做到哪了       | "登录接口好了，注册还在写"          |
| 关键决策  | 定了什么、为什么   | "选 JWT 不选 Session，要无状态" |
| 下一步   | 接下来干嘛      | "做密码重置"                 |
| 关键上下文 | 文件名、函数名、报错 | "主文件 src/auth.ts"       |
