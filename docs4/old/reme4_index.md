# reme4 代码模块索引

> 本文档梳理 `reme4/` 下各能力模块、核心类和代码路径，便于快速定位与扩展。
> 所有路径相对仓库根目录 `/Users/yuli/workspace/ReMe/`。

## 1. 顶层入口与运行流程

| 文件 | 作用 |
| --- | --- |
| `reme4/reme.py` | CLI 入口（`main()`）：`start` 启动应用；`find_reme` 探活；其他动作转发到 client。 |
| `reme4/application.py` | `Application` 基类：解析 config → 注册 service / components / jobs → 拓扑排序启动 → `run_job` / `run_stream_job` / `run_app`。 |
| `reme4/constants.py` | 服务发现常量：`REME_SERVICE_INFO`、默认 host/port (`127.0.0.1:2333`)。 |
| `reme4/__init__.py` | 包入口。 |

启动流程：`reme.main()` → `parse_args` → `resolve_app_config` → `precheck_start` → `ReMe(...).run_app()` → `service.run_app(app)` → `app.start()`（按拓扑序启动 components 与 jobs）。

## 2. 配置与枚举

| 文件 | 作用 |
| --- | --- |
| `reme4/config/default.yaml` | 默认配置：service / jobs / components 全套样例。 |
| `reme4/config/config_parser.py` | `parse_args`、`resolve_app_config`、env 变量展开、点号配置覆盖、YAML/JSON 加载。 |
| `reme4/enumeration/component_enum.py` | `ComponentEnum`：所有组件类型枚举（service/client/job/step/file_*/embedding/keyword_index/tokenizer/as_*）。 |
| `reme4/enumeration/chunk_enum.py` | `ChunkEnum`：流式分块类型 (THINK/CONTENT/TOOL_*/USAGE/ERROR/DONE)。 |

## 3. Schema（Pydantic 数据模型）

代码：`reme4/schema/`

| 类 | 文件 | 说明 |
| --- | --- | --- |
| `ApplicationConfig` / `ComponentConfig` / `JobConfig` | `application_config.py` | 顶层配置模型；包含 service/jobs/components/vault_dir 等字段。 |
| `EmbNode` | `emb_node.py` | 文本+embedding 节点基类，`np.ndarray` 序列化为列表存储。 |
| `FileChunk` | `file_chunk.py` | 文件切片（继承 `EmbNode`）：`path/start_line/end_line/scores`，含 `set_hash_id()`。 |
| `FileNode` | `file_node.py` | 文件级图节点：`path/st_mtime/links/chunk_ids/front_matter`。 |
| `FileLink` | `file_link.py` | 文件间 wikilink 边：`source_path → target_path`，可选 `target_anchor` / `predicate`。 |
| `FileFrontMatter` | `file_front_matter.py` | YAML 头：`name/description`，`extra="allow"` 保留未知键。 |
| `Request` / `Response` | `request.py` / `response.py` | 服务端请求/响应封装。 |
| `StreamChunk` | `stream_chunk.py` | 流式分块：`chunk_type/chunk/done/metadata`。 |

## 4. Components（核心能力组件）

> 所有组件继承 `BaseComponent`（`reme4/components/base_component.py`），提供：
> - `start/close/restart` 生命周期；
> - `bind(name, base_cls, default_factory, optional)` 声明依赖（启动时通过拓扑排序解析）；
> - `vault_path` / `working_metadata_path` 工作目录；
> - `dump/load` 持久化钩子。
>
> 组件通过 `ComponentRegistry`（`R`，`reme4/components/component_registry.py`）按 `(ComponentEnum, name)` 注册和查找；上下文容器为 `ApplicationContext`（`application_context.py`），运行期上下文为 `RuntimeContext`（`runtime_context.py`）。

### 4.1 Tokenizer — `reme4/components/tokenizer/`

| 类 | 文件 | 说明 |
| --- | --- | --- |
| `BaseTokenizer` | `base_tokenizer.py` | 抽象基类，启动时加载 `stopwords` 文件。 |
| `RegexTokenizer` (`@R "regex"`) | `regex_tokenizer.py` | 正则切词；中文按字符切分，非中文按词切分。 |
| `JiebaTokenizer` (`@R "jieba"`) | `jieba_tokenizer.py` | 基于 jieba 的中文分词。 |

### 4.2 Keyword Index — `reme4/components/keyword_index/`

| 类 | 文件 | 说明 |
| --- | --- | --- |
| `BaseKeywordIndex` | `base_keyword_index.py` | 抽象基类：`add_docs/delete_docs/retrieve/clear/optimize_index`，依赖 tokenizer。 |
| `BM25Index` (`@R "bm25"`) | `bm25_index.py` | 自实现 Okapi BM25 倒排索引：`vocab` / `inverted_index` / `doc_meta`，pickle 持久化，支持增量更新与 `optimize_index` 紧凑化。 |

### 4.3 Embedding — `reme4/components/embedding/`

| 类 | 文件 | 说明 |
| --- | --- | --- |
| `BaseEmbeddingModel` | `base_embedding_model.py` | LRU 缓存 + npz 磁盘持久化、批量、重试、健康检查（`is_healthy`）；提供 `get_embedding/get_embeddings/get_node_embeddings`。 |
| `OpenAIEmbeddingModel` (`@R "openai"`) | `openai_embedding_model.py` | OpenAI 兼容协议（dashscope/qwen 等），`AsyncOpenAI` 客户端。 |

### 4.4 File Graph — `reme4/components/file_graph/`

存储 `FileNode` 节点与 `FileLink` 边，支持「虚节点」(被指但尚未导入的目标占位)。

| 类 | 文件 | 说明 |
| --- | --- | --- |
| `BaseFileGraph` | `base_file_graph.py` | 抽象接口：`upsert_nodes/delete_nodes/get_nodes/rebuild_links/clear/get_outlinks/get_inlinks`。 |
| `LocalFileGraph` (`@R "local"`) | `local_file_graph.py` | 纯 dict 实现 + JSONL 持久化；维护 `_nodes`/`_inverse`/`_pending`。 |
| `NxFileGraph` (`@R "nx"`) | `nx_file_graph.py` | networkx `MultiDiGraph` + pickle 持久化，虚节点用「无 node 属性」标识。 |
| `Neo4jFileGraph` (`@R "neo4j"`) | `neo4j_file_graph.py` | Neo4j 后端（bolt 驱动），`(:File)-[:LINKS]->(:File)`，支持升降级虚节点、`rebuild_links` 修复重建。 |

### 4.5 File Chunker — `reme4/components/file_chunker/`

| 类 | 文件 | 说明 |
| --- | --- | --- |
| `BaseFileChunker` | `base_file_chunker.py` | 抽象接口：`parse(path) -> (FileNode, list[FileChunk])`，提供 `_get_relative_path`。 |
| `DefaultFileChunker` (`@R "default"`) | `default_file_chunker.py` | 字节级带 overlap 切片 + YAML front matter + wikilink 抽取（含 Dataview `predicate::`）。 |
| `MarkdownFileChunker` (`@R "markdown"`) | `markdown_file_chunker.py` | Markdown 专用：mistletoe AST → MdNode 树 → 章节递归分块；每个 chunk 携带完整 heading skeleton（TOC）；wikilink 解析支持隐式 `.md`、folder-note、短路径歧义扇出，需注入 `file_graph` 解析目标。 |

### 4.6 File Store — `reme4/components/file_store/`

聚合 `embedding_model` + `keyword_index` + `file_graph`，统一 chunk 写入与混合检索。

| 类 | 文件 | 说明 |
| --- | --- | --- |
| `BaseFileStore` | `base_file_store.py` | 抽象基类：`upsert_file/delete_by_path/clear/vector_search/keyword_search/rebuild_links/get_nodes/get_outlinks/get_inlinks`；启动时探活 embedding，失败则降级为纯关键字检索。 |
| `LocalFileStore` (`@R "local"`) | `local_file_store.py` | 内存 chunk 字典 + JSONL 持久化；upsert 时复用旧 chunk 的 embedding（按 chunk.id 命中）；`vector_search` 用 `batch_cosine_similarity`，`keyword_search` 委托给 keyword_index。 |

### 4.7 File Watcher — `reme4/components/file_watcher/`

| 类 | 文件 | 说明 |
| --- | --- | --- |
| `BaseFileWatcher` | `base_file_watcher.py` | 抽象接口：`watch_loop/update_store/on_added/on_modified/on_deleted`；启动后台任务先做一次全量同步再进入监听循环。 |
| `LiteFileWatcher` (`@R "lite"`) | `lite_file_watcher.py` | 基于 `watchfiles.awatch` 的轮询监听；变更分类后调用 file_chunker 解析、写 file_store；`update_store` 通过 mtime 对比做增量。 |

### 4.8 Job — `reme4/components/job/`

| 类 | 文件 | 说明 |
| --- | --- | --- |
| `BaseJob` (`@R "base"`) | `base_job.py` | 顺序执行 `steps`：每个 step 共享 `RuntimeContext`，最终返回 `Response`；启动时把 step config 实例化为 `BaseStep`。 |
| `StreamJob` (`@R "stream"`) | `stream_job.py` | 流式执行：异常包装为 ERROR chunk，结束时 emit DONE 终止流。 |

### 4.9 Service — `reme4/components/service/`

把 jobs 暴露给外部协议。

| 类 | 文件 | 说明 |
| --- | --- | --- |
| `BaseService` | `base_service.py` | 抽象接口：`build_service/add_job/start_service`，`run_app` 串起来。 |
| `HttpService` (`@R "http"`) | `http_service.py` | FastAPI + uvicorn；普通 job → POST JSON 端点，stream job → SSE 流；CORS 全开。 |
| `MCPService` (`@R "mcp"`) | `mcp_service.py` | FastMCP；把 job 注册为 MCP `FunctionTool`（StreamJob 跳过）；transport 支持 sse/stdio/streamable-http。 |

### 4.10 Client — `reme4/components/client/`

| 类 | 文件 | 说明 |
| --- | --- | --- |
| `BaseClient` | `base_client.py` | 抽象接口：`__call__` 分发 `list`/`_execute`，`list_actions` 列出 server 能力。 |
| `HttpClient` (`@R "http"`) | `http_client.py` | httpx 异步流式：根据 `Content-Type` 自适应 JSON/SSE；`/openapi.json` 列出 actions；CLI 友好格式化。 |
| `MCPClient` (`@R "mcp"`) | `mcp_client.py` | fastmcp Client 包装；transport=`sse/stdio/streamable-http`；`list_tools` 列出工具。 |

### 4.11 AgentScope 适配（as_*） — `reme4/components/as_*/`

把 AgentScope 的 LLM / Formatter / TokenCounter 包成 ReMe 组件，供 step 通过 `step.as_llm` / `step.as_llm_formatter` / `step.as_token_counter` 访问。

| 类 | 文件 | 说明 |
| --- | --- | --- |
| `BaseAsLLM` / `OpenAIAsLLM` (`openai`) / `AnthropicAsLLM` (`anthropic`) | `as_llm/__init__.py` | 包装 `agentscope.model.OpenAIChatModel / AnthropicChatModel`，启动时实例化 `self.model`。 |
| `BaseAsLLMFormatter` / `AsOpenAIChatFormatter` (`openai`) / `AsAnthropicChatFormatter` (`anthropic`) | `as_llm_formatter/__init__.py` | 包装 AgentScope formatter；OpenAI 版用 `ReMeOpenAIChatFormatter` 扩展 |
| `ReMeOpenAIChatFormatter` | `as_llm_formatter/reme_openai_chat_formatter.py` | OpenAI formatter 扩展：tool_result 中的 image 提升为 user 消息；thinking 块合并为 `reasoning_content`；新增 video block 支持。 |
| `BaseAsTokenCounter` / `EstimatedAsTokenCounter` (`estimated`) | `as_token_counter/__init__.py` | 字符级估算 token 计数器（`encoded_byte_len / divisor`）。 |
| `EstimatedTokenCounter` | `as_token_counter/estimate_token_counter.py` | 实现类。 |

### 4.12 Prompt Handler — `reme4/components/prompt_handler.py`

`PromptHandler`：YAML/JSON 加载或类同名文件加载；多语言后缀（`key_zh/key_en`）；`prompt_format` 支持 `[flag]` 行级条件、`{var}` 参数校验。

## 5. Steps（最小执行单元）

代码：`reme4/steps/`

| 类 | 注册名 | 文件 | 作用 |
| --- | --- | --- | --- |
| `BaseStep` | — | `base_step.py` | 抽象基类：`execute()` + `RuntimeContext` 注入 + `input/output_mapping` + 通过 `_resolve` 自动取组件（`as_llm/as_llm_formatter/as_token_counter/file_chunker/file_store/embedding/file_watcher`）；`add_as_tool(toolkit, job_name)` 把 job 包成 AgentScope tool。 |
| `DemoEchoStep1/2` | `demo_echo_step1` / `demo_echo_step2` | `common/demo.py` | 烟雾测试：query 处理 + 应答。 |
| `HealthCheckStep` | `health_check_step` | `common/health_check.py` | 各组件健康/规模快照（embedding/file_graph/file_store/file_watcher/keyword_index）+ 内存深度估算。 |
| `HelpStep` | `help_step` | `common/help.py` | 一行式列出全部 job 元信息（含参数 schema）。 |
| `ReindexStep` | `reindex_step` | `common/reindex.py` | 全量重建：停 watcher → clear store → `update_store` → 重启 watcher。 |
| `SearchStep` | `search_step` | `common/search.py` | 混合检索：vector_search + keyword_search 并发 → RRF 融合 → 阈值过滤 → 截断 → 可选 outlinks/inlinks 邻居展开（含元数据）。 |
| `StreamDemoStep1/2` | `stream_demo_step1` / `stream_demo_step2` | `common/stream_demo.py` | 流式烟雾测试：逐字符 emit CONTENT。 |
| `VersionStep` | `version_step` | `common/version.py` | 输出 `reme4.__version__`。 |

## 6. Utils — `reme4/utils/`

| 文件 | 主要导出 |
| --- | --- |
| `common_utils.py` | `hash_text`(SHA-256)、`execute_stream_task`(SSE 流转发)、`mock_reme_server`(子进程启动测试服务器)、`call_action` / `call_and_check`(HTTP 调用与断言)。 |
| `service_utils.py` | `find_reme` / `locate_reme` / `precheck_start` / `cli_find_reme`：服务发现，`lsof` + `pgrep` 扫描运行实例，端口冲突预检。 |
| `env_utils.py` | `load_env`：加载 `.env`。 |
| `logger_utils.py` | `get_logger`：loguru 日志（控制台 + 文件）。 |
| `logo_utils.py` | `print_logo`：启动 ASCII logo。 |
| `similarity_utils.py` | `cosine_similarity` / `batch_cosine_similarity`：numpy 向量相似度。 |

## 7. 内置 Jobs（`reme4/config/default.yaml`）

| Job | Backend | 步骤 | 说明 |
| --- | --- | --- | --- |
| `demo` | `base` | `demo_echo_step1` → `demo_echo_step2` | 端到端 demo。 |
| `version` | `base` | `version_step` | 返回包版本。 |
| `health_check` | `base` | `health_check_step` | 组件健康快照。 |
| `help` | `base` | `help_step` | 列出全部 job。 |
| `reindex` | `base` | `reindex_step` | 全量重建索引。 |
| `search` | `base` | `search_step` | 混合检索（vector+keyword RRF，可展开邻居）。 |
| `stream_demo` | `stream` | `stream_demo_step1` → `stream_demo_step2` | 流式 demo。 |

## 8. 默认依赖关系（来自 `default.yaml`）

```
tokenizer (regex)        ──┐
embedding_model (openai) ──┤── file_store (local) ── file_watcher (lite)
file_graph (local)       ──┤            (持有 embedding/keyword_index/file_graph)
file_chunker (default)    ──┘                                    │
keyword_index (bm25)  ── tokenizer ──────────────┘              │
                                                  file_chunker ──┘
```

启动时由 `Application._topological_order()`（Kahn 算法）按依赖拓扑序启动；关闭时反向。

## 9. 扩展点速查

| 需求 | 入口 |
| --- | --- |
| 新增检索后端 | 实现 `BaseFileStore` 子类，`@R.register("xxx")` |
| 新增图后端 | 实现 `BaseFileGraph` 子类（参考 `Neo4jFileGraph` 处理虚节点） |
| 新增分词器 | 实现 `BaseTokenizer.tokenize` |
| 新增解析器 | 实现 `BaseFileChunker.parse`（返回 `(FileNode, list[FileChunk])`） |
| 新增 Job | 在 `default.yaml`（或自定义 yaml）`jobs:` 段声明 + 写步骤实现 |
| 新增 Step | 继承 `BaseStep`，实现 `execute()` 并 `@R.register("xxx_step")` |
| 暴露新协议 | 实现 `BaseService`（参考 `HttpService` / `MCPService`） |
| 接入新 LLM | 实现 `BaseAsLLM` 子类（`agentscope.model` 适配） |
