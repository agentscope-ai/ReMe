ReMe新版本V4

@jinli
新版reme是一个自管理的个人知识库。
- **记忆分层：** → 记忆按"原始 → 加工"两层组织：`resource/`（原始素材）、`daily/`（日记事件）是只增不删的流水帐；`digest/` 是加工层，下分 `personal/`（个性化）、`knowledge/`（主题知识）、`procedural/`（Agent 任务经验）、`proactive/`（主动洞察）四个固定子目录，写入策略和检索权重各有差异。
- **记忆的载体还是 Markdown：** → 所有记忆都是 Obsidian 兼容的 .md 文件——YAML front matter、四种 wikilink（`[[X]]` / `[[X#anchor]]` / `[[X|alias]]` / `![[X]]`）、Dataview 风格 `predicate:: [[X]]` 语义关系全部沿用社区约定。用户可读、可备份、可迁移，对抗黑盒。
- **自我管理进化** → 不需要用户手工整理，Agent 在后台自动对于原始素材进行整理和融合，按照记忆的类型（个性化、程序化、知识类）进行分类整理并更新现有逻辑，同时自动Build link让笔记自己长出结构。这一点把 ReMe 同时与"手动建图的 Obsidian"和"扁平存储的Mem0"拉开。
- **渐进式检索** → 自我管理进化的产出物不是一堆扁平笔记，而是一张可被**渐进式检索**消费的图：向量 + 关键词 + 图谱三路 RRF 融合，返回时通过 1-hop 邻居 meta 让 Agent"先看目录、再决定要不要展开正文"，不像传统 RAG 那样一次性把 top-K 切片塞进上下文。
- **被集成而非内置（分发形态）** → ReMe 不做独立 Agent 产品，而是作为**能力**被任意 Harness 调用：SDK 深度集成（qwenpaw / AgentScope）、MCP Tool + skill.md、CLI + skill.md 三条路径并行，记忆跟着用户走，不绑定任何上层框架。

1. 目标： 构建个人知识库，集成qwenpaw等harness框架中，实现知识/记忆的自进化和自管理，结合graph高效搜索。
2. 新的特性：
   - 支持多种记忆类型，包括个性化记忆、程序化记忆、知识类记忆
     - [❌ 待补充] 当前 reme 中只有统一的 `FileNode/FileChunk` 抽象（`reme/schema/file_node.py`、`reme/schema/file_chunk.py`），尚未在代码层区分"个性化/程序化/知识类"三类记忆，需要在 schema 与 store 中扩展类型字段或子类。
   - 支持memory-self-evolving
     - [❌ 待补充] 没有发现自进化相关的 step/job 实现，目前只有基础的 search/reindex 等 common steps（`reme/steps/common/`）。需要新增 auto-memory/auto-dream 等 step。
   - 支持markdown之间的链接，构建graph，更好的渐进式展开
     - [✅ 已实现 → `reme/components/file_chunker/markdown_file_chunker.py`（wikilink 解析 + Dataview 谓词）、`reme/components/file_graph/`（local/nx/neo4j 三种 graph 后端）]
     - [✅ 已实现 → `reme/steps/common/search.py:109` `_expand_links`、`reme/config/default.yaml:86` `expand_links` 参数（搜索结果可附 outlinks/inlinks 邻居元数据）]
3. 工程实现：
   1. components
      - 支持backend切换
        - [✅ 已实现 → `reme/components/component_registry.py`（`R.register(name)` 装饰器 + `R.get(ctype, backend)` 查找）；`reme/application.py:51` 通过 `config.components` 中的 `backend` 字段动态构造]
      - 生命周期管理 start/close
        - [✅ 已实现 → `reme/components/base_component.py:151` `start()` / `:162` `close()` / `:172` `restart()`，含 `is_started` 幂等保护与 `asyncio.Lock`]
      - components之间相互调用，支持前序依赖还是啥
        - [✅ 已实现 → `reme/components/base_component.py:78` `BaseComponent.bind()` 声明依赖（含 `optional`、`default_factory`），`:98` `_resolve_bindings()` 自动注入；`reme/application.py:80` `_topological_order()` Kahn 算法做拓扑排序、检测循环依赖]
   2. job/step，借鉴自github action
      - step是最小的执行单元，可以自由使用components，不需要管理生命周期
        - [✅ 已实现 → `reme/steps/base_step.py`、`reme/steps/common/`（demo/health_check/help/reindex/search/version/stream_demo 等内置 step）]
      - job是steps的集合，可以自由组合，支持step复用
        - [✅ 已实现 → `reme/components/job/base_job.py`（顺序执行 step）、`reme/components/job/stream_job.py`（流式 job）；`reme/config/default.yaml:5` 通过 yaml 声明 job→steps 组合]
      - 对外job可以封装cli命令，mcp_tool，http服务接口等
        - [✅ 已实现（HTTP / MCP / CLI client） → `reme/components/service/http_service.py:35` `_add_job` 把 job 注册成 POST 端点；`reme/components/service/mcp_service.py`；`reme/components/client/http_client.py`、`reme/components/client/mcp_client.py`；`reme/reme.py:27` `main()` 通过 CLI 子命令 `start` / `find_reme` / `<job_name>` 调用 client]
   3. application：
      - components的生命周期管理，通过前序依赖构建拓扑图启动应用
        - [✅ 已实现 → `reme/application.py:115` `_start()` 按拓扑顺序启动所有 component；`:133` `_close()` 反序关闭]
      - 集成run_job
        - [✅ 已实现 → `reme/application.py:148` `run_job()` / `:154` `run_stream_job()`]
      - 集成Service能力对外提供能力
        - [✅ 已实现 → `reme/application.py:170` `run_app()` 调用 `service.run_app(app=self)`；`reme/components/service/base_service.py`]
   4. 对外接口：
      - skill.md + cli方案，通用方案，支持集成到各种harness框架中
        - [⚠️ 部分实现] CLI 调用通道已具备（`reme/reme.py:17` `call_server()` 通过 `http_client` / `mcp_client` 调任意已注册 job），但 [❌ 待补充] 仓库内未发现 `skill.md` 文件，需要为 Claude Code / 其它 harness 编写 skill 描述文件。
        - 可以选择agent来启动reme服务（后台）
          - [❌ 待补充] 未见"由 agent 自动拉起后台 reme 服务"的脚本/约定，需要补充进程托管或 launchctl/systemd 集成方案。
      - skill.md + mcp-tool方案，通用方案
        - [⚠️ 部分实现] MCP 服务通道存在（`reme/components/service/mcp_service.py`、`reme/components/client/mcp_client.py`），但 [❌ 待补充] 同样缺 `skill.md` 模板。
        - 需要手动启动mcp服务
          - [✅ 已实现 → `reme service` 模式可通过 `reme/config/default.yaml:1` `service.backend: mcp` 切换，`reme/reme.py` `start` 子命令拉起]
      - sdk集成（qwenpaw集成）
        - [❌ 待补充] reme 内未见 qwenpaw / agentscope 相关适配代码（仅 `reme/`、`reme_ai/` 旧版有部分逻辑，但已废弃，按记忆 [[feedback_deprecated_directories]] 不应改动）。需要新建 `reme/integrations/qwenpaw/` 之类的模块。
        - str + 封装AgentscopeTools
          - [❌ 待补充] 没有 `AgentscopeTools` 包装层。
        - 集成auto-memory、auto-dream、auto-memory-search的能力
          - [❌ 待补充] 三个能力均未实现。
4. 记忆存储方案：
   - resource：原始对话日志，上传的文件，原始的html文件等
     - [❌ 待补充] `reme/application.py:20` 仅创建 `metadata_dir` / `daily_dir` / `knowledge_dir`，未见 `resource_dir` 概念；需要在 `ApplicationConfig` 中加入并落地相应目录与抓取/上传逻辑。
   - daily：
     - daily/YYYYMMDD.md：主Agent调用write/edit工具修改，兼容上一版，同时承担了当天其他md的索引
       - [⚠️ 部分实现] `reme/application.py:23` 已建 `daily_dir`，但目录内 markdown 的"主索引"约定与 write/edit 兼容协议无显式实现，主要靠主 agent 自身行为。需要文档化 + 校验。
     - daily/YYYYMMDD/{event}.md auto-memory 针对上下文对话，拆分成不同的事件存储，同时在YYYYMMDD.md构建好索引可以链接过来
       - [❌ 待补充] auto-memory 拆事件的 step / job 不存在。
   - knowledge:
     - knowledge/{topic:-personal/agent/financial/work...}/{xxx}.md 在空闲时间整理记忆，按照主题和事件进行分类存储
       - [⚠️ 部分实现] `knowledge_dir` 已建（`reme/application.py:24`），但"按主题/事件整理"的后台任务、topic 枚举均缺失。
   - proactive:
     - proactive/YYYYMMDD.md 待定。如果存在给用户主动推送的能力，这里可以记录每一天agent给用户推荐的分析和心路历程。
       - [❌ 待补充] 主动推送/proactive 目录与逻辑均未实现。
5. Markdown 格式 & Build Graph
   1. obsidian格式的Markdown文件格式
      - front matter格式
        - [✅ 已实现 → `reme/components/file_chunker/markdown_file_chunker.py:313` `frontmatter.loads(...)`；`reme/schema/file_front_matter.py`]
      - file link格式 4种格式
        - [⚠️ 部分实现] `markdown_file_chunker.py:88` `_WIKILINK_RE` 已支持 `[[target]]` / `[[target#anchor]]` / `[[target|alias]]` / `![[target]]`（嵌入），并支持 Dataview `predicate:: [[X]]` 与 inline `[predicate:: [[X]]]`。但 [❌ 待补充] 标准 Markdown `[text](url.md)` 链接尚未被解析为 graph 边。
   2. 更好的文件chunking机制
      - 旧版 类似rag 带overlap的chunking机制
        - [📌 历史] V3 旧逻辑，对照说明用，无需在 reme 中实现。
      - 解析 Markdown Ast
        - [✅ 已实现 → `markdown_file_chunker.py:308` 使用 `mistletoe` 的 `Document`/`MarkdownRenderer`；`:335` `_build_tree` 把扁平 children 折叠成 section 嵌套树（`MdNode`）]
      - 每一个chunk都带全部标题
        - [✅ 已实现 → `markdown_file_chunker.py:381` `_chunk_node`（`before` 累积已经过的标题、`after` 拼剩余 desc_toc）；`:712` `_make_chunk` 用 `_toc_join(before, content, after)` 把全文目录骨架前后包裹]
   3. 通过link构建graph索引，同时构建反向link索引
      - [✅ 已实现 → `reme/components/file_graph/base_file_graph.py`、`reme/components/file_graph/local_file_graph.py`（含 `get_outlinks`、`get_inlinks` 双向索引）；nx/neo4j 后端同 API；`reme/steps/common/search.py:114-129` 使用双向 link]
   4. link的生成有两种，一种是主agent在生成link；另一种是通过后台任务，自动构建文档之间的link
      - 介绍如何auto-link
        - [⚠️ 部分实现] 主 agent 显式写 `[[link]]` 已经会被 parser 抓为边（`markdown_file_chunker.py:152` `_extract_links`）。但 [❌ 待补充] "后台任务自动补 link" 的实现（实体抽取 / 候选文档相似度匹配 / link 写回 markdown）尚不存在，需要单独的 step/job。
6. 如何做memory自进化
Auto-memory
auto-dream
   - [❌ 待补充] reme 没有 auto-memory / auto-dream 任何代码。需要：
     - 新增 step（如 `reme/steps/auto_memory.py`、`reme/steps/auto_dream.py`），基于现有 `BaseStep` + LLM component；
     - 设计触发机制（job 调度、空闲检测）；
     - 与上面的 daily/knowledge 目录约定打通。
7. 更好的检索：
   - 渐进式展开的检索
     - [⚠️ 部分实现] `reme/steps/common/search.py:14` `SearchStep` 已做 vector + keyword 的 RRF 融合，并支持 `expand_links` 一跳展开（outlinks/inlinks + 邻居 meta）。但 [❌ 待补充] "多跳渐进展开"、"按需要由 agent 主动展开下一层"的交互式 API 尚未实现。
8. 结合外部的Agent工具：
ReMe更加专注于知识加工，而不是知识获取
- 结合qwenpaw
  - sdk集成（qwenpaw集成）
    - [❌ 待补充] 见 §3.4。
  - str + 封装AgentscopeTools
    - [❌ 待补充] 同上。
  - 集成auto-memory、auto-dream、auto-memory-search的能力
    - [❌ 待补充] 同上。
- 结合其他的Agent框架
  - skill.md + cli方案，通用方案，支持集成到各种harness框架中
    - 可以选择agent来启动reme服务（后台）
      - [❌ 待补充] 同 §3.4。
  - skill.md + mcp-tool方案，通用方案
    - 需要手动启动mcp服务
      - [⚠️ 部分实现] MCP 服务可启动，但 skill.md 缺失。

## 更好的性能，更稳定和兼容
V4更加高效的底层记忆索引
- V3版本基于sqlite/chroma等本地数据库
  - 在qwenpaw等低版本linux & win系统存在兼容性问题，会存在core dump等问题
  - 不支持关键词检索，这里需要Keyword倒排索引，对中文的支持较差
  - [📌 历史] 描述 V3 痛点，不需要代码。
- V4版本我们重写了file parser，file store，file graph，file watcher，手写了支持增量更新倒排索引
  - file parser → [✅ `reme/components/file_chunker/`（base/default/chunked/linked 四种）]
  - file store → [✅ `reme/components/file_store/local_file_store.py`]
  - file graph → [✅ `reme/components/file_graph/`（local/nx/neo4j）]
  - file watcher → [✅ `reme/components/file_watcher/lite_file_watcher.py` 基于 watchfiles awatch；`base_file_watcher.py` 抽象接口]
  - 增量倒排索引 → [✅ `reme/components/keyword_index/bm25_index.py`（增量 BM25）；`reme/components/tokenizer/`（regex / jieba 两种 tokenizer，jieba 含 stopwords 子目录）]
- 未来可以使用rust/c++重写，高性能本地知识引擎
  - [📌 规划]

## 知识库应用场景（重点）

### 金融
产业链
   - [❌ 待补充] 没有领域 schema / 产业链知识图谱样例，需要写 demo 数据集 + topic 配置。

### 自己的工作&生活
xxxx
   - [❌ 待补充] 文档本身就是占位，需要补充具体场景描述与对应的 daily/knowledge 目录样例。

---

## 标注小结

### ✅ 已经在 `reme/` 中实现的能力
1. **组件框架**：backend 注册（`component_registry.py`）、生命周期（`base_component.py`）、依赖声明 + 拓扑启动（`application.py:80`）。
2. **Job/Step 体系**：`components/job/base_job.py`、`components/job/stream_job.py`、`steps/base_step.py` 与 `steps/common/*`。
3. **服务/客户端**：HTTP（`service/http_service.py` + `client/http_client.py`）、MCP（`service/mcp_service.py` + `client/mcp_client.py`），CLI 入口 `reme.py:main`。
4. **Markdown 解析**：`file_chunker/markdown_file_chunker.py`，含 frontmatter、wikilink + Dataview 谓词、AST 树、带全标题骨架的 chunking。
5. **Graph**：`file_graph/{local,nx,neo4j}_file_graph.py`，双向链接索引。
6. **存储 / 索引**：`file_store/local_file_store.py` + `keyword_index/bm25_index.py`（增量 BM25）+ `tokenizer/{regex,jieba}_tokenizer.py`。
7. **文件监听**：`file_watcher/lite_file_watcher.py`（watchfiles 轮询）。
8. **混合检索 + 一跳展开**：`steps/common/search.py`（vector + keyword RRF 融合，可附 outlinks/inlinks）。
9. **Embedding / LLM 适配壳**：`components/embedding/openai_embedding_model.py`、`components/as_llm/`、`components/as_llm_formatter/`、`components/as_token_counter/`。

### ❌ 需要额外补充的能力
1. **记忆类型分层**：个性化 / 程序化 / 知识类的 schema 与路由。
2. **memory-self-evolving**：auto-memory（拆事件→ daily/YYYYMMDD/{event}.md）、auto-dream（空闲整理→ knowledge/{topic}/）、对应触发器与调度。
3. **存储目录约定**：`resource/`、`proactive/` 目录、daily 主索引协议、knowledge topic 枚举均未落地。
4. **auto-link 后台任务**：自动从正文挖出实体并写回 wikilink。
5. **多跳渐进展开检索 API**：当前只能一跳。
6. **标准 Markdown `[text](url.md)` 链接**：尚未纳入 graph 边解析。
7. **skill.md 模板**：CLI 与 MCP 两种集成方式都缺 skill 描述文件。
8. **Agent 拉起后台 reme 服务**：缺脚本/约定。
9. **qwenpaw / AgentScope SDK 集成**：包括 `AgentscopeTools` 包装层与 auto-memory/dream/search 暴露。
10. **应用场景样例**：金融产业链、个人工作&生活的 demo 数据 + topic 配置。
