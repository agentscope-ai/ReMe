# 快速测试

```bash
# 终端 A：启动服务
reme start

# 终端 B：调用 version 验证服务可用
reme version
# 预期输出：✅ ReMe v{__version__}
```

# 基础Job

@jinli

入口：`reme/reme.py::main()` → `parse_args(*sys.argv[1:])` 解析首个位置参数为 `action`，后续 `key=value` 解析为 kwargs（支持
`service.port=8080` 的 dot notation；自动剥离 `--` / `-` 前缀；值会做 bool / int / float / JSON 转换）。

调用模式：

- `start`：本地启动 `ReMe(Application)` 服务（不经过 client）
- `find_reme`：本地探测正在运行的 reme，不调用服务
- `list`：在 client 端拦截，不转发到服务端，直接返回 action 目录
- 其他 action：通过 `call_server(action, **kwargs)` → `R.get(ComponentEnum.CLIENT, backend)` 实例化客户端并流式打印（任意未列出的
  step register name 都按本规则透传）

通用可选参数 `backend:str=http`（取值 `http` / `mcp`，对应 `reme/components/client/{http_client,mcp_client}.py` 中
`@R.register` 注册名）；服务端默认 host/port 见 `reme/constants.py`，可由 `start` 端通过 `service.host=` / `service.port=`
覆盖。

说明：📥 输入参数 ｜ 📤 输出 ｜ ⭐ 必填 ｜ 🎚️ 默认值 ｜ 🛠️ 内部行为 ｜ 📊 metadata

| 分类         | 指令 (register name)                               | 入口                                                          | 参数 & 行为                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|------------|--------------------------------------------------|-------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 🚀 本地      | 🟢 `start`                                       | `reme.py:30` → `ReMe(**kwargs).run_app()`                   | 📥 可选 `config=<name\|path>`（默认加载 `reme/config/default.yaml`，`.yaml/.yml/.json` 都支持，含 `${ENV:-default}` 占位符）｜ 可选 `service.host=` / `service.port=` 等任意 dot-notation 覆盖 ｜ 🛠️ 流程：`load_env()` → `resolve_app_config(**kwargs)` deep merge → `precheck_start(svc)`（`utils/service_utils.py:72`：目标 host:port 已有 reme → 打印 `reme already running ...` 直接返回；端口被其他进程占用 → stderr 提示 `port {port} occupied. Start on another port: reme start service.port=<other_port>` 并 `sys.exit(1)`）→ 启动服务                                                                                                                                                                                        |
| 🚀 本地      | 🧭 `find_reme`                                   | `reme.py:36` → `utils/service_utils.py:89`                  | 📥 无 ｜ 📤 发现服务则 stdout 打印 `HOST={host} PORT={port} PID={pid or 'unknown'}`；未发现则 stderr 提示 `reme not started. Try: reme start` 并 `sys.exit(1)` ｜ 🛠️ 流程：先探 `REME_DEFAULT_HOST:REME_DEFAULT_PORT`（`health_check` 命中算 `reme`），再 `pgrep -af "reme.* start"` 扫描其他端口                                                                                                                                                                                                                                                                                                                                                                                                                  |
| 🛰️ 客户端    | 📜 `list`                                        | `components/client/base_client.py:36`                       | 📥 无 ｜ 📤 服务端可用 action 目录（JSON，`indent=2 ensure_ascii=False`）｜ 🛠️ 在 `BaseClient.__call__` 中拦截，不进入 `_execute`，直接调用 `list_actions()`（HTTP/MCP backend 各自实现）                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 🌐 通用 step | 🆘 `help` (`help_step`)                          | `call_server("help")`                                       | 📥 无 ｜ 📤 `answer` 一行一个 job：`🛠️ \`{name}\` — {description} 📥 {params}`，参数渲染为 `name:type*`(必填) / `name:type={default}` / `name:type` ｜ 📊 `metadata.job_count` ｜ 🛠️ 自动跳过名为 `help` 的 job                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 🌐 通用 step | 🩺 `health_check` (`health_check_step`)          | `call_server("health_check")`                               | 📥 无 ｜ 📤 `answer = "✅/❌ ReMe v{version} - healthy/unhealthy"` ｜ 📊 `metadata.health = {version, healthy, components}` ｜ 🧩 覆盖组件：`embedding_model`(🟢 is_started/is_healthy/model_name/dimensions/cache_size/memory) · `file_graph`(🕸️ n_nodes/n_edges/n_virtual\|n_pending/memory) · `file_store`(📦 n_chunks/n_chunks_with_embedding/memory) · `file_watcher`(👀 background_running/watch_paths) · `keyword_index`(🔤 n_docs/vocab_size/memory) ｜ 🛠️ deep sizeof（含 numpy.nbytes），未启动 / 后台未跑 / embedding 不健康 → ❌                                                                                                                                                             |
| 🌐 通用 step | 🏷️ `version` (`version_step`)                   | `call_server("version")`                                    | 📥 无 ｜ 📤 `answer = reme.__version__` ｜ 📊 `metadata.version`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| 🌐 通用 step | 🔄 `reindex` (`reindex_step`)                    | `call_server("reindex")`                                    | 📥 无 ｜ 📤 `answer = "🔄 Reindexed {added} file(s)"` ｜ 📊 `metadata.counts = {added, ...}` ｜ 🛠️ 流程：`file_watcher.close()` → `file_store.clear()` → `file_watcher.update_store()` → `file_watcher.start()`（finally 保证重启）                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 🔎 search  | 🔍 `search` (`search_step`)                      | `call_server("search", query=…, …)`                         | 📥 `query:str` ⭐ ｜ 🎚️ `limit:int=5`(>0) ｜ 🎚️ `min_score:float=0.0` ｜ ⚖️ `vector_weight:float=0.7` ∈[0,1]（keyword 权 = 1-vw）｜ 🔀 `candidate_multiplier:float=3.0`（candidates = min(200, limit×mult)）｜ 🔗 `expand_links:bool=True` ｜ 🔢 `max_links_per_direction:int=10` ｜ 🎚️ `search_filter:dict={}` ｜ 📤 `answer` 每命中一行 `path:start-end [score=… vector=… keyword=…] text` + 缩进的 `→ outlinks (n)` / `← inlinks (n)` + `via predicate=… anchor=#…` ｜ 📊 `metadata.results` / `metadata.link_expansion` / `metadata.counts={vector,keyword,returned,hybrid}` ｜ 🛠️ 并行 `vector_search` + `keyword_search` → RRF 融合（K=60，按 chunk.id 合并）→ `min_score` 过滤 → `limit` 截断 → 邻居 meta 注入 |
| 🧪 demo    | 🪄 `demo_echo` (`demo_echo_step1` + `step2`)     | `call_server("demo_echo", query=…, min_score=…)`            | 📥 `query:str=""` ｜ 🎚️ `min_score:float=0.5` ｜ 🛠️ step1：`processed_query = query.strip().lower()`，`adjusted_min_score = min_score * 0.9`，写回 context ｜ 📤 step2：`answer = "echo: {processed_query} (min_score={adjusted_min_score})"` ｜ 📊 `metadata = {step, query, min_score, processed_query, adjusted_min_score}`                                                                                                                                                                                                                                                                                                                                                          |
| 🌊 demo    | 🌊 `stream_demo` (`stream_demo_step1` + `step2`) | `call_server("stream_demo", query=…, repeat=…, interval=…)` | 📥 `query:str=""` ｜ 🎚️ `repeat:int=10` ｜ 🎚️ `interval:float=0.1`（秒/字符）｜ 🛠️ step1：`stream_text = query * repeat` 写回 context ｜ 📤 step2：按字符 `add_stream_string(ch, ChunkEnum.CONTENT)` 流式输出，`asyncio.sleep(interval)` 节流                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 📂 crud    | 📖 `read` (`read_step`)                          | `call_server("read", path=…, …)`                            | 📥 `path:str` ⭐（**完整相对路径**，相对于 workspace；绝对路径会被拒绝；非 `.md` 后缀拒绝）｜ 🎚️ `start_line:int=null`（1-based, 含端点）｜ 🎚️ `end_line:int=null`（1-based, 含端点）｜ 🎚️ `max_bytes:int=51200`（截断阈值）｜ 📤 `answer = 选中的行内容`，超过 `max_bytes` 时附加 `--- TRUNCATED ---` 续读指引（`start_line=…`）｜ 📊 `metadata.path` / `metadata.total_lines`（出错路径才会附带）｜ 🛠️ 流程：`BaseStep.resolve_path(raw, require_md=True)` → `aiofiles.os.stat` → `read_file_safe`（utf-8-sig BOM 容忍、UnicodeDecodeError fallback `errors=ignore`）→ `split("\n")` 切片 `[s-1:e]` → `truncate_text_output` 按字节截断保行                                                                                              |

使用示例：

```bash
# 启动（默认 default.yaml）
reme start

# 指定 config 与服务端口
reme start config=paw.yaml service.port=8181

# 查找在跑的 reme
reme find_reme
# HOST=127.0.0.1 PORT=8000 PID=12345

# 列出所有可用 action（client 端处理，不转服务端）
reme list

# 转发到服务端的 step：所有 key=value 透传为 step kwargs
reme help
reme health_check
reme version
reme reindex
reme search query="latency 问题" limit=10 min_score=0.2 vector_weight=0.6

# 读取 workspace 下的 markdown（完整相对路径；无后缀自动补 .md；可按行切片或限制字节）
reme read path=Templates/Recipe.md
reme read path=Notes start_line=1 end_line=20
reme read path=Big.md max_bytes=4096

# 通过 MCP backend 调用
reme search query="..." backend=mcp
```

@sen
| file | upload/download/move/delete/stat/list | 文件操作CRUD |
| property | read/update/delete | frontmatter CRUD |                                           |
| graph | traverse/retarget | path="My Note"  directtion=forward/backward depth=1 predicat=xxx |

@wangce
| crud | write | path="New Note" name="xxx" description="xxx" metadata={}, content="# Hello" (4 字段都必填，frontmatter 只写 name/description) |
| crud | read | path="Templates/Recipe.md"                                        |
| crud | edit | path="Templates/Recipe.md" old="xxx" new="xxx"                    |
| crud | append | path="My Note" content="New line"                                 |

| crud | delete | path="My Note
| daily_crud | daily_xxx | 与 crud 参数保持一致 |

- daily_resolve name=xxxx (符合一定规范 win下要求)
- daily_list date=xxxx 返回path
- daily_index

frontmatter read path
frontmatter update path metadata={}
frontmatter delete path keys=[]

delete path
download path=xxx（内部相对路径）download_path=(外部绝对路径，可选)
upload path=xxx（外部绝对路径）description="xxx" metadata=xxx 返回内部相对路径 加metadata
stat path
list path
mv path=xxx new_path=xxx

traverse path=xxx direction=xxx depth=xxx

# 日记类型

| 类型        | 路径                                            | 说明                          |
|-----------|-----------------------------------------------|-----------------------------|
| daily     | {daily}/xxxx-mm-dd.md + xxxx-mm-dd/{event}.md | 按日期归档的原始信息记录                |
| topic     | topic/{topic:-personal(agent)}/{xxxx}.md      | 按主题聚类的二次加工内容                |
| proactive | todo                                          | 基于 daily / topic 思考后主动推送的消息 |

# 生成Job

| 任务                      | 输入            | 输出                                            | 触发时机                        | 说明                                                   |
|-------------------------|---------------|-----------------------------------------------|-----------------------------|------------------------------------------------------|
| 日记summary @sen @wangce  | msg           | {daily}/xxxx-mm-dd.md + xxxx-mm-dd/{event}.md | freq (every_n_turn、compact) | 把 msg 的信息写入 daily 目录                                 |
| 主题dream  + 生成链接 @sen    | daily/xxx     | knowledge/xxx                                 | /dream                      | 把 daily 目录的内容按主题聚类合并到 topic 目录, 主动在文档中建立 [[link]] 关联 |
| 主动proactive     @wangce | daily / topic | proactive_query                               | pre_query                   | 思考 daily / topic 信息，主动决定推送给用户的消息                     |

2. file_chunker
   a. 抽象基类 parse: @jinli
   ⅰ. 输入是path：相对路径
   ⅱ. 输出是FileMetadata & list[FileChunks] & list[FileEdge]
   b. default parser 兼容老方案 @jinli
   ⅰ. 带overlap的chunking策略 ，不输出FileEdge
   c. markdown parser @sen
   ⅰ. 根据markdown ast做chunk，不需要overlap
   ⅱ. 增加一个索引的chunk chunk_type @锦鲤 file_chunk_type content/index
   ⅲ. 增加link的正则解析：predicate:: [[path#anchor]]
3. file_store @sen
   a. 抽象存储：
   ⅰ. filenode = file + path + st_mtime + metadata + list[FileEdge]
   ⅱ. graph=dict[str, filenode] 内存+json
   ⅲ. list[FileChunk] 存db
   b. 抽象基类
   ⅰ. graph：fellow dict的操作 update/get/set
   ⅱ. chunks dict[str, list[chunk]]
    1. delete_chunks_by_path
    2. update_chunks_by_path
    3. list_chunks_by_path
    4. vector_search/keyword_search
       ⅲ. 手写一个bm25检索
       ⅳ. 【核心】检索机制 vector bm25 graph 如何进行融合
4. file_watcher @jinli
   a. 抽象基类
   ⅰ. on_start:
    1. file_store 的start 在前，加载graph，file_watcher在后，递归扫描目录
       a. 通过ms_time对比graph，on_change 进行改动
       ⅱ. on_change:
    1. 更新/增加:
       a. delete_chunks_by_path 更新数据库
       b. upate_chunks_by_path 更新数据库
       c. 更新graph
    2. 删除
       a. delete_chunks_by_path 更新数据库

MemorySchema

1. markdown文件结构 @sen
   a. formatter：
   ⅰ. name
   ⅱ. desc
2. memory文件结构目录
   a. MEMORY.md
   b. msg/files -> daily/YYYYMMDD/YYYYMMDD.md + xxxx.md
   ⅰ. YYYYMMDD.md
    1. xxx -> xxxx.md
    2. xxx -> xxxd.md
       ⅱ.
       c. daily -> topic/topic_l1/topic_l1.md + xxx.md + topic_l2
       d. proactive

steps:

1. 治理（算法+LLM）：
   a. 节点关联P0：现有的链接做补充，挖掘新的LLM的link
   ⅰ. /Users/yuli/workspace/ReMe/reme2/component/edge_extractor/llm_edge_extractor.py
   ⅱ. 移动到steps
   b. 节点整合/节点拆分/节点归档
   c. 健康度检查
2. retrieve 调用store的检索
3. 原子steps：reme edit
4. 组合steps：总结：
   a. - freq (every_n_turn、compact) -> daily_summarizer
   b. topic (/dream ) -> topic_summarizer(daily_xx -> topic_xx)
   c. proactive -> proactive_summarizer(personal_xxx -> proactive_query - pre_query
