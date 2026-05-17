# 基础Job

@jinli
说明：📥 输入参数 ｜ 📤 输出 ｜ ⭐ 必填 ｜ 🎚️ 默认值 ｜ 🛠️ 内部行为

| 分类        | 能力 (register name)                               | 参数 & 行为                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|-----------|--------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 🌐 通用     | 🆘 `help` (`help_step`)                          | 📥 无 ｜ 📤 `answer` 一行一个 job：`🛠️ \`{name}\` — {description} 📥 {params}`，参数渲染为 `name:type*`(必填) / `name:type={default}` / `name:type` ｜ 📊 `metadata.job_count` ｜ 🛠️ 自动跳过名为 `help` 的 job                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 🌐 通用     | 🩺 `health_check` (`health_check_step`)          | 📥 无 ｜ 📤 `answer = "✅/❌ ReMe v{version} - healthy/unhealthy"` ｜ 📊 `metadata.health = {version, healthy, components}` ｜ 🧩 覆盖组件：`embedding_model`(🟢 is_started/is_healthy/model_name/dimensions/cache_size/memory) · `file_graph`(🕸️ n_nodes/n_edges/n_virtual\|n_pending/memory) · `file_store`(📦 n_chunks/n_chunks_with_embedding/memory) · `file_watcher`(👀 background_running/watch_paths) · `keyword_index`(🔤 n_docs/vocab_size/memory) ｜ 🛠️ deep sizeof（含 numpy.nbytes），未启动 / 后台未跑 / embedding 不健康 → ❌                                                                                                                                                             |
| 🌐 通用     | 🏷️ `version` (`version_step`)                   | 📥 无 ｜ 📤 `answer = reme4.__version__` ｜ 📊 `metadata.version`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| 🌐 通用     | 🔄 `reindex` (`reindex_step`)                    | 📥 无 ｜ 📤 `answer = "🔄 Reindexed {added} file(s)"` ｜ 📊 `metadata.counts = {added, ...}` ｜ 🛠️ 流程：`file_watcher.close()` → `file_store.clear()` → `file_watcher.update_store()` → `file_watcher.start()`（finally 保证重启）                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 🔎 search | 🔍 `search` (`search_step`)                      | 📥 `query:str` ⭐ ｜ 🎚️ `limit:int=5`(>0) ｜ 🎚️ `min_score:float=0.0` ｜ ⚖️ `vector_weight:float=0.7` ∈[0,1]（keyword 权 = 1-vw）｜ 🔀 `candidate_multiplier:float=3.0`（candidates = min(200, limit×mult)）｜ 🔗 `expand_links:bool=True` ｜ 🔢 `max_links_per_direction:int=10` ｜ 🎚️ `search_filter:dict={}` ｜ 📤 `answer` 每命中一行 `path:start-end [score=… vector=… keyword=…] text` + 缩进的 `→ outlinks (n)` / `← inlinks (n)` + `via predicate=… anchor=#…` ｜ 📊 `metadata.results` / `metadata.link_expansion` / `metadata.counts={vector,keyword,returned,hybrid}` ｜ 🛠️ 并行 `vector_search` + `keyword_search` → RRF 融合（K=60，按 chunk.id 合并）→ `min_score` 过滤 → `limit` 截断 → 邻居 meta 注入 |
| 🧪 demo   | 🪄 `demo_echo` (`demo_echo_step1` + `step2`)     | 📥 `query:str=""` ｜ 🎚️ `min_score:float=0.5` ｜ 🛠️ step1：`processed_query = query.strip().lower()`，`adjusted_min_score = min_score * 0.9`，写回 context ｜ 📤 step2：`answer = "echo: {processed_query} (min_score={adjusted_min_score})"` ｜ 📊 `metadata = {step, query, min_score, processed_query, adjusted_min_score}`                                                                                                                                                                                                                                                                                                                                                          |
| 🌊 demo   | 🌊 `stream_demo` (`stream_demo_step1` + `step2`) | 📥 `query:str=""` ｜ 🎚️ `repeat:int=10` ｜ 🎚️ `interval:float=0.1`（秒/字符）｜ 🛠️ step1：`stream_text = query * repeat` 写回 context ｜ 📤 step2：按字符 `add_stream_string(ch, ChunkEnum.CONTENT)` 流式输出，`asyncio.sleep(interval)` 节流                                                                                                                                                                                                                                                                                                                                                                                                                                                       |

@sen
| tags | stat | 返回特定tag信息 |
| tags | list | 返回所有tag列表 |
| crud | upload/download | 其他文件 |
| file | stat | path |
| file | list | path |
| property | property:read | |
| property | property:update | path="My Note" status=done xx=xxx |
| property | property:delete | keys="[xxxx, xxxx]"                                               |
| graph | traverse | path="My Note"  directtion=forward/backward depth=1 predicat=xxx |

@wangce
| crud | create | path="New Note" content="# Hello" title="xxx" tags="[]" status="" |
| crud | read | path="Templates/Recipe.md"                                        |
| crud | edit | path="Templates/Recipe.md" old="xxx" new="xxx"                    |
| crud | append | path="My Note" content="New line"                                 |
| crud | prepend | path="My Note" content="New line"                                 |
| crud | delete | path="My Note
| daily:crud | daily:xxx | 与 crud 参数保持一致 |

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

2. file_parser
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
   ⅰ. title
   ⅱ. desc
   ⅲ. tags
   ⅳ.
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
