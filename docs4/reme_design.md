# 基础Job
@jinli
| 分类     | 能力      | 参数                                                        |
|--------|---------|-----------------------------------------------------------|
| 通用     | help    |                                                           |
| 通用     | start   |   支持后台 |
| 通用     | restart |                                                           |
| 通用     | version |                                                           |
| 通用     | reindex |                                                           |
| search | search  | query="search term" limit=10 tag="[]" score=0.1 copy=true |

@sen
| tags | stat |  返回特定tag信息 |
| tags | list |  返回所有tag列表 |
| crud | upload/download | 其他文件                                                   |
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
    ⅱ. chunks 	dict[str, list[chunk]]
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
