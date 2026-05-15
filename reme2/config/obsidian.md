# 通用

| 能力      | 参数 |
|---------|----|
| help    |    |
| start   |    |
| restart |    |
| version |    |
| tags    |    |

# crud

| 能力      | 参数                                                                |
|---------|-------------------------------------------------------------------|
| create  | path="New Note" content="# Hello" title="xxx" tags="[]" status="" |
| read    | path="Templates/Recipe.md"                                        |
| edit    | path="Templates/Recipe.md" old="xxx" new="xxx"                    |
| append  | path="My Note" content="New line"                                 |
| prepend | path="My Note" content="New line"                                 |
| delete  | path="My Note"                                                    |

# daily:crud

| 能力        | 参数            |
|-----------|---------------|
| daily:xxx | 与 crud 参数保持一致 |

# stat / list

| 能力   | 参数   |
|------|------|
| stat | path |
| list | path |

# search

| 能力     | 参数                                                        |
|--------|-----------------------------------------------------------|
| search | query="search term" limit=10 tag="[]" score=0.1 copy=true |

# property

| 能力              | 参数                                |
|-----------------|-----------------------------------|
| property:read   |                                   |
| property:update | path="My Note" status=done xx=xxx |
| property:delete | keys="[xxxx, xxxx]"               |

# link

| 能力        | 参数             |
|-----------|----------------|
| backlinks | path="My Note" |
| links     | path="My Note" |




记忆的类型

1. daily -> daily/xxxx-mm-dd/overview.md -> xxxx.md
2. topic -> topic/personal(agent)/xxxx.md

记忆的被动总结

1. 是否需要：是需要，不会主动触发
2. 什么时候调用：

- freq (every_n_turn、compact) -> daily_summarizer
- topic (/dream ) -> topic_summarizer(daily_xx -> topic_xx)
- proactive -> proactive_summarizer(personal_xxx -> proactive_query)
    - pre_query

记忆的搜索
file watch

- start 全量扫描 -> 文件变化结果
- 增量扫描 -> 文件变化结果

如果有变化，检测 增加、删除、修改 cud：

1. file -> metadata 存json

- path
- mtime
- header: tags/kv/title/desc

2. file -> link 存json

- 读全量修改的文件的全量，识别link

3. file -> chunk 存db/json

- 切片 -> file+start+end: preview(100)





