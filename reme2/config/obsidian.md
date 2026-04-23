# all @jinli

reme help
reme start
reme restart
reme version
reme vault="My Vault"

# daily

daily:path
reme daily:xxx

# crud

reme create file="New Note" content="# Hello" title="xxx" tags="[]" status=""
reme read file=Recipe/path="Templates/Recipe.md"
reme edit file=Recipe/path="Templates/Recipe.md" old="xxx" new="xxx"
reme append file="My Note" content="New line"
reme prepend file="My Note" content="New line"
reme delete file="My Note"/path

# reme

reme stat file/path
reme list file/path

# search

reme search query="search term" limit=10 tag="[]" score=0.1 copy=true

# property

reme property:read
reme property:update file="My Note" status=done xx=xxx
reme property:delete keys="[xxxx, xxxx]"

# 全局所有标签

reme tags

# show link

reme backlinks file="My Note"
reme links file="My Note"

[[Algorithm Notes#Sorting]]

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





