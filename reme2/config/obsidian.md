# 基础Job
@jinli
| 分类     | 能力      | 参数                                                        |
|--------|---------|-----------------------------------------------------------|
| 通用     | help    |                                                           |
| 通用     | start   |                                                           |
| 通用     | restart |                                                           |
| 通用     | version |                                                           |
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





