- 增加agent 的component
- 增加全局 时区time_zone，全局作用


数据结构
- resource
  - YYYYMMDD
    - xxxx.html
    - xxxx.txt
    - xxxx.md
- dialog
  - YYYYMMDD
   - session_{session_id}.jsonl msg->dict格式
- daily
  - YYYYMMDD.md 索引
  - YYYYMMDD
    - session_{session_id}.md 日志本
    - resource_{resource_id}.md
- digest
  - personal 个性化信息
    - xxx.md
  - procedural 程序化记忆
    - xxx.md
  - wiki 知识化记忆
    - xxx.md


监控任务【后台】：
- 构建bm25+emb的index【5s】
  - 目录：daily+digest 的 所有md，使用markdown的ast解析
  - 可选：jsonl，要求保存的时候对工具结果截断，使用rag方案解析
- daily md监控：【1min】
  - 暂无
- digest md监控：【1min】
  - 暂无
- resource 监控：【间隔5min】
  - 针对每一个文件，生成一个hashid，作为session_id
  - 生成一个agent，解读文件（读前1MB内容）防止上下文炸了
  - 把抽取的内容写到daily/YYYYMMDD/resource_{resource_id}.md
  - 调用推送工具：
    - qwenpaw推送收件箱/agent
    - cc可以直接推送agent @sen


hook任务：
- auto-memory：在上下文满/每隔多少轮/session_end
    - qwenpaw：直接传message session_id date，直接append session_agent_{session_id}.jsonl @jinli
    - cc：给出新的path，对比我们保存的session_{session_id}.jsonl，看到增量msg，接着解析path的内容，保存到session_{session_id}.jsonl @sen
    - 注意保存的时候截断工具调用结构，防止太长


定时任务：
- auto-dream：每天夜晚触发
  - 记忆整理：按照session/resource_id去做for循环整理：把YYYYMMDD.md + YYYYMMDD/* 消化更新到 digest下的personal/procedural/wiki
  - 记忆link：markdown之间link，digest内部链接，可以链接到外面。
