- reme_session/
  - agentscope|claude_code / # 使用内置的agent wrapper，session会保存在这里
    {session_id}.jsonl UUID格式要求  # /Users/yuli/workspace/ReMe/reme/components/agent_wrapper
  - dialog/
    {session_id}.jsonl  # auto memory保存  可以监控可以被检索【可选】
- resource/
  - YYYY-MM-DD/
    - {channel}_{xxxx}.html
    - {channel}_{xxxx}.md
- daily/【日记，浅加工】
  - YYYY-MM-DD.md
  - YYYY-MM-DD/
    - {session_id}.md
    - {和resource同名}.md
- digest/
  - personal/
  - procedure/
  - wiki/

函数接口：
- auto_memory
  - message 应该会 会保存到 reme_session/dialog/{session_id}.jsonl
  - 通过 message 更新 daily/YYYY-MM-DD/{session_id}.md
- auto-resource
  - 会保存到 daily/YYYY-MM-DD/{resource_stem}.md
- auto-dream
  - 读取所有的md
  - 会生成link auto-link
  - 会生成topic ？
- proactive
  - 会读取topic ？
- search


后台任务：
- index_update_loop 索引监控
- resource_watch_loop 资源监控
- digest_watch_loop 应该是闲置？
