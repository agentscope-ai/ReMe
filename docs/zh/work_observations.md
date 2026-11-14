# 工作观察

[`proactive.yaml`](../../reme/config/proactive.yaml) 默认每五分钟采集一次多屏截图和本机 Codex 新增对话，
通过一次 `append_event(agent_id=work, events=[...])` 投递给工作 Agent。采集器不调用模型。

先启动服务：

```bash
reme start config=proactive
```

在另一个终端查看工作 Agent：

```bash
reme manage_proactive_agent action=status agent_id=work
```

`work_observation_init` 在启动时建立读取基线；`work_observation_cron` 的周期为 `*/5 * * * *`。
原有 `screen_capture_cron` 保留为关闭的独立截图入口，避免默认重复截图。关闭工作观察时，将
`work_observation_init.enabled` 和 `work_observation_cron.enabled` 都设为 `false`。

## 输入与状态

- Codex 来源默认为 `${CODEX_HOME:-~/.codex}/sessions`，可以在这两个 Job 的 Step 配置中修改 `codex_home`。
  这是只读外部来源，与 ReMe 自身的工作区相对 `codex_app_server.codex_home` 独立。
- 首次启用跳过已有完整记录；重启保留读取位置，补读停机期间的新增内容。
- 当前解析 Codex rollout 的 `event_msg/item_completed`：保留真实 `UserMessage` 和
  `AgentMessage.phase=final_answer`。推理、工具输出、过程说明、原始上下文注入和 ReMe 自身会话不进入对话资源。
- 对话保留原始时间、项目 `cwd`、会话、轮次、消息 ID 和正文。图片附件只记录数量，不复制原图或临时路径。
- 扫描、文件读取、截图和解析在线程中执行；读取位置使用字节偏移，半行留待下一次处理。
  `max_scan_bytes` 默认为每轮 32 MiB，允许一条不超过既有 20 MiB 文件上限的完整记录越过扫描预算。
  大量积压分轮处理，并轮换文件顺序。此方式针对追加日志，不保证检测任意位置的历史原地改写。

文件均位于配置的工作区内：

```text
resource/screenshot/<date>/<batch>/metadata.json、1.png、2.png…
resource/codex_dialogue/<date>/<batch>/metadata.json、dialogue.jsonl
metadata/work_observations/work/baseline.json
metadata/work_observations/work/state.json
metadata/work_observations/work/history.jsonl
metadata/work_observations/work/pending/     # 仅未完成批次存在
```

完整批次先暂存，再发布资源、投递事件、提交游标。失败后先重试原批次，保留相同事件 ID、时间和路径；
某个来源失败时，其他成功来源仍可投递。恢复读取索引时，只需移除可重建的 `state.json`，保留
`baseline.json`、`history.jsonl`、未完成批次和资源文件。损坏的状态会明确报错，不会重置读取位置。

截图需要 `reme-ai[core]` 的 `mss` 及操作系统屏幕录制权限。工作 Agent 按自己的 `MEMORY.md` 分析输入；
对话中尚未完成的请求不代表任务已完成，截图表示采集时的历史观察。
