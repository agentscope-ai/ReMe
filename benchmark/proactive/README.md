# Remy Agent 评测

评测使用当前对话 Agent、定时任务 Agent 和主动 Agent。定时任务 Agent 通过真实 Job 执行 query、发布累计摘要并准备内部交接，
再准备主对话摘要，运行同一会话七轮 query 的主动流程（合并读信息、任务结果、消息历史、调查、消息决策、配置、睡前 TODO），
使用真实结果消费、对话游标、提醒状态和交接文件；`proactive_todo.md` 在下轮注入上下文。
本地评测关闭联网搜索，不验证资料检索阶段的实际搜索与原文核查能力。
定时触发在评测中模拟，独立 Cron 调度由隔离单元测试覆盖。资料、对话会话和时钟为可控输入；
截图、财经采集返回合成资料，不访问真实屏幕或财经服务。
测试事实（如“37 项本地测试通过”）均为合成案例内容，不代表本仓库的真实验证结果。

## 执行

需要明确配置 `LLM_API_KEY`、`LLM_BASE_URL`。命令会调用真实模型并产生费用。默认使用 proactive 配置中的模型。
输出目录必须位于仓库外，源码、案例和 Prompt 哈希写入 manifest；源码变化时需要新目录。

```bash
python -m benchmark.proactive.run --root /tmp/remy-eval --repeats 1
python -m benchmark.proactive.run --root /tmp/remy-eval --cases proactive_known,proactive_timely
python -m benchmark.proactive.reminder_quality --root /tmp/remy-quality
python -m benchmark.proactive.analyze /tmp/remy-eval
```

同一 manifest 下重复执行只跳过已有结果，不覆盖失败结果；独立复测使用新目录。每个案例保留 Job 参数与结果、模型工具事件、
实际投递、工作区前后状态和耗时；统计保留每轮用量以及原生缓存、推理 token 的差量。当前只有 `current` Prompt 组，
不沿用历史双 Agent 的 Prompt 排名。

## 判定范围

- 对话 Agent：更新偏好、保留无关事实、执行遗忘且不转存。
- 主动 Agent：消费新增对话、反馈和未读任务摘要、判断新增价值、避免重复与不必要的打扰。
- 空内容提醒是明确的不发送决定，不算非空发送尝试；被拒绝的非空调用仍纳入检查。
- 未完成流程、工具错误、缺少必要工具调用不能因为没有投递就得到正确静默的评分。
- 注入指令夹带、事实来源与验证范围单独检查。

固定会话的注入在模型边界模拟，后台十秒空闲等待、取消、过期和重启恢复由隔离单元覆盖；本评测不宣称验证真实桌面投递。
历史输出无需重写；旧评测快照应使用原来的代码快照读取，不与当前结果混用。

`reminder_quality` 单独测试定时任务的紧急、普通、重复和期限变更，一次性提醒及结果交付，以及智能主动的新结果与重复结果。
使用执行当天的统一时钟，保存 Prompt、源码哈希、实际消息与工具顺序；`--cases cron_urgent,temporary_due` 可选场景。
它在隔离 Main 中模拟已空闲后补投递，不访问真实邮件、屏幕或用户会话，不验证桌面展示。
`report.md` 展示实际文案供人工检查；自动检查流程、关键事实和长度，不把它等同于自然度评分。
