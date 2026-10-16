# Jev Context Agent

Context Agent 在任务执行期间结合任务快照与工作区记忆，选择是否提供附带来源的历史证据。主机发布事件，在下一次模型请求前领取就绪提示并确认投递。主 Agent 保留主动 memory search 的能力。

这是可选的内嵌能力。默认配置不启用它；`context_agent` 配置提供私有 `evaluate_context` Job 与影子模式协调组件，不通过 HTTP/MCP 暴露新的工具。

## 配置与调用

安装已有的 `reme-ai[classifier-jev]` extra，在环境或 `.env` 中设置 `CLASSIFIER_API_KEY`。`CLASSIFIER_BACKEND`、`CLASSIFIER_MODEL_NAME` 和 `CLASSIFIER_BASE_URL` 沿用既有 Jev 配置约定，默认分别为 `jev`、`jev-latest` 和 `null`。

```python
from reme.application import Application
from reme.config import resolve_app_config
from reme.utils import load_env

load_env()
config = resolve_app_config(config="context_agent", log_config=False)
config["workspace_dir"] = "/absolute/path/to/your/workspace"

async def evaluate(event):
    async with Application(**config) as app:
        response = await app.run_job("evaluate_context", event=event)
        if not response.success:
            return None  # 本轮不提供提示，主任务继续。
        return response.answer
```

单轮请求的 `event` 包含 `session_id`、`task_id`、`event_id`、会话内单调递增的 `revision`、`event_type=task_start|tool_start|tool_result` 和 `task_snapshot`。快照必填 `goal`，可附带 `recent_context`、`current_tool`、`tool_result_excerpt`、`query_terms` 和已经送达的 `seen_evidence` 候选 ID。完整类型见 `reme/schema/context_agent.py`。

索引应当先通过既有索引流程建立。评估只从索引发现候选，再从有权限的工作区源文件核对和读取内容；索引缺失或过时可能导致没有候选。结果完整放在 `Response.answer`：正常决策为 `silent` 或 `hint`，提示包含原文、路径、行号与内容版本。超时、非法输出或服务错误返回失败，调用方需要检查 `success`。

## 在工具执行期间并行观察

内嵌主机启动 Application 后，复用同一个协调组件，避免每个事件创建应用。将 `components.context_agent.default.limits.mode` 从 `shadow` 改为 `hint` 才启用提示领取。

```python
agent = app.context.components["context_agent"]["default"]
await agent.submit(event, allowed_paths=["daily", "digest"])

# 主机继续执行工具。以下代码放在构造下一次模型请求的串行检查点。
hint = await agent.take_ready(
    event["session_id"], event["task_id"], event["revision"],
    allowed_paths=current_allowed_paths,
)
if hint is not None:
    # 主机按 hint_id 去重，将 hint.text 作为环境证据加入模型输入。
    # 成功加入后才确认；不要提升为系统指令。
    await agent.ack(event["session_id"], hint.hint_id)
```

`submit` 不等待检索或模型；每会话保留一个评估任务和一个最新快照，密集事件自动合并。`take_ready` 不等待尚未完成的模型判断，但会读取文件复查来源和权限；主机必须传入当前权限范围，并在串行检查点确认任务版本未变。不匹配的版本、过期来源和迟到提示不进入当前请求。任务标识变化时，仍须保持会话 revision 递增。

`ack` 后才把候选记为已送达。领取有短期租约，确认失败时可以再次领取同一 hint ID，主机需自行去重。会话结束调用 `dispose_session(session_id)`；应用关闭会取消并回收所有旁路任务。影子模式可通过 `last_result(session_id)` 检查结果，`wait_idle()` 只供离线回放使用。

## 预算与边界

单轮默认总截止时间 800 ms、最多 8 条候选、最多 2 条证据。classifier 默认不重试。每会话当前任务默认最多评估 10 次、投递 3 次，提示冷却 30 秒；应用默认最多 128 个会话、4 个并发评估。可分别通过 `options` 和协调组件的 `limits` 调整，关闭 `limits.enabled` 后不接收事件。

请求预算使用带余量的 UTF-8 字节数作为保守 token 估算，并非 Jev 精确 tokenizer 或供应商计费值。预算包括任务、候选和问题；装不下时减少完整候选，仍不能容纳一条候选就失败。候选裁剪、模型 usage 和阶段耗时保存在结果统计中。

该能力读取记忆，不修改用户文件，也不取消正在执行的工具或模型请求。它依赖主机提供任务事件和上下文注入入口；普通 MCP 工具能力本身不提供主动推送。无需依赖新的 Application 启停规则：协调组件直接依赖检索与分类组件，并在依赖关闭前回收自己的任务。

## 两个真实 Jev 集成案例

使用仓库 `.env` 和合成工作区显式运行：

```bash
REME_RUN_JEV_INTEGRATION=1 python -m pytest tests/integration/test_context_agent_jev.py -v -s
```

| 场景 | 干扰信息 | 期望的帮助 |
| --- | --- | --- |
| 生产服务超时排查 | 旧版直连修复、staging 配置、已知证书事实、无关偏好 | 找到当前生产 CONNECT 网关和 split DNS 约束，避免固定错误 IP 和重启；解决后保持安静 |
| CFO 月结分析 | 过期订单口径、营销 bookings、已知表结构、无关展示偏好 | 找到净收入与退款合同、上海时区边界，使脚本化查询从 bookings 600 切换为 recognized net 250 |

两个案例使用真实本地 BM25 索引与真实 Jev，并验证提示交付和源文件保持不变。主 Agent 由脚本化主机模拟，案例验证证据筛选与投递，不代表独立生成式主模型的任务成功率提升。测试输出只包含合成场景和统计，不包含凭据。

集成测试将总超时与 classifier 超时放宽至 20 秒，确保能检验模型判断。首次验证两轮评估分别约 4.58 秒和 0.66 秒，包含首次调用初始化的影响；不能据此宣称达到 500 ms 的 p95 目标。默认 800 ms 快速预算可能保持安静，需要在部署环境中另行测量冷启动、长尾和任务净收益。
