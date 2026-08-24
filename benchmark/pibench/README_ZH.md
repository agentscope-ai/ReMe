# π-Bench 评测说明

[English version](./README.md)

将 **ReMe agent（带持久记忆）** 接入 **π-Bench**（Proactive Personal Assistant
Benchmark）的胶水层评测套件。只含对接所需的最小代码与配置；π-Bench 框架
（`src/`）、评测数据（`data/`）、AppWorld 工具环境、ReMe 本体均为**外部第三方
依赖**，通过符号链接与环境变量原位引用，不随本套件分发。

- π-Bench: https://github.com/Simplified-Reasoning/Pi-Bench （arXiv: 2605.14678）
- ReMe: 你所在 ReMe 仓库的根目录（本套件推荐放在 `ReMe/benchmark/pibench/`）

## 1. 架构总览

```
π-Bench runner (src.main --mode run)
  │  user_agent（模拟用户 LLM）按 data/{persona}/episode.yaml 顺序
  │  逐任务、多轮地与 agent 对话，并在 run 阶段判定隐藏意图(PROC)
  ▼
test server (π-Bench scripts/test_server.py, HTTP 长轮询)
  ▲ /send            │ /poll
  │                  ▼
bridge_reme.py ──────────────► ReMe Application（以库方式内嵌启动）
  │                             ├─ agent_wrapper: 被测 agent（AgentScope）
  │                             ├─ jobs: search / auto_memory / daily_write
  │                             └─ workspace: reme_workspace/{persona}/
  │                                （每 persona 独立持久记忆库，互不可见）
  └──── MCP ────► AppWorld MCP ────► AppWorld API（工具/应用环境）

π-Bench runner (src.main --mode eval)
  judger（裁判 LLM）读取 trace，按 checklist 逐条 YES/NO 打分(COMP)
```

要点：
- bridge 用 **ReMe 自己的 venv python** 运行，把 ReMe 当库用（`resolve_app_config`
  + `Application`），**ReMe 源码零改动**。
- 每条用户消息都会自动触发一次 ReMe memory `search` 并把命中记忆注入当前消息
  （参数见 §8）；任务结束（reset）时会话被 `auto_memory` 提炼为 daily 笔记落盘。
- agent 执行的每一轮工具调用（AppWorld MCP + ReMe job 工具）都会被采集并以
  `tool_steps` 形式写入 trace，供 π-Bench 的 `tools_evaluation_path` 脚本
  对工具行为评分（§7）。
- π-Bench 的 `data/`、`src/`、AppWorld 均不属于本套件，需先装好 π-Bench（§3.1）。

## 2. 目录结构

```
pibench/
├── README.md / README_ZH.md  # 本文档（英文 / 中文）
├── env.sh.example            # 环境配置模板（复制为 env.sh 后填写 TODO 项）
├── bridge_reme.py            # ReMe ↔ test server 桥接（记忆注入/保存、
│                             #   profile 注入、工具调用轨迹采集）
├── run_persona.sh            # 单 persona 全流程（5 个服务 + run + eval）
├── run_all.sh                # 5 个 persona 批跑（fresh/resume，默认 2 并行）
├── resume.py                 # 断点续跑：完成判定 + 中断任务残留记忆的外科清理
├── fix_trace_logs.py         # run 输出 → ~/.nanobot/trace_logs 转换，
│                             #   并把工具轨迹合并进 turn 文件（eval 前置）
├── .gitignore                # 排除 env.sh 与全部运行产物
└── config/
    ├── models/reme.yaml      # runner 模型配置（model_id=reme）
    └── bench/evaluation/trace_history.yaml   # trace 渲染策略（随套件提供，
                                              #   经 --history-config-path 显式传入）
```

运行时自动生成（均被 .gitignore 排除）：`data`（符号链接）、`logs/`、
`outputs/`、`reme_workspace/`、`nanobot_workspace/`。

## 3. 前置依赖（第三方，先装好）

### 3.1 π-Bench 仓库（含 AppWorld）

```bash
git clone https://github.com/Simplified-Reasoning/Pi-Bench.git <pi-bench-dir>
cd <pi-bench-dir>
python3.11 -m venv .venv            # 脚本约定使用 .venv 这个目录名
source .venv/bin/activate
pip install -e .                    # pibench runner（src.main）
bash scripts/setup_appworld.sh      # 安装 AppWorld 并下载其数据（体积较大，需网络）
```

装完自检：
```bash
ls data/                    # 应含 researcher marketer pharmacist law_trainee Financier
.venv/bin/python -c "import src" && echo OK
.venv/bin/appworld --help >/dev/null && echo OK
```

### 3.2 ReMe 仓库

```bash
cd <reme-dir>               # ReMe 仓库根目录（含 reme/ 包）
python3.11 -m venv .venv    # 脚本约定使用 .venv 这个目录名
source .venv/bin/activate
pip install -e .            # 或按 ReMe 自身安装方式，保证 `import reme` 可用
```

自检：`.venv/bin/python -c "import reme; print('ok')"`

## 4. 安装本套件（逐步）

1. **放置套件**（推荐放进 ReMe 仓库，`REME_DIR` 可自动推断）：
   ```bash
   cp -r pibench <reme-dir>/benchmark/pibench
   cd <reme-dir>/benchmark/pibench
   ```
   若放在其他位置，稍后在 env.sh 中显式设置 `REME_DIR`。

2. **创建环境文件并填写自定义参数**：
   ```bash
   cp env.sh.example env.sh
   ```
   打开 `env.sh`，必填项（标 TODO 的）：
   | 变量 | 说明 |
   |---|---|
   | `PI_BENCH_ROOT` | π-Bench 仓库根目录（含 `src/` `data/` `.venv` `third_party/appworld`） |
   | `USER_API_KEY` | 模拟用户 LLM 的 API key（run 阶段判定隐藏意图） |
   | `JUDGER_API_KEY` | 裁判 LLM 的 API key（eval 阶段 checklist 打分） |
   | `BRAVE_SEARCH_API_KEY` | 可选；agent 的 web_search 工具用，不用填 `dummy` |

   可选调整：`REME_MODEL_NAME`（被测 agent 基模）、`REME_DIR`、
   `REME_LLM_BASE_URL`（默认 DashScope OpenAI 兼容端点）。

3. **链接评测数据**（π-Bench 数据原位引用，不复制）：
   ```bash
   ln -s "$PI_BENCH_ROOT/data" data
   ```

4. **（可选）调整模型配置** `config/models/reme.yaml`：
   - `user_agent.model` / `judger.model`：模拟用户与裁判的模型名（字面量，
     π-Bench 仅对 base_url/api_key 做 `${ENV}` 展开）。
   - `run.turn_timeout`、`max_tool_iterations` 等按需。

5. **冒烟自检**（不启动评测）：
   ```bash
   bash -n run_all.sh && bash -n run_persona.sh
   source env.sh && "$REME_DIR/.venv/bin/python" -c "import reme; print('reme ok')"
   ```

## 5. 运行评测

> ⚠️ 长时间运行请放进 `screen`，**不要用 nohup**（nohup 在沙箱/受限环境下
> 会丢失权限上下文导致子进程异常）。

```bash
# 完整正式评测：先清空全部 persona 的记忆/输出/trace，再从头跑（默认 fresh，2 并行）
mkdir -p logs   # 全新部署时 logs/ 尚不存在，先建再重定向
screen -dmS pibench_suite bash -c "cd $(pwd) && bash run_all.sh > logs/run_all_master.log 2>&1"

# 断点续跑（中断后继续；不清记忆，跳过已完成任务）
bash run_all.sh --resume

# 其他用法
bash run_all.sh --parallel 1        # 串行
bash run_all.sh --resume --skip-eval  # 只跑 run 阶段
bash run_persona.sh researcher      # 单 persona（默认 --resume 语义）
bash run_persona.sh researcher --fresh
```

耗时参考：5 persona × 20 任务、2 并行，fresh 全量约 12–14 小时。

任一 persona 失败时 `run_all.sh` 以非零状态退出，上层自动化不会把部分失败
的评测误判为成功。

## 6. 端口分配（多 persona 并行互不冲突）

| persona     | AppWorld API | AppWorld MCP | Test Server | ReMe 内部服务 |
|-------------|------|-------|------|-------|
| marketer    | 9001 | 10001 | 9998 | 18766 |
| law_trainee | 9002 | 10002 | 9997 | 18767 |
| pharmacist  | 9003 | 10003 | 9996 | 18768 |
| researcher  | 9004 | 10004 | 9995 | 18765 |
| Financier   | 9005 | 10005 | 9994 | 18769 |

## 7. 输出与分数

- **结果**：`outputs/reme/{persona}/{task}/eval/results/*_result.json`
  - `overall_average_score`：checklist 完整度（COMP，judger 逐条 YES/NO 按依赖组加权）
  - `overall_proactiveness_average_score`：主动性（PROC，run 阶段 user_agent
    判定隐藏意图覆盖率；每个任务文件同时携带全局均值）
- **trace**：`~/.nanobot/trace_logs/reme/{persona}/{task}/...`（eval 的判分输入）
- **日志**：`logs/`（`suite_<persona>.log` 为每 persona 总日志，`bridge_*`、
  `runner_run/eval_*`、`appworld_*`、`test_server_*` 分服务）
- **记忆库**：`reme_workspace/{persona}/`（daily/digest 笔记、session 原始对话、
  BM25 索引等；跨运行持久，fresh 才清空）

查看汇总：
```bash
grep -h "overall_average_score\|overall_proactiveness" \
  outputs/reme/*/*/eval/results/*_result.json | head
```

### 工具轨迹采集（tools_evaluation 支持）

部分任务定义了 `objectives.tools_evaluation_path`：用 Python 脚本对工具行为
打分（例如"临时 Todoist 看板已创建并被删除"）。这些脚本需要 trace 里有真实
的工具调用记录。采集链路：

1. 每轮 `reply()` 之后，bridge 读取 AgentScope 落盘的会话状态，提取本轮新增
   的 `tool_call` / `tool_result` 块（工具名、参数、结果）。
2. 记录按 turn 编号追加写入
   `outputs/reme/{persona}/{task}/history/{ts}-tools.jsonl`；AgentScope 的
   MCP 工具名（`mcp__AppWorld__<tool>`）会规范化为 π-Bench 约定
   （`mcp_appworld_<tool>`）。
3. `fix_trace_logs.py` 将每个 `{ts}-messages.jsonl` 运行与时间上最接近的
   tools 旁路文件配对，把记录合并进生成的 `turn_N.json` 的 `tool_steps`
   字段——这是 π-Bench `collect_tool_history()` 支持的两种工具轨迹格式之一。
4. eval 阶段 `tool_steps` 既提供给 tools_evaluation 脚本，也会被渲染为
   judger 可见的 `<tool_trace_extracts>`。

## 8. 记忆机制（本套件的核心设计）

- **persona 隔离**：每个 persona 独立 workspace（`reme_workspace/{persona}/`），
  bridge 启动时对 workspace 加 `.bridge.lock` 排他锁，两个 bridge 不可能共用
  同一记忆库；一个 persona 的 memory search 永远接触不到其他 persona 的记忆。
- **写入**：任务结束（runner 发送 reset）时，会话经 `auto_memory` job 提炼为
  daily 笔记落盘，后台 watcher 建 BM25 索引。保存为非阻塞后台任务，
  新会话首条消息会先等待在途写入完成再检索。
- **读取**：bridge 每收到一条用户消息自动 `search` 一次并注入命中记忆
  （`[Relevant memories from previous sessions]` 前缀），无命中则原样透传。
  检索参数（bridge 命令行，可在 run_persona.sh 中调整）：
  - `--search-limit 3`：每条消息最多注入 3 个记忆块；
  - `--search-min-score 2.0`：过滤弱 BM25 命中；
  - `tool_context_id` 按任务轮换：同一任务内已注入的记忆块不重复注入
    （ReMe 自带 seen-chunk 去重，24h TTL），任务边界后恢复正常召回。
- **无自泄漏**：进行中的会话尚未入库（save 发生在 reset），任务不会检索到
  自己未完成的内容。
- agent 同时持有 `search`/`daily_write` 工具，可主动检索/记录。
- **system prompt**：`bridge_reme.py:build_system_prompt()` 内置
  HIDDEN-NEEDS 协议（面向 proactiveness），并把 `data/{persona}/profile.yaml`
  的 persona profile 注入每轮 system prompt。

## 9. 断点续跑与记忆清理语义

- **完成判定**（resume.py）：扫描 `outputs/reme/{persona}/**/history/*-log.jsonl`
  与 `outputs/reme/{persona}/run/*-log.jsonl` 中的
  `Task finished task_id=X status=Y`。每个任务以**事件时间最新**的记录为准
  （优先取记录的 `timestamp`，回退 `timestamp_iso`，再回退日志文件名中的
  时间戳）——文件类别与读取顺序本身不能覆盖更新的记录，因此旧的 run 级
  SUCCESS 不会掩盖更新的 per-task ERROR。`SUCCESS/MAX_TURNS/TIMEOUT` 记为
  完成，`ERROR`/未开始的任务重跑（按 episode 顺序以 `--task-id` 传给 runner）。
- **防答案泄漏**：被中断的任务可能已在优雅退出时提炼成 daily 笔记，直接重跑会
  把答案注入、抬高分数。因此 resume 启动前 `resume.py cleanup` **只删除待重跑
  任务**的残留记忆（daily/digest 笔记、session/dialog、mem_session，按
  `session_id = pibench_{task}_*` 匹配），已完成任务的记忆一律不动。daily
  索引**只刷新实际发生删除的日期**，按完整的 workspace 相对 wikilink 路径
  匹配；当 ReMe 包可导入时，刷新直接复用 ReMe 自带的 daily 索引重建逻辑
  （`refresh_day_index`），不会误改其他日期下的同名笔记条目。
- **fresh vs resume 互斥**：全量清记忆只属于 fresh 模式（`run_all.sh` 默认，
  在任何服务启动前执行）；resume 永不清全量。

## 10. 自定义与调优入口

| 目标 | 位置 |
|---|---|
| 被测 agent 基模 | `env.sh` 的 `REME_MODEL_NAME` |
| user_agent / judger 模型 | `config/models/reme.yaml` |
| agent system prompt | `bridge_reme.py` `build_system_prompt()` |
| 记忆检索条数/阈值 | `run_persona.sh` bridge 启动命令的 `--search-limit/--search-min-score` |
| ReMe 内部参数 | **不要改 ReMe 源码**；仿照 `reme/config/beam.yaml` 写专有配置，经 `resolve_app_config(config=...)` 覆盖（见 bridge `_init_reme_app`） |
| 轮超时/工具迭代上限 | `config/models/reme.yaml` `run.turn_timeout`、`model.max_tool_iterations` |

## 11. 故障排查

- **端口被占用**：脚本会自动 kill 上述 4 组端口上的残留进程；若与其他套件
  （如别的 π-Bench 实验）冲突，请先停掉对方或改 run_persona.sh 的端口表。
- **bridge 启动即退出，提示 workspace locked**：另一个 bridge 正占用同一
  workspace；确认每个 persona 用各自的 `--workspace-dir`（脚本已按 persona 分配）。
- **runner 报 `${USER_API_KEY} ... empty`**：env.sh 未填写或未生效；
  run_persona.sh 会自动 source env.sh，手动运行 runner 时请先 `source env.sh`。
- **`Cannot import 'reme'`**：bridge 必须用 `${REME_DIR}/.venv/bin/python` 运行
  （run_persona.sh 已如此），或检查 `REME_DIR` 是否指向 ReMe 仓库根目录。
- **AppWorld 启动失败**：先在 π-Bench 仓库执行 `bash scripts/setup_appworld.sh`
  下载数据；查看 `logs/appworld_*_<persona>.log`。
- **trace_history.yaml 找不到**：runner 需要
  `config/bench/evaluation/trace_history.yaml`；本套件已随附该文件并通过
  `--history-config-path` 显式传入，run_persona.sh 启动前会做存在性检查，
  缺失时立即报出清晰错误。请始终从套件目录启动 run_persona.sh / run_all.sh。

## 12. 隐私与安全

- 套件代码与配置模板中**不含任何真实 API key、用户名或绝对路径**；
  真实 key 只存在于你本地的 `env.sh`（已被 .gitignore 排除）。
- `logs/`、`outputs/`、`reme_workspace/`、`nanobot_workspace/` 含完整对话内容
  与模型输出，请勿提交仓库或外传。
- `data` 符号链接指向 π-Bench 官方评测数据，请遵守其数据许可条款。
