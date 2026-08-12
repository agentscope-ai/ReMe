# Meta-ReMe 设计与当前实现

## 1. 定位

Meta-ReMe 是 ReMe 的实验与评测底座，目标是让 agent 在可复现、可审计的边界内探索 memory 架构。它把 benchmark 的训练数据、可修改的 ReMe 代码、Git 版本和 sandbox 评测结果收拢到一个 workspace 中。

当前代码已经实现的主链路是：**准备训练集 → 构建最小 ReMe bundle → 创建初始 Git commit → 在隔离 sandbox 中验证 → 将结果和 artifacts 写入 workspace**。此外，已经提供了一个供 AgentScope 调用的同步 validation tool。

“weakness mining → proposal → 多候选搜索 → leaderboard → test set final evaluation”的完整自治搜索循环尚未接入。`models.py` 中为这部分定义了严格的数据契约，但这些模型不是当前 `run.py` 的运行行为。本文以现有可执行代码为准，并将未完成部分明确标注为后续设计。

## 2. 已实现的系统边界

### 2.1 入口与配置

准备入口为：

```bash
python meta-reme/run.py --config benchmark/longmemeval/config_meta_reme.yaml
```

当前 YAML 只消费以下字段：

```yaml
meta_workspace: "benchmark/longmemeval/meta-workspace"
dataset:
  name: "longmemeval"       # 或 "beam"
  source: "..."             # 可省略，使用项目内默认路径
  variant: "1M"              # 仅 BEAM 使用
  train_case_ids: []         # 空列表表示全部可用 case
validation:
  concurrency: 5
  fail_fast: false
```

`run.py` 会验证配置、规范化并安装所选训练 case、创建或打开 workspace、构建对应 bundle，并对 `init` 分支的全部已安装 case 执行一次 initial validation。已存在 `summary.json` 的 initial validation 会被复用。

`meta-reme/validation/run.py` 是独立的评测入口；它只评测已有 workspace，不会准备数据或创建 bundle：

```bash
python meta-reme/validation/run.py \
  --workspace /path/to/workspace \
  --case-id case-1 \
  --case-id case-2 \
  --concurrency 2 \
  --fail-fast
```

### 2.2 支持的数据集与规范化

当前支持 `longmemeval` 和 `beam`。原始数据会转换为公开 Pydantic contract：`CaseSpec`、`SessionSpec` 和 `QuerySpec`，并写为统一 JSON。数据集 manifest 记录 case/query 数和由规范化结果计算的 fingerprint。

- LongMemEval 的训练 case ID 是 question ID；case 会包含问题、golden answer、session 与问题时间。
- BEAM 从指定的 `100K`、`500K`、`1M` 或 `10M` variant 中读取 case；每个 probing question 形成 query。
- 安装后 `dataset/` 被设为只读；未选入的 case 不复制到 workspace。因此，当前实现把 `train_case_ids` 作为 workspace 内唯一可评测的数据范围，而不是在同一 workspace 中维护 search/test 两个集合。

## 3. Workspace

`Workspace.create()` 只接受空目录；workspace manifest 最后原子写入，作为创建完成标记。`Workspace.open()` 会校验目录结构与 `domain_spec.yaml` 的 fingerprint。对外部提供的路径、branch、case 和 validation ID 都执行路径安全检查。

```text
workspace/
  .meta-reme-workspace.json
  .meta-reme.lock                 # 仅持锁时存在
  domain_spec.yaml
  code/
    repo/
      reme/                       # 当前唯一的受管 Git 仓库
  dataset/
    manifest.json
    cases/
      000000.json
      ...
  weaknesses/                     # 为搜索输出预留，当前不写入
  evaluations/
  logs/                           # 已创建，当前没有统一日志写入器
```

workspace 使用原子写入（临时文件、`fsync`、同目录 `rename`）发布 JSON 与文本。锁文件记录 PID、主机名和随机 token；仅能接管确认已经退出的本机进程留下的 stale lock。数据集安装拒绝符号链接，且不会覆盖已安装的数据。

当前并不存在 `run.json`、`events.jsonl`、`leaderboard.json`、harness registry 或自动恢复/重放逻辑；这些不能作为现有行为依赖。

## 4. 可修改代码与版本边界

### 4.1 Bundle

`bundle_builder.py` 根据 [`build_bundle.yaml`](build_bundle.yaml) 的 allowlist 从完整 ReMe 仓库构建最小代码包，而不是先复制整个仓库再删除文件。它会跳过缓存、`.pyc`/`.pyo` 和符号链接，改写被裁剪包的 initializer，并将 bundle 的 service backend 固定为 `cli`，移除 `read_image` 与 `version` job。

可用 target：

| target | 包含的配置与 benchmark steps |
| --- | --- |
| `default` | `reme/config/default.yaml`；不含 benchmark steps |
| `lme` | `reme/config/lme.yaml` 和 `reme/steps/benchmark/lme/` |
| `beam` | `reme/config/beam.yaml` 和 `reme/steps/benchmark/beam/` |

`lme` 与 `beam` 不包含 `auto_memory_cc`；三个 target 都要通过 import、step registration、禁止 backend 与禁止路径检查。构建接口为：

```python
build_bundle(target, output_dir, source_repo=PROJECT_ROOT) -> Path
```

返回值为 `output_dir/reme`。直接运行 `python meta-reme/bundle_builder.py` 会分别构建 `default`、`lme`、`beam` 到 `meta-reme/bundles/<target>/reme`。

### 4.2 当前 Git 能力

准备阶段会在 `code/repo/reme` 初始化 Git，初始分支固定为 `init`，初始提交信息为 `Initial version`。如果仓库已存在并且已有 HEAD，初始化函数只返回该 HEAD，不会覆盖用户的后续文件或 commit。

评测前必须处在已 checkout 的非 detached branch，branch 名必须是路径安全的单段名称（不能含 `/`），并且 Git 工作区必须完全干净：staged、unstaged、untracked 和 submodule 变更都会使 validation 失败。评测器以 `git archive <HEAD>` 创建临时源码快照，再把该快照交给 sandbox；因此评测中工作区随后发生的变化不影响已开始的运行。

`git_manager.py` 当前只实现仓库初始化。它还没有封装 branch/worktree 创建、commit、merge、保护分支、scope diff 检查或候选冻结；agent 的代码修改和版本管理能力需要在此基础上继续实现并作为 AgentScope tools 暴露。

## 5. Validation

### 5.1 执行模型

`validation.evaluator.run_validation()` 与异步版本 `run_validation_async()` 接收 workspace、显式 case ID 列表、并发数和可选 `validation_id`。case ID 必须存在、唯一且非空。结果目录不可覆盖：

```text
evaluations/<branch-name>/<commit-prefix>/<validation-id>/
  manifest.json
  summary.json                    # fail_fast 中止时不生成
  failure.json                    # fail_fast 的运行级失败信息
  cases/
    <case-id>/
      case_result.json
      failure.json                 # 有失败时
      full.tar.gz                  # best-effort 故障导出，若可用
      memory_construction/
        result.json
        build.log
        reme_workspace.tar.gz
        reme_workspace/
      queries/
        result.json
        <query-id>/
          answer.log
          result.json
```

每次 validation 使用 workspace lock，因此同一 workspace 不会同时运行两个 validation。manifest 记录 branch、commit、case IDs、并发、fail-fast 设置、调度策略、数据集和数据/代码/配置/模型/镜像 fingerprint。

执行分为严格的两阶段：

1. 所有 case 先按时间顺序执行 `auto_memory` 与 `index_update`，完成 memory construction。
2. 每个成功 construction 的 workspace 在 query 前导出；所有 construction 到达终态后，才开始回答和 judge query。

成功 case 的 query 从同一个 post-construction workspace 快照开始。worker 优先复用已加载 case；空闲 worker 通过带 token 的单 query lease 窃取剩余 query，因此长 case 不会成为单个 worker 的瓶颈。容器在切换 case 时会 reset；发生 infrastructure error 的容器会被关闭，不再复用。

LongMemEval 会过滤晚于 `question_date` 的 session；BEAM 与 LongMemEval 分别由 adapter 构造 answer/judge 参数及评分字段。

### 5.2 失败、得分与 artifacts

结构化 construction 失败为 `candidate_failure`，该 case 的 query 不执行。宿主、容器、导出或 query 执行异常为 `infra_error`；当前 validation 层不会重试，即使 `SandboxSpec.max_retries` 有值，该字段仅保留兼容性说明。默认模式下，一个 case 的失败不会中止其他 case；`fail_fast=true` 会持久化已观察到的失败、取消同级 worker、写入运行级 `failure.json` 并抛出异常。

`summary.json` 将所有已返回 query 的数值 score 求算术平均；缺失 score 按 `0.0` 计。若没有 query，则平均分为 `null`。summary 会分别统计 `infra_error_count` 与 `candidate_failure_count`；任何 case 为 `infra_error` 时运行状态为 `infra_error`。因此现阶段 summary 是一次执行报告，而不是已实现的 leaderboard 可比性裁决。

每个 session 的 `auto_memory` 与 `index_update` 成功后，validation 会为配置的 `daily_dir` 创建本地 Git checkpoint；build 结果记录 commit SHA、session message 和路径。construction 成功后会立即保存带完整 memory Git 历史的 `reme_workspace.tar.gz`、安全解压副本和 build log。逐 query artifact 通过临时 tar 传输并立即删除压缩包。代码与 artifact 解压拒绝绝对路径、`..`、符号链接、硬链接、设备文件、重复条目和超出 `sandbox.max_artifact_bytes` 的归档。

### 5.3 AgentScope validation tool

`as.tools.validation_tool` 提供同步 validation 接口：

```python
validation_tool(case_ids: list[str] | None = None, fail_fast: bool = False) -> dict
```

未传 `case_ids`（或传空列表）时，它验证全部已安装 case；返回运行状态、请求/实际运行/成功/失败 case 数、平均分，以及完整 artifacts 的 `details_path`。tool 同步执行且标记为非并发安全。

直接调用 `validation_tool()` 时，它使用进程内的 `TOOL_RUNTIME` 获取 workspace 与并发数。`prepare_and_validate_workspace()` 会在当前进程调用 `TOOL_RUNTIME.configure()`；将 tool 接入长期运行的 agent host 时，host 必须先明确配置该 runtime。`create_validation_tool(workspace, concurrency)` 也可以生成显式绑定的实例，供 optimizer 等长期 agent 使用。单独启动一次命令行 `run.py` 后，不会为另一个进程保留该进程内状态。

当前 tool 只接受 case 级选择；它不支持 query 级选择、debug mode、validation policy 或 agent 自己的模型/镜像覆盖。

`as/agent/` 还提供两个由同名 YAML 加载 prompt 的 AgentScope agent：只读的
`diagnostic_agent` 会关联 caller trajectory、全部 validation artifacts、sandbox
memory 文件与候选 Git 历史，`optimizer_agent` 则组合文件/命令工具、validation
tool 和 `diagnostic_subagent_tool` 执行代码—测试—诊断—分支优化循环。factory
可以显式绑定 workspace；优化 agent 的文件写入工作目录限定为
`code/repo/reme`，诊断 agent 使用只读 permission mode。它们是 agent 执行入口，
不等同于下文尚未实现的持久化搜索编排器、预算结算或 leaderboard 裁决。

## 6. 已定义但尚未接入的搜索契约

`models.py` 已为下一阶段提供 strict Pydantic schema（未知字段会被拒绝）：

- `DomainSpec`、`ScopeSpec`、`SandboxSpec`、`ValidationPolicySpec` 与 `BudgetSpec`：描述搜索边界、资源与预算；
- `ValidationSelection`、`ValidationSpec`、`ValidationCoverage`、`ValidationResult`：描述 case/query 子集、coverage 和“仅 full search/replay 可比较”的规则；
- `WeaknessReport`、`Proposal`、`HarnessManifest`、`ScopeCheckResult`：描述 agent 的分析、假设和冻结候选；
- `RunState`、`SearchEvent`、`LeaderboardEntry`：描述持久化搜索状态与排行榜；
- `AttemptCompletion`、`CaseResult`、`QueryResult`：描述可复用 attempt 与结构化结果。

这些 contract 反映目标方向，但当前评测器写入的是兼容的 JSON 字典而非上述完整 `ValidationSpec`/`ValidationResult`/`AttemptCompletion` 流程。特别是 query 级筛选、coverage、正式 comparable 标记、重试、attempt 复用和 completion marker 尚未落地。

## 7. 面向自治 memory 搜索的下一步

要实现“向 agent 提供充足工具，持续探索更好 memory 架构”的目标，应以现有 workspace 和 validation 为基础，优先补齐以下闭环：

1. **代码与 Git tools**：提供读取代码/结果、创建安全 branch 或 worktree、应用修改、查看 diff、运行检查、commit、列出与恢复候选的工具。所有写操作限制在 `code/repo/reme` 或受管 worktree，禁止改写已评测 commit。
2. **候选冻结与 scope guard**：冻结前验证 clean Git 状态、bundle import/registration、允许修改路径和配置节点；把 commit SHA、源码 snapshot hash、配置 fingerprint、父 commit 与 scope 结果写入 `HarnessManifest`。
3. **可控 validation**：将 `ValidationSelection` 接入 evaluator，支持固定的 case/query 子集与 selection fingerprint；screening 仅做淘汰，完整 search validation 才能进入 leaderboard。
4. **搜索编排与可恢复状态**：实现 `run.json`/`events.jsonl` 的原子持久化和 event replay，写入 weakness/proposal 文件，按 `BudgetSpec` 结算预算，并由编排器而非 agent 决定 winner。
5. **隔离 final test**：search 集只用于选择；winner 必须从冻结 commit clean replay 后，才由外部流程加载未暴露给 agent 的 test 集评测 baseline 与 winner。

在这些能力完成前，Meta-ReMe 的正确使用方式是：由人或上层 agent 在受管 Git 仓库中创建干净 commit，再通过 validation tool 或 validation CLI 比较这些显式提交的结果；不要把尚未实现的搜索、scope 或恢复能力当作系统保障。
