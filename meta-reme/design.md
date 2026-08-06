# Meta-ReMe 实现方案设计

## 1. 建设目标

Meta-ReMe 是面向 ReMe auto-memory 机制的自动搜索系统。给定某个 benchmark 的 search set，系统通过“weakness mining → harness proposal → harness validation”循环产生多个候选 harness，以 search set 上所有 query 的平均 score 作为唯一优化目标，选出最佳候选，最终在隔离的 test set 上进行一次正式评测。

Harness 定义为 auto-memory 相关代码、Prompt 和配置。ReMe 的检索、回答、Judge、基础组件及 benchmark 逻辑保持固定。每个 benchmark 对应独立 domain spec，因此 LongMemEval、BEAM 等数据集分别搜索自己的最优 harness。

## 2. 总体架构

Meta-ReMe 放置于 `meta-reme/`，入口为：

```bash
python meta-reme/run.py \
  --config benchmark/longmemeval/config_meta_reme.yaml
```

主要模块如下：

```text
meta-reme/
  run.py
  models.py
  config.py
  workspace.py
  git_manager.py
  scope_guard.py
  search_loop.py
  agent.py
  evaluator.py
  result_store.py
  bundle_builder.py
  data_preparation/
    basic.py
    lme.py
    beam.py
```

`run.py` 负责初始化或恢复搜索；`search_loop.py` 负责外循环；`agent.py` 管理 weakness mining 和 proposal agent；`git_manager.py` 管理分支、提交和 worktree；`evaluator.py` 调用 sandbox；`result_store.py` 持久化结果；`scope_guard.py` 执行修改白名单检查。

## 3. Domain Spec

每个 benchmark 提供一个 `domain_spec.yaml`，内容包括：

- search set 的来源、格式和 fingerprint；
- case、session、query 的转换方法；
- benchmark runner 和评分器；
- baseline bundle 的文件清单；
- auto-memory 正式修改白名单；
- debug 阶段允许使用的临时目录；
- sandbox 镜像、超时、并发和重试策略；
- validation 的默认抽样策略、允许指定的 case/query 以及升级为全量评测的条件；
- proposer 模型、搜索预算和 top-K 数量；
- 唯一目标 `mean_query_score`。

平均分定义为所有 query score 的算术平均。候选造成的超时、空回答或运行错误记为 0；Docker、网络等基础设施故障按固定次数重试，仍失败则记为 `infra_error`，不直接参与候选比较。token、耗时和改动大小只作为诊断信息。

## 4. Workspace 设计

一次搜索的全部状态保存在用户指定 workspace：

```text
workspace/
  domain_spec.yaml
  run.json
  events.jsonl
  leaderboard.json
  code/
    repo/
    worktrees/
  datasets/
    search/
      manifest.json
      cases/
  harnesses/
    registry.jsonl
    <commit_sha>/
      manifest.json
      summary.md
  weaknesses/
  proposals/
  evaluations/
    <commit_sha>/
      search.json
      debug/
      cases/
  logs/
```

search set 被转换为统一 JSON 格式复制到 workspace，并设置为只读。Test set 不复制到该目录，也不向 agent 暴露；只有最佳 harness 冻结后，外部 evaluator 才加载 test set。

`run.json` 保存运行状态、baseline、当前轮次、预算和最佳候选。`events.jsonl` 采用 append-only 方式记录 proposal、commit、debug、validation、失败和 selection 等事件，使搜索可以在进程中断后恢复。

## 5. Baseline Bundle

`bundle_builder.py` 从完整 ReMe 仓库构建 benchmark 所需的最小代码仓库，包括 application、schema、registry、必要组件、auto-memory、search、answer、judge、配置和 packaging 文件。DingTalk、cookbook 等无关代码不进入 bundle。

Bundle 不能依赖人工随意删除文件，而应由 domain spec 中的文件清单确定性生成。生成后执行 import、组件注册、安装和固定 smoke case 检查，并验证精简版本与完整 ReMe baseline 行为一致。随后在 workspace 中初始化 Git，创建不可变 baseline commit。

### 5.1 Bundle Target

每个 bundle 由一个 target 名称标识，对应 ReMe 仓库中已有的配置文件和 benchmark 步骤：

| target | 配置文件 | benchmark 步骤 | 用途 |
| --- | --- | --- | --- |
| `default` | `reme/config/default.yaml` | 无 benchmark 步骤 | 通用最小 ReMe，不含任何 benchmark 专用代码 |
| `lme` | `reme/config/lme.yaml` | `reme/steps/benchmark/lme/` | LongMemEval 评测专用 |
| `beam` | `reme/config/beam.yaml` | `reme/steps/benchmark/beam/` | BEAM 评测专用 |

`default` bundle 只包含 ReMe 核心（application、schema、registry、auto-memory、search、answer 等通用组件），不引入 `reme/steps/benchmark/` 下的任何步骤；`lme` 和 `beam` bundle 在 `default` 基础上额外包含对应 benchmark 子包及其 auto-memory、agentic-answer、judge 步骤和配置。三类 bundle 共享核心代码，仅在 benchmark 专用部分不同。

### 5.2 调用方式

`bundle_builder.py` 支持两种调用方式：独立运行与模块导入。

**独立运行**（`python meta-reme/bundle_builder.py`）：一次性构建全部三类 bundle，输出到脚本同目录下的 `bundles/` 文件夹，目录结构如下：

```text
meta-reme/
  bundle_builder.py
  bundles/
    default/
      reme/          # default bundle 生成的最小代码仓库
    lme/
      reme/          # lme bundle 生成的最小代码仓库
    beam/
      reme/          # beam bundle 生成的最小代码仓库
```

每个 target 子目录下统一为 `reme/` 文件夹，结构与完整 ReMe 仓库中的 `reme/` 保持一致，可直接作为独立代码仓库使用。独立运行主要用于本地预生成、调试和校验 bundle 内容，产物可被后续搜索流程直接引用，避免每次运行重复构建。

**模块导入**（被 `run.py`、`search_loop.py` 等调用）：按指定 target 生成单个 bundle 到调用方指定的输出目录，供 workspace 初始化和 Git baseline commit 使用。接口签名如下：

```python
from meta_reme.bundle_builder import build_bundle

build_bundle(
    target: str,        # "default" | "lme" | "beam"
    output_dir: Path,   # bundle 写入目录，其下会生成 reme/ 子目录
    source_repo: Path,  # 完整 ReMe 仓库根目录，默认为项目根
) -> Path               # 返回生成的 reme/ 目录路径
```

模块模式不写入 `bundles/`，而是完全由调用方控制输出位置，以便将 bundle 直接放置到 workspace 的 `code/repo/` 中并就地初始化 Git。无论哪种方式，构建逻辑均由 domain spec 中的文件清单驱动，保证产物确定且可复现。

## 6. Harness 与 Git 管理

Branch 表示一条可继续演化的 proposal 线，commit SHA 表示一个不可变 harness。Agent 可以：

- 从任意历史 commit fork；
- 在独立 Git worktree 中修改；
- merge 多个历史分支；
- 多次 commit 和运行 debug case；
- 自主决定何时冻结候选。

禁止 force push、改写已登记 commit、删除受保护分支。每个正式候选通过 commit SHA、源码 snapshot SHA256 和配置 fingerprint 唯一标识。

Agent 具有较大的 Git 主动权，但所有操作通过 `git_manager.py` 执行，以防止覆盖历史结果。Merge 后必须重新比较 candidate 与 baseline 的完整文件树，而不是只检查最后一次 commit。

## 7. 修改权限

系统区分 debug scope 和 harness scope。

Debug 阶段允许 Agent 在 `meta_debug/` 或指定临时测试目录中增加诊断脚本、测试代码和辅助文件，也允许产生多个中间 commit。此类版本只能运行 debug evaluation，不能进入排行榜。

正式冻结时，仅允许修改 domain spec 声明的 auto-memory Python、Prompt、配置节点和必要注册文件。配置文件采用语义级检查，例如只允许修改 `jobs.auto_memory`，不能因为整个 `lme.yaml` 被放行而修改 search、answer 或 judge。正式 candidate 必须删除 debug 文件，并从干净 commit 重新执行 scope、import 和 smoke 检查。

## 8. Agent 工作方式

首版可以将 weakness mining agent 与 proposal agent 合并为一个自治 agent，使其能够连续完成分析、选父节点、修改、debug 和提交。为了保证过程可审计，每轮仍需生成两个结构化文件：

- `weaknesses/<round>.json`：问题模式、证据 query、可能原因、回归风险；
- `proposals/<proposal>.json`：选择的 parent、merge 决策、修改假设、预期改善目标。

Agent 可以读取最新 harness、当前 top-K、任意历史候选、代码 diff、search dataset、运行 artifacts 和总结文件。最佳候选仍由编排器根据平均分决定，不能由 agent 自行宣布。

## 9. Sandbox 执行与 Debug

所有代码运行必须进入 sandbox，包括正式 validation 和单 case debug。运行阶段包括：

```text
prepare → install → ingest → index → answer → export
```

Agent 可以自主选择完整 case、指定 session、session prefix 或少量 query 进行 debug。部分 session 结果只用于排查语法、配置、记忆写入和运行错误，不进入正式评分。

除 debug 外，validation 还支持显式传入 case ID 列表，并可为每个 case 指定 query ID 子集。该能力用于在全量评测代价过高时先执行 targeted screening，例如优先验证 weakness report 中的失败 query 和少量回归 case。筛选范围必须在运行前固化为 `ValidationSpec`，记录选择理由及其 fingerprint；不得在看到本次得分后追加或替换 query 来美化结果。

现有 sandbox 需要补充失败导出能力：即使在安装、初始化或 ingest 阶段失败，也应 best-effort 导出 `failure.json`、stdout、stderr、Python traceback、action log、已完成阶段和宿主异常。每次运行结束后，宿主下载原始压缩包，并安全解压到：

```text
evaluations/<commit>/<mode>/cases/<case>/attempt-<n>/
```

解压过程需要拒绝绝对路径、`../`、symlink、设备文件和异常大的压缩包，同时保存 tar 包和 SHA256，确保 artifacts 可验证、可复查。

## 10. Validation 与结果记录

Validation 分为 targeted screening 和 full validation。两者都必须从 frozen commit 创建全新 SourceCandidate，并使用同一套 sandbox、失败分类和结果 schema。不同 harness、case 之间使用独立 runtime workspace；同一 case 本次选中的多个 query 应从相同的 post-ingestion memory snapshot 启动，避免 query 顺序造成状态污染。

`ValidationSpec` 明确记录目标 commit、mode、选中的 case/query、选择理由和 dataset/code/config/model/image fingerprint。未指定 case 表示覆盖全部 case，某个 case 未指定 query 子集表示覆盖该 case 的全部 query。系统在执行前根据 dataset manifest 展开并固化选择结果；空集合、未知 ID、重复 ID，以及不属于所选 case 的 query 均应直接报错。

部分 validation 的聚合结果必须同时记录已选数量、数据集总数量、覆盖率和 `is_full=false`。其平均分只描述本次固定子集，不得与其他选择范围的分数直接比较，也不得进入正式 leaderboard 或用于选择 winner。编排器可以依据预先配置的筛选规则决定淘汰候选或将其升级为 full validation；只有覆盖完整 search set 且没有 `infra_error` 的结果才具有正式可比性。Baseline 至少执行一次 full validation，最终 winner 仍需进行第 11 节规定的 clean replay。

每次 targeted screening 的聚合结果写入 `evaluations/<commit>/screening/<validation_id>.json`；完整 search validation 仍发布为 harness 级 `evaluations/<commit>/search.json`。两类结果均保存：

- commit、parents、changed paths 和 scope 结果；
- dataset、代码、配置、模型和镜像 fingerprint；
- 总平均分、query 数、失败数和运行状态；
- validation spec、选择理由、case/query 覆盖数量、覆盖率和是否具有正式可比性；
- 每个 case/session 的处理结果；
- 每个 query 的问题、golden answer、模型回答、score、Judge 结论、异常、token 和耗时；
- artifacts 路径及 hash。

`summary.md` 由 LLM 根据结构化结果生成，说明该 harness 针对的问题、修改内容、相对 parent/baseline 的改善与退化、代表性 case 和后续建议。JSON 是事实来源，summary 仅用于快速阅读。

## 11. 搜索与最终测试

系统首先评测 baseline，然后循环执行：

```text
读取历史与 top-K
→ weakness mining
→ fork/merge
→ 修改和 sandbox debug
→ 冻结 commit
→ 白名单及接口检查
→ 可选：指定 case/query 的 targeted screening
→ 通过预设升级条件后执行完整 search validation
→ 更新 registry、summary 和 leaderboard
```

Targeted screening 是可选的成本控制阶段。未通过筛选的 candidate 记录结果和淘汰原因，但不消耗“完整 validation 数”预算；其实际 token、时间和费用仍计入对应总预算。筛选规则、样本范围和升级阈值必须在 domain spec 或 proposal 执行前确定，避免根据中间结果反复挑选有利样本。

达到 proposal 数、完整 validation 数、token 或时间预算后，系统选择 search set 平均分最高的候选。完全同分时保留较早的 incumbent，不引入其他优化指标。

最终 winner 必须从 commit SHA 重新构建，并在完整 search set 上进行 clean replay。只有 replay 成功后才加载 test set，分别评测 baseline 和 winner，生成最终报告。整个过程中 test 结果不反馈给 agent，从而保证 test set 只用于衡量泛化效果。

## 12. 中断与恢复

搜索可能持续数小时乃至数天，进程崩溃、机器重启或用户主动终止都不应破坏已完成的成果。恢复机制必须满足三个要求：持久化状态可校验、恢复操作幂等、同一 workspace 同时只有一个编排器写入。

### 12.1 持久化与一致性边界

`events.jsonl` 是搜索状态的事实来源，`run.json` 是由事件日志生成的加速快照。每条事件包含单调递增的 `seq`、唯一 `event_id`、时间戳和关联的 round、proposal、commit、validation、case、attempt 标识。写入事件后再原子替换 `run.json`；恢复时从快照记录的 `last_event_seq` 继续回放。若快照缺失、落后或校验失败，则完全从事件日志重建，不以快照覆盖较新的事件。

`run.json`、case 结果、聚合结果和 registry 均采用“写临时文件 → flush/fsync → 同目录原子 rename”的方式发布。`events.jsonl` 尾部若存在崩溃造成的不完整 JSON 行，恢复时忽略该行并记录诊断信息；中间行损坏则停止恢复并明确报错，避免静默跳过历史。启动时通过 workspace lock 拒绝第二个并发编排器，lock 中记录 PID、主机和启动时间，并允许在确认所属进程已不存在后接管 stale lock。

### 12.2 Case 原子性与结果复用

每个 case 的完整 pipeline（`prepare → install → ingest → index → answer → export`）是最小恢复单元。中断时不尝试复用半写入的记忆、索引或回答；下一次 attempt 在全新的 runtime workspace 中从头执行。中断或基础设施失败的 attempt 目录及原始 artifacts 保留用于审计，不原地覆盖，也不作为有效结果参与聚合；可归因于候选本身的确定性失败仍按第 3 节记为 0 分。

完成的 case 结果发布到第 9 节约定的 attempt 目录，并包含 `case_result.json` 和 `complete.json`。`complete.json` 最后写入，记录结果文件及 artifacts 的 SHA256、dataset/code/config/model/image fingerprint、case ID、attempt ID、所选 query 集合的 fingerprint 和完成状态。仅当完成标志可解析、hash 正确、状态为成功或确定性的候选失败，且所有 fingerprint 与当前 validation 完全一致时，结果才可复用；部分 query 的结果不能冒充该 case 的全量结果。目录或某个 `search.json` 文件的存在本身不能作为完成依据。基础设施故障耗尽重试后仍保留为 `infra_error`，但不把 validation 错误地标记为可聚合完成。

`ValidationSpec` 选中的全部 case/query 达到可聚合状态后，编排器确定性地生成对应聚合结果；targeted screening 写入其 validation ID 对应的结果文件，只有 full validation 才生成 harness 级 `evaluations/<commit>/search.json`。随后记录 `validation_completed` 事件。若聚合阶段中断，可从具有相同选择 fingerprint 的 case 结果重新生成，不重复执行已验证的选择范围。

### 12.3 搜索、Proposal 与 Git 续跑

恢复以最后一个已提交的阶段事件为准，而不是无条件进入“下一轮”：

- **搜索循环**：已记录 `round_completed` 的轮次不再执行；进行中的轮次从其最后一个完成阶段继续。只有尚未产生可恢复产物的 agent 调用才重新发起。
- **Proposal**：已落盘的 weakness 和 proposal 文件必须具有唯一 ID、fingerprint 和对应的完成事件。仅有临时文件或缺少完成事件的 agent 输出视为未提交，可保留为诊断材料，但不进入搜索状态。
- **Harness**：commit 只有通过 scope、import、smoke 检查并写入 registry 后才成为 frozen harness。存在 commit 但尚未登记时，恢复流程根据事件和检查结果决定继续冻结或放弃，不能仅凭分支名推断状态。
- **Git 管理**：恢复时先核对 registry 中的 commit 和 snapshot hash，再清理可证明由本次运行创建且不再被引用的 worktree。未登记分支和 debug commit 默认保留或归档；不得自动删除用户分支、受保护分支或仍被事件、proposal、registry 引用的对象。
- **Validation**：进行中的 validation 只调度缺少有效完成标志的 case/query 选择；恢复时必须使用原 `ValidationSpec`，不能重新抽样。部分 validation 完成后记录覆盖范围和筛选结论；只有 full validation 才更新正式 leaderboard。

预算只由去重后的完成事件结算，`event_id` 重放不得重复计费。已发起但没有产生完成事件的模型调用或 sandbox attempt 仍记录实际可观测的 token、费用和耗时；若外部系统无法确认消耗，则记为 `unknown` 并按 domain spec 的保守策略处理。时间预算使用累计运行时，不包含进程停止期间的墙钟时间，除非 domain spec 明确配置绝对截止时间。

### 12.4 恢复流程

`run.py` 启动时执行以下恢复流程：

```text
1. 获取 workspace lock；若已有存活 owner，则拒绝启动
2. 校验 events.jsonl，并以 run.json 的 last_event_seq 为起点回放；必要时从头重建快照
3. 校验 registry 中的 commit、snapshot hash 和 frozen harness 状态
4. 根据阶段事件定位进行中的 round、proposal 或 validation
5. 校验 case 的 complete.json、fingerprint 和 artifacts hash，复用有效结果
6. 为未完成 case 创建新的 attempt；从有效 case 结果重新执行缺失的聚合
7. 清理无引用且可安全识别的临时 worktree，保留或归档其他 Git 状态
8. 原子写入新快照并追加 resume_completed 事件，从最后一个未完成阶段继续
```

`resume_started` 和 `resume_completed` 事件记录恢复前后的状态摘要及本次采取的动作。上述流程可重复执行：再次中断不会重复登记 harness、重复结算预算或覆盖已有 attempt。无法验证的状态应停止并给出可操作的错误，不得通过猜测或静默删除数据来推进搜索。
