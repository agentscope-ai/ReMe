# Meta-ReMe 实现方案设计

## 1. 建设目标

Meta-ReMe 是面向 ReMe auto-memory 机制的自动搜索系统。给定某个 benchmark 的 search set，系统通过“weakness mining → harness proposal → harness validation”循环产生多个候选 harness，以 search set 上所有 query 的平均 score 作为唯一优化目标，选出最佳候选，最终在隔离的 test set 上进行一次正式评测。

Harness 定义为 auto-memory 相关代码、Prompt 和配置。ReMe 的检索、回答、Judge、基础组件及 benchmark 逻辑保持固定。每个 benchmark 对应独立 domain spec，因此 LongMemEval、BEAM 等数据集分别搜索自己的最优 harness。

## 2. 总体架构

Meta-ReMe 放置于 `meta-reme/`，入口为：

```bash
python meta-reme/run.py \
  --dataset longmemeval \
  --workspace /path/to/workspace \
  --config meta-reme/configs/longmemeval.yaml
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
  datasets/
    base.py
    longmemeval.py
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

现有 sandbox 需要补充失败导出能力：即使在安装、初始化或 ingest 阶段失败，也应 best-effort 导出 `failure.json`、stdout、stderr、Python traceback、action log、已完成阶段和宿主异常。每次运行结束后，宿主下载原始压缩包，并安全解压到：

```text
evaluations/<commit>/<mode>/cases/<case>/attempt-<n>/
```

解压过程需要拒绝绝对路径、`../`、symlink、设备文件和异常大的压缩包，同时保存 tar 包和 SHA256，确保 artifacts 可验证、可复查。

## 10. Validation 与结果记录

正式 validation 必须从 frozen commit 创建全新 SourceCandidate。不同 harness、case 之间使用独立 runtime workspace；同一 case 的多个 query 应从相同的 post-ingestion memory snapshot 启动，避免 query 顺序造成状态污染。

每个 harness 的 `search.json` 保存：

- commit、parents、changed paths 和 scope 结果；
- dataset、代码、配置、模型和镜像 fingerprint；
- 总平均分、query 数、失败数和运行状态；
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
→ 完整 search validation
→ 更新 registry、summary 和 leaderboard
```

达到 proposal 数、完整 validation 数、token 或时间预算后，系统选择 search set 平均分最高的候选。完全同分时保留较早的 incumbent，不引入其他优化指标。

最终 winner 必须从 commit SHA 重新构建，并在完整 search set 上进行 clean replay。只有 replay 成功后才加载 test set，分别评测 baseline 和 winner，生成最终报告。整个过程中 test 结果不反馈给 agent，从而保证 test set 只用于衡量泛化效果。
