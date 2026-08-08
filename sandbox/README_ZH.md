# ReMe Docker 沙箱

[English](README.md)

本目录用于在 AgentScope `DockerWorkspace` 容器中运行基准测试 case。它不会启动
ReMe HTTP 服务；上传到容器的 worker 会直接构造 ReMe `Application` 并调用
`run_job()`。

测试框架使用一个轻量的 `DockerWorkspace` 子类：保留 AgentScope 的 Docker
生命周期和 `DockerBackend`，但跳过 MCP gateway 的构建与启动。直接执行 ReMe job
不需要 gateway；省略它可以避免每个基础镜像额外进行一次依赖网络的镜像构建并启动
后台进程。

安装宿主机侧的 Docker 依赖：

```bash
pip install -e ".[sandbox]"
```

## 隔离模型

- 一个 candidate 可以创建多个容器沙箱。
- `create_cases()` 默认为每个 case 创建独立容器；也可以用 `reset_case()` 复用一个
  容器，按顺序执行多个 case。
- 当前 case 写入 `/workspace/case/reme_workspace`；容器交给下一个 case 前会删除该
  目录。
- 同一个 factory 创建的所有 case 复用同一份不可变 candidate 源码快照字节。
- 必须在关闭容器前完成导出。
- Case ID 必须以 ASCII 字母或数字开头，只能包含字母、数字、`.`、`_` 或 `-`，
  最长 128 个字符。一次 `create_cases()` 调用中的 ID 必须唯一。

## 构建仅包含依赖的基础镜像

在仓库根目录运行：

```bash
docker build \
  -f sandbox/Dockerfile.base \
  -t reme-sandbox-base:agentscope-2.0.4-post1 \
  .
```

该镜像根据 `pyproject.toml` 安装依赖，验证 AgentScope 版本为
`2.0.4.post1`，随后卸载用于安装依赖的占位 ReMe 包。因此，准备 case 后，candidate
源码会成为唯一可导入的 ReMe 实现。

## 模式一：源码持续变化的 candidate

```python
import asyncio
import os

from sandbox import DockerReMeSandboxFactory, SourceCandidate, SourceSnapshot


async def main():
    # 每个 candidate 只创建一次快照，不要为每个 case 重复创建。
    snapshot = SourceSnapshot.from_directory(".")
    factory = DockerReMeSandboxFactory(
        SourceCandidate(snapshot),
        env={
            name: os.environ[name]
            for name in ("LLM_API_KEY", "LLM_BASE_URL")
            if name in os.environ
        },
    )

    # 每个 ID 对应独立容器和运行时工作区，但收到完全相同的快照字节。
    cases = await factory.create_cases(["session-001", "session-002"])
    try:
        for case in cases:
            await case.ingest_session(
                session_id=case.case_id,
                messages=[{"role": "user", "content": "Remember this."}],
            )
            await case.commit_memory_history(f"session: {case.case_id}")
            answer = await case.answer(query="What should you remember?")
            await case.judge(
                query="What should you remember?",
                agent_answer=str(answer.answer),
                golden_answer="Remember this.",
            )
            await case.export(f"artifacts/{case.case_id}.tar.gz")
    finally:
        await asyncio.gather(*(case.close() for case in cases))


asyncio.run(main())
```

源码归档会排除版本控制状态、`.reme`、虚拟环境、缓存、`.env` 凭据文件、基准
数据集、已有 memory workspace、日志和构建产物。其他 secret 不会被自动识别，
因此不能放入 candidate 目录树。数据集输入应显式传给 case，不应复制进每个
candidate。源码快照拒绝符号链接，避免意外收录 candidate 根目录之外的文件。

## 模式二：candidate 已预装在镜像中

对于稳定 candidate，使用仓库的 `.dockerignore`（`sandbox/.dockerignore.example`
保留了一份等价示例），然后运行：

```bash
docker build \
  -f sandbox/Dockerfile.candidate \
  --build-arg BASE_IMAGE=reme-sandbox-base:agentscope-2.0.4-post1 \
  -t reme-candidate:my-candidate \
  .
```

使用该 candidate 时，每个 case 不再上传源码：

```python
from sandbox import DockerReMeSandboxFactory, ImageCandidate

factory = DockerReMeSandboxFactory(
    ImageCandidate("reme-candidate:my-candidate", candidate_id="git-or-content-hash"),
    env={...},
)
case = await factory.create_case("session-001")
```

## 单 job 直接调用与产物契约

原有便捷方法映射到 `lme.yaml` 中的 job：

- `ingest_session()` → `auto_memory`，随后可选执行 `index_update`
- `answer()` → `agentic_answer`
- `judge()` → `answer_judge`，其中 `yes=1`、`no=0`
- `run_job()` → 任意显式指定的 ReMe job

`export()` 默认生成面向上述单 job 流程的分析型 gzip 归档。它包含旧版 `logs/`
目录中累积的日志、命令审计日志、每个 job 的 JSON 结果、标准 answer/score 文件、
持久化的 agent session，以及用户拥有的 workspace 文件——包括写在预期
daily/digest 路径之外的文件。归档还包含经过验证的运行时布局和 manifest；请求
副本、临时文件、源码资源以及可重建的索引/缓存会被排除。配置的
`session_dir/dialog` 路径下的原始对话 session 会被保留。

使用 `export(profile="full")` 或 `export_full()` 可下载完整的临时 case 目录树。
`include_candidate=True` 还会把上传的源码 candidate 附加到归档中；image candidate
不支持这个参数。Manifest 只记录环境变量名，不会记录其秘密值。

下文介绍的批处理产物位于 `build_log/` 和 `queries/`，有意不纳入旧版 analysis
profile。需要这些产物时使用 `export_evaluation()`；需要整个 case 时使用 full
profile。

## Build 与多 Query 批处理

多 query 评测应使用批处理 API，避免每个 job 都重新启动 Application。
`run_build()` 在一个 Application 中执行全部 construction job；随后
`run_queries()` 启动一个新的 Application，并让所有 answer/judge 对复用它。Query
按顺序执行，因此 token 增量和进程级全局日志 sink 都能准确归属于单个 query。
Build/query 边界上的重启还保证评测只依赖已持久化的 workspace 状态，而不依赖
construction 阶段独有的内存状态。

每个方法只为当前 case 发布一个阶段：

- `run_build()` 至少需要一个 job；遇到第一个失败 job 后停止；如果
  `build_log/build.log` 已存在，则拒绝追加。
- `run_queries()` 至少需要一个 query，query ID 必须唯一；单个 query 失败后仍继续
  后续 query；如果 `queries/summary.json` 或任一待运行 query 的目录已经存在，则
  拒绝追加。
- 若要在同一容器中启动另一轮评测，应先调用 `reset_case()`。

```python
from sandbox import EvaluationQuery, JobRequest

build = await case.run_build(
    [
        JobRequest("auto_memory", {"session_id": "session-1", "messages": messages}),
        JobRequest("index_update"),
    ],
)
assert build["success"]
await case.commit_memory_history("constructed memory")

evaluation = await case.run_queries(
    [
        EvaluationQuery(
            query_id="question-1",
            question="What should be remembered?",
            golden_answer="Remember this.",
            judge_arguments={
                "query": "What should be remembered?",
                "golden_answer": "Remember this.",
            },
        ),
    ],
)
assert evaluation["success"]
await case.export_evaluation("artifacts/case-1.tar.gz")
```

如需分别保存两个阶段，应在 `run_build()` 后立即调用
`export_memory_construction()`，冻结 query 执行前的 `reme_workspace/` 和
`build_log/`；在 `run_queries()` 后调用 `export_queries()`，只导出
`queries/`。仅在明确需要合并归档时使用 `export_evaluation()`。

`export_evaluation()` 生成且只生成三个顶层目录：

```text
reme_workspace/
build_log/
  build.log
queries/
  summary.json
  <query-id>/
    answer.log
    result.json
```

与 analysis profile 不同，该导出包含完整的 `reme_workspace/`，包括本地 Git 历史，
以及当时存在的可重建 metadata、resource、索引或缓存。导出要求
`build_log/build.log` 和 `queries/summary.json` 都已存在，因此应在两个批处理阶段
均完成产物发布后调用。

Query ID 会原样用作目录名，因此必须是一个安全的路径组件：不能为空，不能是
`.`、`..` 或 `summary.json`，不能包含 `/`、`\` 或 NUL，UTF-8 编码后不能超过
255 字节。

每个 `answer.log` 只包含对应 query 的 answer 和 judge 日志，不包含 Application
启动和关闭日志。每个 `result.json` 包含问题、标准答案、实际回答、归一化 score、
原始 answer/judge 结果、token 增量和错误信息。`queries/summary.json` 包含
`case_id`、数量统计、平均分，以及每个 query 的 ID 和 score，不包含冗余的目录
字段。失败 query 的 score 为 `null` 并带有 error，后续 query 仍会继续执行。

由于 ID 原样保留，调用方需要自行保证解压平台的文件名兼容性。例如，`:` 在 Linux
容器和归档中合法，但不能作为 Windows 文件名字符。

`EvaluationQuery` 默认使用 LongMemEval 约定：answer 会注入 judge 参数
`agent_answer`；judge 返回的 `yes` 和 `no` 分别映射为 `1.0` 和 `0.0`。
`answer_arguments` 和 `judge_arguments` 用于提供其余 job 参数；可以通过
`answer_job`、`judge_job`、`judge_answer_argument`、`score_path` 和
`score_mapping` 覆盖默认约定。最终 score 必须落在闭区间 `[0, 1]`。

BEAM 风格的 judge 可以从 judge metadata 中选择数值 score：

```python
EvaluationQuery(
    query_id="information_extraction:1",
    question=question,
    golden_answer=rubric,
    judge_answer_argument="llm_response",
    judge_arguments={
        "rubric": rubric,
        "probing_question": question,
        "question_type": "information_extraction",
    },
    score_path="metadata.llm_judge_score",
    score_mapping=None,
)
```

## Memory 的本地 Git 历史

运行时 workspace 同时也是一个本地 Git 仓库。Session ingestion 不会隐式提交；
宿主机通过 `commit_memory_history(message)` 自行选择 checkpoint 边界。可以在每个
session 后提交，也可以将多个 session 合并为一次提交，并完全控制 commit message。

每个 checkpoint 只提交配置的 `daily_dir`。即使没有变化，也会保留空提交，作为
显式边界。导出的 workspace 包含 `.git`，因此无需 remote 或 push，也能离线检查
daily memory 的构建历史。`reset_case()` 会连同旧运行时 workspace 一起删除该
仓库，并为下一个 case 初始化新的仓库。

对于临时 Docker workspace，必须在 `close()` 前下载所有需要保留的产物。根据所需
契约选择 `export()`、`export_full()`、`export_build_log()`、
`export_memory_construction()`、`export_queries()`、`export_evaluation()` 或
`export_workspace()`。

## 复用一次构建的 Memory，并行执行 Query

完成 memory 构建后调用 `export_workspace()`，再用 `upload_workspace()` 把该快照
交给多个独立 query case。默认情况下，上传前会清空目标运行时 workspace，因此
所有 case 从同一 memory 状态开始，但不会共享可变文件或索引。

```python
memory_case = await factory.create_case("build-memory")
try:
    await memory_case.ingest_session(session_id="source", messages=[...])
    await memory_case.run_job("index_update")
    snapshot = await memory_case.export_workspace("artifacts/built-memory.tar.gz")
finally:
    await memory_case.close()

query_cases = await factory.create_cases(["query-1", "query-2"])
try:
    await asyncio.gather(*(case.upload_workspace(snapshot) for case in query_cases))
    answers = await asyncio.gather(
        *(case.answer(query=query) for case, query in zip(query_cases, queries))
    )
finally:
    await asyncio.gather(*(case.close() for case in query_cases))
```

`upload_workspace()` 也接受宿主机目录，或同时包含 `manifest.json` 和
`reme_workspace/` 的旧版 `export()` 归档。它会先验证并重新打包目录或归档，再
提取到容器；符号链接、特殊文件、重复归档路径和路径穿越都会被拒绝。
`export_evaluation()` 归档有意不包含 manifest，因此不能直接用于上传；需要可移植
快照时应使用 `export_workspace()`。仅当明确需要合并到当前 workspace 时才传入
`clear=False`。上传后总会执行 `git init`：已有历史会被保留，没有 `.git` 的目录
则会得到一个新仓库。

## 顺序执行多个 Case 时复用一个容器

当容器启动和源码安装占据大部分评测时间时，可以让一个容器按顺序处理多个 case。
先完成并按需导出当前 case，然后清理临时 case 目录，再上传下一个 case 的 session：

```python
case = await factory.create_case("session-001")
try:
    await case.ingest_session(session_id="session-001", messages=[...])
    await case.commit_memory_history("session: session-001")
    first = await case.answer(query="...")
    await case.export("artifacts/session-001.tar.gz")

    await case.reset_case("session-002")
    await case.ingest_session(session_id="session-002", messages=[...])
    await case.commit_memory_history("session: session-002")
    second = await case.answer(query="...")
finally:
    await case.close()
```

`reset_case()` 会删除 ReMe 运行时 workspace、请求、旧版日志、results、
`build_log/`、`queries/`、manifest、case 级临时文件和临时导出归档；已安装的
candidate、candidate 虚拟环境和 benchmark worker 会被保留。

Job、批处理阶段、导出、上传、Git 提交和 reset 共用同一把锁，因此不会在同一容器
内重叠执行。`reset_case()` 返回后，不要继续持有或使用前一个 case 的路径和状态。
