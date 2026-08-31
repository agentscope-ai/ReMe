# AGENTS.md

This file guides coding agents working in the ReMe repository. Keep changes small, testable, and consistent with the
contracts expressed by the current code.

## Project Principles

ReMe is a local-first, file-native memory system for agents.

- User-owned workspace files are the durable source of truth.
- Indexes, catalogs, graphs, caches, and generated metadata must remain rebuildable.
- Prefer transparent formats and predictable behavior over hidden state.
- Preserve user control over workspace paths, configuration, and service boundaries.
- Keep concepts focused on project intent; let code and schemas describe implementation.

When convenience conflicts with these principles, favor data ownership, recoverability, and explicit behavior.

## Sources of Truth

Use this order when documentation and implementation disagree:

1. Current code and public Pydantic schemas.
2. Tests that describe supported behavior.
3. CLI behavior and the built-in configuration.
4. README files and other development documentation.

Do not duplicate large implementation descriptions in documentation. Express the stable contract and link to the
relevant module where useful. When behavior changes intentionally, update the implementation, schemas, tests, defaults,
and concise documentation together.

## Repository Map

- `reme/reme.py`: CLI entry point; dispatches `start`, `find_reme`, and client calls.
- `reme/application.py`: application assembly, dependency ordering, job execution, and lifecycle.
- `reme/config/config_parser.py`: YAML/JSON loading, environment expansion, dot-notation parsing, and deep config
  merging.
- `reme/config/default.yaml`: default service, jobs, steps, and components. Other files in
  `reme/config/` are named configuration variants.
- `reme/schema/application_config.py`: typed application, component, and job configuration.
- `reme/schema/`: request, response, streaming, memory, graph, and file contracts.
- `reme/components/application_context.py`: application-wide wiring and in-memory shared state.
- `reme/components/runtime_context.py`: request-scoped data, response, streaming queue, and stop event.
- `reme/components/base_component.py`: component lifecycle, dependency binding, and workspace helpers.
- `reme/components/component_registry.py`: the frozen built-in registry template and application-local registry factory.
- `reme/components/job/`: base, stream, background, and cron job implementations.
- `reme/components/service/`: local CLI, HTTP, and MCP service backends.
- `reme/components/`: agent wrappers, model adapters, stores, catalogs, graphs, indexes, clients, tokenizers, and
  outbound proxies.
- `reme/steps/`: registered job steps grouped by common, file I/O, index, evolve, cookbook, benchmark, and transfer
  concerns.
- `reme/utils/`: shared utilities, including service discovery, logging, web-static resolution, session I/O, token
  accounting, and wikilink handling.
- `tests/unit/`: primary fast, isolated validation suite.
- `tests/integration/`: service/model tests that may need credentials or external processes.
- `reme_studio/`: ReMe Studio frontend source plus the independently published `reme_studio` Python package and
  `@agentscope-ai/reme_studio` npm static distribution.
- `typescript/`: the independently published `@agentscope-ai/reme` package, including the shared TypeScript client and
  DeepSeek Harness and OpenClaw adapters.
- `plugins/`: installable ReMe extensions, such as Auto Fin.
- `integrations/`: adapters that connect ReMe to external agent hosts, such as Claude Code, DSH, and Hermes Agent.
- `skills/`: standalone skills; `reme_memory` calls ReMe, while other skills may use separate tools or direct-file
  conventions.
- `benchmark/` and `cookbook/`: runnable evaluations and example workflows.
- `docs/`: README-linked supporting pages and figures.

## Development Setup

ReMe requires Python 3.11 or newer. Install the editable development environment with:

```bash
pip install -e reme_studio -e ".[dev,core]"
```

Before changing behavior, inspect the adjacent implementation, schema, built-in config, and focused tests. Follow
existing async and typing patterns unless the task explicitly requires a new contract.

## Configuration and CLI Contracts

- CLI syntax is `reme ACTION key=value ...`; leading `-` or `--` on arguments is accepted.
- Nested overrides use dot notation. Values support null, booleans, numbers, JSON collections, and quoted JSON strings;
  leading-zero numeric-looking values remain strings.
- `config=<name-or-path>` loads a discovered config name or a `.yaml`, `.yml`, or `.json` file. With no explicit config
  path, `default` is loaded when available.
- Config files expand `${VAR}` and `${VAR:-default}` recursively. An undefined variable without a default is an error.
- CLI/config overrides are deep-merged over the loaded file. Do not silently change this merge behavior or stable
  configuration keys.
- `ApplicationConfig` normalizes `workspace_dir` to an expanded absolute path. `session_dir`
  must remain workspace-relative; standard transcripts live under `{session_dir}/dialog`.
- `reme start` runs the configured service. `reme start job=<name> ...` switches to the one-shot CLI service and runs
  the job through the normal application lifecycle.
- Other actions use a client selected from the running service configuration when discoverable, otherwise from local
  config. Client-selection arguments must not leak into the job payload.

## Registration and Application Lifecycle

Component and Step discovery is import-driven:

- Implementations declare a non-`BASE` `component_type` and register with `@R.register("backend")`
  or `R.register(Class, "backend")`.
- Component packages must be imported through `reme/components/__init__.py`.
- Step packages/modules must be reachable through their package `__init__.py` chain and ultimately
  `reme/steps/__init__.py`.
- Adding an implementation without its registration import leaves it undiscoverable at runtime. Treat implementation,
  registration, import side effect, defaults, and tests as one change.

`Application` validates config through `ApplicationContext`, creates workspace directories, instantiates the service,
configured components, and jobs, and then manages lifecycle as follows:

- Components start in topological dependency order. Missing required dependencies and cycles fail explicitly; optional
  dependencies may resolve to `None`.
- Jobs start after components in this order: base jobs, stream jobs, background jobs, then cron jobs.
- Shutdown closes everything in reverse start order and then shuts down the optional thread pool.
- If startup fails, already-started resources are closed.
- `BaseComponent.start()` and `close()` are lock-protected and idempotent. Dependencies created by a standalone
  `default_factory` are owned and closed by the parent component.

Keep async clients, tasks, executors, and services under this lifecycle. Do not introduce an untracked long-lived
resource.

## Jobs, Steps, and State

`BaseJob` resolves configured Step classes during job startup and constructs fresh Step instances for every invocation.
Job-level kwargs are merged into each `RuntimeContext`, with call-time kwargs taking precedence. Sequential Steps in one
invocation share the same `RuntimeContext` and `Response`.

Treat Step instances as invocation-scoped:

- Constructor fields and `self.kwargs` hold Step configuration and resolved dependencies. They may be cached or adjusted
  during that one invocation, but must not be relied on across Job calls.
- `self.context.data` holds request inputs and intermediate values shared by sequential Steps.
- `self.context.response.answer`, `success`, and `metadata` are request-scoped output. Because the same response travels
  through the Step chain, later Steps may consume metadata produced earlier, but it is not application-lifetime or
  durable storage.
- `self.app_context.metadata` holds in-memory state shared across Job/Step invocations for the life of one
  `Application`, such as counters, tool-context state, session maps, or locks.
- Workspace files or a dedicated Component/store hold durable state that must survive restart.

Use narrow, namespaced keys in `app_context.metadata` and protect shared mutable values against concurrent access. The
search/draft helpers intentionally mirror tool-context state into
`self.kwargs` only when no `ApplicationContext` exists for standalone use and unit tests; do not generalize that
compatibility fallback into persistent runtime state. If shared state becomes a stable service contract or needs
dedicated lifecycle, locking, or persistence, promote it to a typed context field or Component.

Additional Step contracts:

- `Ref` dependencies resolve in this order: Step kwargs, current `RuntimeContext`, then the named application component.
  The value is cached only on the current Step instance and cleared before each call.
- `input_mapping` and `output_mapping` copy keys within `RuntimeContext.data`; missing sources are ignored.
- Dispatched Steps receive the current `RuntimeContext`, so their data and response are shared.
- Base jobs convert uncaught Step errors into `Response(success=False)`; stream jobs emit an error chunk and always a
  terminal `DONE`; background jobs let errors reach their supervisor.
- Background jobs are never service-exposed. MCP also skips stream jobs. Respect `enable_serve`
  and any configured service job allowlist.

## Workspace and File Safety

- Application startup creates the workspace plus configured metadata, session, memory-session, resource, daily, and
  digest directories.
- File-operation paths are resolved against the workspace and must stay inside it. Home-relative paths are unsupported,
  traversal escapes are rejected, and `_allowed_paths` restrictions fail closed when invalid.
- Preserve per-path locking, encoding detection, byte limits, truncation behavior, and optimistic
  `expected_mtime` checks when modifying file operations.
- Do not bypass the existing file steps or stores in a way that weakens workspace containment.
- Never write test state into the repository's `.reme/`; use `tmp_path` or another isolated workspace.
- Do not delete or rewrite user memory to repair an index or make a test pass. Rebuild derived state from source files
  instead.

## Validation

Use the narrowest useful check while iterating, then broaden it according to risk.

Focused test:

```bash
pytest tests/unit/path/to/test_file.py -v
```

Main unit suite:

```bash
pytest tests/unit -v --tb=long -s --log-cli-level=WARNING
```

Repository formatting and lint checks:

```bash
pre-commit run --all-files
```

Black and Flake8 use a 120-character line limit and Python 3.11 formatting; Pylint is also run by pre-commit. If
`reme_studio/` changes, use its Node 22.13+ scripts and run the proportionate checks from that directory, such as
`npm run format:check`, `npm run lint`, or `npm test`.

Integration tests may contact real model providers, services, or agent subprocesses and can require credentials. Do not
run credentialed or externally mutating tests automatically; run them only when the task requires them and the necessary
environment has been supplied or authorized. Mock network, model, and subprocess boundaries in unit tests.

## Change Guardrails

- Preserve unrelated user changes in a dirty working tree.
- Make the smallest coherent change and avoid unrelated cleanup or broad refactors.
- Do not edit generated output when the source can be changed instead. The publish workflow builds
  `reme_studio/dist-static` and stages it under `reme_studio/src/reme_studio/static`; change `reme_studio/` source for
  frontend work.
- Do not silently change CLI flags, configuration keys, workspace layouts, serialized schemas, endpoint shapes,
  streaming termination, or service interfaces. Preserve compatibility where practical and document intentional
  migrations.
- Do not introduce dependencies without a concrete repository-level need.
- Do not commit `.env` files, credentials, runtime memory, logs, indexes, caches, benchmark outputs, or generated
  Studio distributions.
- State which validations passed and which relevant checks were not run in the final handoff.

If a requirement is ambiguous, infer intent from nearby code, schemas, defaults, and tests. Ask the user only when the
remaining choice would materially alter a public contract, user data, or an external system.
