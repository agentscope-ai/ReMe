# AGENTS.md

This file guides coding agents working in the ReMe repository. Keep changes small,
testable, and consistent with the contracts already expressed by the code.

## Project Principles

ReMe is a local-first, file-native memory system for agents.

- User-owned memory files are the source of truth.
- Indexes, caches, metadata, and generated state must be rebuildable.
- Prefer transparent formats and behavior over hidden state.
- Preserve user control over storage, configuration, and service boundaries.
- Keep concepts focused on project intent; let code and schemas describe implementation.

When a proposed convenience conflicts with these principles, favor data ownership,
recoverability, and predictable behavior.

## Sources of Truth

Use this order when documentation and implementation disagree:

1. Current code and public Pydantic schemas.
2. Tests that describe supported behavior.
3. CLI help and the built-in configuration.
4. Development documentation and historical notes.

Do not copy large implementation descriptions into documentation. Link to the relevant
module or express the stable contract instead. If behavior changes intentionally, update
the code, schema, tests, configuration, and concise documentation together as needed.

## Repository Map

- `reme/reme.py`: CLI entry point and client/server dispatch.
- `reme/application.py`: application assembly, dependency ordering, and lifecycle.
- `reme/components/application_context.py`: application-wide wiring and shared in-memory metadata.
- `reme/components/runtime_context.py`: scratch state shared by steps within one execution.
- `reme/config/default.yaml`: built-in jobs, components, and defaults.
- `reme/schema/`: public and runtime Pydantic contracts.
- `reme/components/`: services, stores, clients, jobs, and component registration.
- `reme/steps/`: executable job steps.
- `tests/unit/`: primary fast validation suite.
- `tests/integration/`: tests that may require real credentials or services.
- `tests/vector/` and `tests/light/`: specialized suites.
- `plugins/reme/`: Claude Code integration.
- `skills/reme_memory/`: skill that communicates with the ReMe service.
- `skills/qwenpaw_memory/`: separate direct-file memory convention; it does not call ReMe.
- `docs/`: pages and assets that support the repository README; not the deployed docs site.

## Development Setup

ReMe requires Python 3.11 or newer.

```bash
pip install -e ".[dev,core]"
```

Before changing behavior, inspect the adjacent implementation, schemas, configuration,
and focused tests. Follow existing patterns unless the task explicitly calls for a new
contract or architecture.

## Change Workflow

1. Identify the narrowest supported contract affected by the request.
2. Read the relevant implementation and tests before editing.
3. Make the smallest coherent change; avoid unrelated cleanup.
4. Update related schemas, defaults, registrations, and imports when required.
5. Add or adjust focused tests for observable behavior.
6. Run proportionate validation and report anything not run.

Component and step discovery depends on registration imports:

- Components use `R.register(...)` in `reme/components/component_registry.py`.
- Component packages must be reachable through `reme/components/__init__.py`.
- Step modules must be reachable through `reme/steps/__init__.py`.

Adding an implementation without its registration import can leave it undiscoverable at
runtime. Treat the implementation, registry entry, and import side effect as one change.

Do not silently change stable CLI flags, configuration keys, workspace layouts, serialized
schemas, or service interfaces. When such a change is required, preserve compatibility
where practical and make the migration explicit.

## Step State Model

Treat every Step as stateless. `BaseJob` stores Step specifications and builds fresh Step
instances for each Job invocation. A Step instance must not use `self` or class variables to
retain mutable runtime state between calls.

Place state according to its lifetime:

- Constructor fields on `self`: immutable Step configuration and resolved dependencies only.
- `self.context` (`RuntimeContext`): request data and intermediate results for one Job
  execution; sequential Steps share this context.
- `self.app_context.metadata`: in-memory state that must be shared across Step or Job
  invocations for the lifetime of the Application.
- Workspace files or a dedicated Component/store: durable state that must survive an
  Application restart.

Use narrow, namespaced keys in `app_context.metadata`, following existing patterns such as
`tool_contexts`. The ApplicationContext is shared, so account for
concurrent access when values are mutable. New Step code must not fall back to `self.kwargs`
or another Step field to emulate shared state when `app_context` is absent; tests of shared
state should construct an `ApplicationContext`. If shared state grows into a stable
service-level contract or needs its own lifecycle, locking, or persistence, promote it to a
typed ApplicationContext field or a dedicated Component instead of expanding an ad hoc
metadata bucket.

Do not use `Response.metadata` as a state store. It is request-scoped output for callers and
diagnostics, distinct from `ApplicationContext.metadata`.

## Validation

Use the narrowest useful check while iterating, then broaden it according to risk.

Run a focused test:

```bash
pytest tests/unit/path/to/test_file.py -v
```

Run the main unit suite:

```bash
pytest tests/unit -v --tb=long -s --log-cli-level=WARNING
```

Run repository formatting and lint checks when the change warrants it:

```bash
pre-commit run --all-files
```

Formatting and lint configuration is authoritative. Python code currently uses a maximum
line length of 120 for Black and Flake8, with Pylint also run by pre-commit.

Integration tests may contact real services and require credentials such as
`LLM_API_KEY` or `EMBEDDING_API_KEY`. Do not run credentialed or externally mutating tests
automatically. Run them only when the task requires them and the user has supplied or
authorized the necessary environment.

## Coding and Test Conventions

- Target Python 3.11+ and follow the surrounding typing and async style.
- Steps are stateless. If a step needs to persist state, store it in
  `self.app_context.metadata` rather than on the step instance.
- Keep public schemas explicit and backward-compatible where practical.
- Close async clients, services, tasks, and other lifecycle resources deterministically.
- Prefer clear failures over silently falling back to corrupt or ambiguous state.
- Keep indexes and caches derivable from user-owned source files.
- Use `tmp_path` or another isolated temporary workspace in tests.
- Never write test state into the repository's `.reme/` directory.
- Mock network or model boundaries in unit tests.
- Do not commit `.env` files, credentials, runtime memory, logs, indexes, or caches.

## Documentation Boundaries

ReMe's local docs and the deployed documentation site have separate responsibilities.

- Keep `docs/` focused on content and assets used by `README.md` and `README_ZH.md`.
- Preserve README-linked pages under `docs/en/` and `docs/zh/`, including their relative
  paths, unless the README is updated in the same change.
- Keep README-required images under `docs/figure/`.
- Keep the README's main documentation index pointed at `docs.agentscope.io` or the
  `agentscope-ai/docs` repository, following the existing link style.
- Do not treat local README-supporting pages as the source for the deployed website.

The separate `agentscope-ai/docs` repository owns website content, navigation, versioning,
and deployment. Public ReMe pages live there under `reme/<version>/`. Make website changes
in that repository and follow its existing version-management conventions.

Do not add website build configuration or deployment workflows to ReMe unless the task
explicitly changes this repository boundary.

## Agent Guardrails

- Preserve unrelated user changes in a dirty working tree.
- Do not edit generated output when the source can be changed instead.
- Do not delete or rewrite user data to make a test pass.
- Avoid broad refactors unless they are necessary for the requested outcome.
- Do not introduce dependencies without a concrete need and repository-level justification.
- Treat network access, real credentials, and external service mutations as opt-in.
- State which validations passed and which were not run in the final handoff.

If a requirement is ambiguous, first infer intent from nearby code, tests, and schemas. Ask
the user only when the remaining choice would materially alter a public contract, user data,
or external system.
