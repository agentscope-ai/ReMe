# Open Source and Contributing

ReMe is open source and hosted on GitHub:

**https://github.com/agentscope-ai/ReMe**

---

## How to Contribute

Thank you for your interest in ReMe. ReMe is a file-first, self-evolving memory system for agents. Contributions are welcome
through issue reports, documentation improvements, additional tests, bug fixes, and new capabilities.

If this is your first time running ReMe locally, start with [Quick Start](./quick_start.md). If your change affects runtime
layers, Jobs, Steps, or components, read [ReMe Framework](./framework.md). If it affects workspace directories, frontmatter,
wikilinks, or chunking, read [Memory as File](./memory_as_file.md).

### 1. Before You Begin

Before investing in an implementation:

- Check [Open Issues](https://github.com/agentscope-ai/ReMe/issues) for an existing issue or discussion.
- If a related issue is still open, comment that you would like to work on it to avoid duplicate effort.
- If no issue exists, create one describing the context, expected behavior, possible implementation, and scope of impact.
- For larger feature changes, align with maintainers on interfaces, configuration, compatibility, and test strategy before
  submitting an implementation.

### 2. Local Development Environment

The core ReMe code is located in:

- `reme/`: Python package source, including configuration, components, services, Jobs, Steps, schemas, and utilities.
- `pyproject.toml`: project metadata, dependencies, optional dependencies, command entry points, and test configuration.
- `tests/`: unit and integration tests.

The project requires Python 3.11 or later. A virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,full]"
pre-commit install
```

### 3. Development Model

Before developing ReMe code, read [ReMe Framework](./framework.md). New or modified core capabilities should follow the
layers and call chain described there:

```text
CLI / Client -> Service -> Application -> Job -> Step -> Component / Workspace
```

In practice:

- Capabilities exposed to users or external systems should normally be orchestrated by a Job, then exposed by a Service as a
  CLI-, HTTP-, or MCP-callable interface.
- Reusable infrastructure belongs in `reme/components/`, with dependencies declared through `BaseComponent.bind()`.
- Atomic business operations belong in `reme/steps/` and access the file store, agent wrapper, catalog, LLM, and other
  components through `BaseStep.Ref`.
- Request, response, and persistent data structures belong in `reme/schema/` or `reme/enumeration/`. Do not scatter
  implicit structures through Step implementations.
- Configuration-driven defaults belong in `reme/config/default.yaml`, and the default configuration must remain runnable
  and testable.

When adding a Step or Job, pay particular attention to these conventions:

- Register implementations with `@R.register("<backend_name>")`. Registration names should be stable, clear, and match the
  configured `backend`.
- After adding a Step file, make sure its package `__init__.py` imports the module; otherwise, the registry will not load it.
- A Step should perform one atomic business operation. Cross-step flows belong in Job configuration or a dedicated
  orchestration Step.
- A Job composes Steps and selects normal, streaming, background, or scheduled execution. `enable_serve` controls whether it
  is externally exposed.
- When a Step needs components, prefer `BaseStep.Ref`. Do not reconstruct global components inside a Step or bypass
  `ApplicationContext`.
- File, index, graph, frontmatter, and wikilink behavior must preserve consistent workspace-relative path semantics.
- Add fast tests under `tests/unit/` for new capabilities. Put cross-component, LLM, embedding, or service behavior under
  `tests/integration/` when appropriate.

### 4. Code and Documentation Changes

Choose the appropriate entry point for the type of change:

| Change type | Primary location | Guidance |
|---|---|---|
| Configuration or startup behavior | `reme/config/`, `reme/application.py`, `reme/reme.py` | Keep the default configuration runnable and avoid breaking existing CLI, HTTP, and MCP entry points. |
| Component capability | `reme/components/` | Reuse `BaseComponent`, the registry, and context objects. |
| Job or Step | `reme/components/job/`, `reme/steps/` | Follow the Job -> Step model in [ReMe Framework](./framework.md), keep request and response schemas clear, and add corresponding tests. |
| Data structure | `reme/schema/`, `reme/enumeration/` | Preserve serialization compatibility and existing frontmatter and wikilink semantics. |
| Utility | `reme/utils/` | Keep function boundaries small and cover edge cases with unit tests. |
| User documentation | `docs/en/`, `README.md` | Update documentation when user-visible behavior changes. |

If a change involves an LLM, embeddings, an external service, file watching, or a background task, also describe its
dependencies, failure behavior, and local validation method.

### 5. Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/) to keep history clear.

Format:

```text
<type>(<scope>): <subject>
```

Common types:

- `feat`: new feature
- `fix`: bug fix
- `docs`: documentation only
- `style`: code-style change with no behavior change
- `refactor`: refactoring that neither fixes a bug nor adds a feature
- `perf`: performance improvement
- `test`: add or update tests
- `chore`: build, tooling, or maintenance work

Examples:

```bash
feat(search): add link expansion option
fix(file-graph): handle pending wikilinks after move
docs(memory): update auto memory guide
test(config): cover default yaml parsing
chore(pre-commit): update lint hooks
```

### 6. Pull Request Titles

PR titles should use the same format:

```text
<type>(<scope>): <description>
```

Requirements:

- Use `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `perf`, `style`, `build`, or `revert` as the type.
- Use lowercase letters, numbers, hyphens, or underscores for the scope.
- Keep the description short and state the actual effect of the PR.

Examples:

```text
feat(auto-memory): persist source conversation metadata
fix(markdown): keep wikilink aliases during edit
docs(en): add contribution guide
```

### 7. Pre-submit Checks

Before committing or opening a PR, run at least:

```bash
pre-commit run --all-files
pytest
```

For a localized code change, start with a narrower test set:

```bash
pytest tests/unit/test_search_step.py
pytest tests/unit/test_reme_cli.py
```

If `pre-commit` modifies files automatically, commit those changes and rerun the checks until everything passes.

The current pre-commit configuration includes YAML/TOML/JSON validation, private-key detection, trailing-whitespace checks,
`black`, `flake8`, `pylint`, and `pyroma`. The main formatting rules are:

- `black --line-length=120`
- `flake8 --max-line-length=120`
- `pylint --max-line-length=120`

Some integration tests may require an LLM, embeddings, or external service configuration. If you cannot run them locally,
state why they were skipped and what alternative validation you completed in the PR description.

### 8. Testing Requirements

Add tests according to the risk of the change:

- For a bug fix, first add a regression test that reproduces the issue.
- For a new Step, Job, or component, cover at least the main path and a failure path.
- For changes to shared logic such as indexes, graphs, wikilinks, frontmatter, or file operations, add edge cases.
- For changes to the CLI, services, or configuration parsing, cover the user-visible entry point.
- Documentation-only changes usually do not require new tests, but running `pre-commit run --all-files` is still recommended.

Place tests according to the existing structure:

- `tests/unit/`: fast tests that require no real external service.
- `tests/integration/`: integration tests spanning components or requiring external configuration.

### 9. Documentation Contributions

When a change affects how users install, configure, invoke, or understand ReMe, update the documentation as well.

Documentation lives under:

```text
docs/
```

Documentation should:

- Use clear titles that directly identify a capability or flow.
- Provide commands that can be copied and run.
- Use real repository paths such as `reme/config/default.yaml`, `reme/steps/`, and `tests/unit/`.
- Describe default behavior according to the current code, `pyproject.toml`, and default configuration.

---

## Getting Help

- Bugs and feature requests: [GitHub Issues](https://github.com/agentscope-ai/ReMe/issues)
- Project home: [GitHub Repository](https://github.com/agentscope-ai/ReMe)
- Documentation site: [https://reme.agentscope.io/](https://reme.agentscope.io/)

---

Thank you for contributing to ReMe. Your improvements help make long-term memory for agents more readable, controllable, and
maintainable.
