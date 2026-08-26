## Summary

<!-- Explain the problem and the smallest coherent change that addresses it. -->

## Related issue

<!-- Use "Fixes #123" when applicable. -->

## Contract and data impact

- [ ] No public configuration, schema, CLI, endpoint, streaming, or workspace-layout contract changes
- [ ] No user-owned memory files are deleted or rewritten
- [ ] Derived indexes, catalogs, graphs, caches, and metadata remain rebuildable

<!-- If any item is unchecked, describe the impact and migration or recovery path. -->

## Validation

<!-- List the exact checks run and their results. Explain relevant checks that were not run. -->

- [ ] Focused tests pass
- [ ] Unit tests pass, or omitted tests are explained below
- [ ] `pre-commit run --all-files` passes, or omitted checks are explained below
- [ ] Frontend checks were run when `reme_studio/` changed

## Checklist

- [ ] I reviewed the diff for unrelated changes and sensitive data
- [ ] Tests cover intentional behavior changes
- [ ] Defaults, schemas, and concise documentation were updated together when required
- [ ] Long-lived clients, tasks, services, and executors follow the application lifecycle

## Screenshots or additional notes

<!-- Include UI screenshots, compatibility notes, or follow-up work when relevant. -->
