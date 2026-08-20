# Repository automation

GitHub requires workflow files to live directly in `.github/workflows`, so filenames provide the organization:

- `ci-*`: automatically triggered validation, with a manual rerun entry point where useful.
- `policy-*`: repository contribution policies.
- `deploy-*`: deployment workflows.
- `release-*`: package and integration publishing workflows.
- `security-*`: security scanning and scheduled security checks.
- `_build-*`: reusable implementation workflows called by the entry-point workflows above.

Keep trigger selection, permissions, concurrency, and path filters in the entry-point workflow. Put repeated build and
verification jobs in a reusable workflow. A reusable workflow should use `workflow_call` and must not publish by itself.

Release workflows validate versions and build artifacts before publishing. Publishing credentials and environment
protection remain repository settings and must not be committed here.
