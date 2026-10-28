# Framework compatibility baseline

This baseline precedes directory refactoring and records contracts supported at
`17fb9960`. It changes no runtime code, configuration, serialized schema or workspace layout.

| Contract | Coverage |
| --- | --- |
| Public import aliases retain object identity | `test_framework_contracts.py` |
| Existing built-in `(component_type, backend)` names remain discoverable | `fixtures/builtin_backends.json` and `test_framework_contracts.py` |
| Console target, module one-shot execution and plugin discovery groups | `test_framework_contracts.py` |
| Schema, enumeration and config source dependency boundaries | `test_module_boundaries.py` |
| Fresh Steps, shared per-invocation context, isolated responses and stream failure termination | `test_job_contracts.py` |
| Host model injection and live component replacement | `test_embedded_consumer_compat.py` |
| Standalone dependency ownership and startup rollback | `test_base_component.py` |
| Plugin contributions and application-local registries | `test_framework_contracts.py`, `test_plugin.py` and `test_component_registry.py` |
| CLI argument routing and service exposure | `test_reme_cli.py` and `test_service.py` |

The backend fixture is a reviewed minimum compatibility set. Adding a backend does
not fail the baseline; removing or renaming an existing one requires an intentional
compatibility decision. The fixture deliberately omits implementation paths, so
moving a class does not require changing its configuration name. Do not regenerate
the fixture merely to make a failing import or missing registration pass.

Public imports may become compatibility exports, but both paths must still return
the same object. Tests do not freeze private implementation locations, directory
counts or the complete set of exports.

Boundary checks resolve explicit imports, including relative, function-local and
type-only imports. Config loading retains its existing entry-point and logging
dependencies; schema code retains its existing `utils.hash_text` dependency.
These checks do not establish lightweight package imports: the root
package still bootstraps components and Steps eagerly. Add stronger boundaries
alongside the refactoring that makes them true.

Run focused checks from the repository root:

```bash
pytest tests/unit/test_framework_contracts.py tests/unit/test_module_boundaries.py tests/unit/test_job_contracts.py -v
```

Then run the main unit suite and repository lint as described in [AGENTS.md](../../AGENTS.md).
The module one-shot check uses a temporary config and workspace and runs no server
or model. Credentialed integration tests remain outside this baseline.
