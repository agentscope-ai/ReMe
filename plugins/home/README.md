# Home plugin

Provides the `auto_bash` job through an independently enabled ReMe plugin.

Install it into the same Python environment as ReMe:

```bash
reme plugins install plugins/home
```

Enable it in an application config with `plugins: [home]`. The plugin then
contributes the `auto_bash` job. Start `reme start config=<your-config.yaml>`
in one terminal, then call `reme auto_bash command=pwd` from another.

The job always uses a file sandbox (Bubblewrap on Linux, Seatbelt on macOS),
with `workspace_dir` read-only by default, network access blocked, and a minimal
child environment. Missing or unavailable isolation fails without host execution.
Files inside the workspace remain readable, including any `.env` files placed there.
Keep credentials outside the workspace.

The job also defaults to read-only command validation and a 60-second timeout.
`is_read_only=false` disables command validation while keeping the sandbox active.
Configure `read_only_paths` and `writable_paths` only on the Step; request-level
path grants are rejected. Existing path arguments must move to Step configuration.
For example, a trusted Step can grant workspace writes:

```yaml
steps:
  - backend: auto_bash_step
    read_only_paths: []
    writable_paths: ["/absolute/path/to/workspace"]
```

On Linux, Bubblewrap and a writable delegated cgroup v2 are required. The
`memory_limit_mib` cgroup limit defaults to 4096 MiB; 0 disables it. macOS has
no memory limit and warns when a nonzero value is supplied. Direct `AutoBash`
instances also default to a sandbox with their working directory read-only;
trusted callers may inject a different backend.

Standard output and standard error each have a 1 MiB (1,048,576-byte) limit on
all backends. Exceeding either limit terminates the command and returns a
failure with bounded partial output and a diagnostic in standard error.
