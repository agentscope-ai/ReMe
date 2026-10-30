# Home plugin

Provides the `auto_bash` job through an independently enabled ReMe plugin.

Install it into the same Python environment as ReMe:

```bash
reme plugins install plugins/home
```

Enable it in an application config with `plugins: [home]`. The plugin then
contributes the `auto_bash` job, which can be run with
`reme start config=<your-config.yaml> job=auto_bash command=pwd`.

The job defaults to read-only command validation and a 60-second timeout. Set
`is_read_only=false` to disable command validation. Supplying `read_only_paths`
or `writable_paths` enables a file sandbox (Bubblewrap on Linux, Seatbelt on
macOS), with sandbox paths identical to their host paths. On Linux, the
`memory_limit_mib` cgroup limit defaults to 4096 MiB; 0 disables it. Supplying
the limit requires a path policy. macOS has no memory limit and warns when a
nonzero value is supplied.

Standard output and standard error each have a 1 MiB (1,048,576-byte) limit on
all backends. Exceeding either limit terminates the command and returns a
failure with bounded partial output and a diagnostic in standard error.
