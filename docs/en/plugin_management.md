# Plugin Management

ReMe plugins are ordinary Python distributions discovered through the `reme.plugins` entry-point group. Installing a
plugin makes it available to the current Python environment; it does not enable the plugin in every ReMe application.

Keep these two operations separate:

```text
reme plugins install ...        install a package into the current Python environment
plugins: [auto-fin]             enable an installed plugin for one Application
```

Plugin package management is local-only. It does not run through a ReMe HTTP or MCP service and never edits application
configuration files automatically.

A typical plugin workflow has three stages:

1. Install ReMe and the plugin distribution.
2. Configure the plugin's runtime environment as described in the
   [ReMe environment-variable guide](../../README.md#environment-variables).
3. Start an Application with the plugin explicitly enabled, for example
   `reme start plugins='["auto-fin"]'`.

## List installed plugins

```bash
reme plugins list
```

The table shows the plugin entry-point name, Python distribution, version, and plugin contract:

```text
PLUGIN    DISTRIBUTION   VERSION  FORMAT
--------  -------------  -------  --------
auto-fin  reme-auto-fin  0.1.0    manifest
```

`manifest` plugins use the current package-level `plugin.yaml` contract. `legacy` plugins use the compatible Python
descriptor contract.

A manifest separates backend registration from application configuration:

```yaml
backends:
  example_step: example_plugin.steps:ExampleStep

application_defaults:
  jobs:
    example:
      backend: base
      steps:
        - backend: example_step
```

`application_defaults` is a partial `ApplicationConfig`. It is kept below the manifest's `backends` namespace because
backend import declarations are part of plugin discovery and are not application configuration.

Use JSON when another local tool needs structured output:

```bash
reme plugins list --json
```

To compare installed plugins with one application config:

```bash
reme plugins list --config daily_cookbook
```

The optional `ENABLED` column reflects only the `plugins` list resolved from that config. A command-line override used
by another running process is not a global enable state.

## Install a plugin package

Install a published distribution:

```bash
reme plugins install reme-auto-fin
```

Install or upgrade a pinned version:

```bash
reme plugins install 'reme-auto-fin==0.1.0'
reme plugins install reme-auto-fin --upgrade
```

Install a local plugin project:

```bash
reme plugins install ./plugins/auto-fin
```

Use editable mode while developing it:

```bash
reme plugins install ./plugins/auto-fin --editable
```

ReMe invokes pip through the same Python interpreter that runs the `reme` command. Pip remains responsible for package
resolution, downloads, dependency changes, and build execution. Install only packages and local projects you trust.

After installation, confirm the discovered plugin name:

```bash
reme plugins list
reme plugins validate auto-fin
```

## Inspect a plugin

```bash
reme plugins show auto-fin
```

For a manifest plugin, the result includes its registered backend names and default Job names. JSON output is also
available:

```bash
reme plugins show auto-fin --json
```

`show` identifies the package contract without constructing a ReMe Application.

## Validate a plugin

Validate an installed plugin:

```bash
reme plugins validate auto-fin
```

Validate a local project before installation:

```bash
reme plugins validate ./plugins/auto-fin
```

Validation checks the entry point, `plugin.yaml`, backend imports and component types, registry collisions, merged
`application_defaults`, and the resulting `ApplicationConfig`. Validation imports plugin backend modules, so run it
only for trusted code.

## Enable a plugin in a service

Installation alone does not load plugin code into an Application. Enable plugins explicitly in configuration:

```yaml
plugins:
  - auto-fin
```

Or add them for one service launch:

```bash
reme start plugins='["auto-fin"]'
```

When `config` is omitted, ReMe loads `default.yaml`. The plugin's `application_defaults` are merged below that config,
so explicit config values and CLI overrides win. This mapping is an `ApplicationConfig` fragment, not a separate
configuration schema. The plugin backends are registered only in that Application's local registry.

After the default HTTP service starts, access plugin Jobs through ReMe's CLI client or HTTP:

```bash
reme auto_fin topics="黄金,AI,存储芯片"
```

```bash
curl -s http://127.0.0.1:2333/auto_fin \
  -H 'Content-Type: application/json' \
  -d '{"topics":"黄金,AI,存储芯片"}'
```

When the application uses an MCP service, service-enabled plugin Jobs appear as MCP tools instead.

To add the plugin to another application config, select it explicitly:

```bash
reme start config=daily_cookbook plugins='["auto-fin"]'
```

## Uninstall a plugin

Use the plugin entry-point name, not necessarily the distribution name:

```bash
reme plugins uninstall auto-fin
```

Skip pip's confirmation prompt when needed:

```bash
reme plugins uninstall auto-fin --yes
```

ReMe resolves `auto-fin` to the distribution that provides it, such as `reme-auto-fin`. If one distribution provides
multiple plugin entry points, the command lists the other plugins that will also be removed.

Uninstallation does not rewrite user configuration. Remove the plugin from relevant `plugins` lists yourself;
otherwise the next Application startup fails explicitly because the configured plugin is no longer installed. Restart
already-running ReMe processes after installing, upgrading, or uninstalling packages.

## Troubleshooting

### Plugin is installed but unavailable

Check that the `reme` command and pip package share one Python interpreter:

```bash
reme plugins list
python -c 'import sys; print(sys.executable)'
```

Using `reme plugins install` avoids the most common interpreter mismatch because it runs `python -m pip` with ReMe's
own interpreter.

### Plugin is installed but not loaded

Add its entry-point name to the Application's `plugins` list. ReMe intentionally has no global enable/disable state.

### Startup reports that the plugin is not installed

The active config still enables a missing plugin. Reinstall it or remove the corresponding name from `plugins`.

### Changes are not visible in a running service

Plugin discovery and backend registration happen during Application construction. Restart the service after changing
installed packages.
