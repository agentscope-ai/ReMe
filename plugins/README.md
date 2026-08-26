# ReMe Plugins

This directory contains installable extensions of ReMe itself. A plugin exposes its package through the `reme.plugins`
Python entry-point group and declares backend classes plus a low-priority `ApplicationConfig` fragment under
`application_defaults` in that package's `plugin.yaml`.

Manage plugin packages with the local CLI:

```bash
reme plugins list
reme plugins show auto-fin
reme plugins show daily-paper
reme plugins install reme-auto-fin
reme plugins install reme-daily-paper
reme plugins install ./plugins/auto-fin --editable
reme plugins install ./plugins/daily_paper --editable
reme plugins validate auto-fin
reme plugins validate daily-paper
reme plugins uninstall auto-fin
reme plugins uninstall daily-paper
```

Installation and activation are separate. Enable an installed plugin for one application through its config or CLI:

```bash
reme start plugins='["auto-fin","daily-paper"]'
```

Adapters for external agent hosts belong in [`../integrations`](../integrations/README.md).
