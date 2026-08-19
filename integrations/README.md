# Agent Integrations

This directory contains host-specific adapters that connect external agents to ReMe. An integration may use the host's
plugin API, hooks, MCP configuration, or client interface, but it does not extend ReMe's runtime through the
`reme.plugins` entry-point group.

Installable extensions of ReMe itself belong in [`../plugins`](../plugins/README.md).
