# ReMe status view maintenance

The settings action `reme_status` returns one read-only snapshot and opens the packaged MCP App.
The view consumes that initial result; only the Refresh button makes a further `reme_status` call.
It never calls memory-write tools, changes settings, or contacts the ReMe URL directly.

Edit `src/`, then build the self-contained installed asset:

```bash
npm ci
npm run build
```

Commit the generated `../plugins/reme/ui/status.html` with its source so installing from a checkout
requires only the existing Python runtime. Node is needed only to rebuild the view. The bundle uses
OpenAI MCP Extensions 0.1.0's app transport, includes its license, and loads no external scripts.
The resource advertises fullscreen; the host controls actual placement, including a settings modal.

Validation from the repository root:

```bash
pytest tests/unit/test_codex_plugin_mcp.py -q
REME_CODEX_STATUS_UI=1 pytest tests/integration/test_codex_plugin_status_ui.py -q
```

The opt-in browser test needs Playwright with Chromium and uses a synthetic host for protocol and
rendering checks. Its output is not a real Codex screenshot. Native Codex placement must be checked
in the app when access is available; user documentation keeps genuine screenshot placeholders.
