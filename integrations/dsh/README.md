# ReMe Memory for DeepSeek Harness

This DSH bundle follows the same separation used by QwenPaw's embedded ReMe integration:

- the main agent receives durable memory guidance;
- the model can call `reme_search` explicitly;
- completed user turns are submitted to `auto_memory` in the background;
- `auto_dream` runs as an independent daily maintenance task.

The ReMe HTTP service remains the owner of workspace files, indexes, memory extraction, and dream consolidation. The
plugin does not copy or rewrite those files.

## Requirements

- DeepSeek Harness `0.1.0-rc.7` or later compatible `0.1.x` release
- Node.js `^22.19.0` or `>=24`
- A running ReMe HTTP service with the `search`, `auto_memory`, and `auto_dream` jobs enabled

Start ReMe against the workspace that should own the agent's memory:

```bash
reme start workspace_dir=/absolute/path/to/workspace
```

Install this bundle into a DSH profile:

```bash
dsh plugin --profile default add ./integrations/dsh
```

## Configuration

The default endpoint is `http://127.0.0.1:2333`. Set `REME_URL`, or use the existing `REME_HOST` and `REME_PORT`
variables. Bundle configuration can be added to `cordis.patch.yml`:

```yaml
- insert:
    - id: reme-memory
      name: '@deepseek-ai/cordis-plugin-group'
      group: true
      isolate:
        remeMemory: true
      config:
        - id: reme-memory-runtime
          name: '@agentscope-ai/reme-dsh-memory'
          config:
            endpoint: http://127.0.0.1:2333
            language: zh
            autoMemoryInterval: 5
            autoDreamEnabled: true
            dreamCron: '0 23 * * *'
```

| Option | Default | Meaning |
| --- | --- | --- |
| `endpoint` | `http://127.0.0.1:2333` | ReMe HTTP service URL |
| `language` | `en` | Memory guidance language: `en` or `zh` |
| `autoMemoryEnabled` | `true` | Capture completed main-agent turns |
| `autoMemoryInterval` | `5` | Submit after this many completed user turns |
| `autoDreamEnabled` | `true` | Enable daily dream maintenance |
| `dreamCron` | `0 23 * * *` | Daily schedule in the DSH process's local timezone |
| `rootAgentsOnly` | `true` | Keep prompt injection and capture out of subagents |
| `requestTimeoutMs` | `10000` | Search request timeout |
| `backgroundTimeoutMs` | `3600000` | Auto-memory and auto-dream timeout |

`dreamCron` intentionally accepts only the daily form `<minute> <hour> * * *`. This keeps the bundle dependency-free
and makes the maintenance schedule explicit.

## Runtime behavior

Memory guidance is injected as a source-attributed DSH user message rather than a system-prompt fragment. DSH presets
may declare a complete persona and replace other system prompt contributions; a durable plugin message remains visible,
replayable, and eligible for normal compaction.

Only direct human user messages and assembled assistant messages are sent to `auto_memory`. Plugin context and tool
results are excluded so recalled text cannot be stored again as if the user had said it. DSH event sequence numbers are
used to create stable ReMe message IDs, and DSH session IDs are mapped to fixed-length hashed ReMe session IDs.

Auto-memory calls are serialized per session and are never awaited by the model turn. If a request fails, its turns are
put back into the in-process queue and retried after later activity, with one final attempt during session disposal. The
current first version does not yet persist that retry queue across a DSH process crash.

## Testing

```bash
cd integrations/dsh
npm ci
npm run typecheck
npm test
npm pack --dry-run
```

The plugin is authored in TypeScript under `src/`. `npm run build` emits ESM JavaScript, declarations, and source maps
to the ignored `dist/` directory. The `prepare` lifecycle builds the plugin for local-directory and Git installs, while
the npm tarball contains only the compiled `dist/` output, this README, and the DSH bundle patch.

## Publishing

Publishing is intentionally manual. Update `package.json` and `package-lock.json` to the release version, merge that
change, then run the **Publish ReMe DSH integration to npm** workflow with the same version and the desired npm tag.
The repository must provide an `NPM_TOKEN` Actions secret with publish access to the `@agentscope-ai` scope. The
workflow type-checks, tests, packs, uploads the exact tarball as an artifact, rejects an already published version, and
publishes that tarball with npm provenance.
