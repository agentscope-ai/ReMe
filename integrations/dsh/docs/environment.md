# Environment management

The **Environment** tab is the setup surface for a local ReMe install. It finds a Python interpreter on the
machine that runs DSH, installs `reme-ai[core]` into it, writes the workspace `.env`, and starts or stops the
local ReMe service — the work that otherwise needs several terminals.

Open it from **Plugins → ReMe bundle → reme-memory → Configure → Environment**.

The tab manages a **loopback** endpoint (`http://127.0.0.1:…` or `http://localhost:…`). With a remote service
URL it only reports whether that URL answers, and hides the local steps.

Everything the tab reports comes from the DSH **Host**, not from the browser: Python discovery, file access,
the LLM probe and the service health check all run on the Host machine. A browser on another machine therefore
sees the Host's environment, and the hero and the service card can never disagree about it.

## What each step does

| Section         | Capability                                                                                                                                                                                             |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Local Python    | Discovers PEP 514 registry entries, the `py` launcher, `PATH` and conda (base and named environments), labels their origin and `reme-ai` state, and stores the selection in `pythonPath`               |
| ReMe dependency | Checks the `reme-ai` version, the `[core]` extras and the `reme` CLI for the selected interpreter; installs or upgrades `reme-ai[core]` with an optional package index, a live log and cancel          |
| ReMe parameters | Edits the workspace `.env`; secrets stay in that file and are shown masked. Saving requires a non-empty `EMBEDDING_API_KEY` (use `EMPTY` when vector search is off). Provides an LLM connectivity test |
| Service control | Starts and stops the local ReMe service, showing PID, version and log; the launch injects UTF-8 and `NO_PROXY`, sets `enable_logo=false`, and runs inside the workspace so `.env` discovery works      |

The hero summarises the same state as three chips — service, Python, ReMe — plus the endpoint, the detection
time and the two refresh actions.

## Python discovery

ReMe needs **Python 3.11 or newer**. The tab probes these channels on the Host and keeps only interpreters that
answered its verification snippet (a `-c` probe reporting `sys.executable`, the version and the base prefix):

- Windows: PEP 514 registry entries (`HKLM`/`HKCU`), the `py` launcher list, `PATH`, and `conda env list`.
- macOS and Linux: `PATH` (`which -a`), the well-known `/usr/bin`, `/usr/local`, Homebrew (`/opt/homebrew`),
  `~/.pyenv/shims` and `~/.local/bin` locations, and `conda env list`.

Microsoft Store placeholder links under `WindowsApps` are skipped, because they are shell aliases rather than
interpreters. When conda is present, named environments are listed both under `conda env list` and as
interpreters, and the conda root is reported separately.

Detection is cached for 60 seconds; **Refresh** (in the hero) probes again. The recommended interpreter is
chosen in this order:

1. A named conda environment (`base` excluded) that already has a supported `reme-ai`.
2. Another interpreter that already has a supported `reme-ai`.
3. Any named conda environment, which is ready for a ReMe install.
4. The first usable interpreter.

Hover the **recommended** tag to see which rule applied. Picking another interpreter writes `pythonPath` and
nothing else; leaving it unset keeps following the recommendation. The list shows six rows and expands on
demand, with a filter over version, origin and path.

## ReMe dependency

The check asks the selected interpreter three questions and reports them as metrics:

- `reme-ai` metadata version (`importlib.metadata.version("reme-ai")`), and whether it satisfies the minimum
  this integration targets.
- `[core]` extras: whether `import agentscope` works. Without it, `auto_memory` and `auto_dream` are unavailable.
- `reme` CLI: the console script beside the interpreter, or the Python wrapper fallback used at start time.

**Install reme-ai[core]** runs `python -m pip install reme-ai[core]`, optionally with a package index (PyPI,
Tsinghua, Aliyun). **Upgrade** adds `-U`; **Reinstall** adds `--force-reinstall`, because pip otherwise reports a
present-but-broken install as "already satisfied". Large wheels are downloaded on first use; expect 5–20 minutes
depending on the network.

The installation is a Host-owned job:

- It keeps running when the configuration page closes, and reopening the tab restores it for the same
  interpreter. The Host looks the job up by interpreter, so two interpreters installing at the same time each
  see their own progress.
- Progress is polled by job id, so a second installation for another interpreter cannot replace what you see.
  The install answer is bound to the interpreter that asked for it, so a late response after an interpreter
  switch is dropped instead of showing another interpreter's job.
- A successful install re-checks `reme-ai` for that interpreter, so the dependency card and **Start service**
  follow the install without a manual **Check**.
- **Cancel** stops pip and its build children, and the interpreter stays locked until the process has actually
  exited. A process that ignores the request is force-killed; if even that is not confirmed, the job stays
  _cancelling_ and the interpreter stays locked, because a second installation must not run while the first is
  still unaccounted for. The tab keeps showing the job instead of reporting a completed cancellation.
- The log is written to `$DSH_HOME/reme-memory/install-<job>.log` and its tail is shown in place. Each install
  opens a new file and only the newest 20 are kept; the long-lived service log rotates to `service.log.1` once it
  passes 5 MB.
- When the plugin unloads, running jobs are stopped through the same path: a graceful signal first, then a
  force-kill, so a pip that ignores `SIGTERM` cannot outlive the Host.

## Workspace `.env`

ReMe reads credentials from the `.env` in the workspace directory at service start; the tab never stores them in
DSH settings.

| Group     | Keys                                                                                   |
| --------- | -------------------------------------------------------------------------------------- |
| LLM       | `LLM_BACKEND`, `LLM_MODEL_NAME`, `LLM_API_KEY`, `LLM_BASE_URL`                         |
| Embedding | `EMBEDDING_BACKEND`, `EMBEDDING_MODEL_NAME`, `EMBEDDING_API_KEY`, `EMBEDDING_BASE_URL` |

- The workspace directory defaults to `~/.reme` and must be absolute; changing it reloads the `.env` it points at.
- API keys are write-only: the Host returns a mask (last four characters) and a blank field means “keep the
  stored key”. Leaving a key untouched never rewrites it.
- Emptying a **secret** field and saving keeps the stored key, even when a replacement was typed first and then
  cleared: a credential cannot be deleted by emptying its control, so the blank field always means “keep what is
  stored”. Emptying a **non-secret** field and saving **clears** it. Only fields you edited are written, so the rest
  of the file keeps its comments, unknown keys and order.
- A save writes the edits that were submitted and leaves newer ones alone: typing another field while the save is in
  flight keeps that text, and the card says that unsaved edits are still pending.
- Saving validates the whole merged file, not just the values you typed: ReMe 0.4.0.x instantiates the embedding
  component unconditionally, so `EMBEDDING_API_KEY` must be non-empty even without vector search (`EMPTY` is the
  documented placeholder). A duplicated key is rewritten in every position, and the file keeps its own line
  ending.
- The write is atomic and owner-only (0600 on POSIX); the previous file is kept as `.env.bak`, so a failed save
  cannot lose unrelated keys.
- **Test LLM connection** runs a one-token request from inside the workspace with the selected interpreter, and
  adds a hint for `401`/`403` (key), `404` (model or base URL) and network errors. The request follows the
  protocol of the `LLM_BACKEND` in the `.env`: `openai`, `dashscope`, `deepseek` and `moonshot` use the
  OpenAI-compatible `/chat/completions` (Bearer auth), `anthropic` uses `/v1/messages` (`x-api-key`), `gemini`
  uses `/v1beta/models/<model>:generateContent` (`x-goog-api-key`), `ollama` uses `/api/chat` (a local server
  needs no key) and `xai` uses the OpenAI-compatible `/chat/completions` at `api.x.ai`.
- The base URL follows ReMe's own credentials: `LLM_BASE_URL` is read for the OpenAI-compatible providers and for
  Anthropic, while Gemini, Ollama and xAI carry their own endpoint and ignore it. A stale value left in the `.env`
  therefore cannot send the probe to a host the service never calls, and a configuration ReMe accepts is never
  reported as broken. The base URL is used as the provider's own SDK builds it, so a base that already carries the
  version suffix fails in the probe exactly as it fails in the service.
- ReMe reaches xAI through its gRPC SDK, so the xAI probe is a REST-level approximation of that channel.
- The probe reads the `.env` on disk, so a configuration that is only picked in the form is not tested before the
  file is saved; the button says so while it waits.
- Drafts are bound to the workspace they were typed in: switching the workspace drops them, so a key typed for
  one workspace cannot be saved into another.

## Connection testing

**Test connection** in the hero asks the Host to `POST /health_check` on the configured endpoint. A healthy answer
must contain a ReMe health report — a version, a health flag and its components — so a generic HTTP server that
answers `200` with `{}` is not reported as a running ReMe service. The Host reuses the same ReMe client as the rest
of the plugin, so the request, the authentication and the three-second timeout are the ones the service itself is
called with. When the settings carry an `authToken`, the Host sends it as `Authorization: Bearer`, scoped to that
endpoint.

## Service lifetime

Starting the service here is equivalent to running `reme start` in a terminal — with one deliberate difference:
**the service outlives DSH**. Closing DSH does not stop it, and stopping it is an explicit action in the tab.

That is why the plugin does not hand the service to DSH's local subprocess provider, which terminates the
processes it manages when the Host exits. The plugin owns a small service adapter instead:

- The launch is recorded in `$DSH_HOME/reme-memory/service.json` (pid, endpoint, workspace, command, start time)
  **after** the OS acknowledged the spawn, so a missing interpreter leaves no state behind.
- A service is reported as managed — and offered a **Stop** button — only when the record matches the endpoint on
  screen, the pid is still alive, its image is still a Python or `reme` executable, **and** its command line still
  carries the recorded `workspace_dir` and `service.port`. The command line rules out most recycled pids (a longer
  workspace path or another port never matches), though it cannot tell a pid apart from a user who starts the exact
  same command by hand. A record for port 2333 never claims a service on port 2444.
- When the evidence cannot be read (localized `tasklist`/`ps`, missing permissions), ownership stays unproven. Start
  and Stop both leave the process and the record alone: Stop refuses, and Start reports an error instead of dropping
  the record. Discarding it would orphan a live service that the tab could never stop again. The card then offers
  **Forget this record**, which deletes only `service.json` and never signals the process; the user confirms what to do
  with it themselves. The Host names the record that blocked the action, and **Forget** targets exactly that record —
  which may belong to another port, not to the endpoint on screen — together with its pid. When the record changed in
  the meantime, the Host refuses instead of dropping the new record silently.
- A service that occupies the port but fails the health check is reported as a conflict; if it is provably this
  plugin's service, the tab still offers **Stop**.
- **Stop** tree-kills the recorded process (process group on POSIX, `taskkill /T /F` on Windows), waits for it to
  exit, escalates (POSIX: `SIGTERM` then `SIGKILL`; Windows: force `taskkill` again), and clears the record only
  after the pid is gone. A service started elsewhere stays untouched and is only observed.
- A recorded launch that is still alive but never became healthy is shown as **Last launch not ready** with both
  **Stop** and **Start**: Stop clears the record, while Start makes the Host stop a provably owned previous launch or
  discard a proven-foreign one. It is not shown as "Starting" unless a start from this tab is actually in flight.
- **Start** is serialized: two requests cannot spawn two services, and a second request observes the launch
  already in flight. Before replacing a recorded launch that never became healthy, Start only kills it when
  ownership is proven; an unproven record is kept and reported, never silently dropped.
- **Auto-start** (`autoStart`, off by default) pulls the service up when DSH starts, if it is not already running.

Python probes and pip installs are bounded Host operations; the service is the only process that must outlive DSH.

## Platform support

| Area          | Notes                                                                                                                                                           |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DSH           | `0.1.7-rc.2`, `0.2.0-rc.2`, `0.2.1-alpha.2`, `0.2.1-alpha.4`, `0.2.1-rc.2`, `0.2.2-alpha.2`                                                                     |
| Node.js       | `^22.19.0` or `>=24.0.0`; current DSH releases need Node 24.9 or newer, and the compatibility matrix is validated with Node 24.16                               |
| Windows       | Tested on Windows 11 with Miniconda: registry/PATH/conda discovery, `tasklist`/`taskkill` for identity and stop, `PYTHONUTF8`/`PYTHONIOENCODING` for pip output |
| macOS / Linux | Discovery, detached spawn, identity lookup and stop are covered by unit tests that run a real child process; still less exercised on real machines than Windows |

## Troubleshooting

**No usable Python found.** Install Miniconda or any Python 3.11+, then press **Refresh**. Store placeholders
are filtered by design, so install a real distribution instead of the Store alias.

**The selected interpreter is not in the list.** The tab shows it with a “not detected” tag and still uses it.
Refresh, or pick a detected interpreter.

**Installation fails.** Open the log tail: missing wheels, a stale proxy or a restricted network usually show
there. Try a mirror. In a conda environment `pip` works, but a dedicated environment avoids dependency conflicts
with other projects.

**Saving `.env` is refused.** `LLM_API_KEY` and `EMBEDDING_API_KEY` must both be non-empty in the saved file;
use `EMPTY` for the embedding key when vector search is off. Emptying a secret field never reaches the file, so
the stored key is kept; a required key that is not stored at all still fails the save, and the file is left
unchanged.

**The service reports “Port occupied”.** Something else listens on the endpoint. Change the port in the service
URL (Memory settings) or stop the other program.

**The service does not become healthy.** The Start action polls for 120 seconds. If the port is answered but
`/health_check` never passes, the card switches to the conflict state so **Stop** stays available. Use **View log**
(the workspace `service.log`) for the startup trace — a missing `reme-ai`, an invalid `.env`, or a port conflict all
appear there.

**The service shows “Last launch not ready”.** A launch recorded by the plugin is still alive but never answered
`/health_check` (for example DSH closed while it was booting). Use **Stop** to clear it and start again. If the pid
was recycled by another program, that record is provably foreign: **Start** discards it and launches a fresh service.
If the identity cannot be verified at all, use **Forget this record** — it deletes only the plugin's record and
leaves the process untouched.

**The service shows as foreign.** It was started outside the plugin, so it is observed but not managed. Stop it
in its own terminal if you want the tab to manage the next launch.

**Stop is disabled.** The running service does not belong to this endpoint, or its record is gone. Restart it
from the tab to make it managed.

**LLM test fails.** The result distinguishes `401`/`403` (key), `404` (model name or base URL) and network
errors. The probe reads the workspace `.env`, so save before testing, and it follows the `LLM_BACKEND` there: a
native provider (Anthropic, Gemini, Ollama) is probed with its own protocol, not with an OpenAI-style request.

## Where the code lives

```text
integrations/dsh/src/environment/
├── types.ts          shared, browser-safe RPC types
├── env-fields.ts     the managed `.env` keys and the rules for updating them
├── dependency.ts     what one interpreter can host
├── endpoint-rules.ts the endpoints a local service can be managed on
├── host/             detection, install, service adapter, manager, gateway
└── client/           the tab, one module per step, copy and styles
```

The Host modules are the only place that touches processes or files; the client renders what the Host returns.
