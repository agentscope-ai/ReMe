# ReMe Studio

English | [简体中文](https://github.com/agentscope-ai/ReMe/blob/main/reme_studio/README_ZH.md)

ReMe Studio is the local web workspace for ReMe. It lets you browse and edit user-owned workspace files, explore memory
links, and chat with the ReMe Agent without moving durable memory into a separate application database. Search indexes,
graphs, and other derived metadata remain rebuildable from the source files.

![ReMe Studio workspace](https://raw.githubusercontent.com/agentscope-ai/ReMe/main/reme_studio/public/og.jpg)

## Installation

Install Studio together with ReMe's optional integrations:

```bash
pip install "reme-ai[core]"
```

For Studio without the other optional integrations, use `pip install "reme-ai[web]"`. The base `reme-ai` package is
headless and does not include the frontend assets.

The same prebuilt static workspace is available for Node.js applications:

```bash
npm install @agentscope-ai/reme_studio
```

Its static entry point is installed at `@agentscope-ai/reme_studio/dist-static/index.html`.

## Features

- **Workspace browsing**: browse the full workspace or focus on journal and knowledge files through dedicated views. The
  navigator refreshes as files change on disk.
- **Markdown editing and preview**: open multiple files in tabs, render Markdown front matter and GitHub Flavored
  Markdown, edit with Monaco, save with optimistic modification-time checks, and download files locally.
- **Memory graph**: inspect indexed wikilinks under the `wiki`, `personal`, and
  `procedure` knowledge roots, follow inbound and outbound links, and open the corresponding Markdown source.
- **Agent chat**: stream conversations with the read-only workspace Agent, see tool calls and token usage, and drag
  workspace files into the conversation as references.
- **Service management**: inspect service and component memory usage, review the effective redacted configuration and
  version, and rebuild derived indexes without modifying source memory files.
- **Personalization**: switch between English and Chinese, and use light, dark, or system appearance.

## Requirements

- Python 3.11 or newer with ReMe installed.
- A running ReMe HTTP service. Agent chat additionally requires a working Agent and model configuration.
- Node.js 22.13 or newer is required only when developing or building Studio from source.

See the [repository README](https://github.com/agentscope-ai/ReMe#readme) for ReMe installation and backend configuration.

## Development

Start ReMe from the repository root, then run the frontend in another terminal:

```bash
# Terminal 1, from the repository root
reme start

# Terminal 2
cd reme_studio
npm install
npm run dev
```

Open <http://localhost:3000>. The frontend connects to
`http://127.0.0.1:2333` by default. Override it when needed:

```bash
NEXT_PUBLIC_REME_API_URL=http://127.0.0.1:8000 npm run dev
```

## ReMe-hosted static build

ReMe can serve Studio from the same FastAPI process as its HTTP API. Build the static variant and restart ReMe:

```bash
cd reme_studio
npm ci
npm run build:static
cd ..
reme start
```

Open <http://127.0.0.1:2333>. The static build uses same-origin requests by default. For standalone static development,
run `npm run dev:static` and set
`VITE_REME_API_URL` to the running ReMe service URL when necessary.

The regular `npm run build` command remains the vinext/Sites deployment build;
`npm run build:static` creates `dist-static/` exclusively for FastAPI and Python package distribution.

## Configuration

The workspace hides dotfiles and dot-directories. It displays only Markdown and text files by default. Configure the
allowed extensions as a comma-separated list in `.env.local`:

```bash
NEXT_PUBLIC_REME_WORKSPACE_EXTENSIONS=md,txt,mdx
```

The memory graph requires an index built by ReMe. Rebuilding the index from the Studio settings regenerates derived data
from workspace files and does not modify the source memory.

## Checks

```bash
npm run format:check
npm run lint
npm run build
npm run build:static
npm test
```
