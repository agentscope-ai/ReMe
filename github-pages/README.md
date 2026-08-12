# ReMe GitHub Pages

This directory contains the standalone Vite documentation site published at <https://reme.agentscope.io>. The
GitHub Pages fallback is <https://agentscope-ai.github.io/ReMe/>. It does not depend on the ReMe Studio application in
`website/`.

## Requirements

- Node.js 22.13 or newer
- npm

## Local development

From the repository root:

```bash
cd github-pages
npm install
npm run dev
```

Open the URL printed by Vite, normally <http://localhost:5173/>. The development server watches the frontend source.
When a repository Markdown file changes, restart the development command to regenerate the documentation content.

For subsequent installs or CI-compatible dependency installation, use:

```bash
npm ci
```

## Preview the production build

Build and start the preview server:

```bash
npm run build
npm run preview
```

Open the URL printed by Vite, normally <http://localhost:4173/>. Production assets use relative paths so the same build
works on both the custom domain and the GitHub Pages project path.

The generated `dist/` and `.generated/` directories are disposable build output and are excluded from Git.

## Documentation sources

The build script reads the canonical repository files directly. Do not edit generated copies under `.generated/` or
`dist/`.

- `README.md` and `README_ZH.md`: project introductions
- `docs/en/` and `docs/zh/`: English and Chinese guides
- `docs/figure/`: documentation images
- `website/README.md` and `website/README_ZH.md`: ReMe Studio guide
- `cookbook/*/README*.md`: research workflow guides
- `benchmark/{beam,longmemeval,pibench,toolmemory}/README*.md`: benchmark guides and results
- `skills/reme_memory/SKILL.md`: ReMe Memory skill guide
- `AGENTS.md`: repository development guide

To add or reorganize a document in the site navigation, update
[`scripts/generate-content.mjs`](./scripts/generate-content.mjs). Presentation and interaction code lives in `src/`.

## Project structure

```text
github-pages/
├── index.html
├── package.json
├── scripts/
│   └── generate-content.mjs
├── src/
│   ├── main.js
│   └── styles.css
└── vite.config.js
```

## Deployment

The repository workflow `.github/workflows/pages.yml` builds this directory and publishes `dist/` to GitHub Pages.
It runs after relevant documentation or site files change on `main`, and it can also be started manually from the
GitHub Actions page.

The repository's **Settings → Pages → Build and deployment → Source** must be set to **GitHub Actions**. Its custom
domain must be set to `reme.agentscope.io`; `public/CNAME` preserves that domain in the published artifact.

Useful links:

- ReMe documentation: <https://reme.agentscope.io>
- GitHub Pages fallback: <https://agentscope-ai.github.io/ReMe/>
- ReMe repository: <https://github.com/agentscope-ai/ReMe>
