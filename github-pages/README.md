# ReMe documentation site

This package builds the VitePress site published at <https://reme.agentscope.io>. The canonical documentation lives in
`docs/`; selected product, integration, plugin, and benchmark READMEs are mirrored into a disposable source tree during
the build. Do not edit `.generated/` or `dist/`.

## Requirements

- Node.js 22.13 or newer
- npm

## Local development

```bash
cd github-pages
npm ci
npm run dev
```

The development server prints its local URL. Restart it after changing a mirrored README or `reme/config/default.yaml`
so the generated source tree and Job reference are refreshed. Changes under `docs/` are also refreshed on restart.

## Validation

```bash
npm test
npm run build
npm run preview
```

The test suite verifies bilingual core pages, canonical-source mappings, generated Job coverage, and disposable output.
The production build is written to `github-pages/dist/` for the existing GitHub Pages workflow.

## Sources

- `docs/`: canonical guides, VitePress configuration, theme, and brand assets
- `reme/config/default.yaml`: generated callable Job reference
- `reme_studio/README*.md`: ReMe Studio
- `typescript/README*.md`: TypeScript client and adapters
- `plugins/*/README*.md`: plugin guides
- `benchmark/*/README*.md`: benchmark guides
- `scripts/generate-content.mjs`: source mirroring and reference generation

When adding a canonical guide, add both `docs/zh/<name>.md` and `docs/en/<name>.md`, then include it in the appropriate
sidebar in `docs/.vitepress/config.mts`. Add repository-owned READMEs to `externalDocuments` in the generator rather than
duplicating their full content under `docs/`.

## Deployment

`.github/workflows/deploy-docs.yml` uses the reusable documentation build workflow and publishes `dist/` to GitHub
Pages. `public/CNAME` preserves the `reme.agentscope.io` custom domain.
