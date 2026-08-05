# ReMe Workspace

Local web workspace for browsing ReMe files, editing and previewing Markdown,
and streaming conversations with the ReMe Agent.

## Development

Requirements: Node.js 22.13+ and a running ReMe HTTP service.

```bash
# From the repository root, start ReMe in one terminal.
reme start

# Start the web interface in another terminal.
cd website
npm install
npm run dev
```

Open <http://localhost:3000>. The frontend connects to
`http://127.0.0.1:2333` by default. Override it when needed:

```bash
NEXT_PUBLIC_REME_API_URL=http://127.0.0.1:8000 npm run dev
```

The workspace hides dotfiles and dot-directories. It displays only Markdown and
text files by default. Configure the allowed extensions as a comma-separated
list in `.env.local`:

```bash
NEXT_PUBLIC_REME_WORKSPACE_EXTENSIONS=md,txt,mdx
```

Useful checks:

```bash
npm run lint
npx tsc --noEmit
npm run build
npm test
```
