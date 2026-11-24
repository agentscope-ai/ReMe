# ReMe Studio for desktop

See the [Studio guide](../README.md) to install dependencies and start the service.
Enter `npm run dev:desktop` from `reme_studio/` to open the companion and workbench.

The main process owns the service connection, notifications, and companion state. The sandboxed preload exposes typed
IPC to the windows. Remy, Proactive AI, Notebook, and Knowledge render through the shared workbench.

On macOS, closing the workbench hides its window. Click the ReMe Studio icon in the Dock or choose **Chat with Remy**
from the ReMe menu bar icon to reopen it.

From `reme_studio/`:

```bash
npm exec -w desktop -- install-electron
npm test -w desktop
npm run dist -w desktop
```

Building a desktop installer requires the host platform. The Electron app uses the ReMe release version.
