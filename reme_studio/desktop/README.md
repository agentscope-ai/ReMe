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

`pack` and `dist` reuse `dist.noindex/`, keeping generated apps out of macOS Spotlight. Use `npm run dev:desktop`
for previews instead of creating a separate packaged app for every change. Custom preview output directories should
also end in `.noindex`.

To upgrade on macOS, quit ReMe Studio and replace `/Applications/ReMe Studio.app` with the new app from the DMG.
On Windows, run the new installer with the existing installation directory and user scope, or use the in-app updater.
Windows and Linux AppImage in-app updates replace the installed app and restart it without an installation wizard.
Upgrades preserve Desktop settings and the ReMe workspace. GitHub preview releases do not require platform signing
credentials; macOS uses the downloads page. See the [release guide](../README.md#release) for distribution details.
