import { createRoot } from "react-dom/client";
import { useEffect, useSyncExternalStore } from "react";
import Workbench from "../../workbench/src/workspace.tsx";
import { HostContext, workbenchHost } from "../../workbench/src/host.ts";
import { BrowserHost } from "../../client/src/browser-host.ts";
import { defaultSettings } from "../../client/src/settings.ts";
import "../../workbench/src/style.css";
import "../../workbench/src/materials.css";
import "../../workbench/src/workspace.css";

const endpoint = (
  import.meta.env.VITE_REME_API_URL ||
  (import.meta.env.DEV ? defaultSettings.endpoint : location.origin)
).replace(/\/$/, "");
const browser = new BrowserHost({
  defaultEndpoint: endpoint,
  fetchRequest: import.meta.env.DEV
    ? (input, init) => {
        const url = new URL(
          input instanceof Request ? input.url : input,
          location.href,
        );
        if (url.origin !== defaultSettings.endpoint) return fetch(input, init);
        // Keep local management requests same-origin without exposing the dev proxy as the service URL.
        const target = new URL(
          `/api${url.pathname}${url.search}`,
          location.origin,
        );
        return fetch(
          input instanceof Request ? new Request(target, input) : target,
          init,
        );
      }
    : undefined,
});
const host = workbenchHost(browser);

function App() {
  const state = useSyncExternalStore(browser.subscribe, browser.snapshot);
  const { theme, language } = state.settings;
  useEffect(() => browser.start(), []);
  useEffect(() => {
    document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme =
      theme === "system" ? "light dark" : theme;
  }, [language, theme]);
  return (
    <HostContext.Provider value={host}>
      <Workbench state={state} version={__REME_VERSION__} />
    </HostContext.Provider>
  );
}

document.documentElement.dataset.view = "chat";
document.documentElement.dataset.platform = "web";
createRoot(document.getElementById("root")!).render(<App />);
