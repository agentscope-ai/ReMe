import { createRoot } from "react-dom/client";
import { useEffect, useSyncExternalStore } from "react";
import Workbench from "../../workbench/src/workspace.tsx";
import { HostContext, workbenchHost } from "../../workbench/src/host.ts";
import { BrowserHost } from "../../client/src/browser-host.ts";
import "../../workbench/src/style.css";
import "../../workbench/src/materials.css";
import "../../workbench/src/workspace.css";

const demo = import.meta.env.MODE === "demo";
const endpoint = (
  import.meta.env.VITE_REME_API_URL ||
  location.origin + (import.meta.env.DEV && !demo ? "/api" : "")
).replace(/\/$/, "");
const fetchRequest = demo
  ? (await import("../demo/service.ts")).demoService()
  : window.fetch.bind(window);
const browser = new BrowserHost(endpoint, undefined, fetchRequest, demo);
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
