/// <reference types="vite/client" />

import { useEffect, useLayoutEffect, useSyncExternalStore } from "react";
import { createRoot } from "react-dom/client";
import { BrowserHost } from "../../client/src/browser-host.ts";
import { normalizeEndpoint } from "../../client/src/settings.ts";
import {
  HostContext,
  workbenchHost,
  type WorkbenchHost,
} from "../../workbench/src/host.ts";
import {
  iosFetch,
  nativeRequest,
  openNativeRecorder,
  platform,
  requestLocalNetwork,
} from "./native-bridge.ts";
import Workbench from "../../workbench/src/workspace.tsx";
import "../../workbench/src/style.css";
import "../../workbench/src/materials.css";
import "../../workbench/src/workspace.css";
import "../../workbench/src/mobile.css";

declare const __REME_VERSION__: string;

document.documentElement.dataset.platform = platform;

const browser = new BrowserHost({
  fetchRequest: iosFetch,
  async openExternal(url) {
    if (!(await nativeRequest({ action: "openExternal", value: url })).success)
      throw new Error(
        document.documentElement.lang === "zh-CN"
          ? "无法在浏览器中打开链接。"
          : "Could not open the link in the browser.",
      );
  },
});

const workbench = workbenchHost(browser);
const host: WorkbenchHost = {
  ...workbench,
  openRecorder: openNativeRecorder,
  async action(action) {
    if (action.type === "settings.update" && action.endpoint !== undefined)
      await requestLocalNetwork?.(normalizeEndpoint(action.endpoint));
    await workbench.action(action);
  },
};
window.remeHandleBack = () => {
  const dialog = [
    ...document.querySelectorAll<HTMLDialogElement>("dialog[open]"),
  ].at(-1);
  if (dialog) {
    if (dialog.dispatchEvent(new Event("cancel", { cancelable: true })))
      dialog.close();
    return true;
  }
  const menu = Array.from(
    document.querySelectorAll<HTMLElement>(".context-menu"),
  ).find(
    (element) =>
      !element.closest("[inert]") && element.getClientRects().length > 0,
  );
  if (menu) {
    menu.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "Escape",
        bubbles: true,
        cancelable: true,
      }),
    );
    return true;
  }
  return !window.dispatchEvent(
    new Event("reme-mobile-back", { cancelable: true }),
  );
};

function App() {
  const state = useSyncExternalStore(browser.subscribe, browser.snapshot);
  const { theme, language } = state.settings;
  useEffect(() => browser.start(), []);
  useLayoutEffect(() => {
    document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
    document.documentElement.dataset.theme = theme;
    const scheme = matchMedia("(prefers-color-scheme: dark)");
    const update = () =>
      void nativeRequest({
        action: "theme",
        value: theme === "system" ? (scheme.matches ? "dark" : "light") : theme,
      });
    update();
    scheme.addEventListener("change", update);
    return () => scheme.removeEventListener("change", update);
  }, [language, theme]);
  return (
    <HostContext.Provider value={host}>
      <Workbench state={state} version={__REME_VERSION__} mobile />
    </HostContext.Provider>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
