/// <reference types="vite/client" />

import { useEffect, useLayoutEffect, useSyncExternalStore } from "react";
import { createRoot } from "react-dom/client";
import { defaultSettings } from "../../client/src/settings.ts";
import { BrowserHost } from "../../client/src/browser-host.ts";
import { HostContext, workbenchHost } from "../../workbench/src/host.ts";
import {
  nativeRequest,
  openNativeRecorder,
  platform,
} from "./native-bridge.ts";
import Workbench from "../../workbench/src/workspace.tsx";
import "../../workbench/src/style.css";
import "../../workbench/src/materials.css";
import "../../workbench/src/workspace.css";
import "../../workbench/src/mobile.css";

declare const __REME_VERSION__: string;

document.documentElement.dataset.platform = platform;

const browser = new BrowserHost(defaultSettings.endpoint, async (url) => {
  if (!(await nativeRequest({ action: "openExternal", value: url })).success)
    throw new Error(
      document.documentElement.lang === "zh-CN"
        ? "无法在浏览器中打开链接。"
        : "Could not open the link in the browser.",
    );
});

const host = {
  ...workbenchHost(browser),
  openRecorder: openNativeRecorder,
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
  const menu = document.querySelector<HTMLElement>(".context-menu");
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
