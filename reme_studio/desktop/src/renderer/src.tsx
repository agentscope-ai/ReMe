import {
  HostContext,
  workbenchHost,
  type WorkbenchHost,
} from "../../../workbench/src/host.ts";
import { createRoot } from "react-dom/client";
import { useEffect } from "react";
import { defaultLanguage, translate } from "../../../client/src/i18n.ts";
import { action, useDesktop } from "./desktop.ts";
import Companion from "./companion/index.tsx";
import remyAvatar from "../../../workbench/src/assets/remy-avatar.png";
import Workbench from "../../../workbench/src/workspace.tsx";
import "../../../workbench/src/style.css";
import "../../../workbench/src/materials.css";
import "../../../workbench/src/workspace.css";
import "./companion/companion.css";
import DesktopUpdate from "./desktop-update.tsx";

const view =
  new URLSearchParams(location.search).get("view") === "chat"
    ? "chat"
    : "companion";
document.documentElement.dataset.view = view;

const host: WorkbenchHost = {
  ...workbenchHost(window.desktop),
  desktop: {
    companionAvatar: remyAvatar,
    minimize: () => action({ type: "window.minimize" }),
    toggleMaximize: () => action({ type: "window.maximize" }),
    close: () => action({ type: "chat.close" }),
    setCompanionVisible: (visible) =>
      action({ type: "companion.visibility", visible }),
    resetCompanion: () => action({ type: "companion.reset" }),
  },
};

function App() {
  const bootstrap = useDesktop();
  const language = bootstrap?.state.settings.language ?? defaultLanguage;
  const theme = bootstrap?.state.settings.theme ?? "system";
  useEffect(() => {
    document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme =
      theme === "system" ? "light dark" : theme;
    if (bootstrap)
      document.documentElement.dataset.platform = bootstrap.platform;
  }, [language, theme, bootstrap?.platform]);
  if (!bootstrap)
    return view === "chat" ? (
      <p className="loading">{translate(language, "opening")}</p>
    ) : null;
  return view === "chat" ? (
    <HostContext.Provider value={host}>
      <Workbench
        {...bootstrap}
        companionVisible={bootstrap.layout.visible}
        railAccessory={
          <DesktopUpdate language={language} version={bootstrap.version} />
        }
      />
    </HostContext.Provider>
  ) : (
    <Companion {...bootstrap} />
  );
}

const root =
  import.meta.hot?.data.root ?? createRoot(document.getElementById("root")!);
if (import.meta.hot) {
  import.meta.hot.data.root = root;
  import.meta.hot.accept();
}
root.render(<App />);
