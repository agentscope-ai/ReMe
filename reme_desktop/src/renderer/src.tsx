import { createRoot } from "react-dom/client";
import { useEffect } from "react";
import { defaultLanguage, translate } from "../shared/i18n.ts";
import { useDesktop } from "./desktop.ts";
import Companion from "./companion/index.tsx";
import Workspace from "./workspace.tsx";
import "./style.css";
import "./materials.css";
import "./workspace.css";
import "./companion/companion.css";

const view =
  new URLSearchParams(location.search).get("view") === "chat"
    ? "chat"
    : "companion";
document.documentElement.dataset.view = view;

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
    <Workspace {...bootstrap} />
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
