import { createRoot } from "react-dom/client";
import { useEffect } from "react";
import { defaultLanguage, translate } from "../shared/i18n.ts";
import { useDesktop } from "./desktop.ts";
import Companion from "./companion/index.tsx";
import Chat from "./chat/index.tsx";
import "./style.css";

const view =
  new URLSearchParams(location.search).get("view") === "chat"
    ? "chat"
    : "companion";
document.documentElement.dataset.view = view;

function App() {
  const bootstrap = useDesktop();
  const language = bootstrap?.state.settings.language ?? defaultLanguage;
  useEffect(() => {
    document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
  }, [language]);
  if (!bootstrap)
    return view === "chat" ? (
      <p className="loading">{translate(language, "opening")}</p>
    ) : null;
  return view === "chat" ? (
    <Chat state={bootstrap.state} />
  ) : (
    <Companion {...bootstrap} />
  );
}

createRoot(document.getElementById("root")!).render(<App />);
