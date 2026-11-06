import { createRoot } from "react-dom/client";
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
  if (!bootstrap)
    return view === "chat" ? <p className="loading">正在打开 Remy…</p> : null;
  return view === "chat" ? (
    <Chat state={bootstrap.state} />
  ) : (
    <Companion {...bootstrap} />
  );
}

createRoot(document.getElementById("root")!).render(<App />);
