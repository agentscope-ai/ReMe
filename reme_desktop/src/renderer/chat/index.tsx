import { useEffect, useRef, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { action } from "../desktop.ts";
import { displayMessages } from "../../shared/messages.ts";
import type { DesktopState } from "../../shared/protocol.ts";

export default function Chat({ state }: { state: DesktopState }) {
  const [query, setQuery] = useState("");
  const [error, setError] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [endpoint, setEndpoint] = useState(state.settings.endpoint);
  const [scale, setScale] = useState(state.settings.scale);
  const history = useRef<HTMLDivElement | null>(null);
  const input = useRef<HTMLTextAreaElement | null>(null);
  const follow = useRef(true);
  const sending = useRef(false);
  const messages = displayMessages(state);
  const proactiveSignature = state.messages
    .filter((message) => message.source === "proactive")
    .map((message) => `${message.id}:${message.completed}`)
    .join("\n");
  const busy = Boolean(state.pending?.streaming) || state.status === "active";
  const ready = state.connection === "connected" && state.status !== "error";
  const run = async (value: Parameters<typeof action>[0]) => {
    setError("");
    try {
      await action(value);
      return true;
    } catch (error) {
      setError(
        error instanceof Error
          ? error.message.replace(
              /^Error invoking remote method '[^']+': Error: /,
              "",
            )
          : String(error),
      );
      return false;
    }
  };

  useEffect(
    () => setEndpoint(state.settings.endpoint),
    [state.settings.endpoint],
  );
  useEffect(() => setScale(state.settings.scale), [state.settings.scale]);
  useEffect(
    () =>
      window.desktop.onCommand((command) => {
        if (command.type === "settings.open") setSettingsOpen(true);
      }),
    [],
  );
  useEffect(() => {
    if (!state.chatVisible || !state.focusMessageId) return;
    const element = [
      ...(history.current?.querySelectorAll<HTMLElement>("[data-message-id]") ??
        []),
    ].find((node) => node.dataset.messageId === state.focusMessageId);
    element?.scrollIntoView({ block: "center" });
    follow.current = false;
  }, [state.focusMessageId, state.chatVisible]);
  useEffect(() => {
    if (follow.current && history.current)
      history.current.scrollTop = history.current.scrollHeight;
  }, [state.messages, state.pending?.text, state.chatVisible]);
  useEffect(() => {
    if (state.chatVisible && !settingsOpen) input.current?.focus();
  }, [state.chatVisible, settingsOpen]);

  useEffect(() => {
    const root = history.current;
    if (!root || !state.sessionId || !state.chatVisible || !state.chatFocused)
      return;
    const visible = new Set<string>();
    const acknowledged = new Set<string>();
    let timer: ReturnType<typeof setTimeout>;
    const acknowledge = () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        if (!document.hasFocus() || document.visibilityState !== "visible")
          return;
        const ids = [...visible].filter((id) => !acknowledged.has(id));
        if (ids.length) {
          ids.forEach((id) => acknowledged.add(id));
          void action({
            type: "messages.read",
            sessionId: state.sessionId!,
            ids,
          }).catch(() => ids.forEach((id) => acknowledged.delete(id)));
        }
      }, 600);
    };
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          const id = (entry.target as HTMLElement).dataset.messageId!;
          if (entry.isIntersecting) visible.add(id);
          else visible.delete(id);
        }
        acknowledge();
      },
      { root, threshold: 0.3 },
    );
    root
      .querySelectorAll<HTMLElement>("[data-proactive=true] .proactive-label")
      .forEach((element) => observer.observe(element));
    window.addEventListener("focus", acknowledge);
    return () => {
      clearTimeout(timer);
      observer.disconnect();
      window.removeEventListener("focus", acknowledge);
    };
  }, [
    state.sessionId,
    proactiveSignature,
    state.chatVisible,
    state.chatFocused,
  ]);

  const send = async () => {
    const value = query.trim();
    if (!value || busy || !ready || sending.current) return;
    sending.current = true;
    setQuery("");
    follow.current = true;
    if (!(await run({ type: "chat.send", query: value })))
      setQuery((current) => current || value);
    sending.current = false;
  };

  return (
    <main className="chat-shell">
      <header className="chat-titlebar">
        <div className="brand-mark" aria-hidden>
          R<span>·</span>
        </div>
        <div className="chat-heading">
          <h1>Remy</h1>
          <span>
            {state.connection !== "connected"
              ? "等待连接 ReMe"
              : state.behavior.label}
          </span>
        </div>
        <nav aria-label="对话操作">
          <button
            className="icon-button"
            aria-label="新建对话"
            disabled={busy || !state.workspace}
            onClick={() => {
              follow.current = true;
              void run({ type: "session.new" });
            }}
          >
            ＋
          </button>
          <button
            className={`icon-button ${settingsOpen ? "selected" : ""}`}
            aria-label="设置"
            aria-pressed={settingsOpen}
            onClick={() => setSettingsOpen(!settingsOpen)}
          >
            ⚙
          </button>
          <button
            className="icon-button"
            aria-label="收起聊天"
            onClick={() => void run({ type: "chat.close" })}
          >
            ×
          </button>
        </nav>
      </header>
      {settingsOpen && (
        <section className="settings" aria-label="桌面设置">
          <label>
            ReMe 服务地址
            <div className="endpoint-row">
              <input
                value={endpoint}
                onChange={(event) => setEndpoint(event.target.value)}
                spellCheck={false}
                aria-label="ReMe 服务地址"
              />
              <button
                className="primary"
                disabled={Boolean(state.pending?.streaming)}
                onClick={() => void run({ type: "settings.update", endpoint })}
              >
                连接
              </button>
            </div>
          </label>
          <label className="scale-row">
            角色大小 <span>{Math.round(scale * 100)}%</span>
            <input
              type="range"
              aria-label="角色大小"
              min="0.7"
              max="1.5"
              step="0.05"
              value={scale}
              onChange={(event) => setScale(Number(event.target.value))}
              onPointerUp={() => void run({ type: "settings.update", scale })}
              onKeyUp={() => void run({ type: "settings.update", scale })}
            />
          </label>
          <div className="quiet-row">
            <span>主动提醒</span>
            <button
              className="secondary"
              onClick={() =>
                void run({
                  type: "settings.update",
                  quietUntil:
                    state.settings.quietUntil > Date.now()
                      ? 0
                      : Date.now() + 30 * 60_000,
                })
              }
            >
              {state.settings.quietUntil > Date.now()
                ? "恢复提醒"
                : "安静 30 分钟"}
            </button>
          </div>
          <label className="quiet-row">
            <span>
              系统通知 <small>长任务完成、需要处理或角色隐藏时提醒</small>
            </span>
            <input
              type="checkbox"
              aria-label="系统通知"
              checked={state.settings.notifications}
              onChange={(event) =>
                void run({
                  type: "settings.update",
                  notifications: event.target.checked,
                })
              }
            />
          </label>
          {state.workspace && (
            <p className="workspace-path" title={state.workspace}>
              工作区：{state.workspace}
            </p>
          )}
          <p className="settings-note">
            快捷键 ⌘/Ctrl + Shift + Space 打开对话
          </p>
        </section>
      )}
      {state.connectionError && (
        <div className="connection-banner" role="status">
          <span>{state.connectionError}</span>
          <button onClick={() => setSettingsOpen(true)}>连接设置</button>
        </div>
      )}
      {error && (
        <div className="action-error" role="alert">
          <span>{error}</span>
          <button aria-label="关闭提示" onClick={() => setError("")}>
            ×
          </button>
        </div>
      )}
      <div
        ref={history}
        className="chat-history"
        aria-label="对话记录"
        onScroll={() => {
          if (history.current)
            follow.current =
              history.current.scrollHeight -
                history.current.scrollTop -
                history.current.clientHeight <
              48;
        }}
      >
        {messages.length === 0 && (
          <div className="empty-chat">
            <div className="empty-mark" aria-hidden>
              R<span>·</span>
            </div>
            <h2>有什么想和我说的？</h2>
            <p>从一句话开始，让记忆慢慢积累。</p>
            {state.connection !== "connected" && (
              <p className="start-hint">
                先启动 ReMe 服务：<code>reme start config=proactive</code>
              </p>
            )}
          </div>
        )}
        {messages.map((message) => (
          <article
            key={message.id}
            data-message-id={message.id}
            data-proactive={Boolean(message.proactive)}
            className={`message ${message.role}`}
          >
            {message.proactive && (
              <div className="proactive-label" data-message-id={message.id}>
                <span aria-hidden>✦</span> Remy 的主动提醒
              </div>
            )}
            <div className="message-body">
              {message.role === "user" ? (
                <p>{message.content}</p>
              ) : message.content ? (
                <Markdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    a: ({ href, children }) => (
                      <a
                        href={href}
                        onClick={(event) => {
                          event.preventDefault();
                          if (href) void run({ type: "link.open", url: href });
                        }}
                      >
                        {children}
                      </a>
                    ),
                    img: ({ alt }) => (
                      <span className="image-placeholder">{alt || "图片"}</span>
                    ),
                  }}
                >
                  {message.content}
                </Markdown>
              ) : message.streaming ? (
                <span className="thinking">
                  正在思考<span aria-hidden> · · ·</span>
                </span>
              ) : null}
            </div>
            {message.error && <p className="message-error">{message.error}</p>}
          </article>
        ))}
      </div>
      <footer className="chat-composer">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void send();
          }}
        >
          <textarea
            ref={input}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onPointerDown={() => void action({ type: "interaction" })}
            placeholder={busy ? "可以先写下，等我说完…" : "和 Remy 说点什么…"}
            aria-label="消息"
            rows={2}
            maxLength={100000}
            onKeyDown={(event) => {
              void action({ type: "interaction" });
              if (
                event.key === "Enter" &&
                !event.shiftKey &&
                !event.nativeEvent.isComposing &&
                event.nativeEvent.keyCode !== 229
              ) {
                event.preventDefault();
                void send();
              }
            }}
          />
          {state.pending?.streaming ? (
            <button
              type="button"
              className="stop-button"
              aria-label="停止回复"
              onClick={() => void run({ type: "chat.stop" })}
            >
              ■
            </button>
          ) : (
            <button
              type="submit"
              className="send-button"
              aria-label="发送消息"
              disabled={busy || !ready || !query.trim()}
            >
              ↑
            </button>
          )}
        </form>
        <div className="composer-meta">
          <span
            className={`connection-dot ${state.connection === "connected" ? "online" : ""}`}
          />
          <span>
            {state.connection === "connected"
              ? "记忆保存在 ReMe 工作区"
              : "正在等待 ReMe 服务"}
          </span>
          <span className="keyboard-hint">Shift + Enter 换行</span>
        </div>
      </footer>
    </main>
  );
}
