import { translate, useLanguageStore } from "../app/i18n";
import { WORKSPACE_CHANGED } from "../app/studio-mode";
import type { AppConfig, StreamChunk } from "../app/types";
import { createDemoWorkspace, DEMO_STORAGE_KEY } from "./workspace";
import { scenarios } from "./scenarios";

const content = import.meta.glob<string>("./content/**/*.md", {
  eager: true,
  query: "?raw",
  import: "default",
});
const workspaces = new Map<string, ReturnType<typeof createDemoWorkspace>>();
export function demoWorkspace() {
  const language = useLanguageStore.getState().language;
  if (!workspaces.has(language)) {
    const prefix = `./content/${language}/`;
    const seeds = Object.fromEntries(
      Object.entries(content)
        .filter(([path]) => path.startsWith(prefix))
        .map(([path, text]) => [path.slice(prefix.length), text]),
    );
    let storage: Storage | undefined;
    try {
      storage = localStorage;
    } catch {
      /* Editing remains available in memory. */
    }
    workspaces.set(
      language,
      createDemoWorkspace(seeds, storage, `${DEMO_STORAGE_KEY}-${language}`),
    );
  }
  return workspaces.get(language)!;
}
export const REME_API_ENDPOINT = translate(
  useLanguageStore.getState().language,
  "demoTitle",
);
export async function getAppConfig(): Promise<AppConfig> {
  return {
    app_name: "ReMe Studio Demo",
    workspace_dir: "/demo",
    daily_dir: "daily",
    digest_dir: "digest",
    resource_dir: "resources",
  };
}
export async function listWorkspaceFiles(extensions: string[]) {
  return {
    paths: Object.keys(demoWorkspace().files()).filter((path) =>
      extensions.includes(path.split(".").pop()!),
    ),
    limited: false,
  };
}
export async function readWorkspaceFile(path: string) {
  return demoWorkspace().read(path);
}
export async function saveWorkspaceFile(
  path: string,
  content: string,
  expectedMtime?: string,
) {
  const stat = demoWorkspace().save(path, content, expectedMtime);
  window.dispatchEvent(new Event(WORKSPACE_CHANGED));
  return stat;
}
export async function getGraphSnapshot() {
  return demoWorkspace().graph();
}
export async function streamChat(
  query: string,
  _sessionId: string | undefined,
  signal: AbortSignal,
  onChunk: (chunk: StreamChunk) => void,
) {
  const language = useLanguageStore.getState().language;
  const scenario = scenarios[language].find((item) => item.question === query);
  const emit = (
    chunk_type: StreamChunk["chunk_type"],
    chunk: StreamChunk["chunk"],
    extra = {},
  ) => {
    signal.throwIfAborted();
    onChunk({ chunk_type, chunk, done: chunk_type === "done", ...extra });
  };
  emit("reply_start", "");
  if (scenario) {
    emit(
      "tool_call",
      { query: scenario.tag, tags: [scenario.tag] },
      { tool_call_id: "search", tool_call_name: "search" },
    );
    emit("tool_result", demoWorkspace().search(scenario.tag, [scenario.tag]), {
      tool_call_id: "search",
      tool_call_name: "search",
    });
    for (const path of scenario.paths) {
      emit(
        "tool_call",
        { path },
        { tool_call_id: path, tool_call_name: "read" },
      );
      emit("tool_result", demoWorkspace().read(path).content, {
        tool_call_id: path,
        tool_call_name: "read",
      });
    }
  }
  const answer =
    scenario?.answer ||
    (language === "zh"
      ? "这是预设对话演示，未运行模型。请选择上方的示例问题，或使用「搜索 / 标签」检索和编辑工作区。"
      : "This is a scripted conversation without a model. Choose a sample question above, or use Search / Tags to explore and edit the workspace.");
  for (const part of answer.match(/[\s\S]{1,24}/gu) || [answer]) {
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => {
        signal.removeEventListener("abort", abort);
        resolve();
      }, 20);
      function abort() {
        clearTimeout(timer);
        reject(signal.reason);
      }
      signal.addEventListener("abort", abort, { once: true });
    });
    emit("content", part);
  }
  emit("reply_end", "");
  emit("done", "");
}
