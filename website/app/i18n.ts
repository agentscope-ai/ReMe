"use client";

import { create } from "zustand";

export type Language = "zh" | "en";

const messages = {
  zh: {
    memoryWorkspace: "记忆工作区",
    newAgentChat: "新建 Agent 对话",
    newConversation: "新对话",
    workspace: "工作区",
    daily: "日记",
    knowledgeBase: "知识库",
    workspaceTab: "工作区",
    chatTab: "对话",
    dailyTab: "日记",
    knowledgeTab: "知识库",
    memoryGraph: "记忆图谱",
    memoryGraphShort: "图谱",
    memoryGraphLoading: "正在加载记忆图谱…",
    memoryGraphLoadFailed: "记忆图谱加载失败",
    memoryGraphEmpty: "这个分类还没有已索引的记忆链接",
    memoryGraphCounts: "{nodes} 个节点 · {edges} 条链接",
    memoryGraphZoomOut: "缩小",
    memoryGraphZoomIn: "放大",
    memoryGraphFit: "适应画布",
    memoryGraphIndexed: "已索引文件",
    memoryGraphDirection: "Wiki Link 方向",
    memoryGraphOpenFile: "打开 Markdown",
    memoryGraphOutbound: "出链 · {count}",
    memoryGraphInbound: "入链 · {count}",
    download: "下载",
    loadingWorkspace: "读取工作区…",
    connectionFailed: "无法连接 ReMe",
    emptyWorkspace: "工作区暂无文件",
    fileReadFailed: "文件读取失败",
    edit: "编辑",
    split: "分栏",
    preview: "预览",
    save: "保存",
    saving: "保存中",
    saved: "已保存",
    saveFailed: "保存失败：{error}",
    unknownError: "未知错误",
    closeCurrentTab: "关闭当前文件",
    closeOtherTabs: "关闭其他文件",
    discardUnsavedConfirm: "将丢弃 {count} 个文件中未保存的修改，确定关闭吗？",
    workspaceFileLimit:
      "仅显示最近的 {limit} 个文件，工作区中可能还有其他文件。",
    openingFile: "正在打开 {path}",
    chatFailed: "对话失败",
    chatTitle: "和你的记忆对话",
    chatDescription: "搜索、阅读和整理 ReMe 工作区中的内容。",
    promptRecent: "总结最近的记录",
    promptTasks: "有哪些事项值得继续？",
    promptIdeas: "查找关于 Agent Memory 的想法",
    you: "你",
    thinking: "思考过程",
    dataChunk: "数据",
    approvalChunk: "审批",
    usageChunk: "模型用量",
    unknownChunk: "事件 · {type}",
    toolCall: "调用参数",
    toolResult: "执行结果",
    inputTokens: "输入",
    outputTokens: "输出",
    streaming: "进行中",
    completed: "已完成",
    askWorkspace: "询问你的工作区…",
    send: "发送",
    composerHint: "Enter 发送 · Shift + Enter 换行",
    toggleNavigator: "切换文件导航",
    resizeNavigator: "调整工作区导航宽度",
    welcomeDescription: "从左侧打开 Markdown，或者开始一段 Agent 对话。",
    startChat: "开始对话",
    localFiles: "文件保留在你的本地工作区",
    switchLanguage: "切换语言",
    appearance: "外观",
    lightTheme: "浅色",
    darkTheme: "深色",
    systemTheme: "跟随系统",
    documentation: "文档资料",
    github: "GitHub",
    settings: "设置",
    settingsTitle: "ReMe 设置",
    settingsDescription: "查看运行状态并管理本地工作区服务。",
    settingsStatus: "状态",
    settingsIndex: "索引",
    settingsConfig: "配置",
    settingsVersion: "版本",
    closeSettings: "关闭设置",
    refresh: "刷新",
    loadingSettings: "正在读取…",
    processMemory: "进程内存",
    componentMemory: "组件内存",
    componentDetails: "组件明细",
    serviceOnline: "服务运行正常",
    indexTitle: "工作区索引",
    indexDescription: "从现有文件重新构建搜索索引。记忆文件不会被修改。",
    rebuildIndex: "重建索引",
    rebuildingIndex: "正在重建…",
    confirmReindexTitle: "确定重建索引？",
    confirmReindexDescription:
      "现有派生索引会被清空，然后根据工作区文件重新生成。",
    cancel: "取消",
    confirmReindex: "确认重建",
    indexRebuilt: "索引重建完成",
    effectiveConfig: "当前生效配置",
    redactedConfig: "敏感字段已由 ReMe 后端隐藏。",
    currentVersion: "当前版本",
    apiEndpoint: "服务地址",
    invalidResponse: "ReMe 返回了无法解析的响应（HTTP {status}）",
    requestFailed: "请求失败（HTTP {status}）",
    agentUnavailable: "无法连接 Agent（HTTP {status}）",
  },
  en: {
    memoryWorkspace: "Memory workspace",
    newAgentChat: "New Agent chat",
    newConversation: "New chat",
    workspace: "Workspace",
    daily: "Journal",
    knowledgeBase: "Knowledge base",
    workspaceTab: "Files",
    chatTab: "Chat",
    dailyTab: "Daily",
    knowledgeTab: "Knowledge",
    memoryGraph: "Memory graph",
    memoryGraphShort: "Graph",
    memoryGraphLoading: "Loading memory graph…",
    memoryGraphLoadFailed: "Unable to load memory graph",
    memoryGraphEmpty: "No indexed memory links in this category",
    memoryGraphCounts: "{nodes} nodes · {edges} links",
    memoryGraphZoomOut: "Zoom out",
    memoryGraphZoomIn: "Zoom in",
    memoryGraphFit: "Fit canvas",
    memoryGraphIndexed: "Indexed file",
    memoryGraphDirection: "Wiki Link direction",
    memoryGraphOpenFile: "Open Markdown",
    memoryGraphOutbound: "Outbound · {count}",
    memoryGraphInbound: "Inbound · {count}",
    download: "Download",
    loadingWorkspace: "Loading workspace…",
    connectionFailed: "Unable to connect to ReMe",
    emptyWorkspace: "No files in this workspace",
    fileReadFailed: "Unable to read file",
    edit: "Edit",
    split: "Split view",
    preview: "Preview",
    save: "Save",
    saving: "Saving",
    saved: "Saved",
    saveFailed: "Save failed: {error}",
    unknownError: "Unknown error",
    closeCurrentTab: "Close current file",
    closeOtherTabs: "Close other files",
    discardUnsavedConfirm:
      "Discard unsaved changes in {count} file(s) and close?",
    workspaceFileLimit:
      "Showing the {limit} most recent files. More files may exist in this workspace.",
    openingFile: "Opening {path}",
    chatFailed: "Chat failed",
    chatTitle: "Chat with your memory",
    chatDescription:
      "Search, read, and organize content in your ReMe workspace.",
    promptRecent: "Summarize my recent notes",
    promptTasks: "What should I follow up on?",
    promptIdeas: "Find my Agent Memory ideas",
    you: "You",
    thinking: "Thinking",
    dataChunk: "Data",
    approvalChunk: "Approval",
    usageChunk: "Model usage",
    unknownChunk: "Event · {type}",
    toolCall: "Call",
    toolResult: "Result",
    inputTokens: "input",
    outputTokens: "output",
    streaming: "Streaming",
    completed: "Completed",
    askWorkspace: "Ask about your workspace…",
    send: "Send",
    composerHint: "Enter to send · Shift + Enter for a new line",
    toggleNavigator: "Toggle file navigator",
    resizeNavigator: "Resize workspace navigator",
    welcomeDescription:
      "Open a Markdown file from the left, or start an Agent conversation.",
    startChat: "Start chatting",
    localFiles: "Files stay in your local workspace",
    switchLanguage: "Switch language",
    appearance: "Appearance",
    lightTheme: "Light",
    darkTheme: "Dark",
    systemTheme: "System",
    documentation: "Documentation",
    github: "GitHub",
    settings: "Settings",
    settingsTitle: "ReMe Settings",
    settingsDescription:
      "Inspect runtime status and manage the local workspace service.",
    settingsStatus: "Status",
    settingsIndex: "Index",
    settingsConfig: "Configuration",
    settingsVersion: "Version",
    closeSettings: "Close settings",
    refresh: "Refresh",
    loadingSettings: "Loading…",
    processMemory: "Process memory",
    componentMemory: "Component memory",
    componentDetails: "Component details",
    serviceOnline: "Service is running",
    indexTitle: "Workspace index",
    indexDescription:
      "Rebuild the search index from existing files. Memory files are not modified.",
    rebuildIndex: "Rebuild index",
    rebuildingIndex: "Rebuilding…",
    confirmReindexTitle: "Rebuild the index?",
    confirmReindexDescription:
      "The derived index will be cleared and regenerated from workspace files.",
    cancel: "Cancel",
    confirmReindex: "Confirm rebuild",
    indexRebuilt: "Index rebuilt",
    effectiveConfig: "Effective configuration",
    redactedConfig: "Sensitive fields are redacted by the ReMe backend.",
    currentVersion: "Current version",
    apiEndpoint: "Service endpoint",
    invalidResponse: "ReMe returned an invalid response (HTTP {status})",
    requestFailed: "Request failed (HTTP {status})",
    agentUnavailable: "Unable to connect to Agent (HTTP {status})",
  },
} as const;

export type TranslationKey = keyof typeof messages.zh;

export function translate(
  language: Language,
  key: TranslationKey,
  values: Record<string, string> = {},
): string {
  let text: string = messages[language][key];
  for (const [name, value] of Object.entries(values))
    text = text.replace(`{${name}}`, value);
  return text;
}

interface LanguageState {
  language: Language;
  hydrate: () => void;
  setLanguage: (language: Language) => void;
}

const applyLanguage = (language: Language) => {
  if (typeof document !== "undefined")
    document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
};

export const useLanguageStore = create<LanguageState>((set) => ({
  language: "zh",
  hydrate: () => {
    const saved = localStorage.getItem("reme-language");
    const language: Language =
      saved === "zh" || saved === "en"
        ? saved
        : navigator.language.startsWith("zh")
        ? "zh"
        : "en";
    applyLanguage(language);
    set({ language });
  },
  setLanguage: (language) => {
    localStorage.setItem("reme-language", language);
    applyLanguage(language);
    set({ language });
  },
}));

export function useI18n() {
  const language = useLanguageStore((state) => state.language);
  const setLanguage = useLanguageStore((state) => state.setLanguage);
  return {
    language,
    setLanguage,
    t: (key: TranslationKey, values?: Record<string, string>) =>
      translate(language, key, values),
  };
}
