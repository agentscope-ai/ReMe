export const languages = ["zh", "en"] as const;
export type Language = (typeof languages)[number];
export const defaultLanguage: Language = "zh";
export type TranslationValues = Record<string, string | number>;

/** Chinese and English live under the same key, so the catalogs cannot drift. */
export const messages = {
  language: ["语言", "Language"],
  chinese: ["简体中文", "简体中文"],
  english: ["English", "English"],
  openChat: ["和 Remy 聊聊", "Chat with Remy"],
  unreadReminders: ["未读提醒（{count}）", "Unread reminders ({count})"],
  unreadBadge: ["{count} 条未读提醒", "Unread reminders: {count}"],
  noUnread: ["暂无未读提醒", "No unread reminders"],
  playAnimation: ["播放动作", "Play animation"],
  quiet: ["安静 30 分钟", "Quiet for 30 minutes"],
  resumeReminders: ["恢复主动提醒", "Resume proactive reminders"],
  resume: ["恢复提醒", "Resume reminders"],
  characterSize: ["角色大小", "Character size"],
  notifications: ["系统通知", "System notifications"],
  settings: ["设置", "Settings"],
  settingsMenu: ["设置…", "Settings…"],
  resetPosition: ["回到屏幕右下角", "Return to the bottom-right corner"],
  hideCharacter: ["隐藏角色", "Hide character"],
  showCharacter: ["显示角色", "Show character"],
  quit: ["退出 ReMe Desktop", "Quit ReMe Desktop"],
  operationFailed: ["桌面操作失败", "Desktop action failed"],
  startupFailed: ["ReMe Desktop 启动失败", "ReMe Desktop could not start"],
  trayTooltip: ["Remy · ReMe 桌面助手", "Remy · ReMe desktop companion"],
  animationIdle: ["待机", "Idle"],
  animationListening: ["聆听", "Listening"],
  animationThinking: ["思考", "Thinking"],
  animationRemembering: ["回忆", "Remembering"],
  animationSpeaking: ["对话", "Speaking"],
  animationReminding: ["主动提醒", "Proactive reminder"],
  animationHappy: ["完成", "Happy"],
  animationConfused: ["疑惑", "Confused"],
  animationSleeping: ["睡觉", "Sleeping"],
  animationDragged: ["拖动", "Dragged"],
  thinking: ["正在认真想…", "Thinking it through…"],
  working: ["正在查找和处理…", "Searching and working…"],
  remembering: ["正在回忆…", "Recalling…"],
  answering: ["正在回复…", "Replying…"],
  waiting: ["等你确认…", "Waiting for you…"],
  completed: ["完成啦", "All done"],
  failed: ["这次没有完成", "Could not finish this time"],
  interrupted: ["回复已停止", "Reply stopped"],
  idle: ["你的记忆，在这里", "Your memories, right here"],
  dragging: ["轻轻挪一挪", "Moving gently"],
  waitingConnection: ["等待连接 ReMe", "Waiting for ReMe"],
  reminderStatus: [
    "想到一件值得告诉你的事",
    "Something worth sharing with you",
  ],
  listening: ["我在听，慢慢说", "I’m listening. Take your time"],
  background: ["在整理一些信息…", "Organizing a few things…"],
  sleeping: ["休息一会儿", "Resting for a while"],
  ambient: ["安静陪着你", "Quietly keeping you company"],
  animationUnavailable: [
    "部分角色动画无法加载，已切换为待机动作",
    "Some animations could not load. Using idle instead",
  ],
  notifyWaiting: ["Remy 需要你处理一下", "Remy needs your attention"],
  notifyFailed: ["Remy 的回复未完成", "Remy could not finish the reply"],
  notifyCompleted: ["Remy 已完成回复", "Remy has finished replying"],
  notifyOpen: ["点击返回对应对话查看", "Click to return to the conversation"],
  reminderTitle: ["Remy 想到一件事", "Remy has something to share"],
  companion: ["Remy 桌面助手", "Remy desktop companion"],
  reminder: ["主动提醒", "Proactive reminder"],
  reminderHeading: ["REMY 想到一件事", "REMY HAS SOMETHING TO SHARE"],
  dismissReminder: ["收起提醒", "Dismiss reminder"],
  expandReminder: ["展开聊聊", "Open conversation"],
  snoozeReminder: ["10 分钟后", "In 10 minutes"],
  statusOpen: [
    "{text}，点击查看对话",
    "{text}. Click to view the conversation",
  ],
  characterAction: [
    "和 Remy 聊聊，按住拖动可移动角色",
    "Chat with Remy; hold and drag to move",
  ],
  characterHint: [
    "左键开聊 · 按住拖动 · 右键菜单",
    "Click to chat · Hold to drag · Right-click for menu",
  ],
  offlineAction: [
    "服务未连接，点击查看",
    "Service disconnected. Click for details",
  ],
  openStatus: ["打开对话查看状态", "Open chat to view status"],
  opening: ["正在打开 Remy…", "Opening Remy…"],
  chatActions: ["对话操作", "Conversation actions"],
  newConversation: ["新建对话", "New conversation"],
  closeChat: ["收起聊天", "Hide chat"],
  desktopSettings: ["桌面设置", "Desktop settings"],
  endpoint: ["ReMe 服务地址", "ReMe service URL"],
  connect: ["连接", "Connect"],
  notificationHint: [
    "长任务完成、需要处理或角色隐藏时提醒",
    "For long replies, attention requests, or hidden-character reminders",
  ],
  workspace: ["工作区：{path}", "Workspace: {path}"],
  shortcutHint: [
    "快捷键 ⌘/Ctrl + Shift + Space 打开对话",
    "Open chat with ⌘/Ctrl + Shift + Space",
  ],
  connectionSettings: ["连接设置", "Connection settings"],
  closeNotice: ["关闭提示", "Dismiss notice"],
  history: ["对话记录", "Conversation history"],
  emptyTitle: ["有什么想和我说的？", "What would you like to talk about?"],
  emptyDescription: [
    "从一句话开始，让记忆慢慢积累。",
    "Start with a message and let memories grow",
  ],
  startService: ["先启动 ReMe 服务：", "Start the ReMe service first:"],
  proactiveLabel: ["Remy 的主动提醒", "Proactive reminder from Remy"],
  image: ["图片", "Image"],
  thinkingPlaceholder: ["正在思考", "Thinking"],
  draftWhileBusy: ["可以先写下，等我说完…", "You can draft while I finish…"],
  messagePlaceholder: ["和 Remy 说点什么…", "Say something to Remy…"],
  message: ["消息", "Message"],
  stopReply: ["停止回复", "Stop reply"],
  sendMessage: ["发送消息", "Send message"],
  memoryLocation: [
    "记忆保存在 ReMe 工作区",
    "Memories stay in your ReMe workspace",
  ],
  waitingService: ["正在等待 ReMe 服务", "Waiting for the ReMe service"],
  newlineHint: ["Shift + Enter 换行", "Shift + Enter for a new line"],
  characterAnimation: ["Remy 角色动画", "Remy character animation"],
  assetFailed: [
    "角色素材加载失败，请检查安装文件",
    "Character assets could not load. Check the installation",
  ],
  assetLoadFailed: [
    "角色素材加载失败：{file}",
    "Could not load character asset: {file}",
  ],
  streamTooLarge: ["ReMe 流事件过大", "ReMe stream event is too large"],
  invalidStream: [
    "ReMe 返回了无效的流事件",
    "ReMe returned an invalid stream event",
  ],
  streamInterrupted: [
    "连接在结束标记之前中断，正在等待会话同步",
    "Connection interrupted before completion; waiting for session sync",
  ],
  replyError: ["回复出错", "Reply failed"],
  saveSettingsFailed: [
    "保存桌面设置失败：{error}",
    "Could not save desktop settings: {error}",
  ],
  missingJob: [
    "服务未启用 {job}，请使用 reme start config=proactive 启动",
    "The service has not enabled {job}. Start it with reme start config=proactive",
  ],
  missingWorkspace: [
    "ReMe 未返回工作区路径",
    "ReMe did not return a workspace path",
  ],
  subscriptionError: ["订阅出错", "Subscription failed"],
  subscriptionEnded: [
    "主动消息订阅已结束，正在重新连接",
    "The proactive subscription ended. Reconnecting",
  ],
  closed: ["桌面客户端已关闭", "The desktop client is closed"],
  stopBeforeEndpoint: [
    "请先停止当前回复，再切换服务",
    "Stop the current reply before switching services",
  ],
  invalidScale: [
    "角色大小需要在 70%–150% 之间",
    "Character size must be between 70% and 150%",
  ],
  invalidQuiet: ["免打扰时间无效", "Invalid quiet period"],
  invalidNotifications: ["系统通知开关无效", "Invalid notification setting"],
  invalidLanguage: ["语言需要是中文或英文", "Choose Chinese or English"],
  sessionBusy: [
    "请等待连接建立、当前会话空闲后再新建",
    "Wait until connected and the conversation is idle before starting a new one",
  ],
  invalidQuery: [
    "消息不能为空，且不超过 100000 字符",
    "Enter a message of no more than 100000 characters",
  ],
  sessionUnavailable: [
    "会话暂不可用，请检查连接或新建对话",
    "Conversation unavailable. Check the connection or start a new conversation",
  ],
  busy: [
    "Remy 正在忙，请稍等或先停止回复",
    "Remy is busy. Please wait or stop the current reply",
  ],
  replyIncomplete: [
    "回复未完成，可查看已同步的会话后再继续",
    "The reply is incomplete. Review the synced conversation before continuing",
  ],
  invalidEndpoint: [
    "服务地址需要是 HTTP/HTTPS 地址，不包含账号、查询参数或片段",
    "Use an HTTP/HTTPS service URL without credentials, query parameters, or fragments",
  ],
  invalidSettingsJson: [
    "配置文件不是有效 JSON，请修复后重试：{file}",
    "Invalid settings JSON. Fix the file and retry: {file}",
  ],
  invalidPresentation: [
    "presentation.json 需要是对象；原文件未修改",
    "presentation.json must contain an object; the file was preserved",
  ],
  invalidReceipts: [
    "presentation.json 的展示记录无效；原文件未修改",
    "Invalid presentation receipts; the file was preserved",
  ],
  invalidMessage: ["ReMe 会话消息格式无效", "Invalid ReMe session message"],
  invalidSnapshot: ["ReMe 会话快照格式无效", "Invalid ReMe session snapshot"],
  duplicateMessages: [
    "ReMe 会话快照包含重复消息",
    "ReMe session snapshot contains duplicate messages",
  ],
  invalidEvent: ["ReMe 主动消息事件格式无效", "Invalid ReMe proactive event"],
  invalidStatus: ["ReMe 会话状态无效", "Invalid ReMe session status"],
  missingActivities: [
    "ReMe 未返回活动状态，请更新服务与桌面客户端",
    "ReMe did not return activity state. Update the service and desktop client",
  ],
  invalidActivity: ["ReMe 活动状态无效", "Invalid ReMe activity state"],
  sessionMismatch: [
    "ReMe 事件与当前会话不一致，需要重新同步",
    "ReMe event does not match the current session. Resynchronization is required",
  ],
  invalidManifest: [
    "角色素材 manifest.json 无效",
    "Invalid character asset manifest.json",
  ],
  invalidTiming: [
    "角色动作时间表无效：{id}",
    "Invalid character animation timing: {id}",
  ],
  invalidSegments: [
    "角色分段时间表无效：{id}",
    "Invalid character animation segments: {id}",
  ],
  invalidFrame: [
    "角色帧图路径无效：{frame}",
    "Invalid character frame path: {frame}",
  ],
  missingAnimation: [
    "角色素材缺少必需动作或提醒分段",
    "Character assets are missing required animations or reminder segments",
  ],
  unknownAnimation: ["未知的角色动作", "Unknown character animation"],
  invalidSender: ["无效的桌面消息来源", "Invalid desktop message sender"],
  invalidAction: ["无效的桌面操作", "Invalid desktop action"],
  menuFromCharacter: ["请从角色打开菜单", "Open the menu from the character"],
  textRequired: ["消息需要是文本", "Message must be text"],
  invalidRead: ["无效的已读消息", "Invalid read acknowledgement"],
  unknownAction: ["未知的桌面操作", "Unknown desktop action"],
  httpError: ["ReMe {job}：HTTP {status}", "ReMe {job}: HTTP {status}"],
  jobFailed: ["ReMe {job} 执行失败", "ReMe {job} failed"],
  idleConnection: [
    "ReMe 连接长时间没有响应",
    "The ReMe connection has not responded for too long",
  ],
  expectedStream: [
    "ReMe {job} 未返回事件流，请使用 proactive 配置启动服务",
    "ReMe {job} did not return an event stream. Start the service with the proactive config",
  ],
} as const satisfies Record<string, readonly [string, string]>;

export type TranslationKey = keyof typeof messages;

export const animationLabels: Record<string, TranslationKey> = {
  idle: "animationIdle",
  listening: "animationListening",
  thinking: "animationThinking",
  remembering: "animationRemembering",
  speaking: "animationSpeaking",
  reminding: "animationReminding",
  happy: "animationHappy",
  confused: "animationConfused",
  sleeping: "animationSleeping",
  dragged: "animationDragged",
};

export function translate(
  language: Language,
  key: TranslationKey,
  values: TranslationValues = {},
): string {
  return messages[key][language === "zh" ? 0 : 1].replace(
    /\{(\w+)\}/g,
    (placeholder, name: string) =>
      Object.hasOwn(values, name) ? String(values[name]) : placeholder,
  );
}

export const translator =
  (language: Language) => (key: TranslationKey, values?: TranslationValues) =>
    translate(language, key, values);

export class UiError extends Error {
  constructor(
    readonly key: TranslationKey,
    readonly values: TranslationValues = {},
  ) {
    super(translate(defaultLanguage, key, values));
  }
}

export type ErrorData =
  { key: TranslationKey; values: TranslationValues } | { message: string };

export function serializeError(error: unknown): ErrorData {
  return error instanceof UiError
    ? { key: error.key, values: error.values }
    : { message: error instanceof Error ? error.message : String(error) };
}

export const deserializeError = (error: ErrorData): Error =>
  "key" in error
    ? new UiError(error.key, error.values)
    : new Error(error.message);

export function errorText(error: unknown, language: Language): string {
  return error instanceof UiError
    ? translate(language, error.key, error.values)
    : error instanceof Error
      ? error.message
      : String(error);
}
