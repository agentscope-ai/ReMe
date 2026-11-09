export const languages = ["zh", "en"] as const;
export type Language = (typeof languages)[number];
export const defaultLanguage: Language = "zh";
export type TranslationValues = Record<string, string | number>;

/** Chinese and English live under the same key, so the catalogs cannot drift. */
export const messages = {
  documentProperties: ["文档属性", "Document properties"],
  invalidFrontmatter: [
    "文档属性无法解析，已保留原始内容供检查。",
    "Document properties could not be parsed. The original content is shown below.",
  ],
  createdTime: ["创建时间", "Created"],
  modifiedTime: ["修改时间", "Modified"],
  resizeSidebar: ["调整侧栏宽度", "Resize sidebar"],
  hideSidebar: ["收起侧栏", "Hide sidebar"],
  showSidebar: ["展开侧栏", "Show sidebar"],
  knowledge: ["知识库", "Knowledge"],
  workspaceTab: ["工作区", "Workspace"],
  dailyTab: ["每日记忆", "Daily"],
  digestTab: ["知识库", "Knowledge"],
  navigation: ["页面导航", "Navigation"],
  companionCaption: ["你的记忆，日渐生长", "A little more you, every day"],
  inbox: ["收件箱", "Inbox"],
  reminders: ["提醒列表", "Reminders"],
  remyUpdates: ["消息", "Updates"],
  taskResults: ["任务结果", "Task results"],
  filterTask: ["按任务筛选", "Filter by task"],
  allTasks: ["全部任务", "All tasks"],
  backToChat: ["返回当前对话", "Back to current chat"],
  continueConversation: ["继续这段对话", "Continue this conversation"],
  inboxCaption: ["值得告诉你的新发现", "Good things worth sharing"],
  taskInboxCaption: [
    "定时任务的结果都在这里",
    "Results from your scheduled tasks",
  ],
  noTaskResults: [
    "定时任务的结果会出现在这里",
    "Scheduled task results will appear here",
  ],
  inboxUserUnread: ["未读", "Unread"],
  inboxUserRead: ["已读", "Read"],
  inboxShowMore: ["显示更多", "Show more"],
  inboxShowLess: ["收起列表", "Show less"],
  inboxPrevious: ["上一页", "Previous page"],
  inboxNext: ["下一页", "Next page"],
  inboxRetry: ["重试", "Retry"],
  inboxRefresh: ["刷新", "Refresh"],
  inboxDetails: ["查看原因", "View details"],
  inboxPartial: [
    "部分收件箱内容暂时无法读取",
    "Some inbox content could not be read",
  ],
  noUpdates: ["新的发现会出现在这里", "New discoveries will find you here"],
  noUpdatesHint: [
    "Remy 会在合适的时候，带来值得关注的信息。",
    "Remy will share something useful when the time is right.",
  ],
  earlier: ["更早", "Earlier"],
  today: ["今天", "Today"],
  conversation: ["对话", "Conversation"],
  privateMemory: ["本地记忆 · 与你同行", "Local memory. Always with you."],
  welcomeHint: [
    "你的 Proactive Agent，记住偏好、执行任务，在合适的时候主动提醒。",
    "Your proactive agent: remembers you, handles tasks, and shares timely updates.",
  ],
  promptScheduledQuery: [
    "每天早上 8 点，帮我总结昨晚美股表现。",
    "Every morning at 8, summarize the previous night's US stock market.",
  ],
  promptTemporaryQuery: [
    "5 分钟后提醒我起来活动一下。",
    "In 5 minutes, remind me to get up and stretch.",
  ],
  promptProactiveQuery: [
    "帮我关注黄金价格，有重要变化时提醒我。",
    "Follow gold prices and let me know about significant changes.",
  ],
  general: ["通用", "General"],
  appearance: ["外观", "Appearance"],
  about: ["关于", "About"],
  settingsCaption: ["让 Remy 更合你的习惯", "Make yourself at home"],
  generalHint: [
    "连接你的 ReMe 服务与本地知识。",
    "Connect to your ReMe service and local knowledge.",
  ],
  appearanceHint: [
    "为你的工作空间，选择舒服的样子。",
    "A quieter, more comfortable space to think.",
  ],
  remyHint: [
    "留一点空间，让 Remy 陪着你。",
    "A small companion for your everyday.",
  ],
  notificationSettings: ["通知", "Notifications"],
  notificationSettingsHint: [
    "决定 Remy 如何把新发现告诉你。",
    "Choose how Remy shares new discoveries.",
  ],
  theme: ["显示模式", "Theme"],
  systemTheme: ["跟随系统", "System"],
  lightTheme: ["浅色", "Light"],
  darkTheme: ["深色", "Dark"],
  interfaceLanguage: ["界面语言", "Interface language"],
  serviceConnection: ["服务连接", "Service connection"],
  workspaceLocation: ["工作区位置", "Workspace location"],
  connected: ["已连接", "Connected"],
  reconnecting: ["连接中", "Connecting"],
  showPet: ["显示桌面角色", "Show desktop companion"],
  hidePet: ["隐藏桌面角色", "Hide desktop companion"],
  quietHint: [
    "暂时收起提醒，记忆仍会继续生长。",
    "Pause reminders while your memories keep growing.",
  ],
  aboutHint: [
    "一个懂得记忆，也懂得陪伴的本地助手。",
    "A local companion that remembers, and grows with you.",
  ],
  openGithub: ["在 GitHub 上查看", "View on GitHub"],
  openDocs: ["使用文档", "Documentation"],
  versionLabel: ["版本", "Version"],
  minimizeWindow: ["最小化", "Minimize"],
  maximizeWindow: ["最大化或还原", "Maximize or restore"],
  knowledgeCaption: [
    "让每一次记录，都有所归属",
    "A place for everything you remember",
  ],
  searchKnowledge: ["搜索记忆与文档…", "Search memories and documents…"],
  searchAction: ["搜索", "Search"],
  refreshFiles: ["刷新知识库", "Refresh knowledge"],
  files: ["文档", "Documents"],
  memoryTags: ["记忆标签", "Memory tags"],
  tagsGlobalHint: [
    "全工作区的标签与文档数；多个标签匹配任意一个。",
    "Workspace-wide tags and file counts. Multiple tags match any.",
  ],
  moreTags: ["更多标签", "More tags"],
  clearFilters: ["清除筛选", "Clear filters"],
  noFiles: ["这里还没有文档", "No documents here yet"],
  noMatches: ["没有找到匹配的文档", "No matching documents"],
  selectDocument: [
    "每一段记忆，都值得再次发现",
    "Every memory is worth rediscovering",
  ],
  selectDocumentHint: [
    "从左侧打开文档，或通过标签寻找相关记忆。",
    "Open a document on the left, or follow a tag to something related.",
  ],
  reading: ["阅读", "Reading"],
  documentLoading: ["正在打开文档…", "Opening document…"],
  retry: ["重试", "Try again"],
  limitedFiles: [
    "已展示最近 5000 个文件，可切换目录缩小范围。",
    "Showing the latest 5,000 files. Choose a directory to narrow the view.",
  ],
  nextPage: ["下一页", "Next page"],
  previousPage: ["上一页", "Previous page"],
  pageLabel: ["第 {page} / {total} 页", "Page {page} of {total}"],
  tagsUnavailable: [
    "标签浏览暂不可用，请检查服务是否启用标签索引与 list_tag_files。",
    "Tag browsing is unavailable. Check the tag index and list_tag_files job.",
  ],
  workspaceUnavailable: [
    "当前服务未启用文档浏览功能。",
    "Document browsing is not enabled on this service.",
  ],
  invalidWorkspaceRequest: [
    "文档请求或路径无效。",
    "Invalid document request or path.",
  ],
  invalidTheme: ["无效的显示模式。", "Invalid theme."],
  tagFilterUnavailable: [
    "标签筛选未生效，请检查标签索引后重试。",
    "The tag filter could not be applied. Check the tag index and retry.",
  ],
  speechSettings: ["语音", "Voice"],
  voiceAutoSend: ["语音识别后自动发送", "Send after transcription"],
  voiceEnabled: ["自动语音播报", "Automatic narration"],
  voiceReplies: ["朗读新的对话回复", "Read new replies aloud"],
  voiceReminders: ["朗读主动提醒", "Read proactive reminders aloud"],
  voiceVolume: ["语音音量", "Voice volume"],
  voiceHint: [
    "自动播报和手动朗读都会读完整消息；长文分段准备，按顺序播放。安静模式会暂停自动播报。",
    "Automatic and manual narration read the complete message, preparing long text in parts and playing them in order. Quiet mode pauses automatic narration.",
  ],
  voiceInputHint: [
    "按住话筒说话，松开识别后默认自动发送；关闭自动发送后只填入输入框。录音发送到当前 ReMe 服务识别。",
    "Hold the microphone, then release to transcribe and send by default. Turn off auto-send to insert an editable draft instead. Recordings are sent to your connected ReMe service for recognition.",
  ],
  voiceShortcut: [
    "按住 ⌘/Ctrl + R 说话，松开结束；⌘/Ctrl + Shift + M 切换录音，Esc 取消",
    "Hold ⌘/Ctrl + R to speak, release to finish; ⌘/Ctrl + Shift + M toggles recording; Esc cancels",
  ],
  holdToSpeak: ["按住说话", "Hold to speak"],
  finishRecording: ["结束录音", "Finish recording"],
  cancelRecording: ["取消录音", "Cancel recording"],
  readAloud: ["朗读这条回复", "Read this reply aloud"],
  stopSpeech: ["停止朗读", "Stop narration"],
  speechOpeningMic: ["正在打开麦克风…", "Opening microphone…"],
  speechListening: ["我在听你说…", "Listening to you…"],
  speechRecording: [
    "录音中 {seconds} / 24 秒 · 松开结束",
    "Recording {seconds} / 24 seconds · Release to finish",
  ],
  speechTranscribing: ["正在识别…", "Transcribing…"],
  speechLoading: ["正在准备语音…", "Preparing narration…"],
  speechNotReady: [
    "语音功能正在准备，请稍后再试。",
    "Voice is getting ready. Please try again shortly.",
  ],
  speechPreparingProgress: [
    "正在准备第 {part} / {total} 段",
    "Preparing part {part} of {total}",
  ],
  speechPaused: ["朗读已暂停", "Narration paused"],
  speechRetry: ["继续朗读", "Resume narration"],
  speechPlaying: ["正在朗读…", "Reading aloud…"],
  speechProgress: ["第 {part} / {total} 段", "Part {part} of {total}"],
  speechAsrUnavailable: [
    "当前 ReMe 服务未启用语音识别",
    "Speech recognition is not enabled on this ReMe service",
  ],
  speechTtsUnavailable: [
    "当前 ReMe 服务未启用语音合成",
    "Speech synthesis is not enabled on this ReMe service",
  ],
  speechCancelled: ["语音操作已取消", "Voice operation cancelled"],
  speechNoText: [
    "这条回复没有可朗读的文字",
    "This reply has no text to read aloud",
  ],
  speechNoInput: ["未检测到语音", "No speech detected"],
  speechMicDenied: [
    "无法使用麦克风。请在系统设置中允许 ReMe Desktop 使用麦克风，再重试。",
    "Microphone access is unavailable. Allow ReMe Desktop to use the microphone in system settings, then try again.",
  ],
  speechMicMissing: [
    "没有找到可用麦克风，请连接设备后重试",
    "No microphone was found. Connect one and try again",
  ],
  speechMicFailed: [
    "录音未能启动，请检查麦克风后重试",
    "Recording could not start. Check your microphone and try again",
  ],
  speechInvalidRecording: [
    "录音格式无效或超过时长限制，请重新录音",
    "The recording is invalid or too long. Please record again",
  ],
  speechInvalidTranscript: [
    "识别服务返回了无效结果，请重试",
    "The recognition service returned an invalid result. Please try again",
  ],
  speechInvalidAudio: [
    "语音合成返回了不完整或无效的音频，请重试",
    "Speech synthesis returned incomplete or invalid audio. Please try again",
  ],
  speechAudioTooLarge: [
    "语音结果过大，请尝试较短的内容",
    "The audio response is too large. Try shorter text",
  ],
  speechPlaybackFailed: [
    "语音播放失败，可以重新点击喇叭重试",
    "Narration could not play. Select the speaker to retry",
  ],
  invalidVoiceSettings: [
    "语音设置无效，音量需要在 0 到 100% 之间",
    "Invalid voice settings. Volume must be between 0 and 100%",
  ],
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
  activityLevel: ["空闲动作", "Idle activity"],
  activityLevelHint: [
    "偶尔摸摸鱼、翻翻便签，动作之间会休息一会儿",
    "An occasional fish cuddle or note to read, with a rest between gestures",
  ],
  activityCalm: ["安静", "Calm"],
  activityNormal: ["自然", "Natural"],
  activityLively: ["活泼", "Lively"],
  sleepAfter: ["多久后打盹", "Nap after"],
  sleepAfterHint: [
    "从最后一次互动开始计时，忙碌时会保持清醒",
    "Time since your last interaction; Remy stays awake while working",
  ],
  sleepNever: ["不自动打盹", "Never automatically"],
  sleepAfterMinutes: ["{minutes} 分钟", "{minutes} minutes"],
  invalidAnimationSettings: [
    "萌宠动作设置无效",
    "Invalid companion animation settings",
  ],
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
  animationPettingFish: ["摸鱼", "Petting a fish"],
  animationWalking: ["走路", "Walking"],
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
  taskBackground: ["正在执行任务：{task}", "Working on task: {task}"],
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
  dismissBubble: ["收起气泡", "Dismiss bubble"],
  reply: ["Remy 的回复", "Remy’s reply"],
  replyHeading: ["REMY 的回复", "REMY’S REPLY"],
  expandReminder: ["展开聊聊", "Open conversation"],
  snoozeReminder: ["10 分钟后", "In 10 minutes"],
  statusOpen: [
    "{text}，点击查看对话",
    "{text}. Click to view the conversation",
  ],
  characterAction: [
    "打开主窗口并置前，按住拖动可移动角色",
    "Bring the main window to the front; hold and drag to move",
  ],
  characterHint: [
    "单击展示主窗口并置前 · 按住拖动 · 右键菜单",
    "Click to bring the main window to the front · Hold to drag · Right-click for menu",
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
  explainReminder: ["详细解读", "Explain in detail"],
  explainReminderQuery: [
    "请详细解读下面这条主动提醒；涉及外部事实或数据时，请核实来源并附链接：\n\n{content}",
    "Explain this proactive reminder in detail. Verify external facts or data and include source links:\n\n{content}",
  ],
  image: ["图片", "Image"],
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
  voiceShortcutHint: [
    "按住 Ctrl + R 语音输入",
    "Hold Ctrl + R for voice input",
  ],
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
  petting_fish: "animationPettingFish",
  walking: "animationWalking",
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
