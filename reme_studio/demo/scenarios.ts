export const scenarios = {
  zh: [
    {
      question: "Aurora 为什么从云端方案改为本地优先？",
      tag: "local_first",
      paths: [
        "digest/wiki/local-first.md",
        "resources/offline-feedback.md",
        "daily/2026-09-15/decision.md",
      ],
      answer:
        "**以下是基于初始示例文件的预设回答，未运行模型。**\n\nAurora 最初考虑云端集中存储。现场反馈发现断网和内部资料外发限制，因此 9 月 15 日确认本地优先：Markdown 是持久来源，索引和图谱可重建。\n\n来源：[[resources/offline-feedback.md|现场反馈]]、[[daily/2026-09-15/decision.md|决策记录]]、[[digest/wiki/local-first.md|长期架构知识]]。\n\n模型推理仍取决于配置的提供商，跨设备同步不在首期范围。展开上面的工具块，可以检查当前浏览器中的文件内容。",
    },
    {
      question: "用 memory tags 查找发布相关的记忆。",
      tag: "release",
      paths: [
        "digest/wiki/release.md",
        "digest/procedure/release-checklist.md",
      ],
      answer:
        "**这是预设对话演示。**\n\n`release` 标签把验收资料、日记、发布状态和操作清单聚在一起。标签来自文件的 `memory_tags`，因此不受目录边界限制。\n\n[[digest/wiki/release.md|发布状态]] 记录当前结论；[[digest/procedure/release-checklist.md|检查清单]] 保存可复用经验；[[resources/pilot-results.md|原始结果]] 提供证据。\n\n试着打开「搜索 / 标签」，选择 `release`，再编辑某篇文件的标签并保存。计数与匹配结果会更新。",
    },
    {
      question: "林澄喜欢怎样的项目汇报？",
      tag: "personal",
      paths: ["digest/personal/communication.md", "daily/2026-09-18/review.md"],
      answer:
        "**这是基于初始示例的预设回答。**\n\n林澄希望先结论、后证据，尽量三段以内；行动项带负责人和截止日期，未验证项明确标注。\n\n这条偏好来自 [[daily/2026-09-18/review.md|试点复盘]]，保存于 [[digest/personal/communication.md|个人记忆]]。用户可以直接检查和修改它。\n\n演示中的工具结果读取当前文件，预设回答不会随编辑自动重新生成。",
    },
  ],
  en: [
    {
      question: "Why did Aurora switch from cloud-first to local-first?",
      tag: "local_first",
      paths: [
        "digest/wiki/local-first.md",
        "resources/offline-feedback.md",
        "daily/2026-09-15/decision.md",
      ],
      answer:
        "**Scripted answer based on the initial example files. No model is running.**\n\nAurora initially considered central cloud storage. Field feedback revealed connectivity failures and restrictions on external sharing. The September 15 decision made Markdown the durable source and indexes and graphs rebuildable.\n\nSources: [[resources/offline-feedback.md|field feedback]], [[daily/2026-09-15/decision.md|the decision]], and [[digest/wiki/local-first.md|durable architecture knowledge]].\n\nModel reasoning still depends on the configured provider; multi-device sync is outside the pilot. Expand the tool blocks to inspect the current browser files.",
    },
    {
      question: "Find release memories using memory tags.",
      tag: "release",
      paths: [
        "digest/wiki/release.md",
        "digest/procedure/release-checklist.md",
      ],
      answer:
        "**This is a scripted conversation.**\n\nThe `release` tag connects acceptance resources, daily notes, release status, and the operating checklist. It comes from each file's `memory_tags`, so it crosses directory boundaries.\n\n[[digest/wiki/release.md|Release status]] captures the conclusion, [[digest/procedure/release-checklist.md|the checklist]] preserves reusable experience, and [[resources/pilot-results.md|original results]] supply evidence.\n\nOpen Search / Tags, select `release`, then edit a file's tags and save. Counts and matches update.",
    },
    {
      question: "How does Lin prefer project updates?",
      tag: "personal",
      paths: ["digest/personal/communication.md", "daily/2026-09-18/review.md"],
      answer:
        "**Scripted answer based on the initial examples.**\n\nLin prefers conclusions before evidence, usually within three paragraphs. Actions need owners and deadlines; unverified claims must be marked.\n\nThe preference comes from [[daily/2026-09-18/review.md|the pilot review]] and lives in [[digest/personal/communication.md|personal memory]], where users can inspect and edit it.\n\nTool results read the current files. This scripted answer does not regenerate when you edit them.",
    },
  ],
};
