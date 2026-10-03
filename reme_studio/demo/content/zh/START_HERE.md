---
name: 从一条记录到可复用的记忆
description: 探索 Aurora 的资料、日记、长期知识、标签与关联图谱。
memory_tags: ["aurora", "memory", "guide"]
---

# 欢迎体验 ReMe Studio

这里是虚构的 **Aurora 团队知识助手**工作区。团队希望把散落的会议、决策和经验变成可追溯、可复用的记忆。你可以在浏览器里打开、编辑、保存和下载这些 Markdown，无需启动服务。

## 1. 跟随一条记忆的来源

打开 [[digest/wiki/local-first.md|本地优先决策]]。这篇长期知识来自 [[resources/offline-feedback.md|现场反馈]] 与 [[daily/2026-09-15/decision.md|决策日记]]。早期 [[resources/requirements.md|需求草案]] 仍保留云端方案，因此你可以看到旧计划和新结论的区别。

## 2. 用标签跨目录查找

点击顶部 **搜索 / 标签**，选择 `local_first`。同一个主题会同时找到资料、日记、知识和流程；再输入「离线」，缩小到包含这个词的文件。标签来自每篇文件 front matter 中的 `memory_tags`，不是另一份手工维护的分类。

## 3. 修改自己的浏览器副本

打开 [[digest/personal/communication.md|沟通偏好]]，切换到 Edit。将标签 `communication` 改为 `writing`，保存，再查看标签数量和筛选结果。下载按钮可以导出文件；「重置示例」恢复最初的内容。

## 4. 探索关系与回答

在 Knowledge 分类旁点击 Graph，选中节点查看入链、出链，再打开源文件。创建 Chat，发送预设问题，展开 `search` 和 `read` 查看实际示例文件。

> 图谱和关键词筛选在浏览器里计算。对话回答是预设演示，修改文档不会自动生成新的模型回答。数据仅保存在当前浏览器；清除网站数据会清除修改。

安装真实 ReMe：打开顶部文档，运行 `reme start`，在本机 Studio 使用自己的工作区。
