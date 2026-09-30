---
name: 同步范围与边界
description: 把延期事项与当前支持的能力区分清楚。
memory_tags: ["aurora", "local_first", "planning"]
---

# 同步范围

首期 Aurora 只承诺单机工作区，不包含自动多设备同步。此边界由 [[daily/2026-09-15/decision.md|本地优先决策]] 确认。

## 后续需要解决

跨设备保存冲突、重复记录合并、敏感资料的同步范围，以及用户如何恢复旧版本。当前试点结果未验证这些问题。

## 现阶段操作

用户可以下载 Markdown 副本，但手工导出不等于同步机制。浏览器演示中的保存仅属于当前浏览器，也不能当作真实工作区备份。

发布条件见 [[digest/wiki/release.md|发布状态]]。数据所有权原则见 [[digest/wiki/local-first.md|架构知识]]。
