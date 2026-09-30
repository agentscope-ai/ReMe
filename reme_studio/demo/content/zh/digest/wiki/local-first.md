---
name: 本地优先架构决策
description: 记录当前结论、变更原因与可追溯来源。
memory_tags: ["aurora", "local_first", "decision"]
---

# 本地优先

**当前结论：工作区 Markdown 是持久来源；索引、目录和图谱是可重建的派生数据。**

## 为什么改变方案

[[resources/requirements.md|9 月 14 日草案]] 提出云端集中存储。[[resources/offline-feedback.md|9 月 15 日现场反馈]] 发现弱网与资料外发限制，[[daily/2026-09-15/decision.md|当天决策]] 因此确认本地优先。

## 对用户意味着什么

停止服务不影响直接读取文件。索引损坏时应从源文件恢复，流程见 [[digest/procedure/recovery.md|恢复指南]]。联网模型不可用时，模型回答可能不可用；这不改变文件所有权。

同步单独规划，见 [[digest/wiki/sync.md|同步范围]]。此结论来源于明确证据，不应把早期草案误当成当前方案。
