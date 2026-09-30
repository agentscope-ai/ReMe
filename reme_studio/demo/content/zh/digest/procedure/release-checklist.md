---
name: 可复用的发布检查清单
description: 将验收经验写成有证据的操作步骤。
memory_tags: ["aurora", "release", "procedure"]
---

# 发布检查清单

适用于 Aurora 内部试点，由 [[digest/wiki/people.md|周宁]] 执行，林澄确认范围。来源：[[daily/2026-09-18/review.md|复盘]]。

- [ ] 保留工作区文件副本，确认 Markdown 可直接读取。
- [ ] 检查 [[digest/wiki/local-first.md|架构结论]] 的来源链接。
- [ ] 用 `local_first` 和 `release` 标签检查跨目录检索。
- [ ] 停止服务后验证已有文件仍可离线打开。
- [ ] 根据 [[digest/procedure/recovery.md|恢复流程]] 检查派生数据可恢复。
- [ ] 将通过项、失败项和未做项写入验收记录。
- [ ] 更新 [[digest/wiki/release.md|发布状态]]，附原始结果。

清单描述可复用经验，不表示勾选动作已触发后台任务。最新原始结果见 [[resources/pilot-results.md|验收记录]]。
