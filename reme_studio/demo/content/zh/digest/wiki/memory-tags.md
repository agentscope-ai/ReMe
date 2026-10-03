---
name: 用 memory tags 找到同一主题
description: 演示真实标签字段、规范化、文件计数和组合筛选。
memory_tags: ["aurora", "memory", "guide"]
---

# Memory tags

标签保存在 Markdown front matter 的 `memory_tags` 列表中。ReMe 的标签索引来自文件，可以重新构建。

```yaml
memory_tags: [aurora, local_first, decision]
```

## 标签和目录、链接的区别

目录表达文件用途；`local_first` 标签跨 resources、daily 和 digest 聚合主题；[[digest/wiki/local-first.md|wikilink]] 表达具体来源或关系。

## 试一试

1. 在顶部搜索面板选择 `local_first`，观察各目录的命中与标签文件数。
2. 同时选择 `release`：多标签为「任意一个匹配」，与真实 search Job 的 tags 筛选一致。
3. 输入「离线」：关键词与标签共同约束结果。
4. 编辑 [[digest/personal/communication.md|沟通偏好]] 的标签并保存，计数立即变化。

演示支持行内列表和缩进列表。空格转为下划线、大小写统一、重复项合并，每篇默认最多三个有效标签。这里运行关键词筛选；真实 ReMe 的混合向量与 BM25 检索需要服务。
