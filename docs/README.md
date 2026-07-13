# ReMe 开发文档

本目录保存与 ReMe 源码一起维护的设计资料、历史决策和 README 图片，不再作为文档站点的构建或部署来源。

面向用户发布的中英文文档位于 [agentscope-ai/docs](https://github.com/agentscope-ai/docs) 仓库，并由该仓库统一完成版本管理和 Mintlify 部署。

## 目录用途

```text
docs/
├── README.md   本目录的维护说明
├── doc.md      当前文档设计与维护边界
├── figure/     ReMe README 使用的图片资源
├── old/        历史设计、调研和方案记录
```

## 维护原则

- 产品理念和长期设计判断记录在 [`doc.md`](./doc.md) 或对应设计文档中。
- 具体实现以源码、schema、测试和运行时帮助为准，避免维护重复且容易过期的开发手册。
- README 引用的图片保留在 `figure/`；发布文档需要图片时，在统一文档仓库的 `images/reme/` 中维护对应副本。
- 用户可见的安装、接入、Reference 和 Support 内容在统一文档仓库中更新。
- `old/` 只保存历史背景，不代表当前行为契约。
