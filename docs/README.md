# ReMe 仓库文档

本目录保存 ReMe 仓库 README 直接引用的中英文补充说明和图片资源，不作为文档站点的构建或部署来源。

面向用户发布的中英文文档位于 [agentscope-ai/docs](https://github.com/agentscope-ai/docs) 仓库，并由该仓库统一完成版本管理和 Mintlify 部署。

## 目录用途

```text
docs/
├── README.md   本目录的维护说明
├── doc.md      当前文档设计与维护边界
├── en/         README 引用的英文补充说明
├── zh/         README 引用的中文补充说明
└── figure/     ReMe README 使用的图片资源
```

## 维护原则

- `en/` 和 `zh/` 保持精简，服务 README 中需要进一步解释的功能与场景；修改路径时同步更新 README 链接。
- 具体实现以源码、schema、测试和运行时帮助为准，避免维护重复且容易过期的开发手册。
- README 引用的图片保留在 `figure/`；发布文档需要图片时，在统一文档仓库的 `images/reme/` 中维护对应副本。
- 网页文档、导航、版本和部署在统一文档仓库中维护。
