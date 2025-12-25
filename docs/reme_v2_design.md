# ReMeV2的设计文档

## 现在的问题
1. 外功修炼：当前server-client的模式不太适合新手，需要对开发者更加友好的接口设计。
2. 内功修炼：Inspired by skills & agentic memory，我们考虑开发基于渐进式检索/总结的agentic memory方案。

## 竞品的调研
1. todo 对比5家，出参，入参，竞品对比的逻辑
尤其是mem0

## 我们的方案

### 对外接口设计



### 算法设计





## 改造计划
1. 合并flowllm中reme必要的代码，保留现在server-client的依赖，兼容现在各个仓库的依赖代码
2. 新的ReMe接口设计，支持summary，retrieve，context_offload, context_reload 4个核心接口
3. 新的agentic算法方案开发
4. benchmark
   - halumem
   - locomo
   - longmemevel
   - personal-v2 ?
   - appworld/bfcl-v3
5. 技术报告
6. 更新各个仓库的依赖代码
   - agentscope
   - agentscope-runtime
   - evotraders
   - alias(tool-memory)
   - agentscope-java
   - AgentEvolver
   - cookbook: reme procedural memory paper
   - tool-memory-upgrade（将要合并）

春节前有一个小版本？

## 我们优势
1. 融合了多种记忆的渐进式agentic方案【最重要】
2. 同时支持长期记忆和短期记忆
3. 开发者友好
   1. 提供简介的接口设计，全异步更高性能
   2. 提供cli直接体验
   3. 提供和agentscope、langchain融合的方案
   4. 支持agentic的二开方案
