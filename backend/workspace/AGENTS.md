# Agent Guide

## 核心原则

1. 文件优先。会话、长期记忆、技能和知识都应该落在本地文件中。
2. 技能优先。若现有技能能解决问题，先读取对应 `SKILL.md` 再执行。
3. 透明优先。工具调用、检索结果和记忆注入都要尽量可解释。
4. 当用户表达 稳定偏好、长久设置或个人习惯 时：
    先在当前回答中 复述并确认；
    然后用简短的一两句话总结这条偏好；
    将总结文本写入 workspace/USER.md，追加在“会话偏好记录”部分（通过提供的工具）。
## 工具使用协议

- `read_file`: 读取技能、工作区文档和知识文件。
- `terminal`: 仅在需要运行本地命令时使用。
- `python_repl`: 仅用于短脚本和数据处理。
- `fetch_url`: 获取网页或 JSON 接口内容。
- `search_memory`: （仅 MEMORY_BACKEND=v2 + MEMORY_V2_INJECT=tool 时可用）检索跨会话的长期记忆，返回历史对话的蒸馏摘要和原始片段。当用户提到历史话题或需要回忆过去讨论时主动调用。
- `distill_session`: （仅 MEMORY_BACKEND=v2 时可用）手动触发当前会话的蒸馏，将对话压缩为结构化记忆。

## Memory 协议

### v1 模式 (MEMORY_BACKEND=v1)
- `memory_module_v1/long_term_memory/MEMORY.md` 是长期记忆主文件。
- 通过 Chroma 向量检索动态注入相关片段。
- 未检索到的记忆不应被假设为仍然可用。

### v2 模式 (MEMORY_BACKEND=v2)
- 长期记忆存储在 PostgreSQL + pgvector 中，由结构化蒸馏（distillation）产生。
- 每次对话结束后自动异步蒸馏新的交换片段。
- 检索方式取决于 `MEMORY_V2_INJECT` 配置：
  - `tool`: agent 自主决定何时调用 `search_memory` 检索。
  - `always`: 系统每轮自动检索并注入上下文。
- 检索结果包含蒸馏对象（摘要）和原始对话证据（verbatim），优先引用证据。
