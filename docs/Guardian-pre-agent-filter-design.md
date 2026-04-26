# Guardian 前置安全过滤设计

日期: 2026-03-25  
状态: Draft (已与用户确认关键决策)

## 1. 目标与范围

在主 Agent 执行前接入一个 OpenAI 兼容小模型作为安全卫士，基于用户输入判定:

- `安全`: 放行给主 Agent
- `危险`: 硬拦截并返回固定拒答

本设计仅覆盖主对话链路(`POST /chat`)的前置过滤，不引入与本目标无关的重构。

## 2. 已确认决策

1. 拦截策略: **A. 硬拦截**
2. 接入位置: **主 Agent 前置 middleware**
3. 配置要求: 增加显式开关控制是否启用安全卫士
4. 模型接口: OpenAI 兼容格式

## 3. 方案对比与结论

### 方案 1: Guardian Middleware 前置拦截 (推荐)

在 `backend/graph/agent_factory.py` 的 middleware 链最前挂载 Guardian。危险请求直接中断，不进入主 Agent 推理。

优点:
- 与现有架构一致(当前已使用 `SummarizationMiddleware`)
- 保护边界统一，可复用到其他 Agent 入口
- 后续可叠加多类安全中间件

缺点:
- 需要对齐当前 `langchain` 版本 Guardian API 细节

### 方案 2: API 层前置拦截

在 `backend/api/chat.py` 调用 `agent_manager.astream(...)` 之前先判定。

优点: 实现直观、调试简单。  
缺点: 容易遗漏其他调用入口，防护边界分散。

### 方案 3: API + Middleware 双层

优点: 防旁路能力最强。  
缺点: 复杂度和延迟更高，需避免重复调用卫士模型。

### 结论

优先落地方案 1；后续如出现旁路入口，再补方案 3。

## 4. 目标架构

新增模块:

- `backend/graph/guardian.py` (建议)
  - `GuardianVerdict`: `安全|危险` + `reason_code` + `latency_ms`
  - `GuardianClient`: 调用小模型并解析强约束输出
  - `GuardianMiddleware`: 在主 Agent 前置执行，危险即短路

接入点:

- `backend/graph/agent_factory.py`
  - 在 middleware 列表中将 Guardian 放在最前
- `backend/graph/agent.py`
  - 保持现有 `astream` 事件契约，支持 Guardian 短路返回
- `backend/api/chat.py`
  - 无需改入口协议；消费既有 `done/error` 事件即可

## 5. 配置设计

在 `backend/config/config.py` 与 `backend/config/.env.example` 增加:

- `GUARDIAN_ENABLED=true`
- `GUARDIAN_PROVIDER=openai`
- `GUARDIAN_MODEL=<small-model>`
- `GUARDIAN_API_KEY=<api-key>`
- `GUARDIAN_BASE_URL=<openai-compatible-url>`
- `GUARDIAN_TIMEOUT_MS=1500`
- `GUARDIAN_FAIL_MODE=closed` (`closed|open`)
- `GUARDIAN_BLOCK_MESSAGE=检测到潜在提示词攻击风险，本次请求已被拦截。`

默认建议:

- `GUARDIAN_ENABLED=true`
- `GUARDIAN_FAIL_MODE=closed` (超时/异常默认拦截)

## 6. 请求处理流程

1. 用户消息进入 `POST /chat`
2. 请求进入 agent middleware 链
3. Guardian 调用小模型，仅输出 `安全` 或 `危险`
4. 若 `危险`:
   - 立即终止后续流程
   - 不触发 memory 注入、tool 调用、主模型推理
   - 返回 `GUARDIAN_BLOCK_MESSAGE`
5. 若 `安全`: 原流程继续，行为与当前一致
6. 若 Guardian 异常:
   - `closed`: 拦截
   - `open`: 放行并记录告警

## 7. 提示词与输出约束

Guardian 判定提示词建议:

- 系统提示明确任务为二分类安全审查
- 明确输出格式只能是单词: `安全` 或 `危险`
- 禁止输出解释、标点、额外文本
- 解析层做二次校验，不匹配则按异常处理

## 7.1 OpenAI 兼容接口最小契约

为避免实现歧义，Guardian 调用接口约定如下:

- 请求最小字段: `model`、`messages`、`temperature`
- `temperature` 固定为 `0`，确保分类稳定
- `messages` 仅包含系统提示与当前待检测用户消息(可选最近 1 轮上下文)
- 响应解析: 仅取第一候选文本，执行 `安全|危险` 严格枚举校验
- 若响应为空、结构异常、非枚举值: 进入 fail mode 分支处理
- 错误映射:
  - 401/403: 认证失败，按 fail mode 处理并告警
  - 429: 限流，按 fail mode 处理并记录限流指标
  - 5xx/网络超时: 按 fail mode 处理并记录上游可用性指标

## 8. 可观测性与审计

记录字段(内部日志/埋点):

- `session_id`
- `guardian_verdict`
- `reason_code` (内部使用，不返回给用户)
- `guardian_latency_ms`
- `guardian_fail_mode`
- `guardian_fallback_triggered`

指标建议:

- `guardian_blocked_rate`
- `guardian_timeout_rate`
- `guardian_parse_error_rate`
- `guardian_added_latency_p95`

治理建议:

- 生产环境默认使用 `GUARDIAN_FAIL_MODE=closed`
- `open` 仅允许在灰度或故障应急窗口短时启用，并配套告警

## 9. 测试计划

单元测试:

- 判定结果解析: 正常值/非法值/空响应
- fail mode 行为: `closed` 与 `open`
- 开关行为: `GUARDIAN_ENABLED=false` 完全旁路

集成测试:

- 正常问答可放行
- 典型 prompt injection 被硬拦截
- Guardian 超时在 `closed` 下拦截
- SSE 流在拦截时可正常结束，前端无挂流
- 拦截时不触发工具调用与 memory 注入

回归测试:

- 关闭 Guardian 后主流程与基线一致
- 与 `SummarizationMiddleware` 并存时无顺序冲突

SSE 拦截场景示例(建议约定):

```text
event: done
data: {"content":"检测到潜在提示词攻击风险，本次请求已被拦截。"}
```

## 10. 风险与缓解

- 误杀正常请求:  
  通过灰度阶段记录样本，优化提示词和模型，必要时引入白名单规则。

- 模型不可用导致可用性下降:  
  对生产环境可阶段性采用 `open`，并结合告警；最终回归 `closed`。

- 被反向探测规则:  
  对外统一拦截文案，不暴露 `reason_code` 与策略细节。

## 11. 实施步骤(高层)

1. 扩展配置模型与 `.env.example`
2. 新增 Guardian 客户端与 middleware
3. 在 `agent_factory` 组装 middleware 顺序(Guardian 在前)
4. 在 `agent.py` 对拦截结果做事件兼容输出
5. 增加单元/集成测试
6. 灰度观察指标并调优

## 12. 验收标准

- 危险输入 100% 在主 Agent 前被拦截
- 拦截场景下无工具调用与记忆检索
- 正常请求端到端成功率不显著下降
- P95 延迟增量在可接受范围(目标 <= 200ms，视小模型实际表现调整)

