<!--
  本模板参照社区实践制定：
  - xarray / numpy / holoviews：PR 模板内置 AI Disclosure 勾选区
  - Biome：模板顶部以 IMPORTANT 强制声明 AI 辅助
  - TypeScript：未披露 AI 参与的 PR 直接关闭（强执行）
  - Kubernetes：描述里一句话即可；spec-kit：要求写明 agent/模型/设置/范围
  完整规则见 docs/AI_POLICY.md 与 AGENTS.md。
-->

## 变更概要 / Summary
<!-- 用中文与英文简述本 PR 的目标与改动范围 -->

## AI 参与披露 / AI Participation
- [ ] <!-- ai-pr --> 本 PR 含 AI 参与（AI agent 按 AGENTS.md 自动标记，或人工标记） / This PR involves AI assistance

<!--
  本仓库的 AI 披露由 AGENTS.md 约束：AI 编码 Agent 在产出 PR 时会自动生成
  `reviews/PR-<编号>.html` 双语说明，无需你手动勾选上方方框。
  CI 判定「AI PR」的条件 = 勾选本框 或 已附 reviews/PR-<编号>.html（二者其一即可），
  因此即使不勾选，只要附了说明文件也会被识别并校验。
  人类若使用 AI 协助但未走 Agent 流程，建议手动勾选以便审阅。
  完整规则见 docs/AI_POLICY.md 与 AGENTS.md。
-->

## 测试与验证 / Testing & Verification
<!-- 如何验证本 PR 的改动，含命令与结果 -->

## 关联 / Related
<!-- 关联 Issue、任务编号（如 E0–E5）、文档链接 -->
