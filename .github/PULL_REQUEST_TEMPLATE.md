<!--
  本模板参照社区实践制定：
  - xarray / numpy / holoviews：PR 模板内置 AI Disclosure 勾选区
  - Biome：模板顶部以 IMPORTANT 强制声明 AI 辅助
  - TypeScript：未披露 AI 参与的 PR 直接关闭（强执行）
  - Kubernetes：描述里一句话即可；spec-kit：要求写明 agent/模型/设置/范围
  完整规则见 docs/AI_POLICY.md。
-->

## 变更概要 / Summary
<!-- 用中文与英文简述本 PR 的目标与改动范围 -->

## AI 参与披露 / AI Participation
- [ ] <!-- ai-pr --> 本 PR 由 AI 参与完成 / This PR was produced with AI assistance

<!--
  若上方已勾选：
  必须随本 PR 提交一份「双语 HTML reviewer 说明」，路径固定为
      reviews/PR-<编号>.html
  模板见 docs/reviewer-note.template.html。
  缺少该文件，或其内容非双语（中英均有、且含概述/AI 参与披露/验证等小节），
  合并将被分支保护（main）阻止，由 CI 检查 ai-pr-reviewer-note 判定的。
-->

## 测试与验证 / Testing & Verification
<!-- 如何验证本 PR 的改动，含命令与结果 -->

## 关联 / Related
<!-- 关联 Issue、任务编号（如 E0–E5）、文档链接 -->
