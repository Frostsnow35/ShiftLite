# AI 参与 PR 政策 / AI Participation Policy

> 本仓库要求：**AI 参与的 Pull Request 必须随 PR 提交一份双语（中文 + English）HTML reviewer 说明**，否则合并将被 `main` 分支保护阻止。

## 适用范围 / Scope

勾选 PR 模板中的「本 PR 由 AI 参与完成 / This PR was produced with AI assistance」即视为 AI 参与 PR。
未勾选则不受此政策约束，正常评审即可。

## 提交要求 / Requirement

1. 在 PR 模板勾选 AI 参与框。
2. 随 PR 在仓库内提交文件 `reviews/PR-<编号>.html`（编号与 PR 号一致）。
3. 该 HTML 须同时包含中文与英文内容，并至少涵盖以下小节：
   - 概述 / Summary
   - AI 参与披露（工具·模型·范围·人工复核）/ AI Participation Disclosure
   - 验证与测试 / Verification & Testing
4. 模板见 [`docs/reviewer-note.template.html`](./reviewer-note.template.html)，复制后填写。

CI 检查 [`ai-pr-reviewer-note`](.github/workflows/ai-pr-reviewer-note.yml) 会在每个 PR 上运行：
检测到 AI 参与勾选但缺文件 / 文件非双语时判失败并在 PR 评论说明，分支保护据此阻止合并。

## 设计参照 / Community References

本政策的格式与执行力度参考了以下权威、知名开源仓库的既有实践：

- **xarray / numpy / holoviews**：在 PR 模板内置 *AI Disclosure* 勾选区（是否含 AI 内容、是否已测试、谁负责、用了什么工具/模型）——轻量、低摩擦，本模板即采用此结构。
- **Biome**：在 PR 模板顶部以 `IMPORTANT!!` 强制声明 AI 辅助，强调披露用于帮助 reviewer 施加适当审查力度。
- **TypeScript (microsoft)**：未披露 AI 参与的 PR 直接关闭、不予评审——强执行的一端。
- **Kubernetes**：描述中包含一句「本 PR 部分由生成式 AI 协助完成」即为充分披露——极简一端。
- **spec-kit**：要求写明 agent、模型、设置与参与范围，甚至在每条 AI 生成的提交上加 `Assisted-by:` 尾注。

本仓库取中间偏严：默认不强制所有 PR，但对 AI 参与的 PR 要求结构化双语披露，并辅以 CI 硬校验。

## 原则 / Principles

- 披露不改变评审标准：AI 写或人写，合入门槛一致（参照 Google 代码健康标准）。
- 披露用于让 reviewer 在 AI 易出错的环节（虚构 API、多余抽象、测试套话）多加留意，而非惩罚。
- 作者对所提交的每一行负责；AI 产出须经过人工复核与测试。
