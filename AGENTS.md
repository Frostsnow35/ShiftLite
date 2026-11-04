# AGENTS.md — 面向 AI 编码 Agent 的仓库约定

本文件约束**任何在本仓库工作的 AI 编码 Agent**（WorkBuddy / Claude Code / Codex / Cursor / Copilot 等）。人类贡献者不受本文件约束，但鼓励遵循同样的披露习惯。

## 核心规则 / Core rule

**只要 AI 参与了某个 Pull Request 的产出，该 PR 必须随提交一份双语（中文 + English）HTML reviewer 说明**，路径固定为：

```
reviews/PR-<编号>.html
```

缺少该文件、或文件非双语（须同时含中文与英文、且含 概述 / AI 参与披露 / 验证 等小节），`main` 分支保护会阻止合并（由 CI 检查 `ai-pr-reviewer-note` 判定）。

> 人类是否手动勾选 PR 模板里的「AI 参与」**不是硬性前置**。本规则的首要执行机制就是本文件：Agent 按本约定自动产出说明文件，而说明文件的存在本身即被 CI 识别为 AI PR 并触发校验。勾选框只是可选的辅助声明。

## Agent 必须执行的步骤 / What the agent MUST do

当你的产出将进入一个 PR 时：

1. 开 PR（或确认目标 PR 编号）。
2. 复制模板并填写：
   ```
   cp docs/reviewer-note.template.html reviews/PR-<编号>.html
   ```
   模板结构与必填小节见该文件；须同时用中文与英文填写。
3. 在 `reviews/PR-<编号>.html` 中如实披露：使用的工具 / 模型、参与范围、是否经过人工复核与测试。
4. 把该文件随 PR 一起提交并推送。CI 会在 `synchronize` 事件重跑并校验双语结构。
5. （可选但推荐）在 PR 模板勾选「本 PR 含 AI 参与」以利人工审阅；不勾选也不影响 CI——说明文件本身已是信号。

## 不要做的事 / What NOT to do

- 不要在没有 `reviews/PR-<编号>.html` 的情况下，把 AI 产出的 PR 标记为可合入。
- 不要用空壳 HTML、纯机器翻译占位、或缺少中文 / 英文任一语言的内容应付校验。
- 不要把 reviewer 说明写进 PR 描述正文（GitHub 会净化 HTML，版式与机器校验都不可靠）。

## 相关文件 / Related

- 政策全文：`docs/AI_POLICY.md`
- 说明模板：`docs/reviewer-note.template.html`
- CI 校验：`.github/workflows/ai-pr-reviewer-note.yml`
- PR 模板：`.github/PULL_REQUEST_TEMPLATE.md`
