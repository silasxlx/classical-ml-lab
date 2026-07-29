# Contributing

感谢你考虑参与 `classical-ml-lab`。项目欢迎小而清晰、带测试、可复现的改进。

## 开始之前

1. 搜索现有 Issue，确认问题尚未被讨论。
2. 对新实验、公开接口或架构变化先创建 Issue；重要取舍需要新增 ADR。
3. 不要提交真实客户数据、大型原始数据、模型文件、凭证或个人信息。

## 本地开发

```bash
git clone https://github.com/silasxlx/machinelearning-blob.git classical-ml-lab
cd classical-ml-lab
uv sync --locked --all-extras
```

提交 PR 前运行：

```bash
uv run ruff check .
uv run mypy src
uv run pytest --cov=classical_ml_lab --cov-branch --cov-fail-under=85
```

## TDD 与代码质量

- 行为变化先写一个能稳定失败的测试，再实现最小代码，最后重构。
- 核心路径必须测试，不要求 100% 覆盖率；总体门槛为 85%。
- 公共 API 和关键内部结构必须提供类型标注和 docstring。
- 不提交 dead code、大段注释代码或全局 warning 抑制。
- `TODO`、`FIXME`、`HACK` 必须包含对应 Issue，例如 `TODO(#123): ...`。

## Commit message

所有新增 commit 和 PR 标题遵循 [Conventional Commits](https://www.conventionalcommits.org/)：

```text
feat(cli): add reproducible all-experiment runner
fix(metrics): compute auc from decision scores
docs(adr): explain offline dataset policy
test(data): reject non-binary credit targets
```

允许的常用类型：`feat`、`fix`、`docs`、`test`、`refactor`、`perf`、`build`、`ci`、`chore`、`revert`。

## 版本与发布

- 使用 SemVer：破坏兼容为 major，向后兼容功能为 minor，向后兼容修复为 patch。
- 用户可见变化写入 `CHANGELOG.md` 的 `Unreleased`。
- Release 必须包含迁移提示、验证结果、已知限制和安全影响。

## Pull Request

PR 应保持单一目的，并说明：需求/Issue、先失败的测试、实现、重构和验证命令。维护者优先使用 squash merge，最终提交信息必须符合 Conventional Commits。
