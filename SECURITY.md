# Security Policy

## Supported versions

安全修复只保证覆盖最新发布版本。教学代码不构成任何生产决策保证。

## Reporting a vulnerability

请不要创建公开 Issue 披露未修复漏洞、凭证或敏感数据。优先使用 GitHub 仓库的 **Report a vulnerability** 私密报告功能；如果该功能尚未启用，请通过维护者 GitHub 主页公开的联系方式私下报告。

报告请包含受影响版本、复现步骤、影响和可行缓解方式，但不要附带真实客户数据。维护者目标是在 7 天内确认报告，并在验证后协调修复与披露。

## Security boundaries

- 项目默认不访问网络，也不读取未显式提供的本地数据。
- 不接受真实客户、授信或医疗数据作为 Issue、fixture 或示例。
- 生成物不得包含凭证、环境变量值、数据行或用户绝对路径。
- 模型输出不得直接用于真实授信、医疗或其他高风险决策。
