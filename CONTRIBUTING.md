# 贡献指南（CONTRIBUTING）

感谢参与本项目！本仓库服务于本科生大创（CCUS 方向），欢迎队友提交改进。

## 开发流程
1. 从 `main` 切出功能分支：`git checkout -b feature/你的功能`
2. 遵循 Conventional Commits 提交（见 README 提交规范）
3. 保证 `pytest tests/` 通过
4. 合并回 `main` 前，说明做了什么、附指标/截图

## 代码风格
- Python 用 4 空格缩进，关键函数写中文 docstring（便于本科生阅读）
- 模型与数据管线改动需补充对应测试

## 数据合规
- 严禁提交任何真实油田/封存监测数据
- 严禁提交 `数据集/` 里的官方 U-FNO `.pt`（体积数十 GB，且许可条款未确认）；已在 `.gitignore` 内
- 实验产物只提交可由 `tools/` 重新生成的 json/csv 指标，`.npz` 大文件不入库
- 早期 `sim/` 合成数据路线已停用，仅作历史脚手架保留，不再作为结论依据
