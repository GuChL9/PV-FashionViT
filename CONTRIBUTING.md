# 协作约定

## 开始工作

1. 从 `main` 拉取最新代码。
2. 为一个明确任务创建分支，例如 `feature/dataset-preview`、`experiment/cnn-center` 或 `docs/final-report`。
3. 提交信息使用 `类型: 简短说明`，例如 `feat: implement grid evaluation`。
4. 推送分支并创建 Pull Request，至少由另一位成员检查后合并。

## 分工建议

- 数据负责人：数据集、样本可视化和数据口径。
- 模型负责人：MLP/CNN 与 ViT/Hybrid 模型。
- 实验负责人：六组训练、网格评估、结果表格和图。
- 报告负责人：方法、结果分析、演示材料与最终整合。

实际贡献请按 commit、PR、实验记录和文档修改如实填写 `report/contribution.md`。

## 仓库规则

- 不提交 `data/`、`outputs/checkpoints/`、训练日志或临时目录。
- 若必须交付权重，使用 Git LFS 或 GitHub Release，不要直接放进普通 Git 历史。
- 禁止把测试集表现用于选择训练 epoch 或调参；best checkpoint 只依据验证集准确率。
- 结果表中的数值必须能追溯到配置和 checkpoint，不手工编造或覆盖原始记录。

