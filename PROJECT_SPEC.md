# PV-FashionViT 实施规范

本文件是原项目说明书的可执行补充。原说明书的任务主线保持不变；以下定义用于消除实现和评估歧义。

## 数据划分与复现

- FashionMNIST 官方训练集按固定随机种子划分为 54,000 个训练样本和 6,000 个验证样本；官方测试集只用于最终评估。
- 随机位置由 `seed + epoch + sample_index` 确定，因而在多进程 DataLoader 中仍可复现；训练集每个 epoch 更新位置，验证和测试固定为 epoch 0。
- 28x28 图像放入 56x56 画布。`dx=0, dy=0` 表示居中；正 `dx` 向右，正 `dy` 向下。
- 由于无裁剪平移上限只有 14 像素，`|dx|` 或 `|dy|` 大于 14 时前景会在边界被裁剪；本项目将这种遮挡视为 large-shift 难度的一部分。
- `small_shift` 在 `[-8, 8]` 均匀采样，`large_shift` 在 `[-18, 18]` 均匀采样。
- `grid_shift` 评估对 49 个固定位置分别测试完整官方测试集；Grid Accuracy 是 49 个位置准确率的宏平均。
- Clipping Split 将同一份 49 点结果拆成两组：`|dx|, |dy| <= 12` 的内层 5×5 网格保证前景完整，至少一个坐标为 `±18` 的 24 点外圈可能发生裁剪。两组分别取宏平均，并报告 `Inner - Outer` 的百分点差；该差值同时包含离中心更远和可见信息减少的影响，不作为裁剪的纯因果效应。
- Matched Clipping 在整数坐标 `[-18,18]^2` 中筛选同一 `dx^2+dy^2` 下同时包含完整与可能裁剪位置的半径。当前共有 7 个精确匹配半径、104 个坐标；先在每个半径内分别宏平均两类位置，再对半径作宏平均。该设计精确控制位移半径，但方向集合仍有差异，因此属于更严格的距离匹配对照而非纯因果估计。

## 增强顺序

仿射旋转/缩放先作用于原始 28x28 前景，随后粘贴到画布，再进行 Random Erasing。`random_shift` 是画布放置策略，不与仿射变换中的二次平移叠加。
正式增强模型使用 `shift_rotation`：平移范围为 `[-18, 18]`，旋转范围为 `[-45°, 45°]`，不额外混入缩放。固定角度评估使用 `[-45°, -30°, -15°, 0°, 15°, 30°, 45°]`。

增强逐项消融保持 Tiny ViT、CLS pooling、优化器和最多 12 epochs 的预算不变，只依次扩大训练分布：Center、仅随机平移、随机平移+旋转、随机平移+旋转+Random Erasing。支线使用 seed 42、2026、3407，报告均值与样本标准差；它比单种子更稳定，但仍不替代六组主实验的五随机种子正式均值。所有阶段设置 `output.publish_global=false`，避免覆盖主实验汇总表。

位置编码消融同样使用 seed 42、2026、3407，并固定 Center 训练、CLS pooling、patch size、embedding、Encoder、优化器与训练预算，只切换 `learnable`、`sincos` 和 `none`。原有 `configs/vit_sincos.yaml` 同时使用增强与 Mean Pooling，不作为这一控制变量表的证据。

## 指标口径

- Center / Small Shift / Large Shift Accuracy：相应固定测试协议下完整测试集准确率。
- Rotation Accuracy：居中放置并在 `[-45°, 45°]` 随机旋转时的准确率；固定角度曲线另按七个角度逐点统计。
- Robust Drop：`Center Accuracy - Large Shift Accuracy`，使用百分点表示。
- Per-class Accuracy：每类正确数除以该类样本总数；不存在样本时记为 `NaN`。
- 混淆矩阵按真实类别为行、预测类别为列。

## 工程约定

- 所有训练配置都继承 `configs/base.yaml`，差异只写在各模型配置中。
- 每次实验写入独立目录 `outputs/<run_name>/`，避免多个模型相互覆盖。
- checkpoint 同时保存模型、优化器、scheduler、epoch、最佳验证准确率和完整配置。
- 仓库只提交小型表格和图像；原始数据、日志和模型权重默认忽略，避免 Git 仓库膨胀。

## CPU 优先配置

- 默认配置固定使用 CPU，batch size 为 64，`num_workers=0`，避免 Windows 多进程启动和内存问题。
- Tiny ViT 与 HybridConv-ViT 默认使用 `patch_size=7`、`embed_dim=64`、`depth=2` 和 `mlp_ratio=2`，把 token 数从 196 降为 64。
- 默认最多训练 12 epochs，并在至少 5 epochs 后以验证准确率执行 patience=3 的早停。
- 首轮训练只自动执行 Center、Small Shift 和 Large Shift；49 点 Grid 通过 `--grid` 单独运行。
- CPU 效率表统一使用 8 个 PyTorch 线程、batch size 64、5 次预热和 30 次计时，只统计 `eval()`/`inference_mode()` 下的模型前向。表中延迟不含数据读取、预处理、反向传播和优化器更新；不同机器上的绝对数值不可直接横比。
