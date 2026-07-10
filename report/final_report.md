# PV-FashionViT：基于 Transformer 的位置可变 FashionMNIST 图像分类

> 状态：方法与工程部分已完成；所有标记为“待实验”的位置必须在正式训练后填写，不得用 smoke test 数值代替。

## 摘要

本项目基于 FashionMNIST 构造 PV-FashionMNIST 位置可变分类任务，在 56x56 画布上比较 MLP、CNN、Tiny ViT、位置增强 ViT、Mean-Pooling ViT 与 HybridConv-ViT。我们使用 Center、Small Shift、Large Shift 和 49 点 Grid 协议评估位置鲁棒性，并以 Robust Drop、类别准确率、混淆矩阵和位置热力图进行分析。正式实验结果与结论将在六组配置完成后补充。

## 1. 项目背景

标准图像分类数据通常将主体放在相对稳定的位置，这可能掩盖模型对平移和边界遮挡的敏感性。本项目通过只改变目标在画布中的位置，观察不同架构利用局部模式与绝对位置的方式。

## 2. 任务定义

输入为单通道 56x56 图像，输出为 FashionMNIST 的 10 个类别。原始 28x28 前景居中位置定义为 `(dx, dy)=(0, 0)`。正 `dx` 向右，正 `dy` 向下。平移达到正负 18 时会有 4 像素越过画布并被裁剪，因此 Large Shift 同时测试平移与轻微边界遮挡。

## 3. 数据集构造

- 原始数据：FashionMNIST 官方 60,000 张训练图像和 10,000 张测试图像。
- 划分：官方训练集固定划分为 54,000 训练和 6,000 验证；官方测试集只用于最终评估。
- Center：`dx=dy=0`。
- Small Shift：`dx,dy` 从 `[-8,8]` 均匀采样。
- Large/Random Shift：`dx,dy` 从 `[-18,18]` 均匀采样。
- Grid Shift：两个方向都取 `{-18,-12,-6,0,6,12,18}`。
- Affine：随机平移外，前景进行正负 10 度旋转和 `[0.9,1.1]` 缩放，再执行 Random Erasing。

数据可视化：待运行 `python src/visualize_data.py` 后插入 `outputs/figures/data_preview.png` 与 `position_variation_demo.png`。

## 4. 模型设计

### 4.1 MLP 与 CNN

MLP 将整个画布展平后使用一个隐藏层分类。CNN 使用两组 `3x3 Conv-ReLU-MaxPool`，随后使用全连接分类头。

### 4.2 Tiny ViT

```mermaid
flowchart LR
    A["56x56 grayscale image"] --> B["4x4 patch embedding"]
    B --> C["CLS token + learned position embedding"]
    C --> D["4 Transformer encoder blocks"]
    D --> E["CLS or mean pooling"]
    E --> F["10-class head"]
```

### 4.3 HybridConv-ViT

```mermaid
flowchart LR
    A["56x56 grayscale image"] --> B["two-layer convolution stem"]
    B --> C["stride-4 patch projection"]
    C --> D["learned position embedding"]
    D --> E["Transformer encoder"]
    E --> F["mean pooling + classifier"]
```

## 5. 训练流程

CPU 默认实验最多训练 12 epochs，batch size 为 64，使用 AdamW、初始学习率 `3e-4`、权重衰减 `0.05`、CosineAnnealingLR 和 `0.1` label smoothing。为控制普通电脑上的耗时，Tiny ViT 和 HybridConv-ViT 使用 7x7 patch、64 维 embedding 和 2 层 encoder，并在至少 5 epochs 后采用 patience=3 的早停。best checkpoint 依据验证集准确率选择；last checkpoint 用于恢复训练状态。说明书推荐的 30-epoch 大模型作为后续 GPU 配置保留。

## 6. 实验设置

| 实验 | 模型 | 训练数据 | 测试协议 |
|---|---|---|---|
| E1 | MLP | Center | Center/Small/Large/Grid |
| E2 | CNN | Center | Center/Small/Large/Grid |
| E3 | ViT-AbsPos | Center | Center/Small/Large/Grid |
| E4 | ViT-Aug | Affine + Erasing | Center/Small/Large/Grid |
| E5 | ViT-MeanPool | Affine + Erasing | Center/Small/Large/Grid |
| E6 | HybridConv-ViT | Affine + Erasing | Center/Small/Large/Grid |

## 7. 实验结果

正式训练后引用 `outputs/tables/main_results.csv`。以下表格不得提前填入推测值。

| Model | Center Acc | Small Shift Acc | Large Shift Acc | Grid Acc | Robust Drop |
|---|---:|---:|---:|---:|---:|
| MLP | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |
| CNN | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |
| ViT-AbsPos | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |
| ViT-Aug | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |
| ViT-MeanPool | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |
| HybridConv-ViT | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |

## 8. 可视化分析

训练完成后依次分析 loss/accuracy 曲线、模型 Center Accuracy、Robust Drop、Grid heatmap、混淆矩阵与错误样本。分析应引用具体模型和位置，不仅复述图表高低。

## 9. 错误分析

重点检查 Shirt、T-shirt/top、Pullover 与 Coat 等视觉相近类别，以及边缘裁剪是否移除了鞋底、包带或袖口等判别区域。待实验后补充代表性失败案例。

## 10. 总结

待正式结果完成后回答：位置变化的影响、最稳定的架构、数据增强的效果、HybridConv-ViT 是否改善鲁棒性，以及局限性。应区分“平移不变性”和“边界遮挡鲁棒性”，因为正负 18 设置同时包含二者。

## 11. 成员贡献

见 [contribution.md](contribution.md)。
