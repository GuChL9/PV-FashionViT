# PV-FashionViT

基于 PyTorch 的位置可变 FashionMNIST 分类项目。项目把 28x28 的服饰图像放入 56x56 画布，比较 MLP、CNN、Tiny ViT、ViT-MeanPool 和 HybridConv-ViT 在中心、随机平移及固定位置网格下的表现。

> 当前状态：工程代码、CPU 默认配置、测试和报告骨架已完成。RE3:报告已完成。 

## 研究问题

1. 目标位置变化会造成多大的准确率下降？
2. 使用绝对位置编码的 Tiny ViT 对位移是否敏感？
3. 随机位置增强、Mean Pooling 和卷积 stem 能否提升位置鲁棒性？

## 项目结构

```text
configs/       六组实验配置
src/datasets/  FashionMNIST 与 PV-FashionMNIST
src/models/    MLP、CNN、Tiny ViT、HybridConv-ViT
src/engine/    训练、评估与指标
src/utils/     配置、复现、checkpoint、日志和绘图
scripts/       多随机种子正式流程与单模型调试脚本
tests/         数据集和模型单元测试
notebooks/     数据、结果和注意力分析入口
outputs/       自动生成的表格、图片、日志和权重
report/        中文 XeLaTeX 报告模板
```

更精确的实验口径和对原说明书的修订见 [PROJECT_SPEC.md](PROJECT_SPEC.md)。

## 安装

```bash
conda env create -f environment.yml
conda activate pv-vit
```

首次运行时 `torchvision` 会自动下载 FashionMNIST 到 `data/`。

## 数据预览

```bash
python src/visualize_data.py
```

生成：

- `outputs/figures/data_preview.png`
- `outputs/figures/position_variation_demo.png`
- `outputs/figures/angle_variation_demo.png`

## 训练

默认配置面向普通 CPU：`device=cpu`、batch size 64、DataLoader 单进程、ViT 使用 7x7 patch、64 维 embedding、2 层 encoder，并在至少 5 个 epoch 后启用早停。默认最多训练 12 个 epoch，训练结束后先执行 Center、Small Shift 和 Large Shift 验证，不自动运行耗时较长的 49 点网格。

正式结果统一采用五个随机种子，不再额外训练一套内容重复的单种子模型。Windows PowerShell 一键运行 6 个模型、5 个 seed，共 30 个 run：

```powershell
.\scripts\run_multiseed_experiments.ps1
```

其中 seed 42 同时作为训练曲线、位置热力图、逐类结果和 attention 图的代表性 run；它已经包含在 30 个 run 中。

位置编码消融的可选扩展（不计入上面的六组主实验）为：

```bash
python src/main.py --config configs/vit_sincos.yaml
```

数据增强还提供一条逐项消融支线。它保持 Tiny ViT、优化器与训练预算不变，只补训“仅平移”和“平移+旋转”两个中间阶段，并与已有的 Center、完整增强端点组成四阶段对照：

```powershell
.\scripts\run_augmentation_ablation.ps1
```

默认只运行 seed 42，作为低成本的探索性证据；如需把中间阶段扩展到三个随机种子，可运行：

```powershell
.\scripts\run_augmentation_ablation.ps1 -Seeds 42,2026,3407
```

脚本不会改写六组主实验的全局汇总表。seed 42 完成后会生成 `outputs/tables/augmentation_ablation_s42.csv`、`report/tables/augmentation_stages.tex` 和 `report/figures/augmentation_stages.png`。

统一 CPU 前向基准可独立运行：

```bash
python src/benchmark_cpu.py
```

它在 8 个 PyTorch 线程、batch size 64 下，对五种不同推理结构分别预热 5 次并测量 30 个 batch，报告参数量、平均 batch 延迟和吞吐量。ViT-AbsPos 与 ViT-Aug 共用同一 Tiny ViT (CLS) 结构，因此只计一次；数据增强不会增加推理结构。该结果只测模型前向，不包含数据加载，也不等同于完整训练耗时。

快速验证整个程序链路：

```bash
python src/main.py --config configs/vit_abspos.yaml --smoke
```

## 测试与网格评估

```bash
pytest -q
bash scripts/eval_grid_shift.sh configs/seeds/vit_abspos_s42.yaml outputs/vit_abspos_center_cpu_s42/checkpoints/best.pt
```

49 点网格会对完整测试集重复评估 49 次，CPU 上应在模型训练完成后单独运行。

Windows PowerShell：

```powershell
.\scripts\eval_grid_shift.ps1
```

30 个正式 checkpoint 均已训练完成后，统一生成 49 点网格、固定角度结果、报告图与 LaTeX 表格：

```powershell
.\scripts\eval_multiseed_grid.ps1
```

该脚本不会重新训练模型。它会依次加载 30 个 `best.pt` 完成网格与角度评估，然后从 seed 42 生成细节图，并汇总五个 seed 的主表与统计图。CPU 上这一步耗时较长；请避免中途终止，否则对应 run 不会写出完整的 49 行网格结果。

`eval_grid_shift.ps1` 只评估一个 checkpoint，保留作局部调试；正式报告只使用 `eval_multiseed_grid.ps1`。

只重新评估已有模型：

```bash
python src/main.py --config configs/seeds/vit_abspos_s42.yaml --eval-only --checkpoint outputs/vit_abspos_center_cpu_s42/checkpoints/best.pt
```

强制执行 49 点网格：

```bash
python src/main.py --config configs/seeds/vit_abspos_s42.yaml --eval-only --grid --checkpoint outputs/vit_abspos_center_cpu_s42/checkpoints/best.pt
```

## 日后切换 GPU

`configs/gpu/` 保留了说明书推荐的 4x4 patch、128 维 embedding、4 层 encoder 和 30 epochs 配置。例如：

```bash
python src/main.py --config configs/gpu/vit_abspos.yaml
```

## 输出

每次实验写入 `outputs/<run_name>/`，包含 best/last checkpoint、训练日志、曲线、混淆矩阵、网格热力图和逐样本预测。跨模型汇总写入：

- `outputs/tables/main_results.csv`
- `outputs/tables/grid_accuracy.csv`
- `outputs/tables/angle_accuracy.csv`

多种子网格评估脚本会自动依次运行：

```bash
python src/analyze_results.py
python src/analyze_multiseed_results.py
python src/analyze_clipping_effect.py
```

第一个脚本只读取六个 seed 42 run，生成训练曲线、位置网格、角度图和逐类图；第二个脚本读取全部 30 个 run，生成五种子主表、误差图和模型差值图；第三个脚本把 seed 42 的 49 点位置网格拆成无裁剪内层 5×5 与可能裁剪的外圈，量化边缘区域的额外下降。三者共享同一批正式训练结果，不再依赖无 seed 后缀的旧单种子目录。

### 正式多种子评估完成后的核心产物

当 30 个 run 都有完整的 `tables/grid_accuracy.csv` 和 `tables/angle_accuracy.csv` 后，自动汇总会生成：

- `outputs/figures/grid_accuracy_panel.png`：六个模型的 7×7 Grid Accuracy heatmap 面板；
- `outputs/figures/grid_accuracy_comparison.png`：六个模型的 Grid Accuracy 对比；
- `report/figures/clipping_effect_comparison.png`：seed 42 的无裁剪内层与可能裁剪外圈对比；
- `outputs/figures/angle_robustness_heatmap.png`：固定角度准确率热力图，并附平均与最差角度表现；
- `outputs/figures/model_accuracy_comparison.png`：Center Accuracy 对比；
- `outputs/figures/robust_drop_comparison.png`：Robust Drop 对比；
- `outputs/figures/training_curves_overview.png`：六组验证曲线总览；
- `outputs/figures/per_class_accuracy_heatmap.png`：Large Shift 条件下的类别准确率热力图。

这些图会同步复制到 `report/figures/`；正式结果表会写入 `report/tables/`。若只完成部分网格评估，panel 中对应模型会显示缺失状态，不能用于最终报告。

## 工程改进与使用边界

- 所有绘图脚本固定使用无界面 `Agg` 后端，因此不依赖 Windows 的 Tcl/Tk 图形环境。
- `--smoke` 只验证训练链路，不会写入正式 `outputs/tables/`；正式图表只读取有 checkpoint 的非 smoke run。
- `analyze_results.py` 固定读取多种子流程中的 seed 42 run，`analyze_multiseed_results.py` 固定读取全部 30 个正式 run，避免旧单种子目录混入报告。
- Sin-Cos 位置编码配置 `configs/vit_sincos.yaml` 是可选消融，不替代六组主实验。
- Attention Rollout 是展示模型行为的进阶分析，不是分类性能的核心证据；报告中应将其表述为辅助解释，而非因果证明。

## 报告可围绕的关键启示

- 不应只比较中心位置 Accuracy。`Robust Drop` 与 7×7 heatmap 才能直接说明模型是否依赖“目标位于中心”的训练偏置。
- 49 点网格还应区分无裁剪内层与可能裁剪外圈；两者的差距能量化边缘区域的额外困难，但由于位置距离同时变化，不能把差值全部归因于裁剪。
- MLP、CNN 与 ViT-AbsPos 的作用是建立对照：它们说明局部归纳偏置和绝对位置编码在位置扰动下各自的局限，而非单纯争夺最高中心准确率。
- 数据增强、Mean Pooling 与卷积 stem 分别对应数据、聚合方式和结构三个工程层面的改进；消融表应按这一控制变量逻辑解释。
- 类别准确率、混淆矩阵与错误样本需要共同解释：衣物外形相近、边缘裁剪和只看到局部纹理都会造成错误，不能把所有错误简单归因于模型结构。

## Attention Rollout

Attention Rollout 用于辅助检查模型关注区域是否随衣物位置移动；它不是分类性能的核心指标，也不能单独解释模型决策因果。

报告优先使用下面的对照图：脚本会从四个极端角落自动筛选约 10 个 ``ViT-AbsPos 分类错误、HybridConv-ViT 分类正确`` 的样本，优先覆盖不同类别与不同角落。筛选规则仅写入样本清单；报告图本身按两张、每张五例的干净三栏版式展示输入、ViT-AbsPos rollout 与 HybridConv-ViT rollout。

```bash
python src/visualize_attention_comparison.py
```

生成：

- `outputs/figures/attention_abspos_vs_hybrid_01.png`
- `outputs/figures/attention_abspos_vs_hybrid_02.png`
- `report/figures/attention_abspos_vs_hybrid_01.png`
- `report/figures/attention_abspos_vs_hybrid_02.png`
- `outputs/tables/attention_abspos_wrong_hybrid_correct_samples.csv`

如需查看单一模型在任意位置的 rollout，可使用下面的通用命令。输出包含原图、rollout 与叠加图：

```bash
python src/visualize_attention.py --config configs/seeds/hybrid_vit_s42.yaml \
  --checkpoint outputs/hybrid_vit_cpu_s42/checkpoints/best.pt --mode center
python src/visualize_attention.py --config configs/seeds/hybrid_vit_s42.yaml \
  --checkpoint outputs/hybrid_vit_cpu_s42/checkpoints/best.pt --mode grid_shift --dx 18 --dy 18
```

## 评价指标

- Center、Small Shift、Large Shift 和 49 点 Grid Accuracy
- Robust Drop = Center Accuracy - Large Shift Accuracy
- 每类别准确率与混淆矩阵
