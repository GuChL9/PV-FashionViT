# PV-FashionViT

基于 PyTorch 的位置可变 FashionMNIST 分类项目。项目把 28x28 的服饰图像放入 56x56 画布，比较 MLP、CNN、Tiny ViT、ViT-MeanPool 和 HybridConv-ViT 在中心、随机平移及固定位置网格下的表现。

> 当前状态：工程代码、CPU 默认配置、测试和报告骨架已完成。RE1:可视化已微调更新,报告前期铺垫工作已完成。

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
scripts/       单模型与一键实验脚本
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

## 训练

默认配置面向普通 CPU：`device=cpu`、batch size 64、DataLoader 单进程、ViT 使用 7x7 patch、64 维 embedding、2 层 encoder，并在至少 5 个 epoch 后启用早停。默认最多训练 12 个 epoch，训练结束后先执行 Center、Small Shift 和 Large Shift 验证，不自动运行耗时较长的 49 点网格。

建议先运行计算量较小的 MLP 和 CNN：

```bash
bash scripts/run_cpu_baselines.sh
```

Windows PowerShell：

```powershell
.\scripts\run_cpu_baselines.ps1
```

也可以逐个运行全部模型：

```bash
python src/main.py --config configs/mlp.yaml
python src/main.py --config configs/cnn.yaml
python src/main.py --config configs/vit_abspos.yaml
python src/main.py --config configs/vit_aug.yaml
python src/main.py --config configs/vit_meanpool.yaml
python src/main.py --config configs/hybrid_vit.yaml
```

位置编码消融的可选扩展（不计入上面的六组主实验）为：

```bash
python src/main.py --config configs/vit_sincos.yaml
```

快速验证整个程序链路：

```bash
python src/main.py --config configs/vit_abspos.yaml --smoke
```

Linux/macOS 一键运行全部 CPU 实验：

```bash
bash scripts/run_all_experiments.sh
```

Windows PowerShell 一键运行全部 CPU 实验：

```powershell
.\scripts\run_all_experiments.ps1
```

## 测试与网格评估

```bash
pytest -q
bash scripts/eval_grid_shift.sh configs/vit_abspos.yaml outputs/vit_abspos_center_cpu/checkpoints/best.pt
```

49 点网格会对完整测试集重复评估 49 次，CPU 上应在模型训练完成后单独运行。

Windows PowerShell：

```powershell
.\scripts\eval_grid_shift.ps1
```

六组正式 checkpoint 均已训练完成后，可统一生成 49 点网格结果、报告图与 LaTeX 表格：

```powershell
.\scripts\eval_all_grid.ps1
```

该脚本不会重新训练模型。它会依次加载六个 `best.pt`，完成每个模型的 49 点 Grid 评估，并自动运行结果汇总。CPU 上这一步耗时明显长于单次训练；请避免中途终止，否则对应模型不会写出完整的 49 行网格结果。

`eval_grid_shift.ps1` 只评估一个模型（默认是 ViT-AbsPos），适合调试某个 checkpoint；`eval_all_grid.ps1` 是六组正式实验的批量版本，适合生成最终对比结果。

只重新评估已有模型：

```bash
python src/main.py --config configs/vit_abspos.yaml --eval-only --checkpoint outputs/vit_abspos_center_cpu/checkpoints/best.pt
```

强制执行 49 点网格：

```bash
python src/main.py --config configs/vit_abspos.yaml --eval-only --grid --checkpoint outputs/vit_abspos_center_cpu/checkpoints/best.pt
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

六组实验全部完成后运行：

```bash
python src/analyze_results.py
```

该脚本只读取正式 run，自动忽略名称以 `_smoke` 结尾的链路测试输出，并同步生成
`outputs/figures/` 与 `report/figures/` 中的报告图片、`report/tables/` 中的正式表格。

### 六组网格评估完成后的核心产物

当六个 run 都有完整的 `tables/grid_accuracy.csv` 后，自动汇总会生成：

- `outputs/figures/grid_accuracy_panel.png`：六个模型的 7×7 Grid Accuracy heatmap 面板；
- `outputs/figures/grid_accuracy_comparison.png`：六个模型的 Grid Accuracy 对比；
- `outputs/figures/model_accuracy_comparison.png`：Center Accuracy 对比；
- `outputs/figures/robust_drop_comparison.png`：Robust Drop 对比；
- `outputs/figures/training_curves_overview.png`：六组验证曲线总览；
- `outputs/figures/per_class_accuracy_heatmap.png`：Large Shift 条件下的类别准确率热力图。

这些图会同步复制到 `report/figures/`；正式结果表会写入 `report/tables/`。若只完成部分网格评估，panel 中对应模型会显示缺失状态，不能用于最终报告。

## 工程改进与使用边界

- 所有绘图脚本固定使用无界面 `Agg` 后端，因此不依赖 Windows 的 Tcl/Tk 图形环境。
- `--smoke` 只验证训练链路，不会写入正式 `outputs/tables/`；正式图表只读取有 checkpoint 的非 smoke run。
- `analyze_results.py` 从每个 run 的 `evaluation.json`、配置和网格 CSV 重建汇总，避免旧 CSV 或临时实验污染结果。
- Sin-Cos 位置编码配置 `configs/vit_sincos.yaml` 是可选消融，不替代六组主实验。
- Attention Rollout 是展示模型行为的进阶分析，不是分类性能的核心证据；报告中应将其表述为辅助解释，而非因果证明。

## 报告可围绕的关键启示

- 不应只比较中心位置 Accuracy。`Robust Drop` 与 7×7 heatmap 才能直接说明模型是否依赖“目标位于中心”的训练偏置。
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
python src/visualize_attention.py --config configs/hybrid_vit.yaml \
  --checkpoint outputs/hybrid_vit_cpu/checkpoints/best.pt --mode center
python src/visualize_attention.py --config configs/hybrid_vit.yaml \
  --checkpoint outputs/hybrid_vit_cpu/checkpoints/best.pt --mode grid_shift --dx 18 --dy 18
```

## 评价指标

- Center、Small Shift、Large Shift 和 49 点 Grid Accuracy
- Robust Drop = Center Accuracy - Large Shift Accuracy
- 每类别准确率与混淆矩阵
