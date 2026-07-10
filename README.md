# PV-FashionViT

基于 PyTorch 的位置可变 FashionMNIST 分类项目。项目把 28x28 的服饰图像放入 56x56 画布，比较 MLP、CNN、Tiny ViT、ViT-MeanPool 和 HybridConv-ViT 在中心、随机平移及固定位置网格下的表现。

> 当前状态：工程代码、CPU 默认配置、测试和报告骨架已完成。

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
report/        报告与成员贡献
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

## 评价指标

- Center、Small Shift、Large Shift 和 49 点 Grid Accuracy
- Robust Drop = Center Accuracy - Large Shift Accuracy
- 每类别准确率与混淆矩阵
