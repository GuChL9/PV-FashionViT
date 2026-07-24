# PV-FashionViT

PV-FashionViT 是一个基于 PyTorch 的位置可变 FashionMNIST 分类项目。项目将原始 $28\times28$ 服饰图像放入 $56\times56$ 画布，比较 MLP、CNN、Tiny ViT 及 HybridConv-ViT 在中心、平移和旋转条件下的表现。

最终报告见 [report/main.pdf](report/main.pdf)，更详细的任务定义见 [PROJECT_SPEC.md](PROJECT_SPEC.md)。

## 研究问题

1. 目标位置变化会造成多大的准确率下降？
2. 使用绝对位置编码的 Tiny ViT 对位移是否敏感？
3. 位置与旋转增强、Mean Pooling 和卷积 stem 能否改善位置鲁棒性？

## 核心结果

六组主实验使用 5 个随机种子。下表列出部分主要指标，完整均值、标准差、角度评估和分类别分析见报告。

| 模型 | Center | Large Shift | Grid |
| --- | ---: | ---: | ---: |
| CNN | 90.93% | 13.80% | 13.09% |
| ViT-AbsPos | 86.38% | 20.03% | 20.68% |
| ViT-Aug | 62.81% | 64.60% | 63.77% |
| ViT-MeanPool | 63.51% | 66.29% | 65.17% |
| HybridConv-ViT | 81.42% | 81.33% | 81.05% |

中心准确率不能代表位置变化下的表现。平移与旋转增强带来的改善最大，Mean Pooling 有小幅收益；加入卷积 stem 后，HybridConv-ViT 在多种测试条件下均保持约 $81\%$ 的准确率。三个随机种子的逐项消融显示，Random Erasing 在当前设置下影响较小。

## 项目结构

```text
configs/       主实验、多随机种子、消融与 GPU 备用配置
src/datasets/  FashionMNIST 与 PV-FashionMNIST 数据集
src/models/    MLP、CNN、Tiny ViT 与 HybridConv-ViT
src/engine/    训练、评估与指标
src/utils/     配置、随机种子、检查点、日志和绘图
scripts/       PowerShell 与 Bash 运行脚本
tests/         单元测试
notebooks/     数据、结果和注意力分析入口
outputs/       本地生成的结果；权重和大部分运行文件不入库
report/        中文 XeLaTeX 报告源码、图表与 PDF
```

## 环境安装

```bash
conda env create -f environment.yml
conda activate pv-vit
```

首次读取数据时，`torchvision` 会下载 FashionMNIST 到 `data/`。

## 快速检查

生成数据预览：

```bash
python src/visualize_data.py
```

运行一次小规模训练以检查完整流程：

```bash
python src/main.py --config configs/vit_abspos.yaml --smoke
```

运行测试：

```bash
pytest -q
```

## 主实验

Windows PowerShell 下，运行六组模型的五随机种子训练：

```powershell
.\scripts\run_multiseed_experiments.ps1
```

训练完成后，运行 49 点位置网格、固定角度评估和结果汇总：

```powershell
.\scripts\eval_multiseed_grid.ps1
```

单个检查点也可使用 `--eval-only` 重新评估：

```bash
python src/main.py --config configs/seeds/vit_abspos_s42.yaml --eval-only --grid --checkpoint outputs/vit_abspos_center_cpu_s42/checkpoints/best.pt
```

## 补充实验

逐项增强消融、位置编码消融和等半径裁剪对照分别使用：

```powershell
.\scripts\run_augmentation_ablation.ps1
.\scripts\run_position_encoding_ablation.ps1
.\scripts\run_matched_clipping.ps1
```

两组消融默认使用随机种子 42、2026 和 3407。等半径裁剪对照复用随机种子 42 的 Center-trained ViT 与 ViT-Aug 检查点。

统一 CPU 前向基准：

```bash
python src/benchmark_cpu.py
```

## 输出与报告

每次实验写入 `outputs/<run_name>/`，包括配置、检查点、训练日志、评估结果、表格和图片。分析脚本将报告所需的表格与图片复制到 `report/tables/` 和 `report/figures/`。

报告使用 XeLaTeX 编译：

```bash
cd report
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

GitHub Actions 中的 `build-report` 工作流也会编译 `report/main.tex` 并上传 PDF 构建产物。

Attention Rollout 仅用于观察注意力分布，不作为额外的准确率指标。位置网格的外圈可能同时包含更大位移和图像裁剪，相关对照与限制见报告第 7 节。
