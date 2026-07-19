# PV-FashionViT 最终实验报告

本目录包含中文 XeLaTeX 报告源码、实验图表和编译后的 [main.pdf](main.pdf)，内容包括任务定义、实验设计、五随机种子主实验、三随机种子补充消融、误差分析、局限与成员贡献。

报告主文件是 `main.tex`，章节正文位于 `sections/`，结果表位于 `tables/`，图片位于 `figures/`。实验数值由分析脚本汇总。

## 本地编译

本地安装 TeX Live 后，在 `report/` 目录运行：

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

也可以使用：

```bash
xelatex main.tex
xelatex main.tex
```

推送到 GitHub 后，`build-report` 工作流会自动编译，并在该次 Actions 运行的 Artifacts 区域提供 `PV-FashionViT-report`。

## 内容结构

- `main.tex`：文档入口、封面信息和章节顺序；
- `sections/`：摘要、背景、方法、实验、分析、总结与成员贡献；
- `tables/`：由结果分析脚本生成的 LaTeX 表格；
- `figures/`：数据示例、主结果、消融、误差样本和 Attention Rollout；
- `references.bib`：报告引用的文献条目；
- `pvfashionvit.sty`：报告版式与命令定义。

最终提交前建议以一次成功的 `build-report` 产物为准，并确认 PDF、源代码与仓库中的实验表格来自同一提交。
