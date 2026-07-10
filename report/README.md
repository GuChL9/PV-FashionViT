# LaTeX 报告模板

报告主文件是 `main.tex`，使用 XeLaTeX 编译。模板只包含章节、表格、图片和成员信息占位，不包含实验结论或正式结果。

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

填写规则：

1. 在 `main.tex` 顶部填写课程、小组和提交信息。
2. 在 `sections/` 中逐章替换灰色填写提示。
3. 将正式图片放入 `figures/`，结果表写入 `tables/`。
4. 在 `references.bib` 中添加文献，并按 `main.tex` 末尾注释启用 BibTeX。
