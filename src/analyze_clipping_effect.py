from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

MODEL_ORDER = [
    "mlp_center_cpu",
    "cnn_center_cpu",
    "vit_abspos_center_cpu",
    "vit_aug_cpu",
    "vit_meanpool_cpu",
    "hybrid_vit_cpu",
]

MODEL_LABELS = {
    "mlp_center_cpu": "MLP",
    "cnn_center_cpu": "CNN",
    "vit_abspos_center_cpu": "ViT-AbsPos",
    "vit_aug_cpu": "ViT-Aug",
    "vit_meanpool_cpu": "ViT-MeanPool",
    "hybrid_vit_cpu": "HybridConv-ViT",
}


def load_grid_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def summarize_clipping_effect(
    rows: list[dict[str, object]], no_clip_limit: int = 14
) -> list[dict[str, object]]:
    """Split each model's grid into a no-clipping inner grid and outer ring.

    On a 56x56 canvas, a centered 28x28 foreground can move by at most 14
    pixels without losing pixels.  The current seven-point grid uses
    {-18, -12, -6, 0, 6, 12, 18}, so the inner 5x5 grid is guaranteed not to
    clip while every point on the outer ring can clip on at least one axis.
    """

    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"inner": [], "outer": []}
    )
    for row in rows:
        model = str(row["model"])
        dx, dy = int(row["dx"]), int(row["dy"])
        accuracy = float(row["accuracy"])
        region = "inner" if abs(dx) <= no_clip_limit and abs(dy) <= no_clip_limit else "outer"
        grouped[model][region].append(accuracy)

    summaries = []
    model_names = [name for name in MODEL_ORDER if name in grouped]
    model_names.extend(sorted(set(grouped) - set(model_names)))
    for model in model_names:
        inner, outer = grouped[model]["inner"], grouped[model]["outer"]
        if not inner or not outer:
            raise ValueError(
                f"Model {model!r} needs both inner and outer grid points; "
                f"got {len(inner)} inner and {len(outer)} outer"
            )
        inner_accuracy = sum(inner) / len(inner)
        outer_accuracy = sum(outer) / len(outer)
        summaries.append(
            {
                "model": model,
                "inner_points": len(inner),
                "outer_points": len(outer),
                "inner_accuracy": inner_accuracy,
                "outer_accuracy": outer_accuracy,
                "edge_gap": inner_accuracy - outer_accuracy,
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "inner_points",
        "outer_points",
        "inner_accuracy",
        "outer_accuracy",
        "edge_gap",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{随机种子 42 的无裁剪内层与可能裁剪外圈对照。内层为 $|dx|,|dy|\leq12$ 的 25 个点，外圈为至少一个坐标等于 $\pm18$ 的 24 个点。}",
        r"  \label{tab:clipping-effect}",
        r"  \renewcommand{\arraystretch}{1.12}",
        r"  \begin{tabular}{lccc}",
        r"    \toprule",
        r"    Model & Inner $5\times5$ Acc & Outer Ring Acc & Inner--Outer (pp) \\",
        r"    \midrule",
    ]
    for row in rows:
        label = MODEL_LABELS.get(str(row["model"]), str(row["model"]).replace("_", r"\_"))
        inner = 100 * float(row["inner_accuracy"])
        outer = 100 * float(row["outer_accuracy"])
        gap = 100 * float(row["edge_gap"])
        lines.append(f"    {label} & {inner:.2f}\\% & {outer:.2f}\\% & {gap:.2f} \\\\")
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_comparison(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [MODEL_LABELS.get(str(row["model"]), str(row["model"])) for row in rows]
    inner = [100 * float(row["inner_accuracy"]) for row in rows]
    outer = [100 * float(row["outer_accuracy"]) for row in rows]
    gaps = [100 * float(row["edge_gap"]) for row in rows]
    x = list(range(len(rows)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    inner_bars = ax.bar(
        [value - width / 2 for value in x],
        inner,
        width,
        label="Inner 5x5 (no clipping)",
        color="#244A73",
    )
    outer_bars = ax.bar(
        [value + width / 2 for value in x],
        outer,
        width,
        label="Outer ring (clipping possible)",
        color="#C58B32",
    )
    ax.bar_label(inner_bars, fmt="%.1f", padding=2, fontsize=8)
    ax.bar_label(outer_bars, fmt="%.1f", padding=2, fontsize=8)
    for index, (inside, outside, gap) in enumerate(zip(inner, outer, gaps)):
        ax.text(
            index,
            max(inside, outside) + 6.0,
            f"gap {gap:.1f} pp",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#5F6B76",
        )

    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the no-clipping inner grid with the clipped outer ring"
    )
    parser.add_argument("--grid-csv", type=Path, default=Path("outputs/tables/grid_accuracy.csv"))
    parser.add_argument("--no-clip-limit", type=int, default=14)
    parser.add_argument("--csv-output", type=Path, default=Path("outputs/tables/clipping_effect.csv"))
    parser.add_argument(
        "--table-output", type=Path, default=Path("report/tables/clipping_effect.tex")
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("outputs/figures/clipping_effect_comparison.png"),
    )
    parser.add_argument(
        "--report-figure",
        type=Path,
        default=Path("report/figures/clipping_effect_comparison.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = summarize_clipping_effect(load_grid_rows(args.grid_csv), args.no_clip_limit)
    write_csv(args.csv_output, rows)
    write_latex_table(args.table_output, rows)
    plot_comparison(args.figure_output, rows)
    plot_comparison(args.report_figure, rows)
    print(
        f"Wrote clipping analysis for {len(rows)} models to "
        f"{args.csv_output}, {args.table_output}, and {args.report_figure}"
    )


if __name__ == "__main__":
    main()
