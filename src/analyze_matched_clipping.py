from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


MODEL_LABELS = {
    "vit_center_stage_cpu_s42": "Center-trained ViT",
    "vit_shift_rotation_erasing_cpu_s42": "Augmented ViT",
}


def load_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = []
        for row in csv.DictReader(stream):
            rows.append(
                {
                    "model": row["model"],
                    "seed": int(row["seed"]),
                    "dx": int(row["dx"]),
                    "dy": int(row["dy"]),
                    "radius_squared": int(row["radius_squared"]),
                    "clipped": row["clipped"].lower() in {"true", "1", "yes"},
                    "accuracy": float(row["accuracy"]),
                }
            )
    return rows


def summarize_matched_rows(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, int, int], dict[bool, list[float]]] = defaultdict(
        lambda: {False: [], True: []}
    )
    for row in rows:
        key = (str(row["model"]), int(row["seed"]), int(row["radius_squared"]))
        grouped[key][bool(row["clipped"])].append(float(row["accuracy"]))

    radius_rows = []
    for (model, seed, radius_squared), conditions in sorted(grouped.items()):
        no_clip, clipped = conditions[False], conditions[True]
        if not no_clip or not clipped:
            raise ValueError(
                f"{model} seed {seed} radius^2={radius_squared} lacks a matched condition"
            )
        no_clip_accuracy = statistics.fmean(no_clip)
        clipped_accuracy = statistics.fmean(clipped)
        radius_rows.append(
            {
                "model": model,
                "seed": seed,
                "radius_squared": radius_squared,
                "radius": math.sqrt(radius_squared),
                "no_clip_points": len(no_clip),
                "clipped_points": len(clipped),
                "no_clip_accuracy": no_clip_accuracy,
                "clipped_accuracy": clipped_accuracy,
                "matched_gap": no_clip_accuracy - clipped_accuracy,
            }
        )

    summary_rows = []
    model_keys = sorted({(str(row["model"]), int(row["seed"])) for row in radius_rows})
    for model, seed in model_keys:
        subset = [
            row for row in radius_rows if row["model"] == model and row["seed"] == seed
        ]
        gaps = [float(row["matched_gap"]) for row in subset]
        summary_rows.append(
            {
                "model": model,
                "seed": seed,
                "matched_radii": len(subset),
                "no_clip_accuracy": statistics.fmean(
                    float(row["no_clip_accuracy"]) for row in subset
                ),
                "clipped_accuracy": statistics.fmean(
                    float(row["clipped_accuracy"]) for row in subset
                ),
                "matched_gap": statistics.fmean(gaps),
                "gap_std_across_radii": statistics.stdev(gaps) if len(gaps) > 1 else 0.0,
            }
        )
    return radius_rows, summary_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{seed 42 的等半径裁剪对照。每个半径内分别宏平均无裁剪与可能裁剪坐标，再对七个完全匹配半径作宏平均；Gap 为无裁剪减可能裁剪。}",
        r"  \label{tab:matched-clipping}",
        r"  \renewcommand{\arraystretch}{1.12}",
        r"  \begin{tabular}{lcccc}",
        r"    \toprule",
        r"    Model & Matched radii & No-clip Acc & Clipping-possible Acc & Gap (pp) \\",
        r"    \midrule",
    ]
    for row in rows:
        label = MODEL_LABELS.get(str(row["model"]), str(row["model"]).replace("_", r"\_"))
        lines.append(
            f"    {label} & {int(row['matched_radii'])} & "
            f"{100 * float(row['no_clip_accuracy']):.2f}\\% & "
            f"{100 * float(row['clipped_accuracy']):.2f}\\% & "
            f"{100 * float(row['matched_gap']):.2f} \\\\"
        )
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(path: Path, radius_rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    colors = ["#244A73", "#C58B32", "#4E8B66"]
    for (model, seed), color in zip(
        sorted({(str(row["model"]), int(row["seed"])) for row in radius_rows}),
        colors,
    ):
        subset = sorted(
            (row for row in radius_rows if row["model"] == model and row["seed"] == seed),
            key=lambda row: int(row["radius_squared"]),
        )
        ax.plot(
            [float(row["radius"]) for row in subset],
            [100 * float(row["matched_gap"]) for row in subset],
            marker="o",
            linewidth=2.0,
            color=color,
            label=MODEL_LABELS.get(model, model),
        )
    ax.axhline(0, color="#222222", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Exact displacement radius (pixels)")
    ax.set_ylabel("No-clip - clipping-possible accuracy (pp)")
    ax.set_title("Distance-matched clipping comparison (seed 42)")
    ax.grid(linestyle="--", alpha=0.28)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze exact-distance clipping matches")
    parser.add_argument(
        "--input", type=Path, default=Path("outputs/tables/matched_clipping_raw.csv")
    )
    parser.add_argument(
        "--radius-output",
        type=Path,
        default=Path("outputs/tables/matched_clipping_by_radius.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/tables/matched_clipping_summary.csv"),
    )
    parser.add_argument(
        "--table-output", type=Path, default=Path("report/tables/matched_clipping.tex")
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("report/figures/matched_clipping.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    radius_rows, summary_rows = summarize_matched_rows(load_rows(args.input))
    write_csv(args.radius_output, radius_rows)
    write_csv(args.summary_output, summary_rows)
    write_latex(args.table_output, summary_rows)
    plot(args.figure_output, radius_rows)
    print(
        f"Wrote {len(radius_rows)} radius comparisons for {len(summary_rows)} model/seed pairs"
    )


if __name__ == "__main__":
    main()
