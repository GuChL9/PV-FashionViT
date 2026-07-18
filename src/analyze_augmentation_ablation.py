from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stage:
    key: str
    label: str
    short_label: str
    run_base: str
    global_fallback: str | None = None


STAGES = [
    Stage("center", "Center", "Center", "vit_abspos_center_cpu", "vit_abspos_center_cpu"),
    Stage("shift", "Shift only", "Shift", "vit_shift_cpu"),
    Stage("shift_rotation", "Shift + Rotation", "Shift+Rot", "vit_shift_rotation_cpu"),
    Stage(
        "shift_rotation_erasing",
        "Shift + Rotation + Erasing",
        "+Erase",
        "vit_aug_cpu",
        "vit_aug_cpu",
    ),
]

METRICS = [
    ("center_accuracy", "Center"),
    ("large_shift_accuracy", "Large Shift"),
    ("rotation_accuracy", "Rotation"),
    ("shift_rotation_accuracy", "Shift+Rotation"),
]


def load_global_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        return {row["model"]: row for row in csv.DictReader(stream)}


def load_stage_summary(stage: Stage, seed: int, outputs_dir: Path, global_rows: dict) -> dict:
    run_name = f"{stage.run_base}_s{seed}"
    evaluation = outputs_dir / run_name / "evaluation.json"
    if evaluation.exists():
        with evaluation.open("r", encoding="utf-8") as stream:
            return json.load(stream)["summary"]
    if stage.global_fallback and stage.global_fallback in global_rows:
        return global_rows[stage.global_fallback]
    raise FileNotFoundError(
        f"Missing {evaluation}. Train the ablation stage first; no global fallback is available."
    )


def collect_rows(seed: int, outputs_dir: Path, global_table: Path) -> list[dict[str, object]]:
    global_rows = load_global_rows(global_table)
    rows = []
    for stage in STAGES:
        summary = load_stage_summary(stage, seed, outputs_dir, global_rows)
        row: dict[str, object] = {
            "stage": stage.key,
            "label": stage.label,
            "short_label": stage.short_label,
            "seed": seed,
        }
        for metric, _ in METRICS:
            row[metric] = float(summary[metric])
        row["robust_drop"] = float(summary["robust_drop"])
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stage",
        "label",
        "seed",
        *(metric for metric, _ in METRICS),
        "robust_drop",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_latex(path: Path, rows: list[dict[str, object]], seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[H]",
        r"  \centering",
        rf"  \caption{{随机种子 {seed} 的数据增强逐项消融。四行使用相同 Tiny ViT、优化器和训练预算，只逐步扩大训练分布。}}",
        r"  \label{tab:augmentation-stages}",
        r"  \renewcommand{\arraystretch}{1.12}",
        r"  \resizebox{\textwidth}{!}{%",
        r"    \begin{tabular}{lccccc}",
        r"      \toprule",
        r"      Training distribution & Center Acc & Large Shift Acc & Rotation Acc & Shift+Rotation Acc & Robust Drop (pp) \\",
        r"      \midrule",
    ]
    for row in rows:
        values = [100 * float(row[metric]) for metric, _ in METRICS]
        drop = 100 * float(row["robust_drop"])
        label = str(row["label"]).replace("+", r"$+$")
        lines.append(
            f"      {label} & {values[0]:.2f}\\% & {values[1]:.2f}\\% & "
            f"{values[2]:.2f}\\% & {values[3]:.2f}\\% & {drop:.2f} \\\\"
        )
    lines.extend(
        [
            r"      \bottomrule",
            r"    \end{tabular}%",
            r"  }",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(path: Path, rows: list[dict[str, object]], seed: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    x = list(range(len(rows)))
    offsets = [-0.27, -0.09, 0.09, 0.27]
    width = 0.18
    colors = ["#244A73", "#2F7F83", "#C58B32", "#4E8B66"]
    fig, ax = plt.subplots(figsize=(10.5, 5.1))
    for (metric, label), offset, color in zip(METRICS, offsets, colors):
        values = [100 * float(row[metric]) for row in rows]
        bars = ax.bar([value + offset for value in x], values, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x, [str(row["short_label"]) for row in rows])
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.legend(loc="upper left", ncols=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"Augmentation-stage ablation (seed {seed})")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the staged augmentation ablation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--global-table", type=Path, default=Path("outputs/tables/main_results.csv")
    )
    parser.add_argument(
        "--csv-output", type=Path, default=Path("outputs/tables/augmentation_ablation_s42.csv")
    )
    parser.add_argument(
        "--table-output", type=Path, default=Path("report/tables/augmentation_stages.tex")
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("report/figures/augmentation_stages.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.seed, args.outputs_dir, args.global_table)
    write_csv(args.csv_output, rows)
    write_latex(args.table_output, rows, args.seed)
    plot(args.figure_output, rows, args.seed)
    print(f"Wrote {len(rows)} augmentation stages for seed {args.seed}")


if __name__ == "__main__":
    main()
