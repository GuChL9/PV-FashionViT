from __future__ import annotations

import argparse
import csv
import json
import statistics
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
    Stage(
        "center",
        "Center",
        "Center",
        "vit_center_stage_cpu",
        "vit_abspos_center_cpu",
    ),
    Stage("shift", "Shift only", "Shift", "vit_shift_cpu"),
    Stage(
        "shift_rotation",
        "Shift + Rotation",
        "Shift+Rot",
        "vit_shift_rotation_cpu",
    ),
    Stage(
        "shift_rotation_erasing",
        "Shift + Rotation + Erasing",
        "+Erase",
        "vit_shift_rotation_erasing_cpu",
        "vit_aug_cpu",
    ),
]

METRICS = [
    ("center_accuracy", "Center"),
    ("large_shift_accuracy", "Large Shift"),
    ("rotation_accuracy", "Rotation"),
    ("shift_rotation_accuracy", "Shift+Rotation"),
]
SUMMARY_METRICS = [metric for metric, _ in METRICS] + ["robust_drop"]


def load_global_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        return {row["model"]: row for row in csv.DictReader(stream)}


def load_stage_summary(
    stage: Stage,
    seed: int,
    outputs_dir: Path,
    global_rows: dict[str, dict[str, str]],
) -> dict[str, object]:
    run_name = f"{stage.run_base}_s{seed}"
    evaluation = outputs_dir / run_name / "evaluation.json"
    if evaluation.exists():
        with evaluation.open("r", encoding="utf-8") as stream:
            return json.load(stream)["summary"]
    if seed == 42 and stage.global_fallback and stage.global_fallback in global_rows:
        return global_rows[stage.global_fallback]
    raise FileNotFoundError(
        f"Missing {evaluation}. Train the ablation stage for seed {seed} first."
    )


def collect_rows(
    seed: int,
    outputs_dir: Path,
    global_table: Path,
) -> list[dict[str, object]]:
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
        for metric in SUMMARY_METRICS:
            row[metric] = float(summary[metric])
        rows.append(row)
    return rows


def collect_multiseed_rows(
    seeds: list[int],
    outputs_dir: Path,
    global_table: Path,
) -> list[dict[str, object]]:
    return [
        row
        for seed in seeds
        for row in collect_rows(seed, outputs_dir, global_table)
    ]


def summarize_rows(
    rows: list[dict[str, object]], seeds: list[int]
) -> list[dict[str, object]]:
    summaries = []
    for stage in STAGES:
        subset = [row for row in rows if row["stage"] == stage.key]
        observed = sorted(int(row["seed"]) for row in subset)
        if observed != sorted(seeds):
            raise ValueError(
                f"Stage {stage.key} has seeds {observed}; expected {sorted(seeds)}"
            )
        summary: dict[str, object] = {
            "stage": stage.key,
            "label": stage.label,
            "short_label": stage.short_label,
            "seed_count": len(subset),
        }
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in subset]
            summary[f"{metric}_mean"] = statistics.fmean(values)
            summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty augmentation table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format_mean_std(row: dict[str, object], metric: str) -> str:
    mean = 100 * float(row[f"{metric}_mean"])
    std = 100 * float(row[f"{metric}_std"])
    return f"{mean:.2f} $\\pm$ {std:.2f}"


def write_latex(
    path: Path,
    rows: list[dict[str, object]],
    seeds: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seed_text = ", ".join(str(seed) for seed in seeds)
    lines = [
        r"\begin{table}[H]",
        r"  \centering",
        rf"  \caption{{数据增强逐项消融（随机种子 {seed_text}）。单元格为均值 $\pm$ 样本标准差，单位为百分比或百分点。}}",
        r"  \label{tab:augmentation-stages}",
        r"  \renewcommand{\arraystretch}{1.12}",
        r"  \resizebox{\textwidth}{!}{%",
        r"    \begin{tabular}{lccccc}",
        r"      \toprule",
        r"      Training distribution & Center Acc & Large Shift Acc & Rotation Acc & Shift+Rotation Acc & Robust Drop (pp) \\",
        r"      \midrule",
    ]
    for row in rows:
        label = str(row["label"]).replace("+", r"$+$")
        values = [_format_mean_std(row, metric) for metric, _ in METRICS]
        drop = _format_mean_std(row, "robust_drop")
        lines.append(f"      {label} & " + " & ".join([*values, drop]) + r" \\")
    lines.extend(
        [
            r"      \bottomrule",
            r"    \end{tabular}%",
            r"  }",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(path: Path, rows: list[dict[str, object]], seeds: list[int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    x = list(range(len(rows)))
    offsets = [-0.27, -0.09, 0.09, 0.27]
    width = 0.18
    colors = ["#244A73", "#2F7F83", "#C58B32", "#4E8B66"]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for (metric, label), offset, color in zip(METRICS, offsets, colors):
        means = [100 * float(row[f"{metric}_mean"]) for row in rows]
        stds = [100 * float(row[f"{metric}_std"]) for row in rows]
        positions = [value + offset for value in x]
        bars = ax.bar(
            positions,
            means,
            width,
            yerr=stds,
            capsize=2.5,
            label=label,
            color=color,
            error_kw={"elinewidth": 0.9, "capthick": 0.9},
        )
        ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x, [str(row["short_label"]) for row in rows])
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.legend(loc="upper left", ncols=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"Augmentation-stage ablation ({len(seeds)} seeds, mean +/- SD)")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the staged augmentation ablation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2026, 3407])
    parser.add_argument("--seed", type=int, help="Legacy single-seed shortcut")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--global-table", type=Path, default=Path("outputs/tables/main_results.csv")
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=Path("outputs/tables/augmentation_ablation_3seed_raw.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/tables/augmentation_ablation_3seed_summary.csv"),
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
    seeds = [args.seed] if args.seed is not None else list(dict.fromkeys(args.seeds))
    raw = collect_multiseed_rows(seeds, args.outputs_dir, args.global_table)
    summary = summarize_rows(raw, seeds)
    write_csv(args.raw_output, raw)
    write_csv(args.summary_output, summary)
    write_latex(args.table_output, summary, seeds)
    plot(args.figure_output, summary, seeds)
    print(f"Wrote {len(STAGES)} augmentation stages across {len(seeds)} seed(s)")


if __name__ == "__main__":
    main()
