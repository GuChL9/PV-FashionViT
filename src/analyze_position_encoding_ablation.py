from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PositionProfile:
    key: str
    label: str
    short_label: str
    run_base: str


PROFILES = [
    PositionProfile("learnable", "Learnable absolute", "Learnable", "vit_center_stage_cpu"),
    PositionProfile("sincos", "Fixed 2-D Sin-Cos", "Sin-Cos", "vit_sincos_center_cls_cpu"),
    PositionProfile("none", "No positional encoding", "No Pos", "vit_no_pos_center_cls_cpu"),
]

METRICS = [
    ("center_accuracy", "Center"),
    ("large_shift_accuracy", "Large Shift"),
    ("rotation_accuracy", "Rotation"),
    ("shift_rotation_accuracy", "Shift+Rotation"),
]
SUMMARY_METRICS = [metric for metric, _ in METRICS] + ["robust_drop"]


def collect_rows(seeds: list[int], outputs_dir: Path) -> list[dict[str, object]]:
    rows = []
    missing = []
    for profile in PROFILES:
        for seed in seeds:
            run_name = f"{profile.run_base}_s{seed}"
            evaluation = outputs_dir / run_name / "evaluation.json"
            if not evaluation.exists():
                missing.append(str(evaluation))
                continue
            with evaluation.open("r", encoding="utf-8") as stream:
                result = json.load(stream)["summary"]
            row: dict[str, object] = {
                "profile": profile.key,
                "label": profile.label,
                "short_label": profile.short_label,
                "run_name": run_name,
                "seed": seed,
            }
            for metric in SUMMARY_METRICS:
                row[metric] = float(result[metric])
            rows.append(row)
    if missing:
        raise FileNotFoundError("Missing position-encoding runs:\n" + "\n".join(missing))
    return rows


def summarize_rows(
    rows: list[dict[str, object]], seeds: list[int]
) -> list[dict[str, object]]:
    summaries = []
    for profile in PROFILES:
        subset = [row for row in rows if row["profile"] == profile.key]
        observed = sorted(int(row["seed"]) for row in subset)
        if observed != sorted(seeds):
            raise ValueError(
                f"Profile {profile.key} has seeds {observed}; expected {sorted(seeds)}"
            )
        summary: dict[str, object] = {
            "profile": profile.key,
            "label": profile.label,
            "short_label": profile.short_label,
            "seed_count": len(subset),
        }
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in subset]
            summary[f"{metric}_mean"] = statistics.fmean(values)
            summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _cell(row: dict[str, object], metric: str) -> str:
    mean = 100 * float(row[f"{metric}_mean"])
    std = 100 * float(row[f"{metric}_std"])
    return f"{mean:.2f} $\\pm$ {std:.2f}"


def write_latex(path: Path, rows: list[dict[str, object]], seeds: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seed_text = ", ".join(str(seed) for seed in seeds)
    lines = [
        r"\begin{table}[H]",
        r"  \centering",
        rf"  \caption{{位置编码控制变量实验（随机种子 {seed_text}）。三组均使用 Center 训练、CLS pooling 与相同 Tiny ViT；单元格为均值 $\pm$ 样本标准差。}}",
        r"  \label{tab:position-encoding-ablation}",
        r"  \renewcommand{\arraystretch}{1.12}",
        r"  \resizebox{\textwidth}{!}{%",
        r"    \begin{tabular}{lccccc}",
        r"      \toprule",
        r"      Positional encoding & Center Acc & Large Shift Acc & Rotation Acc & Shift+Rotation Acc & Robust Drop (pp) \\",
        r"      \midrule",
    ]
    for row in rows:
        values = [_cell(row, metric) for metric, _ in METRICS]
        lines.append(
            f"      {row['label']} & "
            + " & ".join([*values, _cell(row, "robust_drop")])
            + r" \\"
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


def plot(path: Path, rows: list[dict[str, object]], seeds: list[int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = list(range(len(rows)))
    offsets = [-0.27, -0.09, 0.09, 0.27]
    colors = ["#244A73", "#2F7F83", "#C58B32", "#4E8B66"]
    fig, ax = plt.subplots(figsize=(9.4, 5.1))
    for (metric, label), offset, color in zip(METRICS, offsets, colors):
        means = [100 * float(row[f"{metric}_mean"]) for row in rows]
        stds = [100 * float(row[f"{metric}_std"]) for row in rows]
        bars = ax.bar(
            [position + offset for position in x],
            means,
            0.18,
            yerr=stds,
            capsize=2.5,
            label=label,
            color=color,
            error_kw={"elinewidth": 0.9, "capthick": 0.9},
        )
        ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=7)
    ax.set_xticks(x, [str(row["short_label"]) for row in rows])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Position-encoding ablation ({len(seeds)} seeds, mean +/- SD)")
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.legend(loc="upper left", ncols=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize position-encoding ablations")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2026, 3407])
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=Path("outputs/tables/position_encoding_3seed_raw.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/tables/position_encoding_3seed_summary.csv"),
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=Path("report/tables/position_encoding_ablation.tex"),
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("report/figures/position_encoding_ablation.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = list(dict.fromkeys(args.seeds))
    raw = collect_rows(seeds, args.outputs_dir)
    summary = summarize_rows(raw, seeds)
    write_csv(args.raw_output, raw)
    write_csv(args.summary_output, summary)
    write_latex(args.table_output, summary, seeds)
    plot(args.figure_output, summary, seeds)
    print(f"Wrote {len(PROFILES)} position profiles across {len(seeds)} seed(s)")


if __name__ == "__main__":
    main()
