from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import statistics
import time
from pathlib import Path

import torch

try:
    from models import build_model
    from utils.config import load_config
except ModuleNotFoundError:  # Support importing this script as src.benchmark_cpu in tests.
    from src.models import build_model
    from src.utils.config import load_config


MODEL_CONFIGS = [
    ("MLP", Path("configs/mlp.yaml")),
    ("CNN", Path("configs/cnn.yaml")),
    ("Tiny ViT (CLS)", Path("configs/vit_abspos.yaml")),
    ("ViT-MeanPool", Path("configs/vit_meanpool.yaml")),
    ("HybridConv-ViT", Path("configs/hybrid_vit.yaml")),
]


def processor_name() -> str:
    if platform.system() == "Windows":
        try:
            import winreg

            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except (ImportError, OSError):
            pass
    return (
        platform.processor()
        or os.environ.get("PROCESSOR_IDENTIFIER")
        or "unknown"
    ).strip()


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def benchmark_model(
    label: str,
    config_path: Path,
    batch_size: int,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    config = load_config(config_path)
    model = build_model(config["model"]).cpu().eval()
    channels = int(config["model"].get("in_channels", 1))
    image_size = int(config["model"].get("img_size", 56))
    inputs = torch.randn(batch_size, channels, image_size, image_size)
    total, trainable = count_parameters(model)

    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs)
        timings = []
        for _ in range(repeats):
            started = time.perf_counter()
            model(inputs)
            timings.append(time.perf_counter() - started)

    mean_seconds = statistics.fmean(timings)
    median_seconds = statistics.median(timings)
    row = {
        "model": label,
        "parameters": total,
        "trainable_parameters": trainable,
        "batch_size": batch_size,
        "mean_latency_ms": mean_seconds * 1000,
        "median_latency_ms": median_seconds * 1000,
        "throughput_images_per_s": batch_size / mean_seconds,
        "repeats": repeats,
    }
    del inputs, model
    gc.collect()
    return row


def run_benchmark(
    batch_size: int = 64,
    warmup: int = 5,
    repeats: int = 30,
    threads: int = 8,
) -> list[dict[str, object]]:
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(max(1, min(4, threads)))
    torch.manual_seed(42)
    return [
        benchmark_model(label, config, batch_size, warmup, repeats)
        for label, config in MODEL_CONFIGS
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(path: Path, rows: list[dict[str, object]], threads: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    batch_size = int(rows[0]["batch_size"])
    repeats = int(rows[0]["repeats"])
    lines = [
        r"\begin{table}[H]",
        r"  \centering",
        rf"  \caption{{统一 CPU 前向基准。使用 {threads} 个 PyTorch 线程、batch size {batch_size}，预热后重复 {repeats} 次；延迟不含数据加载。}}",
        r"  \label{tab:cpu-benchmark}",
        r"  \renewcommand{\arraystretch}{1.12}",
        r"  \begin{tabular}{lrrr}",
        r"    \toprule",
        r"    Model & Parameters & Mean latency (ms/batch) & Throughput (images/s) \\",
        r"    \midrule",
    ]
    for row in rows:
        lines.append(
            f"    {row['model']} & {int(row['parameters']):,} & "
            f"{float(row['mean_latency_ms']):.2f} & "
            f"{float(row['throughput_images_per_s']):,.1f} \\\\"
        )
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark project models on CPU")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--csv-output", type=Path, default=Path("outputs/tables/cpu_benchmark.csv")
    )
    parser.add_argument(
        "--table-output", type=Path, default=Path("report/tables/cpu_benchmark.tex")
    )
    parser.add_argument(
        "--metadata-output", type=Path, default=Path("outputs/tables/cpu_benchmark_meta.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_benchmark(args.batch_size, args.warmup, args.repeats, args.threads)
    write_csv(args.csv_output, rows)
    write_latex(args.table_output, rows, args.threads)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "processor": processor_name(),
        "threads": args.threads,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
    }
    args.metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
