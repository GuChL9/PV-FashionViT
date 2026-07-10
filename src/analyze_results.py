from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def bar_chart(table, column, title, output):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(table["model"], table[column])
    ax.set(title=title, ylabel=column.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    root = Path("outputs")
    table = pd.read_csv(root / "tables" / "main_results.csv")
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    bar_chart(table, "center_accuracy", "Center Accuracy by Model", figures / "model_accuracy_comparison.png")
    bar_chart(table, "robust_drop", "Robust Drop by Model", figures / "robust_drop_comparison.png")


if __name__ == "__main__":
    main()

