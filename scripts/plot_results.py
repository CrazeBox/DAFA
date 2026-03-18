#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot publication-style figures from DAFA summaries")
    parser.add_argument("--best_runs", type=str, default="results/summary/best_runs.json")
    parser.add_argument("--output_dir", type=str, default="results/summary/plots")
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png", "svg"])
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--style", type=str, default="whitegrid", choices=["whitegrid", "darkgrid", "ticks"])
    return parser.parse_args()


def load_best_runs(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_by_dataset(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("dataset", "unknown")), []).append(row)
    return grouped


def save_accuracy_bar(rows: List[Dict[str, Any]], output_dir: Path, fmt: str, dpi: int) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda x: (str(x.get("dataset", "")), -float(x.get("best_accuracy", 0.0))))
    labels = [f"{r.get('dataset')}-{r.get('method')}" for r in rows]
    values = [float(r.get("best_accuracy", 0.0)) for r in rows]
    plt.figure(figsize=(max(10, len(labels) * 0.5), 6))
    bars = plt.bar(range(len(labels)), values, color=sns.color_palette("husl", len(labels)))
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Best Accuracy")
    plt.title("Best Accuracy by Dataset-Method")
    plt.ylim(0, 1.0)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"best_accuracy_bar.{fmt}", dpi=dpi)
    plt.close()


def save_dataset_panels(rows: List[Dict[str, Any]], output_dir: Path, fmt: str, dpi: int) -> None:
    grouped = group_by_dataset(rows)
    for dataset, items in grouped.items():
        items = sorted(items, key=lambda x: -float(x.get("best_accuracy", 0.0)))
        methods = [str(x.get("method", "unknown")) for x in items]
        accs = [float(x.get("best_accuracy", 0.0)) for x in items]
        plt.figure(figsize=(8, 4))
        bars = plt.bar(methods, accs, color=sns.color_palette("tab10", len(methods)))
        plt.ylabel("Best Accuracy")
        plt.title(f"{dataset}: method ranking")
        plt.ylim(0, 1.0)
        for bar, val in zip(bars, accs):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.4f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset}_ranking.{fmt}", dpi=dpi)
        plt.close()


def main() -> None:
    args = parse_args()
    sns.set_theme(style=args.style, context="talk")
    best_runs_path = Path(args.best_runs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_best_runs(best_runs_path)
    save_accuracy_bar(rows, output_dir, args.format, args.dpi)
    save_dataset_panels(rows, output_dir, args.format, args.dpi)
    print(f"saved plots to {output_dir}")


if __name__ == "__main__":
    main()
