#!/usr/bin/env python3
"""Unified result summarization, selection, and plotting entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze DAFA experiment results")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize = subparsers.add_parser("summarize", help="Scan result files and write all/best-run summaries")
    summarize.add_argument("--results_root", type=str, default="results")
    summarize.add_argument("--output_dir", type=str, default="results/summary")

    select_best = subparsers.add_parser("select-best", help="Alias of summarize for compatibility")
    select_best.add_argument("--results_root", type=str, default="results")
    select_best.add_argument("--output_dir", type=str, default="results/summary")

    plot = subparsers.add_parser("plot", help="Plot summary figures from best-runs JSON")
    plot.add_argument("--best_runs", type=str, default="results/summary/best_runs.json")
    plot.add_argument("--output_dir", type=str, default="results/summary/plots")
    plot.add_argument("--format", type=str, default="pdf", choices=["pdf", "png", "svg"])
    plot.add_argument("--dpi", type=int, default=220)
    plot.add_argument("--style", type=str, default="whitegrid", choices=["whitegrid", "darkgrid", "ticks"])

    compare = subparsers.add_parser("compare", help="Compare multiple result directories or result files")
    compare.add_argument("--inputs", type=str, nargs="+", required=True)
    compare.add_argument("--output_dir", type=str, default="analysis_output")
    compare.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"])
    compare.add_argument("--dpi", type=int, default=220)
    compare.add_argument("--style", type=str, default="whitegrid", choices=["whitegrid", "darkgrid", "ticks"])
    compare.add_argument("--smooth_window", type=int, default=1)

    return parser.parse_args()


def _metric_name(data: Dict[str, Any]) -> str:
    return str(data.get("primary_metric_name") or ("perplexity" if data.get("best_perplexity") is not None else "accuracy"))


def _metric_value(data: Dict[str, Any]) -> float:
    metric_name = _metric_name(data)
    if metric_name == "perplexity":
        value = data.get("best_perplexity")
        return float(value) if value is not None else float("inf")
    return float(data.get("best_accuracy", 0.0))


def load_result(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    config_path = path.parent / "config.json"
    config: Dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    metric_name = _metric_name(data)
    return {
        "path": str(path),
        "metric_name": metric_name,
        "metric_value": _metric_value(data),
        "best_accuracy": float(data.get("best_accuracy", 0.0)),
        "best_perplexity": float(data.get("best_perplexity")) if data.get("best_perplexity") is not None else None,
        "final_round": int(data.get("final_round", 0)),
        "total_time": float(data.get("total_time", 0.0)),
        "method": config.get("method", "unknown"),
        "dataset": config.get("dataset", "unknown"),
        "seed": config.get("seed", "unknown"),
        "alpha": config.get("alpha", "unknown"),
        "run_group": config.get("run_group", "default"),
        "run_name": config.get("run_name", path.parent.name),
    }


def load_results_file_or_dir(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a single run results.json, got {path.name} with top-level type {type(data).__name__}")
        return data
    results_file = path / "results.json"
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a single run results.json under {path}")
        return data
    raise FileNotFoundError(f"Results not found: {path}")


def summarize_results(results_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(results_root.glob("**/results.json"))
    rows: List[Dict[str, Any]] = [load_result(p) for p in files]

    best: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = f"{row['run_group']}|{row['dataset']}|{row['method']}|{row['alpha']}"
        if key not in best:
            best[key] = row
            continue

        current = best[key]
        if row["metric_name"] == "perplexity":
            if row["metric_value"] < current["metric_value"]:
                best[key] = row
        else:
            if row["metric_value"] > current["metric_value"]:
                best[key] = row

    all_path = output_dir / "all_runs.json"
    best_path = output_dir / "best_runs.json"
    csv_path = output_dir / "best_runs.csv"

    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    sorted_best = sorted(best.values(), key=lambda x: (x["dataset"], x["method"], str(x["alpha"])))
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(sorted_best, f, indent=2)

    header = [
        "run_group", "dataset", "method", "alpha", "seed", "metric_name", "metric_value",
        "best_accuracy", "best_perplexity", "final_round", "total_time", "run_name", "path",
    ]
    lines = [",".join(header)]
    for row in sorted_best:
        lines.append(",".join(str(row.get(name, "")) for name in header))
    csv_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Scanned runs: {len(rows)}")
    print(f"Best groups: {len(best)}")
    print(f"Saved: {all_path}")
    print(f"Saved: {best_path}")
    print(f"Saved: {csv_path}")


def load_best_runs(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_by_dataset(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("dataset", "unknown")), []).append(row)
    return grouped


def _classification_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("metric_name") != "perplexity"]


def save_accuracy_bar(rows: List[Dict[str, Any]], output_dir: Path, fmt: str, dpi: int) -> None:
    rows = _classification_rows(rows)
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


def save_perplexity_bar(rows: List[Dict[str, Any]], output_dir: Path, fmt: str, dpi: int) -> None:
    rows = [row for row in rows if row.get("metric_name") == "perplexity"]
    if not rows:
        return
    rows = sorted(rows, key=lambda x: float(x.get("best_perplexity") or np.inf))
    labels = [f"{r.get('dataset')}-{r.get('method')}" for r in rows]
    values = [float(r.get("best_perplexity") or np.inf) for r in rows]
    plt.figure(figsize=(max(8, len(labels) * 0.5), 5))
    bars = plt.bar(range(len(labels)), values, color=sns.color_palette("crest", len(labels)))
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Best Perplexity")
    plt.title("Best Perplexity by Dataset-Method")
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"best_perplexity_bar.{fmt}", dpi=dpi)
    plt.close()


def save_dataset_panels(rows: List[Dict[str, Any]], output_dir: Path, fmt: str, dpi: int) -> None:
    grouped = group_by_dataset(rows)
    for dataset, items in grouped.items():
        metric_name = items[0].get("metric_name", "accuracy")
        reverse = metric_name != "perplexity"
        sort_key = (lambda x: float(x.get("metric_value", 0.0)))
        items = sorted(items, key=sort_key, reverse=reverse)
        methods = [str(x.get("method", "unknown")) for x in items]
        values = [float(x.get("metric_value", 0.0)) for x in items]
        plt.figure(figsize=(8, 4))
        bars = plt.bar(methods, values, color=sns.color_palette("tab10", len(methods)))
        plt.ylabel(f"Best {metric_name.title()}")
        plt.title(f"{dataset}: method ranking")
        if metric_name != "perplexity":
            plt.ylim(0, 1.0)
        for bar, val in zip(bars, values):
            text = f"{val:.2f}" if metric_name == "perplexity" else f"{val:.4f}"
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), text, ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset}_ranking.{fmt}", dpi=dpi)
        plt.close()


def plot_best_runs(best_runs_path: Path, output_dir: Path, fmt: str, dpi: int, style: str) -> None:
    sns.set_theme(style=style, context="talk")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_best_runs(best_runs_path)
    save_accuracy_bar(rows, output_dir, fmt, dpi)
    save_perplexity_bar(rows, output_dir, fmt, dpi)
    save_dataset_panels(rows, output_dir, fmt, dpi)
    print(f"Saved plots to {output_dir}")


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1 or len(values) <= 2:
        return values
    result = []
    for i in range(len(values)):
        left = max(0, i - window + 1)
        result.append(float(np.mean(values[left:i + 1])))
    return result


def compare_runs(inputs: List[str], output_dir: Path, fmt: str, dpi: int, style: str, smooth_window: int) -> None:
    sns.set_theme(style=style, context="talk")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_list = [load_results_file_or_dir(Path(path)) for path in inputs]
    names = [Path(path).name for path in inputs]
    metric_names = [_metric_name(result) for result in results_list]
    common_metric_name = metric_names[0] if len(set(metric_names)) == 1 else "primary_metric"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    palette = sns.color_palette("tab10", len(names))
    summary: Dict[str, Any] = {"methods": names, "metrics": {}}

    for idx, (result, name, metric_name) in enumerate(zip(results_list, names, metric_names)):
        history = result.get("history", [])
        metric_values = [h.get(metric_name, 0.0) for h in history]
        losses = [h.get("loss", 0.0) for h in history]
        rounds = list(range(1, len(history) + 1))
        axes[0].plot(rounds, moving_average(metric_values, smooth_window), linewidth=2, label=name, color=palette[idx])
        axes[1].plot(rounds, moving_average(losses, smooth_window), linewidth=2, label=name, color=palette[idx])
        best_value = min(metric_values) if metric_name == "perplexity" else max(metric_values) if metric_values else 0.0
        summary["metrics"][name] = {
            "primary_metric_name": metric_name,
            "best_primary_metric": best_value,
            "final_primary_metric": metric_values[-1] if metric_values else 0.0,
        }

    axes[0].set_xlabel("Round")
    axes[0].set_ylabel(common_metric_name.replace("_", " ").title())
    axes[0].set_title(f"{common_metric_name.replace('_', ' ').title()} Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss Comparison")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"comparison_curves.{fmt}", dpi=dpi)
    plt.close()

    with open(output_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved comparison outputs to {output_dir}")


def main() -> None:
    args = parse_args()
    if args.command in {"summarize", "select-best"}:
        summarize_results(Path(args.results_root), Path(args.output_dir))
    elif args.command == "plot":
        plot_best_runs(Path(args.best_runs), Path(args.output_dir), args.format, args.dpi, args.style)
    elif args.command == "compare":
        compare_runs(args.inputs, Path(args.output_dir), args.format, args.dpi, args.style, args.smooth_window)


if __name__ == "__main__":
    main()
