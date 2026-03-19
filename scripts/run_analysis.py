#!/usr/bin/env python3
"""
Analysis script for federated learning experiment results.

Usage:
    python run_analysis.py --results_dir results/experiment_name
    python run_analysis.py --results_file results/experiment_name/results.json
"""

import argparse
import logging
import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze FL experiment results")
    
    parser.add_argument("--results_dir", type=str,
                       help="Directory containing experiment results")
    parser.add_argument("--results_file", type=str,
                       help="Specific results file to analyze")
    parser.add_argument("--output_dir", type=str, default="analysis_output",
                       help="Output directory for analysis results")
    
    parser.add_argument("--plot", action="store_true",
                       help="Generate plots")
    parser.add_argument("--plot_format", type=str, default="png",
                       choices=["png", "pdf", "svg"],
                       help="Plot output format")
    parser.add_argument("--plot_dpi", type=int, default=220,
                       help="DPI for saved plots")
    parser.add_argument("--smooth_window", type=int, default=1,
                       help="Moving average window for curves")
    parser.add_argument("--style", type=str, default="whitegrid",
                       choices=["whitegrid", "darkgrid", "ticks"],
                       help="Seaborn plotting style")
    
    parser.add_argument("--compare", type=str, nargs="+",
                       help="Compare multiple experiment results")
    
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    return parser.parse_args()


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1 or len(values) <= 2:
        return values
    result = []
    for i in range(len(values)):
        left = max(0, i - window + 1)
        chunk = values[left:i + 1]
        result.append(float(np.mean(chunk)))
    return result


def get_primary_metric_name(results: Dict[str, Any]) -> str:
    """Infer the primary metric stored in a results file."""
    if results.get("primary_metric_name"):
        return str(results["primary_metric_name"])
    history = results.get("history", [])
    if history and "perplexity" in history[0]:
        return "perplexity"
    return "accuracy"


def load_results(path: str) -> Dict[str, Any]:
    """Load experiment results from file."""
    path = Path(path)
    
    if path.is_file():
        with open(path, "r") as f:
            return json.load(f)
    elif path.is_dir():
        results_file = path / "results.json"
        if results_file.exists():
            with open(results_file, "r") as f:
                return json.load(f)
        
        all_results = {}
        for json_file in path.glob("**/*.json"):
            with open(json_file, "r") as f:
                all_results[json_file.stem] = json.load(f)
        return all_results
    else:
        raise FileNotFoundError(f"Results not found: {path}")


def analyze_single_experiment(
    results: Dict[str, Any],
    output_dir: Path,
    generate_plots: bool = True,
    plot_format: str = "png",
    plot_dpi: int = 220,
    smooth_window: int = 1,
    style: str = "whitegrid",
) -> Dict[str, Any]:
    """Analyze a single experiment's results."""
    analysis = {}
    
    history = results.get("history", [])
    if not history:
        logger.warning("No training history found")
        return analysis
    
    rounds = [h.get("round", i) for i, h in enumerate(history)]
    metric_name = get_primary_metric_name(results)
    metric_values = [h.get(metric_name, 0) for h in history]
    losses = [h.get("loss", 0) for h in history]
    
    analysis["convergence"] = {
        "primary_metric_name": metric_name,
        "final_primary_metric": metric_values[-1] if metric_values else 0,
        "best_primary_metric": min(metric_values) if metric_name == "perplexity" else max(metric_values) if metric_values else 0,
        "final_loss": losses[-1] if losses else 0,
        "convergence_round": None,
    }
    
    if metric_values:
        target = min(metric_values) * 1.05 if metric_name == "perplexity" else 0.95 * max(metric_values)
        for i, value in enumerate(metric_values):
            if (metric_name == "perplexity" and value <= target) or (metric_name != "perplexity" and value >= target):
                analysis["convergence"]["convergence_round"] = rounds[i]
                break
    
    analysis["training_stats"] = {
        "total_rounds": len(rounds),
        f"mean_{metric_name}": np.mean(metric_values),
        f"std_{metric_name}": np.std(metric_values),
        "mean_loss": np.mean(losses),
        "std_loss": np.std(losses),
    }
    
    if generate_plots:
        sns.set_theme(style=style, context="talk")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        smooth_metric = moving_average(metric_values, smooth_window)
        smooth_loss = moving_average(losses, smooth_window)
        
        axes[0].plot(rounds, smooth_metric, "b-", linewidth=2)
        axes[0].set_xlabel("Round")
        axes[0].set_ylabel(metric_name.replace("_", " ").title())
        axes[0].set_title(f"Test {metric_name.replace('_', ' ').title()} over Rounds")
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(rounds, smooth_loss, "r-", linewidth=2)
        axes[1].set_xlabel("Round")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Test Loss over Rounds")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"training_curves.{plot_format}", dpi=plot_dpi)
        plt.close()
    
    return analysis


def compare_experiments(
    results_list: List[Dict[str, Any]],
    names: List[str],
    output_dir: Path,
    plot_format: str = "png",
    plot_dpi: int = 220,
    smooth_window: int = 1,
    style: str = "whitegrid",
) -> Dict[str, Any]:
    """Compare multiple experiments."""
    comparison = {
        "methods": names,
        "metrics": {},
    }
    
    sns.set_theme(style=style, context="talk")
    metric_names = [get_primary_metric_name(results) for results in results_list]
    common_metric_name = metric_names[0] if len(set(metric_names)) == 1 else "primary_metric"
    all_metric_values = []
    all_losses = []
    
    for results, name, metric_name in zip(results_list, names, metric_names):
        history = results.get("history", [])
        metric_values = [h.get(metric_name, 0) for h in history]
        losses = [h.get("loss", 0) for h in history]
        
        all_metric_values.append(metric_values)
        all_losses.append(losses)
        
        best_value = min(metric_values) if metric_name == "perplexity" else max(metric_values) if metric_values else 0
        comparison["metrics"][name] = {
            "primary_metric_name": metric_name,
            "best_primary_metric": best_value,
            "final_primary_metric": metric_values[-1] if metric_values else 0,
            "mean_primary_metric": np.mean(metric_values) if metric_values else 0,
        }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    palette = sns.color_palette("tab10", len(names))
    
    for idx, (metric_values, name) in enumerate(zip(all_metric_values, names)):
        rounds = list(range(1, len(metric_values) + 1))
        axes[0].plot(rounds, moving_average(metric_values, smooth_window), linewidth=2, label=name, color=palette[idx])
    
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel(common_metric_name.replace("_", " ").title())
    axes[0].set_title(f"{common_metric_name.replace('_', ' ').title()} Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    for idx, (losses, name) in enumerate(zip(all_losses, names)):
        rounds = list(range(1, len(losses) + 1))
        axes[1].plot(rounds, moving_average(losses, smooth_window), linewidth=2, label=name, color=palette[idx])
    
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss Comparison")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"comparison_curves.{plot_format}", dpi=plot_dpi)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    best_values = [comparison["metrics"][name]["best_primary_metric"] for name in names]
    bars = ax.bar(names, best_values, color=sns.color_palette("husl", len(names)))
    
    ax.set_ylabel(f"Best {common_metric_name.replace('_', ' ').title()}")
    ax.set_title(f"Best {common_metric_name.replace('_', ' ').title()} Comparison")
    if common_metric_name != "perplexity":
        ax.set_ylim([0, 1])
    
    for bar, value in zip(bars, best_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )
    
    plt.tight_layout()
    plt.savefig(output_dir / f"accuracy_bar.{plot_format}", dpi=plot_dpi)
    plt.close()
    
    metrics_csv = output_dir / "comparison_metrics.csv"
    lines = ["method,primary_metric_name,best_primary_metric,final_primary_metric,mean_primary_metric"]
    for name in names:
        m = comparison["metrics"][name]
        lines.append(
            f"{name},{m['primary_metric_name']},{m['best_primary_metric']:.6f},{m['final_primary_metric']:.6f},{m['mean_primary_metric']:.6f}"
        )
    metrics_csv.write_text("\n".join(lines), encoding="utf-8")
    
    return comparison


def generate_report(
    analysis: Dict[str, Any],
    output_dir: Path,
) -> str:
    """Generate a text report of the analysis."""
    report_lines = [
        "=" * 60,
        "FEDERATED LEARNING EXPERIMENT ANALYSIS REPORT",
        "=" * 60,
        "",
    ]
    
    if "convergence" in analysis:
        conv = analysis["convergence"]
        report_lines.extend([
            "CONVERGENCE ANALYSIS",
            "-" * 40,
            f"  Primary Metric: {conv.get('primary_metric_name', 'accuracy')}",
            f"  Final Value: {conv.get('final_primary_metric', 0):.4f}",
            f"  Best Value: {conv.get('best_primary_metric', 0):.4f}",
            f"  Final Loss: {conv.get('final_loss', 0):.4f}",
            f"  Convergence Round: {conv.get('convergence_round', 'N/A')}",
            "",
        ])
    
    if "training_stats" in analysis:
        stats = analysis["training_stats"]
        primary_metric_name = analysis.get("convergence", {}).get("primary_metric_name", "accuracy")
        report_lines.extend([
            "TRAINING STATISTICS",
            "-" * 40,
            f"  Total Rounds: {stats.get('total_rounds', 0)}",
            f"  Mean {primary_metric_name.title()}: {stats.get(f'mean_{primary_metric_name}', 0):.4f}",
            f"  Std {primary_metric_name.title()}: {stats.get(f'std_{primary_metric_name}', 0):.4f}",
            f"  Mean Loss: {stats.get('mean_loss', 0):.4f}",
            f"  Std Loss: {stats.get('std_loss', 0):.4f}",
            "",
        ])
    
    if "methods" in analysis:
        report_lines.extend([
            "METHOD COMPARISON",
            "-" * 40,
        ])
        
        for method in analysis["methods"]:
            metrics = analysis["metrics"].get(method, {})
            report_lines.extend([
                f"  {method}:",
                f"    Metric: {metrics.get('primary_metric_name', 'accuracy')}",
                f"    Best Value: {metrics.get('best_primary_metric', 0):.4f}",
                f"    Final Value: {metrics.get('final_primary_metric', 0):.4f}",
            ])
        
        report_lines.append("")
    
    report_lines.extend([
        "=" * 60,
        f"Report generated at: {output_dir}",
        "=" * 60,
    ])
    
    report = "\n".join(report_lines)
    
    with open(output_dir / "analysis_report.txt", "w") as f:
        f.write(report)
    
    return report


def main():
    """Main entry point."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.compare:
        results_list = []
        for path in args.compare:
            results = load_results(path)
            results_list.append(results)
        
        names = [Path(p).stem for p in args.compare]
        
        comparison = compare_experiments(
            results_list,
            names,
            output_dir,
            args.plot_format,
            args.plot_dpi,
            args.smooth_window,
            args.style,
        )
        
        with open(output_dir / "comparison_results.json", "w") as f:
            json.dump(comparison, f, indent=2, default=str)
        
        print(f"Comparison saved to {output_dir}")
    
    elif args.results_file or args.results_dir:
        path = args.results_file or args.results_dir
        results = load_results(path)
        
        analysis = analyze_single_experiment(
            results,
            output_dir,
            generate_plots=args.plot,
            plot_format=args.plot_format,
            plot_dpi=args.plot_dpi,
            smooth_window=args.smooth_window,
            style=args.style,
        )
        
        report = generate_report(analysis, output_dir)
        
        with open(output_dir / "analysis_results.json", "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(report)
        print(f"\nAnalysis saved to {output_dir}")
    
    else:
        print("Please specify --results_dir, --results_file, or --compare")
        sys.exit(1)


if __name__ == "__main__":
    main()
