#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract best DAFA experiment runs")
    parser.add_argument("--results_root", type=str, default="results", help="Root directory containing results.json files")
    parser.add_argument("--output_dir", type=str, default="results/summary", help="Output directory for summaries")
    return parser.parse_args()


def load_result(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    config_path = path.parent / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    return {
        "path": str(path),
        "best_accuracy": float(data.get("best_accuracy", 0.0)),
        "final_round": int(data.get("final_round", 0)),
        "total_time": float(data.get("total_time", 0.0)),
        "method": config.get("method", "unknown"),
        "dataset": config.get("dataset", "unknown"),
        "seed": config.get("seed", "unknown"),
        "alpha": config.get("alpha", "unknown"),
        "run_group": config.get("run_group", "default"),
        "run_name": config.get("run_name", path.parent.name),
    }


def key_fn(item: Dict[str, Any]) -> str:
    return f"{item['run_group']}|{item['dataset']}|{item['method']}|{item['alpha']}"


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(results_root.glob("**/results.json"))
    rows: List[Dict[str, Any]] = [load_result(p) for p in files]

    best: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = key_fn(row)
        if key not in best or row["best_accuracy"] > best[key]["best_accuracy"]:
            best[key] = row

    all_path = output_dir / "all_runs.json"
    best_path = output_dir / "best_runs.json"
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(sorted(best.values(), key=lambda x: (-x["best_accuracy"], x["dataset"], x["method"])), f, indent=2)

    csv_path = output_dir / "best_runs.csv"
    header = ["run_group", "dataset", "method", "alpha", "seed", "best_accuracy", "final_round", "total_time", "run_name", "path"]
    lines = [",".join(header)]
    for row in sorted(best.values(), key=lambda x: (-x["best_accuracy"], x["dataset"], x["method"])):
        values = [str(row.get(h, "")) for h in header]
        lines.append(",".join(values))
    csv_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Scanned runs: {len(rows)}")
    print(f"Best groups: {len(best)}")
    print(f"Saved: {all_path}")
    print(f"Saved: {best_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
