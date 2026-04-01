#!/usr/bin/env python3

import argparse
import itertools
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DAFA five-stage experiments")
    parser.add_argument("--stages", type=str, default="all", help="all or comma-separated from 1,2,3,4,5")
    parser.add_argument("--output_dir", type=str, default="results/five_stages")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_rounds", type=int, default=100)
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def parse_seeds(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def run_cmd(cmd: List[str], dry_run: bool = False) -> int:
    print("RUN:", " ".join(cmd))
    if dry_run:
        return 0
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode


def run_cmd_detailed(cmd: List[str], dry_run: bool = False) -> Dict[str, Any]:
    print("RUN:", " ".join(cmd))
    if dry_run:
        return {"code": 0, "stdout": "", "stderr": "", "dry_run": True}
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return {
        "code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "dry_run": False,
    }


def find_results_json(base_dir: Path) -> Path:
    candidates = sorted(base_dir.glob("**/results.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No results.json found under {base_dir}")
    return candidates[0]


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean_std(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {"mean": float(statistics.mean(values)), "std": float(statistics.stdev(values))}


def get_primary_metric_name(result: Dict[str, Any], dataset: str) -> str:
    """Return the primary metric name for a dataset/result pair."""
    return str(result.get("primary_metric_name") or ("perplexity" if dataset == "shakespeare" else "accuracy"))


def get_primary_metric_value(result: Dict[str, Any], dataset: str) -> float:
    """Extract the best primary metric from a result file."""
    metric_name = get_primary_metric_name(result, dataset)
    if metric_name == "perplexity":
        value = result.get("best_perplexity")
        return float(value) if value is not None else float("inf")
    return float(result.get("best_accuracy", 0.0))


def run_experiment(
    output_dir: Path,
    method: str,
    dataset: str,
    seed: int,
    num_rounds: int,
    device: str,
    alpha: float = 0.5,
    extra_args: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
) -> Tuple[int, Optional[Path]]:
    extra_args = extra_args or {}
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_experiment.py"),
        "--method",
        method,
        "--dataset",
        dataset,
        "--alpha",
        str(alpha),
        "--seed",
        str(seed),
        "--num_rounds",
        str(num_rounds),
        "--device",
        device,
        "--output_dir",
        str(output_dir),
        "--track_dsnr",
        "true",
        "--track_variance",
        "true",
        "--track_convergence",
        "true",
        "--track_fairness",
        "true",
    ]
    for k, v in extra_args.items():
        cmd.extend([f"--{k}", str(v)])
    cmd_result = run_cmd_detailed(cmd, dry_run=dry_run)
    code = int(cmd_result["code"])
    if code != 0 or dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        failure_path = output_dir / "failure.json"
        with open(failure_path, "w", encoding="utf-8") as f:
            json.dump({
                "code": code,
                "command": cmd,
                "stderr": cmd_result.get("stderr", ""),
                "stdout": cmd_result.get("stdout", ""),
            }, f, indent=2)
        return code, None
    return code, find_results_json(output_dir)


def run_stage1(base_dir: Path, seeds: List[int], num_rounds: int, device: str, dry_run: bool) -> Dict[str, Any]:
    stage_dir = base_dir / "stage1_tuning"
    stage_dir.mkdir(parents=True, exist_ok=True)
    methods_grid = {
        "fedavg": [{}],
        "fedprox": [{"mu": x} for x in [0.001, 0.01, 0.1]],
        "scaffold": [{"server_lr": x} for x in [0.5, 1.0, 2.0]],
        "fednova": [{}],
        "fedavgm": [{"server_momentum": x} for x in [0.9, 0.99]],
        "fedadam": [{"server_lr": x} for x in [0.01, 0.1]],
        "dir_weight": [{"gamma": x, "use_pi_weighting": True} for x in [1.0, 2.0]],
        "dafa": [{"gamma": g, "beta": b, "mu": 0.01, "use_pi_weighting": True} for g, b in itertools.product([1.0, 2.0], [0.9, 0.95])],
    }
    records: List[Dict[str, Any]] = []
    best_config: Dict[str, Dict[str, Any]] = {}
    for method, configs in methods_grid.items():
        best_mean = -1.0
        best_item: Dict[str, Any] = {}
        for idx, cfg in enumerate(configs):
            scores: List[float] = []
            for seed in seeds:
                exp_dir = stage_dir / f"{method}_cfg{idx}_seed{seed}"
                code, result_path = run_experiment(
                    output_dir=exp_dir,
                    method=method,
                    dataset="cifar10",
                    seed=seed,
                    num_rounds=num_rounds,
                    device=device,
                    alpha=0.1,
                    extra_args=cfg,
                    dry_run=dry_run,
                )
                if code == 0 and result_path:
                    results = load_json(result_path)
                    scores.append(float(results.get("best_accuracy", 0.0)))
                records.append({"method": method, "config_index": idx, "seed": seed, "status": code, "config": cfg})
            ms = mean_std(scores)
            if ms["mean"] > best_mean:
                best_mean = ms["mean"]
                best_item = dict(cfg)
        best_config[method] = best_item
    summary = {"records": records, "best_config": best_config}
    with open(stage_dir / "stage1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(stage_dir / "best_hyperparams_generated.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(best_config, f, sort_keys=False, allow_unicode=True)
    return summary


def _latest_bottom10(history: List[Dict[str, Any]]) -> float:
    for item in reversed(history):
        if "bottom_10_accuracy" in item:
            return float(item["bottom_10_accuracy"])
    return 0.0


def run_stage2(base_dir: Path, seeds: List[int], num_rounds: int, device: str, dry_run: bool, stage1_best: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    stage_dir = base_dir / "stage2_comparison"
    stage_dir.mkdir(parents=True, exist_ok=True)
    scenarios = [("cifar10", 0.5), ("cifar10", 0.1), ("femnist", 0.5), ("shakespeare", 0.5)]
    methods = ["fedavg", "fedprox", "scaffold", "fednova", "fedavgm", "fedadam", "dir_weight", "dafa"]
    best_cfg = (stage1_best or {}).get("best_config", {})
    records: List[Dict[str, Any]] = []
    for dataset, alpha in scenarios:
        for method in methods:
            cfg = dict(best_cfg.get(method, {}))
            scores: List[float] = []
            bottoms: List[float] = []
            metric_name = "accuracy" if dataset != "shakespeare" else "perplexity"
            for seed in seeds:
                exp_dir = stage_dir / f"{dataset}_a{alpha}_{method}_seed{seed}"
                code, result_path = run_experiment(
                    output_dir=exp_dir,
                    method=method,
                    dataset=dataset,
                    seed=seed,
                    num_rounds=num_rounds,
                    device=device,
                    alpha=alpha,
                    extra_args=cfg,
                    dry_run=dry_run,
                )
                if code == 0 and result_path:
                    data = load_json(result_path)
                    metric_name = get_primary_metric_name(data, dataset)
                    scores.append(get_primary_metric_value(data, dataset))
                    bottoms.append(_latest_bottom10(data.get("history", [])))
            record = {
                "dataset": dataset,
                "alpha": alpha,
                "method": method,
                "primary_metric_name": metric_name,
                "primary_metric": mean_std(scores),
                "bottom_10_accuracy": mean_std(bottoms),
                "runs": len(scores),
            }
            record[metric_name] = record["primary_metric"]
            records.append(record)
    out = {"records": records}
    for name in ["stage2_summary.json", "stage2_table1_summary.json"]:
        with open(stage_dir / name, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    return out


def run_stage3(base_dir: Path, seeds: List[int], num_rounds: int, device: str, dry_run: bool) -> Dict[str, Any]:
    stage_dir = base_dir / "stage3_mechanism"
    stage_dir.mkdir(parents=True, exist_ok=True)
    methods = ["fedavg", "fedavgm", "scaffold", "dafa"]
    result_dirs: List[Path] = []
    dsnr_corr: List[float] = []
    for method in methods:
        for seed in seeds:
            exp_dir = stage_dir / f"{method}_seed{seed}"
            code, result_path = run_experiment(
                output_dir=exp_dir,
                method=method,
                dataset="cifar10",
                seed=seed,
                num_rounds=num_rounds,
                device=device,
                alpha=0.1,
                extra_args={},
                dry_run=dry_run,
            )
            if code == 0 and result_path:
                result_dirs.append(result_path.parent)
                if method == "dafa":
                    data = load_json(result_path)
                    hist = data.get("history", [])
                    a = [float(h.get("dsnr", 0.0)) for h in hist if "dsnr" in h and "decentralized_dsnr" in h]
                    b = [float(h.get("decentralized_dsnr", 0.0)) for h in hist if "dsnr" in h and "decentralized_dsnr" in h]
                    if len(a) >= 3 and len(a) == len(b):
                        ma = statistics.mean(a)
                        mb = statistics.mean(b)
                        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
                        den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
                        if den > 1e-12:
                            dsnr_corr.append(num / den)
    if result_dirs:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "analyze_results.py"),
            "compare",
            "--inputs",
            *[str(x) for x in result_dirs[: min(8, len(result_dirs))]],
            "--output_dir",
            str(stage_dir / "analysis"),
            "--format",
            "png",
        ]
        run_cmd(cmd, dry_run=dry_run)
    out = {"dafa_dsnr_corr_mean": mean_std(dsnr_corr)}
    with open(stage_dir / "stage3_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def run_stage4(base_dir: Path, seeds: List[int], num_rounds: int, device: str, dry_run: bool) -> Dict[str, Any]:
    stage_dir = base_dir / "stage4_ablation"
    stage_dir.mkdir(parents=True, exist_ok=True)
    tasks: List[Tuple[str, str, float, Dict[str, Any]]] = []
    tasks.extend([("4A_femnist_da", "dafa", 0.5, {"dataset": "femnist", "gamma": 1.0, "beta": 0.9, "use_pi_weighting": use_pi}) for use_pi in [False, True]])
    tasks.extend([("4B_mu", "dafa", 0.1, {"dataset": "cifar10", "gamma": 1.0, "beta": 0.9, "mu": mu}) for mu in [0.0, 0.01, 0.05, 0.1]])
    tasks.extend([("4C_gamma_cifar", "dafa", 0.1, {"dataset": "cifar10", "beta": 0.9, "gamma": gamma}) for gamma in [0.1, 0.5, 1.0, 2.0, 5.0]])
    tasks.extend([("4C_gamma_fem", "dafa", 0.5, {"dataset": "femnist", "beta": 0.9, "gamma": gamma}) for gamma in [0.1, 0.5, 1.0, 2.0, 5.0]])
    tasks.extend([("4D_beta_cifar", "dafa", 0.1, {"dataset": "cifar10", "gamma": 1.0, "beta": beta}) for beta in [0.0, 0.5, 0.7, 0.9, 0.99]])
    tasks.extend([("4D_beta_fem", "dafa", 0.5, {"dataset": "femnist", "gamma": 1.0, "beta": beta}) for beta in [0.0, 0.5, 0.7, 0.9, 0.99]])
    records: List[Dict[str, Any]] = []
    for tag, method, alpha, cfg in tasks:
        dataset = cfg.pop("dataset")
        for seed in seeds:
            exp_dir = stage_dir / f"{tag}_{method}_{dataset}_seed{seed}"
            code, _ = run_experiment(
                output_dir=exp_dir,
                method=method,
                dataset=dataset,
                seed=seed,
                num_rounds=num_rounds,
                device=device,
                alpha=alpha,
                extra_args=cfg,
                dry_run=dry_run,
            )
            records.append({"tag": tag, "method": method, "dataset": dataset, "seed": seed, "status": code, "config": cfg})
    out = {"records": records}
    with open(stage_dir / "stage4_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def run_stage5(base_dir: Path, seeds: List[int], num_rounds: int, device: str, dry_run: bool) -> Dict[str, Any]:
    stage_dir = base_dir / "stage5_extension"
    stage_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []

    for seed in seeds:
        exp_dir = stage_dir / f"5A_dafa_seed{seed}"
        code, result_path = run_experiment(
            output_dir=exp_dir,
            method="dafa",
            dataset="cifar10",
            seed=seed,
            num_rounds=max(num_rounds, 200),
            device=device,
            alpha=0.1,
            extra_args={},
            dry_run=dry_run,
        )
        corr = None
        if code == 0 and result_path:
            data = load_json(result_path)
            hist = data.get("history", [])
            x = [float(h.get("alignment_mean", 0.0)) for h in hist if "alignment_mean" in h and "update_variance" in h]
            y = [float(h.get("update_variance", 0.0)) for h in hist if "alignment_mean" in h and "update_variance" in h]
            if len(x) >= 3:
                mx = statistics.mean(x)
                my = statistics.mean(y)
                num = sum((a - mx) * (b - my) for a, b in zip(x, y))
                den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
                corr = num / den if den > 1e-12 else None
        records.append({"experiment": "5A", "seed": seed, "status": code, "correlation": corr})

    for attack in ["random", "reverse"]:
        for method in ["fedavg", "fedavgm", "dafa"]:
            for seed in seeds:
                exp_dir = stage_dir / f"5B_{attack}_{method}_seed{seed}"
                code, _ = run_experiment(
                    output_dir=exp_dir,
                    method=method,
                    dataset="cifar10",
                    seed=seed,
                    num_rounds=num_rounds,
                    device=device,
                    alpha=0.1,
                    extra_args={"malicious_client_fraction": 0.1, "attack_type": attack},
                    dry_run=dry_run,
                )
                records.append({"experiment": "5B", "attack": attack, "method": method, "seed": seed, "status": code})

    for num_clients in [100, 500, 1000]:
        for method in ["fedavg", "dafa"]:
            for seed in seeds:
                exp_dir = stage_dir / f"5C_n{num_clients}_{method}_seed{seed}"
                code, _ = run_experiment(
                    output_dir=exp_dir,
                    method=method,
                    dataset="cifar10",
                    seed=seed,
                    num_rounds=num_rounds,
                    device=device,
                    alpha=0.1,
                    extra_args={"num_clients": num_clients, "clients_per_round": max(1, int(num_clients * 0.1))},
                    dry_run=dry_run,
                )
                records.append({"experiment": "5C", "num_clients": num_clients, "method": method, "seed": seed, "status": code})

    for alpha in [0.05, 0.1, 0.3, 0.5, 1.0]:
        for method in ["fedavg", "fedavgm", "dafa"]:
            for seed in seeds:
                exp_dir = stage_dir / f"5D_a{alpha}_{method}_seed{seed}"
                code, _ = run_experiment(
                    output_dir=exp_dir,
                    method=method,
                    dataset="cifar10",
                    seed=seed,
                    num_rounds=num_rounds,
                    device=device,
                    alpha=alpha,
                    extra_args={},
                    dry_run=dry_run,
                )
                records.append({"experiment": "5D", "alpha": alpha, "method": method, "seed": seed, "status": code})

    cmd_5e = [
        sys.executable,
        str(ROOT / "scripts" / "run_experiment_5e.py"),
        "--setting",
        "all",
        "--output_dir",
        str(stage_dir / "5E"),
    ]
    code_5e = run_cmd(cmd_5e, dry_run=dry_run)
    records.append({"experiment": "5E", "status": code_5e})

    out = {"records": records}
    with open(stage_dir / "stage5_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    base_dir = (ROOT / args.output_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    selected = {"1", "2", "3", "4", "5"} if args.stages == "all" else {x.strip() for x in args.stages.split(",")}
    summary: Dict[str, Any] = {}
    stage_status: Dict[str, Any] = {}
    stage1_best = None
    if "1" in selected:
        started_at = datetime.now().isoformat()
        try:
            summary["stage1"] = run_stage1(base_dir, seeds, args.num_rounds, args.device, args.dry_run)
            stage1_best = summary["stage1"]
            stage_status["stage1"] = {"started_at": started_at, "finished_at": datetime.now().isoformat(), "status": "completed"}
        except Exception as exc:
            stage_status["stage1"] = {"started_at": started_at, "finished_at": datetime.now().isoformat(), "status": "failed", "error": str(exc)}
    if "2" in selected:
        started_at = datetime.now().isoformat()
        try:
            summary["stage2"] = run_stage2(base_dir, seeds, max(args.num_rounds, 200), args.device, args.dry_run, stage1_best)
            stage_status["stage2"] = {"started_at": started_at, "finished_at": datetime.now().isoformat(), "status": "completed"}
        except Exception as exc:
            stage_status["stage2"] = {"started_at": started_at, "finished_at": datetime.now().isoformat(), "status": "failed", "error": str(exc)}
    if "3" in selected:
        started_at = datetime.now().isoformat()
        try:
            summary["stage3"] = run_stage3(base_dir, seeds, max(args.num_rounds, 200), args.device, args.dry_run)
            stage_status["stage3"] = {"started_at": started_at, "finished_at": datetime.now().isoformat(), "status": "completed"}
        except Exception as exc:
            stage_status["stage3"] = {"started_at": started_at, "finished_at": datetime.now().isoformat(), "status": "failed", "error": str(exc)}
    if "4" in selected:
        started_at = datetime.now().isoformat()
        try:
            summary["stage4"] = run_stage4(base_dir, seeds, max(args.num_rounds, 200), args.device, args.dry_run)
            stage_status["stage4"] = {"started_at": started_at, "finished_at": datetime.now().isoformat(), "status": "completed"}
        except Exception as exc:
            stage_status["stage4"] = {"started_at": started_at, "finished_at": datetime.now().isoformat(), "status": "failed", "error": str(exc)}
    if "5" in selected:
        started_at = datetime.now().isoformat()
        try:
            summary["stage5"] = run_stage5(base_dir, seeds, max(args.num_rounds, 200), args.device, args.dry_run)
            stage_status["stage5"] = {"started_at": started_at, "finished_at": datetime.now().isoformat(), "status": "completed"}
        except Exception as exc:
            stage_status["stage5"] = {"started_at": started_at, "finished_at": datetime.now().isoformat(), "status": "failed", "error": str(exc)}
    with open(base_dir / "five_stages_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(base_dir / "pipeline_status.json", "w", encoding="utf-8") as f:
        json.dump({
            "selected_stages": sorted(list(selected)),
            "device": args.device,
            "dry_run": args.dry_run,
            "stage_status": stage_status,
        }, f, indent=2)
    print("Completed. Summary:", base_dir / "five_stages_summary.json")


if __name__ == "__main__":
    main()
