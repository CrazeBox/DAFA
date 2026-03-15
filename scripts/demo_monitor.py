#!/usr/bin/env python3
"""
Demo script for real-time monitoring panel.

This script demonstrates the real-time monitoring panel functionality
without running actual federated learning training.
"""

import time
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.monitor import (
    MonitorPanel,
    DataItem,
    DataStatus,
    ClientStatus,
    create_monitor,
)


def simulate_training():
    monitor = create_monitor(
        total_rounds=100,
        num_clients=10,
        refresh_rate=0.3,
    )
    
    monitor.start()
    
    try:
        accuracy = 0.1
        loss = 2.5
        
        for round_num in range(1, 101):
            for client_id in range(10):
                monitor.update_client(
                    client_id=client_id,
                    state="training",
                    progress=random.uniform(0, 1),
                    loss=random.uniform(0.5, 2.0),
                )
            
            time.sleep(0.1)
            
            for client_id in range(10):
                monitor.update_client(
                    client_id=client_id,
                    state="completed",
                    progress=1.0,
                    loss=random.uniform(0.3, 1.5),
                    samples=random.randint(100, 500),
                )
            
            accuracy += random.uniform(-0.02, 0.05)
            accuracy = max(0.1, min(0.95, accuracy))
            
            loss -= random.uniform(-0.1, 0.15)
            loss = max(0.1, min(3.0, loss))
            
            dsnr = random.uniform(1.0, 5.0)
            variance = random.uniform(0.001, 0.01)
            
            monitor.add_round_data(
                round_num=round_num,
                accuracy=accuracy,
                loss=loss,
                dsnr=dsnr,
                variance=variance,
            )
            
            monitor.update_gpu(
                utilization=random.uniform(60, 95),
                memory=random.uniform(40, 80),
            )
            
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    
    finally:
        monitor.stop()
        
        stats = monitor.get_stats()
        print("\n" + "=" * 60)
        print("TRAINING STATISTICS")
        print("=" * 60)
        print(f"Total rounds: {stats['count']}")
        if stats.get('accuracy'):
            print(f"Accuracy: {stats['accuracy']['min']:.4f} ~ {stats['accuracy']['max']:.4f}")
            print(f"  Mean: {stats['accuracy']['mean']:.4f}")
        if stats.get('loss'):
            print(f"Loss: {stats['loss']['min']:.4f} ~ {stats['loss']['max']:.4f}")
            print(f"  Mean: {stats['loss']['mean']:.4f}")
        print("=" * 60)


def demo_data_export():
    print("\nDemo: Data Export")
    print("-" * 40)
    
    monitor = MonitorPanel(total_rounds=10)
    
    for i in range(10):
        monitor.add_round_data(
            round_num=i + 1,
            accuracy=random.uniform(0.5, 0.9),
            loss=random.uniform(0.3, 1.5),
            dsnr=random.uniform(1.0, 5.0),
        )
    
    json_output = monitor.export_data(format="json")
    print("\nJSON Export (first 500 chars):")
    print(json_output[:500] + "...")
    
    csv_output = monitor.export_data(format="csv")
    print("\nCSV Export (first 5 lines):")
    print('\n'.join(csv_output.split('\n')[:5]))


def demo_filter_sort():
    print("\nDemo: Filter and Sort")
    print("-" * 40)
    
    monitor = MonitorPanel(total_rounds=10)
    
    for i in range(10):
        accuracy = 0.5 + i * 0.05
        monitor.add_round_data(
            round_num=i + 1,
            accuracy=accuracy,
            loss=2.0 - i * 0.15,
            dsnr=1.0 + i * 0.3,
        )
    
    print("\nOriginal data (by round):")
    items = monitor.get_data(apply_filter=False, apply_sort=False)
    for item in items[:5]:
        print(f"  Round {item.round_num}: ACC={item.accuracy:.4f}")
    
    monitor.set_sorter(sort_by="accuracy", ascending=True)
    print("\nSorted by accuracy (ascending):")
    items = monitor.get_data(apply_filter=False, apply_sort=True)
    for item in items[:5]:
        print(f"  Round {item.round_num}: ACC={item.accuracy:.4f}")
    
    monitor.set_filter(min_accuracy=0.65)
    print("\nFiltered (accuracy >= 0.65):")
    items = monitor.get_data(apply_filter=True, apply_sort=False)
    for item in items:
        print(f"  Round {item.round_num}: ACC={item.accuracy:.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time Monitor Demo")
    parser.add_argument("--demo", choices=["training", "export", "filter", "all"],
                       default="training", help="Demo to run")
    
    args = parser.parse_args()
    
    if args.demo == "training":
        simulate_training()
    elif args.demo == "export":
        demo_data_export()
    elif args.demo == "filter":
        demo_filter_sort()
    else:
        simulate_training()
        demo_data_export()
        demo_filter_sort()
