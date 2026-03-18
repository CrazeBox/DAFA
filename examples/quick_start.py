"""
DAFA Quick Example

This script demonstrates the basic usage of DAFA for federated learning.
"""

import sys
sys.path.insert(0, '.')

import torch

from src.data import get_cifar10_loaders
from src.models.cnn import SimpleCNN
from src.methods.dafa import DAFAAggregator, DAFAConfig
from src.core import FederatedTrainer, TrainerConfig


def main():
    print("=" * 50)
    print("DAFA Quick Example")
    print("=" * 50)
    
    # 1. Load dataset
    print("\n[1/4] Loading CIFAR-10 dataset...")
    client_loaders, val_loader, test_loader, _ = get_cifar10_loaders(
        num_clients=10,
        alpha=0.5,
        batch_size=64,
    )
    print(f"      Loaded {len(client_loaders)} clients")
    
    # 2. Create model
    print("\n[2/4] Creating CNN model...")
    model = SimpleCNN(num_classes=10)
    print(f"      Model: {model.__class__.__name__}")
    
    # 3. Create aggregator
    print("\n[3/4] Creating DAFA aggregator...")
    aggregator = DAFAAggregator(DAFAConfig(gamma=1.0, beta=0.9, mu=0.01))
    print(f"      Method: DAFA")
    
    # 4. Train
    print("\n[4/4] Training...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = TrainerConfig(
        num_rounds=10,
        num_clients_per_round=5,
        local_epochs=5,
        local_lr=0.01,
        device=device,
        use_amp=device == "cuda",
        eval_every=2,
    )
    
    trainer = FederatedTrainer(
        model=model,
        aggregator=aggregator,
        client_loaders=client_loaders,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
    )
    
    results = trainer.train()
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Best accuracy: {results['best_accuracy']:.4f}")
    print(f"Total rounds: {results['final_round']}")
    print(f"Total time: {results['total_time']:.2f}s")


if __name__ == "__main__":
    main()
