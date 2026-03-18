#!/usr/bin/env python3
"""
Test suite for DAFA federated learning framework.

Run tests with:
    pytest tests/ -v
    pytest tests/test_models.py -v
    pytest tests/ --cov=src
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModels:
    """Test model architectures."""
    
    def test_resnet18_creation(self):
        """Test ResNet-18 model creation."""
        from src.models.resnet import ResNet18, resnet18
        
        model = ResNet18(num_classes=10)
        assert model is not None
        
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        assert output.shape == (2, 10)
    
    def test_resnet18_features(self):
        """Test ResNet-18 feature extraction."""
        from src.models.resnet import ResNet18
        
        model = ResNet18(num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        
        features = model.get_features(x)
        assert features.shape == (2, 512)
    
    def test_simple_cnn(self):
        """Test SimpleCNN model."""
        from src.models.cnn import SimpleCNN
        
        model = SimpleCNN(num_classes=10, in_channels=1)
        x = torch.randn(2, 1, 28, 28)
        output = model(x)
        assert output.shape == (2, 10)
    
    def test_two_layer_cnn(self):
        """Test TwoLayerCNN model for FEMNIST."""
        from src.models.cnn import TwoLayerCNN
        
        model = TwoLayerCNN(num_classes=62, in_channels=1)
        x = torch.randn(2, 1, 28, 28)
        output = model(x)
        assert output.shape == (2, 62)
    
    def test_lstm_model(self):
        """Test LSTM model for Shakespeare."""
        from src.models.lstm import LSTMModel
        
        vocab_size = 80
        model = LSTMModel(vocab_size=vocab_size)
        
        x = torch.randint(0, vocab_size, (2, 80))
        output, hidden = model(x)
        assert output.shape == (2, vocab_size)


class TestAggregators:
    """Test aggregation methods."""
    
    def test_fedavg_aggregation(self):
        """Test FedAvg aggregation."""
        from src.methods.fedavg import FedAvgAggregator, FedAvgConfig
        from src.methods.base import ClientUpdate
        
        config = FedAvgConfig(device="cpu")
        aggregator = FedAvgAggregator(config)
        
        model = nn.Linear(10, 5)
        params = torch.cat([p.data.view(-1) for p in model.parameters()])
        
        updates = [
            ClientUpdate(client_id=0, update=torch.randn_like(params) * 0.1, num_samples=100),
            ClientUpdate(client_id=1, update=torch.randn_like(params) * 0.1, num_samples=200),
        ]
        
        updated_model = aggregator.aggregate(model, updates)
        assert updated_model is not None
        assert aggregator.round_num == 1
    
    def test_fedprox_aggregation(self):
        """Test FedProx aggregation."""
        from src.methods.fedprox import FedProxAggregator, FedProxConfig
        from src.methods.base import ClientUpdate
        
        config = FedProxConfig(proximal_mu=0.01, device="cpu")
        aggregator = FedProxAggregator(config)
        
        model = nn.Linear(10, 5)
        params = torch.cat([p.data.view(-1) for p in model.parameters()])
        
        updates = [
            ClientUpdate(client_id=0, update=torch.randn_like(params) * 0.1, num_samples=100),
        ]
        
        updated_model = aggregator.aggregate(model, updates)
        assert updated_model is not None
        assert aggregator.proximal_mu == 0.01
    
    def test_scaffold_aggregation(self):
        """Test SCAFFOLD aggregation."""
        from src.methods.scaffold import SCAFFOLDAggregator, SCAFFOLDConfig
        from src.methods.base import ClientUpdate
        
        config = SCAFFOLDConfig(device="cpu")
        aggregator = SCAFFOLDAggregator(config)
        
        model = nn.Linear(10, 5)
        params = torch.cat([p.data.view(-1) for p in model.parameters()])
        
        updates = [
            ClientUpdate(
                client_id=0,
                update=torch.randn_like(params) * 0.1,
                num_samples=100,
                extra_info={"new_control": torch.zeros_like(params)}
            ),
        ]
        
        updated_model = aggregator.aggregate(model, updates)
        assert updated_model is not None
        assert aggregator.global_control is not None
    
    def test_fednova_aggregation(self):
        """Test FedNova aggregation."""
        from src.methods.fednova import FedNovaAggregator, FedNovaConfig
        from src.methods.base import ClientUpdate
        
        config = FedNovaConfig(device="cpu")
        aggregator = FedNovaAggregator(config)
        
        model = nn.Linear(10, 5)
        params = torch.cat([p.data.view(-1) for p in model.parameters()])
        
        updates = [
            ClientUpdate(client_id=0, update=torch.randn_like(params) * 0.1, num_samples=100, num_steps=5),
            ClientUpdate(client_id=1, update=torch.randn_like(params) * 0.1, num_samples=200, num_steps=10),
        ]
        
        updated_model = aggregator.aggregate(model, updates)
        assert updated_model is not None
    
    def test_fedavgm_aggregation(self):
        """Test FedAvgM aggregation."""
        from src.methods.fedavgm import FedAvgMAggregator, FedAvgMConfig
        from src.methods.base import ClientUpdate
        
        config = FedAvgMConfig(server_momentum=0.9, device="cpu")
        aggregator = FedAvgMAggregator(config)
        
        model = nn.Linear(10, 5)
        params = torch.cat([p.data.view(-1) for p in model.parameters()])
        
        updates = [
            ClientUpdate(client_id=0, update=torch.randn_like(params) * 0.1, num_samples=100),
        ]
        
        updated_model = aggregator.aggregate(model, updates)
        assert updated_model is not None
        assert aggregator.velocity is not None
    
    def test_fedadam_aggregation(self):
        """Test FedAdam aggregation."""
        from src.methods.fedadam import FedAdamAggregator, FedAdamConfig
        from src.methods.base import ClientUpdate
        
        config = FedAdamConfig(server_lr=0.001, device="cpu")
        aggregator = FedAdamAggregator(config)
        
        model = nn.Linear(10, 5)
        params = torch.cat([p.data.view(-1) for p in model.parameters()])
        
        updates = [
            ClientUpdate(client_id=0, update=torch.randn_like(params) * 0.1, num_samples=100),
        ]
        
        updated_model = aggregator.aggregate(model, updates)
        assert updated_model is not None
        assert aggregator.m is not None
        assert aggregator.v is not None
    
    def test_dafa_aggregation(self):
        """Test DAFA aggregation."""
        from src.methods.dafa import DAFAAggregator, DAFAConfig
        from src.methods.base import ClientUpdate
        
        config = DAFAConfig(mu=0.5, device="cpu")
        aggregator = DAFAAggregator(config)
        
        model = nn.Linear(10, 5)
        params = torch.cat([p.data.view(-1) for p in model.parameters()])
        
        base_update = torch.randn_like(params) * 0.1
        updates = [
            ClientUpdate(client_id=0, update=base_update + torch.randn_like(params) * 0.01, num_samples=100),
            ClientUpdate(client_id=1, update=base_update + torch.randn_like(params) * 0.01, num_samples=100),
            ClientUpdate(client_id=2, update=base_update + torch.randn_like(params) * 0.01, num_samples=100),
        ]
        
        updated_model = aggregator.aggregate(model, updates)
        assert updated_model is not None
        
        alignment_scores = aggregator.get_alignment_scores()
        assert len(alignment_scores) >= 1


class TestDSNRAnalyzer:
    """Test DSNR analysis module."""
    
    def test_dsnr_computation(self):
        """Test DSNR computation."""
        from src.analysis.analyzer import DSNRAnalyzer
        
        analyzer = DSNRAnalyzer()
        
        base = torch.randn(100)
        updates = [base + torch.randn(100) * 0.1 for _ in range(10)]
        
        dsnr = analyzer.compute_dsnr(updates)
        assert isinstance(dsnr, float)
        assert dsnr > 0
    
    def test_alignment_scores(self):
        """Test alignment score computation."""
        from src.analysis.analyzer import DSNRAnalyzer
        
        analyzer = DSNRAnalyzer()
        
        global_dir = torch.randn(100)
        global_dir = global_dir / global_dir.norm()
        
        aligned_update = global_dir * 0.5
        orthogonal_update = torch.randn(100)
        orthogonal_update = orthogonal_update - torch.dot(orthogonal_update, global_dir) * global_dir
        
        updates = [aligned_update, orthogonal_update]
        scores = analyzer.compute_alignment_scores(updates, global_dir)
        
        assert len(scores) == 2
        assert scores[0] > scores[1]
    
    def test_round_analysis(self):
        """Test round analysis."""
        from src.analysis.analyzer import DSNRAnalyzer
        
        analyzer = DSNRAnalyzer()
        
        base = torch.randn(100)
        updates = [base + torch.randn(100) * 0.1 for _ in range(5)]
        
        results = analyzer.analyze_round(updates)
        
        assert "dsnr" in results
        assert "alignment_scores" in results
        assert "update_variance" in results
        assert len(results["alignment_scores"]) == 5


class TestCheckpoint:
    """Test checkpoint functionality."""
    
    def test_checkpoint_save_load(self, tmp_path):
        """Test checkpoint save and load."""
        from src.utils.checkpoint import CheckpointManager
        
        manager = CheckpointManager(save_dir=str(tmp_path), max_checkpoints=3)
        
        state = {
            "round": 10,
            "model_state": {"weight": torch.randn(10, 5)},
            "best_accuracy": 0.85,
        }
        
        manager.save(state, round_num=10)
        
        checkpoints = manager.checkpoints
        assert len(checkpoints) == 1
        
        loaded = manager.load()
        assert loaded is not None
        assert loaded.get("round") == 10
    
    def test_max_checkpoints(self, tmp_path):
        """Test maximum checkpoint limit."""
        from src.utils.checkpoint import CheckpointManager
        
        manager = CheckpointManager(save_dir=str(tmp_path), max_checkpoints=2)
        
        for i in range(5):
            state = {"round": i, "data": f"round_{i}"}
            manager.save(state, round_num=i)
        
        checkpoints = manager.checkpoints
        assert len(checkpoints) <= 2


class TestSeed:
    """Test random seed management."""
    
    def test_seed_reproducibility(self):
        """Test that seed produces reproducible results."""
        from src.utils.seed import set_seed
        
        set_seed(42)
        a = torch.randn(10)
        
        set_seed(42)
        b = torch.randn(10)
        
        assert torch.allclose(a, b)
    
    def test_different_seeds(self):
        """Test that different seeds produce different results."""
        from src.utils.seed import set_seed
        
        set_seed(42)
        a = torch.randn(10)
        
        set_seed(123)
        b = torch.randn(10)
        
        assert not torch.allclose(a, b)


class TestMetrics:
    """Test metrics computation."""
    
    def test_accuracy_computation(self):
        """Test accuracy computation."""
        from src.utils.metrics import compute_accuracy
        
        predictions = torch.tensor([0, 1, 2, 0, 1])
        targets = torch.tensor([0, 1, 1, 0, 2])
        
        accuracy = compute_accuracy(predictions, targets)
        assert isinstance(accuracy, list)
        assert accuracy[0] == 0.6
    
    def test_perplexity_computation(self):
        """Test perplexity computation."""
        from src.utils.metrics import compute_perplexity
        import torch.nn.functional as F
        
        logits = torch.randn(10, 80)
        targets = torch.randint(0, 80, (10,))
        
        loss = F.cross_entropy(logits, targets).item()
        perplexity = compute_perplexity(loss)
        assert perplexity > 0


class TestIntegration:
    """Integration tests."""
    
    @pytest.mark.slow
    def test_mini_training_run(self, tmp_path):
        """Test a minimal training run."""
        from src.models.cnn import SimpleCNN
        from src.methods.fedavg import FedAvgAggregator, FedAvgConfig
        from src.core.trainer import FederatedTrainer, TrainerConfig
        from src.methods.base import ClientUpdate
        
        model = SimpleCNN(num_classes=10, in_channels=1)
        
        config = FedAvgConfig(device="cpu")
        aggregator = FedAvgAggregator(config)
        
        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 100
            def __getitem__(self, idx):
                return torch.randn(1, 28, 28), torch.randint(0, 10, ()).item()
        
        client_loaders = {
            i: torch.utils.data.DataLoader(DummyDataset(), batch_size=10)
            for i in range(5)
        }
        test_loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=10)
        
        trainer_config = TrainerConfig(
            num_rounds=2,
            num_clients_per_round=2,
            local_epochs=1,
            local_lr=0.01,
            device="cpu",
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )
        
        trainer = FederatedTrainer(
            model=model,
            aggregator=aggregator,
            client_loaders=client_loaders,
            test_loader=test_loader,
            config=trainer_config,
        )
        
        results = trainer.train()
        
        assert "best_accuracy" in results
        assert results["final_round"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
