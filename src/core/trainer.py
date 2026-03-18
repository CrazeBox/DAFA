"""Federated learning trainer with comprehensive metrics tracking."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from datetime import datetime
from copy import deepcopy
from contextlib import nullcontext
from tqdm import tqdm

from ..methods.base import ClientUpdate
from ..utils.checkpoint import CheckpointManager
from ..utils.logger import get_logger
from ..utils.printing import print_header, print_line, print_summary
from ..utils.metrics import compute_accuracy, compute_perplexity
from ..analysis.analyzer import ExperimentAnalyzer, DSNRAnalyzer


logger = get_logger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for federated trainer."""
    
    num_rounds: int = 100
    num_clients_per_round: int = 10
    local_epochs: int = 5
    local_lr: float = 0.01
    local_batch_size: int = 64
    
    client_lr_scheduler: str = "constant"
    lr_decay_factor: float = 0.1
    lr_decay_rounds: int = 50
    cosine_decay_T_max: Optional[int] = None
    cosine_decay_eta_min: float = 0.0
    
    server_lr: float = 1.0
    
    eval_every: int = 10
    save_every: int = 10
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = None
    
    log_dir: str = "logs"
    results_dir: str = "results"
    
    track_dsnr: bool = True
    track_variance: bool = True
    track_convergence_speed: bool = True
    convergence_threshold: float = 0.8
    track_fairness: bool = True
    fairness_eval_freq: int = 10
    
    use_monitor: bool = False
    monitor_refresh_rate: float = 0.5
    
    use_amp: bool = True
    num_parallel_clients: int = 4
    malicious_client_fraction: float = 0.0
    attack_type: str = "none"


class LocalTrainer:
    """Local trainer for client-side training."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        config: TrainerConfig,
        client_id: int = 0,
        current_round: int = 0,
    ):
        """
        Initialize local trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            config: Trainer configuration
            client_id: Client identifier
            current_round: Current global round (for learning rate scheduling)
        """
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.client_id = client_id
        self.current_round = current_round
        self.device = config.device
        
        self.model.to(self.device)
        
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.local_lr,
            momentum=0.9,
            weight_decay=1e-4,
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.num_steps = 0
        
        self.scaler = torch.amp.GradScaler('cuda', enabled=config.use_amp)
        
        self.scheduler = None
        if config.client_lr_scheduler == "cosine":
            T_max = config.cosine_decay_T_max or config.num_rounds
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max, eta_min=config.cosine_decay_eta_min
            )
            for _ in range(current_round):
                self.scheduler.step()
    
    def train(
        self,
        global_params: Optional[torch.Tensor] = None,
        proximal_mu: float = 0.0,
        show_progress: bool = False,
        client_control: Optional[torch.Tensor] = None,
        global_control: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Train locally for specified epochs.
        
        Args:
            global_params: Global model parameters for proximal term
            proximal_mu: Proximal term coefficient
            show_progress: Whether to show progress bar
            client_control: Client control variate (for SCAFFOLD)
            global_control: Global control variate (for SCAFFOLD)
        
        Returns:
            Tuple of (update, training_info)
        """
        self.model.train()
        
        initial_params = self._get_params().clone()
        
        total_loss = 0.0
        total_samples = 0
        num_steps = 0
        
        epoch_range = range(self.config.local_epochs)
        if show_progress:
            epoch_range = tqdm(
                epoch_range,
                desc=f"  Client {self.client_id}",
                leave=False,
                dynamic_ncols=True,
            )
        
        for epoch in epoch_range:
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                
                self.optimizer.zero_grad()
                
                with torch.amp.autocast('cuda', enabled=self.config.use_amp):
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    
                    if proximal_mu > 0 and global_params is not None:
                        proximal_term = self._compute_proximal_term(global_params)
                        loss = loss + (proximal_mu / 2) * proximal_term
                
                self.scaler.scale(loss).backward()
                
                if client_control is not None and global_control is not None:
                    self._apply_scaffold_correction(client_control, global_control)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                batch_size = data.size(0)
                epoch_loss += loss.item() * batch_size
                epoch_samples += batch_size
                num_steps += 1
            
            total_loss += epoch_loss
            total_samples += epoch_samples
        
        self.num_steps = num_steps
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        final_params = self._get_params()
        update = final_params - initial_params
        
        new_control = None
        if client_control is not None and global_control is not None:
            new_control = client_control - global_control + update / (num_steps * self.config.local_lr)
        
        info = {
            "loss": total_loss / total_samples if total_samples > 0 else 0.0,
            "num_samples": total_samples,
            "num_steps": num_steps,
            "new_control": new_control,
        }
        
        return update, info
    
    def _apply_scaffold_correction(
        self,
        client_control: torch.Tensor,
        global_control: torch.Tensor,
    ) -> None:
        """Apply SCAFFOLD control variate correction to gradients."""
        control_diff = client_control - global_control
        control_diff = control_diff.to(self.device)
        
        idx = 0
        for param in self.model.parameters():
            if param.grad is not None:
                param_size = param.numel()
                param.grad.data.add_(control_diff[idx:idx + param_size].view_as(param.data))
                idx += param_size
    
    def _get_params(self) -> torch.Tensor:
        """Get flattened model parameters."""
        return torch.cat([p.data.view(-1) for p in self.model.parameters()])
    
    def _compute_proximal_term(self, global_params: torch.Tensor) -> torch.Tensor:
        """Compute proximal regularization term."""
        local_params = self._get_params()
        return torch.norm(local_params - global_params) ** 2


class FederatedTrainer:
    """Main federated learning trainer with comprehensive metrics tracking."""
    
    def __init__(
        self,
        model: nn.Module,
        aggregator: Any,
        client_loaders: Dict[int, DataLoader],
        test_loader: DataLoader,
        config: TrainerConfig,
        validation_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
    ):
        """
        Initialize federated trainer.
        
        Args:
            model: Global model
            aggregator: Aggregation method
            client_loaders: Dictionary of client data loaders
            test_loader: Test data loader (for final evaluation only)
            config: Trainer configuration
            validation_loader: Optional validation loader for drift analysis
            val_loader: Validation loader for hyperparameter selection
        """
        self.model = model
        self.aggregator = aggregator
        self.client_loaders = client_loaders
        self.test_loader = test_loader
        self.val_loader = val_loader or validation_loader
        self.config = config
        
        self.device = config.device
        self.model.to(self.device)
        
        self.checkpoint_manager = CheckpointManager(
            save_dir=Path(config.checkpoint_dir),
            max_checkpoints=5,
        )
        
        self.current_round = 0
        self.best_accuracy = 0.0
        self.best_val_accuracy = 0.0
        self.history: List[Dict[str, Any]] = []
        
        self.start_time = time.time()
        self.client_sample_counts = {
            cid: len(loader.dataset) for cid, loader in client_loaders.items()
        }
        
        self.dsnr_analyzer = DSNRAnalyzer() if config.track_dsnr else None
        
        self.experiment_analyzer = ExperimentAnalyzer(
            output_dir=str(Path(config.results_dir) / "analysis"),
            validation_loader=validation_loader,
            device=config.device,
        )
        
        self.convergence_round: Optional[int] = None
        self.convergence_threshold = config.convergence_threshold
        
        self.monitor = None
        self.monitor_wrapper = None
        if config.use_monitor:
            try:
                from ..monitor import create_monitor, TrainingMonitorWrapper
                self.monitor = create_monitor(
                    total_rounds=config.num_rounds,
                    num_clients=len(client_loaders),
                    refresh_rate=config.monitor_refresh_rate,
                )
                self.monitor_wrapper = TrainingMonitorWrapper(self.monitor)
            except ImportError:
                logger.warning("Monitor module not available, using standard progress bar")
                self.monitor = None
                self.monitor_wrapper = None
    
    def train(self) -> Dict[str, Any]:
        """
        Run federated training.
        
        Returns:
            Training results dictionary
        """
        if self.config.resume_from:
            self._load_checkpoint(self.config.resume_from)
        
        logger.info(f"Starting federated training for {self.config.num_rounds} rounds")
        logger.info(f"Number of clients: {len(self.client_loaders)}")
        logger.info(f"Clients per round: {self.config.num_clients_per_round}")
        
        print_header()
        
        try:
            while self.current_round < self.config.num_rounds:
                self.current_round += 1
                
                round_start = time.time()
                
                selected_clients = self._select_clients()
                
                client_updates = self._train_clients(selected_clients)
                
                self.model = self.aggregator.aggregate(self.model, client_updates)
                
                round_time = time.time() - round_start
                
                train_loss, train_acc = self._evaluate_train_loss(selected_clients)
                
                metrics = self._evaluate(use_val=True)
                metrics["round"] = self.current_round
                metrics["round_time"] = round_time
                metrics["total_time"] = time.time() - self.start_time
                metrics["train_loss"] = train_loss
                metrics["train_accuracy"] = train_acc
                
                if self.config.track_dsnr:
                    dsnr_metrics = self._compute_dsnr_metrics(client_updates)
                    metrics.update(dsnr_metrics)
                
                if self.config.track_convergence_speed and self.convergence_round is None:
                    if metrics["accuracy"] >= self.convergence_threshold:
                        self.convergence_round = self.current_round
                        logger.info(f"Convergence reached at round {self.current_round}")
                
                self.history.append(metrics)
                
                print_line(
                    round_num=self.current_round,
                    total_rounds=self.config.num_rounds,
                    is_train=True,
                    num_clients=len(selected_clients),
                    loss=train_loss,
                    accuracy=train_acc,
                    dsnr=metrics.get("dsnr"),
                    round_time=round_time,
                    total_time=metrics["total_time"],
                    persistent=(self.current_round % self.config.eval_every == 0),
                )
                
                if self.val_loader:
                    print_line(
                        round_num=self.current_round,
                        total_rounds=self.config.num_rounds,
                        is_train=False,
                        num_clients=0,
                        loss=metrics["loss"],
                        accuracy=metrics["accuracy"],
                        dsnr=None,
                        round_time=None,
                        total_time=metrics["total_time"],
                        persistent=(self.current_round % self.config.eval_every == 0),
                    )
                
                if metrics["accuracy"] > self.best_accuracy:
                    self.best_accuracy = metrics["accuracy"]
                    if self.current_round % self.config.eval_every == 0:
                        self._save_checkpoint(is_best=True)
                
                if self.current_round % self.config.save_every == 0:
                    self._save_checkpoint()
        
        except KeyboardInterrupt:
            logger.info("Training interrupted. Saving checkpoint...")
            self._save_checkpoint()
        
        return self._get_results()
    
    def _select_clients(self) -> List[int]:
        """Select clients for current round."""
        all_clients = list(self.client_loaders.keys())
        
        num_select = min(self.config.num_clients_per_round, len(all_clients))
        
        import random
        random.seed(self.config.seed + self.current_round)
        selected = random.sample(all_clients, num_select)
        
        return selected

    def _is_malicious_client(self, client_id: int) -> bool:
        fraction = getattr(self.config, "malicious_client_fraction", 0.0)
        if fraction <= 0:
            return False
        import random
        rng = random.Random(self.config.seed * 1000003 + self.current_round * 1009 + client_id)
        return rng.random() < fraction

    def _apply_attack(self, update: torch.Tensor, is_malicious: bool) -> torch.Tensor:
        if not is_malicious:
            return update
        attack_type = getattr(self.config, "attack_type", "none")
        if attack_type == "reverse":
            return -update
        if attack_type == "random":
            scale = update.norm().item()
            random_update = torch.randn_like(update)
            random_norm = random_update.norm().item() + 1e-8
            return random_update * (scale / random_norm)
        return update
    
    def _train_clients(self, client_ids: List[int]) -> List[ClientUpdate]:
        """Train selected clients and collect updates."""
        client_updates = []
        
        global_params = self._get_model_params()
        
        use_monitor = self.monitor is not None and self.monitor_wrapper is not None
        
        num_parallel = getattr(self.config, 'num_parallel_clients', 1)
        
        if num_parallel > 1 and len(client_ids) > 1:
            client_updates = self._train_clients_parallel(
                client_ids, global_params, use_monitor
            )
        else:
            client_updates = self._train_clients_sequential(
                client_ids, global_params, use_monitor
            )
        
        return client_updates
    
    def _train_clients_sequential(
        self,
        client_ids: List[int],
        global_params: torch.Tensor,
        use_monitor: bool,
    ) -> List[ClientUpdate]:
        """Train clients sequentially."""
        client_updates = []
        
        if not use_monitor:
            client_pbar = tqdm(
                client_ids,
                desc="Training clients",
                leave=False,
                dynamic_ncols=True,
            )
        else:
            client_pbar = client_ids
        
        is_scaffold = hasattr(self.aggregator, 'get_client_control')
        
        for client_id in client_pbar:
            if not use_monitor:
                client_pbar.set_postfix({"client": client_id})
            else:
                self.monitor_wrapper.on_client_start(client_id)
            
            local_model = deepcopy(self.model)
            train_loader = self.client_loaders[client_id]
            
            local_trainer = LocalTrainer(
                model=local_model,
                train_loader=train_loader,
                config=self.config,
                client_id=client_id,
                current_round=self.current_round,
            )
            
            proximal_mu = getattr(self.aggregator, "proximal_mu", 0.0)
            
            client_control = None
            global_control = None
            if is_scaffold:
                client_control = self.aggregator.get_client_control(client_id, self.model)
                global_control = self.aggregator.get_global_control(self.model)
            
            update, info = local_trainer.train(
                global_params=global_params,
                proximal_mu=proximal_mu,
                client_control=client_control,
                global_control=global_control,
            )
            is_malicious = self._is_malicious_client(client_id)
            update = self._apply_attack(update, is_malicious)
            
            if use_monitor:
                self.monitor_wrapper.on_client_complete(
                    client_id=client_id,
                    loss=info["loss"],
                    samples=info["num_samples"],
                )
            
            extra_info = {}
            if is_scaffold and info.get("new_control") is not None:
                extra_info["new_control"] = info["new_control"]
            
            client_update = ClientUpdate(
                client_id=client_id,
                update=update.cpu() if update.is_cuda else update,
                num_samples=info["num_samples"],
                loss=info["loss"],
                num_steps=info["num_steps"],
                extra_info=extra_info,
            )
            
            client_updates.append(client_update)
        
        return client_updates
    
    def _train_clients_parallel(
        self,
        client_ids: List[int],
        global_params: torch.Tensor,
        use_monitor: bool,
    ) -> List[ClientUpdate]:
        """Train clients in parallel using threads with CUDA streams."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        client_updates = []
        lock = threading.Lock()
        
        is_scaffold = hasattr(self.aggregator, 'get_client_control')
        
        use_cuda_stream = self.device.startswith("cuda") and torch.cuda.is_available()

        def train_single_client(client_id: int) -> ClientUpdate:
            if use_monitor:
                self.monitor_wrapper.on_client_start(client_id)
            
            stream_context = torch.cuda.stream(torch.cuda.Stream()) if use_cuda_stream else nullcontext()
            with stream_context:
                local_model = deepcopy(self.model)
                train_loader = self.client_loaders[client_id]
                
                local_trainer = LocalTrainer(
                    model=local_model,
                    train_loader=train_loader,
                    config=self.config,
                    client_id=client_id,
                    current_round=self.current_round,
                )
                
                proximal_mu = getattr(self.aggregator, "proximal_mu", 0.0)
                
                client_control = None
                global_control = None
                if is_scaffold:
                    client_control = self.aggregator.get_client_control(client_id, self.model)
                    global_control = self.aggregator.get_global_control(self.model)
                
                update, info = local_trainer.train(
                    global_params=global_params,
                    proximal_mu=proximal_mu,
                    client_control=client_control,
                    global_control=global_control,
                )
                is_malicious = self._is_malicious_client(client_id)
                update = self._apply_attack(update, is_malicious)
                
                if use_monitor:
                    self.monitor_wrapper.on_client_complete(
                        client_id=client_id,
                        loss=info["loss"],
                        samples=info["num_samples"],
                    )
                
                extra_info = {}
                if is_scaffold and info.get("new_control") is not None:
                    extra_info["new_control"] = info["new_control"]
                
                return ClientUpdate(
                    client_id=client_id,
                    update=update.cpu(),
                    num_samples=info["num_samples"],
                    loss=info["loss"],
                    num_steps=info["num_steps"],
                    extra_info=extra_info,
                )
        
        num_parallel = getattr(self.config, 'num_parallel_clients', 4)
        num_parallel = min(num_parallel, len(client_ids))
        
        if not use_monitor:
            print(f"Training {len(client_ids)} clients with {num_parallel} parallel workers...")
        
        with ThreadPoolExecutor(max_workers=num_parallel) as executor:
            futures = {executor.submit(train_single_client, cid): cid for cid in client_ids}
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    with lock:
                        client_updates.append(result)
                except Exception as e:
                    client_id = futures[future]
                    print(f"Error training client {client_id}: {e}")
        
        client_updates.sort(key=lambda x: client_ids.index(x.client_id))
        
        return client_updates
    
    def _evaluate(self, use_val: bool = True) -> Dict[str, float]:
        """
        Evaluate global model.
        
        Args:
            use_val: If True, use validation set; otherwise use test set
        
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        
        loader = self.val_loader if (use_val and self.val_loader) else self.test_loader
        eval_name = "Validation" if (use_val and self.val_loader) else "Test"
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        criterion = nn.CrossEntropyLoss()
        
        eval_pbar = tqdm(
            loader,
            desc=f"Evaluating ({eval_name})",
            leave=False,
            dynamic_ncols=True,
        )
        
        with torch.no_grad():
            for data, target in eval_pbar:
                data = data.to(self.device)
                target = target.to(self.device)
                
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item() * data.size(0)
                
                pred = output.argmax(dim=1)
                total_correct += (pred == target).sum().item()
                total_samples += data.size(0)
                
                current_acc = total_correct / total_samples
                eval_pbar.set_postfix({"acc": f"{current_acc:.4f}"})
        
        metrics = {
            "accuracy": total_correct / total_samples,
            "loss": total_loss / total_samples,
        }
        
        if self.config.track_fairness and (self.current_round % self.config.fairness_eval_freq == 0):
            fairness_metrics = self._evaluate_fairness()
            metrics.update(fairness_metrics)
        
        return metrics
    
    def _evaluate_train_loss(self, client_ids: List[int]) -> Tuple[float, float]:
        """
        Evaluate training loss on selected clients.
        
        Args:
            client_ids: List of client IDs to evaluate
        
        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for client_id in client_ids:
                loader = self.client_loaders[client_id]
                for data, target in loader:
                    data = data.to(self.device)
                    target = target.to(self.device)
                    
                    output = self.model(data)
                    loss = criterion(output, target)
                    
                    total_loss += loss.item() * data.size(0)
                    pred = output.argmax(dim=1)
                    total_correct += (pred == target).sum().item()
                    total_samples += data.size(0)
        
        if total_samples == 0:
            return 0.0, 0.0
        
        return total_loss / total_samples, total_correct / total_samples
    
    def _evaluate_fairness(self) -> Dict[str, float]:
        """Evaluate per-client accuracy for fairness metrics."""
        from ..utils.metrics import compute_bottom_k_accuracy, compute_fairness
        
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        client_accuracies = {}
        
        with torch.no_grad():
            for client_id, loader in self.client_loaders.items():
                total_correct = 0
                total_samples = 0
                
                for data, target in loader:
                    data = data.to(self.device)
                    target = target.to(self.device)
                    
                    output = self.model(data)
                    pred = output.argmax(dim=1)
                    total_correct += (pred == target).sum().item()
                    total_samples += data.size(0)
                
                if total_samples > 0:
                    client_accuracies[client_id] = total_correct / total_samples
        
        if not client_accuracies:
            return {}
        
        return {
            "bottom_10_accuracy": compute_bottom_k_accuracy(list(client_accuracies.values()), k=0.1),
            "fairness_std": compute_fairness(list(client_accuracies.values()), metric="std"),
            "fairness_min": compute_fairness(list(client_accuracies.values()), metric="min"),
            "num_clients_evaluated": len(client_accuracies),
        }
    
    def _compute_dsnr_metrics(
        self,
        client_updates: List[ClientUpdate],
    ) -> Dict[str, Any]:
        """Compute DSNR and variance metrics for the round."""
        if not client_updates:
            return {}
        
        updates = [u.update for u in client_updates]
        
        if not updates:
            return {}
        
        device = self.device
        updates = [u.to(device) if u.device != device else u for u in updates]
        aggregated_update = torch.stack(updates).mean(dim=0)
        
        dsnr = self.dsnr_analyzer.compute_dsnr(updates, aggregated_update)
        
        momentum = getattr(self.aggregator, "momentum", None)
        decentralized_dsnr = 0.0
        if momentum is not None:
            if momentum.device != device:
                momentum = momentum.to(device)
            decentralized_dsnr = self.dsnr_analyzer.compute_decentralized_dsnr(
                updates, aggregated_update, momentum
            )
        
        proxy_direction = aggregated_update / (aggregated_update.norm() + 1e-10)
        alignment_scores = self.dsnr_analyzer.compute_alignment_scores(
            updates, proxy_direction
        )
        
        variance = torch.stack(updates).var().item()
        
        norms = torch.tensor([u.norm().item() for u in updates])
        
        return {
            "dsnr": dsnr,
            "decentralized_dsnr": decentralized_dsnr,
            "alignment_mean": float(alignment_scores.mean()),
            "alignment_std": float(alignment_scores.std()) if len(alignment_scores) > 1 else 0.0,
            "update_variance": variance,
            "update_norm_mean": float(norms.mean()),
            "update_norm_std": float(norms.std()) if len(norms) > 1 else 0.0,
        }
    
    def _get_model_params(self) -> torch.Tensor:
        """Get flattened model parameters."""
        return torch.cat([p.data.view(-1) for p in self.model.parameters()])
    
    def _save_checkpoint(self, is_best: bool = False) -> None:
        """Save training checkpoint."""
        state = {
            "round": self.current_round,
            "model_state_dict": self.model.state_dict(),
            "aggregator_state": self.aggregator.state_dict(),
            "best_accuracy": self.best_accuracy,
            "history": self.history,
            "config": self.config.__dict__,
            "convergence_round": self.convergence_round,
            "timestamp": datetime.now().isoformat(),
        }
        
        self.checkpoint_manager.save(state, self.current_round, is_best=is_best)
        
        logger.debug(f"Saved checkpoint at round {self.current_round}")
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint."""
        state = self.checkpoint_manager.load(checkpoint_path)
        
        if state is None:
            logger.warning(f"Could not load checkpoint from {checkpoint_path}")
            return
        
        self.model.load_state_dict(state["model_state_dict"])
        self.aggregator.load_state_dict(state["aggregator_state"])
        self.current_round = state["round"]
        self.best_accuracy = state.get("best_accuracy", 0.0)
        self.history = state.get("history", [])
        self.convergence_round = state.get("convergence_round", None)
        
        logger.info(f"Resumed from round {self.current_round}")
    
    def _get_results(self) -> Dict[str, Any]:
        """Get final training results."""
        results = {
            "best_accuracy": self.best_accuracy,
            "final_round": self.current_round,
            "total_time": time.time() - self.start_time,
            "history": self.history,
            "config": self.config.__dict__,
            "convergence_round": self.convergence_round,
        }
        
        if self.history:
            dsnr_values = [h.get("dsnr", 0) for h in self.history if "dsnr" in h]
            variance_values = [h.get("update_variance", 0) for h in self.history if "update_variance" in h]
            
            if dsnr_values:
                results["dsnr_summary"] = {
                    "mean": float(sum(dsnr_values) / len(dsnr_values)),
                    "min": float(min(dsnr_values)),
                    "max": float(max(dsnr_values)),
                }
            
            if variance_values:
                results["variance_summary"] = {
                    "mean": float(sum(variance_values) / len(variance_values)),
                    "min": float(min(variance_values)),
                    "max": float(max(variance_values)),
                }
        
        return results
    
    def save_results(self, path: Optional[str] = None) -> None:
        """Save training results to file."""
        if path is None:
            path = f"{self.config.results_dir}/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        results = self._get_results()
        
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {path}")
    
    def get_convergence_speed(self) -> Optional[int]:
        """Get the round at which convergence threshold was reached."""
        return self.convergence_round
    
    def get_dsnr_history(self) -> List[float]:
        """Get history of DSNR values."""
        return [h.get("dsnr", 0) for h in self.history if "dsnr" in h]
    
    def get_variance_history(self) -> List[float]:
        """Get history of update variance values."""
        return [h.get("update_variance", 0) for h in self.history if "update_variance" in h]
    
    def get_accuracy_history(self) -> List[float]:
        """Get history of accuracy values."""
        return [h.get("accuracy", 0) for h in self.history if "accuracy" in h]
