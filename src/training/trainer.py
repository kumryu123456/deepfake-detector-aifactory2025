"""Trainer module for deepfake detection model.

This module implements the training loop with support for:
- CombinedLoss optimization for Macro F1
- Early stopping on validation Macro F1
- Learning rate scheduling with warmup
- Mixed precision training (FP16)
- Checkpoint management
- TensorBoard logging
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .losses import CombinedLoss, create_loss_function
from .metrics import MetricsCalculator


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving.

    Reference: training_config.yaml early_stopping section
    """

    def __init__(
        self,
        patience: int = 15,
        metric: str = "macro_f1",
        mode: str = "max",
        min_delta: float = 0.0,
        verbose: bool = True,
    ):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            metric: Metric to monitor
            mode: "max" or "min" (maximize or minimize metric)
            min_delta: Minimum change to qualify as improvement
            verbose: Print early stopping messages
        """
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

        # For mode="max", we want higher values; for "min", lower values
        self.monitor_op = np.greater if mode == "max" else np.less

    def __call__(self, current_value: float, epoch: int) -> bool:
        """Check if training should stop.

        Args:
            current_value: Current metric value
            epoch: Current epoch number

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_value
            self.best_epoch = epoch
            return False

        # Check if current value is better than best
        if self.mode == "max":
            is_better = current_value > self.best_score + self.min_delta
        else:
            is_better = current_value < self.best_score - self.min_delta

        if is_better:
            self.best_score = current_value
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"EarlyStopping: New best {self.metric}: {current_value:.4f} (epoch {epoch})")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"EarlyStopping: Stopping early at epoch {epoch}")
                    print(f"Best {self.metric}: {self.best_score:.4f} (epoch {self.best_epoch})")

        return self.early_stop


class Trainer:
    """Trainer for deepfake detection model.

    Implements complete training pipeline with early stopping, LR scheduling,
    checkpointing, and logging optimized for Macro F1 metric.

    Example:
        >>> model = DeepfakeDetector(...)
        >>> train_loader = DataLoader(train_dataset, ...)
        >>> val_loader = DataLoader(val_dataset, ...)
        >>>
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     config=config,
        ...     device="cuda",
        ... )
        >>>
        >>> trainer.train(epochs=100)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict] = None,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        experiment_name: Optional[str] = None,
    ):
        """Initialize trainer.

        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            config: Training configuration dictionary (from training_config.yaml)
            device: Device for training ("cuda" or "cpu")
            checkpoint_dir: Directory for saving checkpoints
            log_dir: Directory for TensorBoard logs
            experiment_name: Name for this experiment (used in logging)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name or f"experiment_{int(time.time())}"

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Extract config parameters
        self.epochs = self.config.get("training", {}).get("epochs", 100)
        self.mixed_precision = self.config.get("training", {}).get("mixed_precision", True)

        # Initialize loss function
        self.criterion = self._create_loss_function()

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Initialize early stopping
        self.early_stopping = self._create_early_stopping()

        # Mixed precision scaler
        self.scaler = GradScaler() if self.mixed_precision else None

        # Metrics calculator
        self.metrics_calculator = MetricsCalculator()

        # TensorBoard writer
        tensorboard_enabled = self.config.get("logging", {}).get("tensorboard", True)
        if tensorboard_enabled:
            self.writer = SummaryWriter(log_dir=self.log_dir / self.experiment_name)
        else:
            self.writer = None

        # Training state
        self.current_epoch = 0
        self.best_metric = -np.inf  # For tracking best validation metric
        self.train_losses = []
        self.val_metrics_history = []

        # Fine-tuning schedule (adjust loss weights after certain epoch)
        self.finetuning_config = self.config.get("loss", {}).get("finetuning", {})
        self.finetuning_enabled = self.finetuning_config.get("enabled", False)
        self.finetuning_start_epoch = self.finetuning_config.get("start_epoch", 80)

        print(f"Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Mixed precision: {self.mixed_precision}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Optimizer: {type(self.optimizer).__name__}")
        print(f"  Scheduler: {type(self.scheduler).__name__}")
        print(f"  Early stopping: {self.early_stopping is not None}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
        print(f"  Log dir: {self.log_dir / self.experiment_name}")

    def _create_loss_function(self) -> nn.Module:
        """Create loss function from config."""
        loss_config = self.config.get("loss", {})
        loss_type = loss_config.get("type", "combined")

        if loss_type == "combined":
            # Get weights
            weights = loss_config.get("weights", {})
            ce_weight = weights.get("cross_entropy", 0.5)
            focal_weight = weights.get("focal", 0.3)
            f1_weight = weights.get("soft_f1", 0.2)

            # Focal loss parameters
            focal_config = loss_config.get("focal", {})
            focal_gamma = focal_config.get("gamma", 2.0)
            focal_alpha = focal_config.get("alpha", 0.25)

            criterion = CombinedLoss(
                ce_weight=ce_weight,
                focal_weight=focal_weight,
                f1_weight=f1_weight,
                focal_gamma=focal_gamma,
                focal_alpha=focal_alpha,
            )
        else:
            criterion = create_loss_function(loss_type)

        return criterion

    def _create_optimizer(self) -> Optimizer:
        """Create optimizer from config."""
        optimizer_config = self.config.get("optimizer", {})
        optimizer_type = optimizer_config.get("type", "adamw")

        lr = optimizer_config.get("learning_rate", 1e-4)
        weight_decay = optimizer_config.get("weight_decay", 0.01)

        if optimizer_type.lower() == "adamw":
            betas = optimizer_config.get("betas", [0.9, 0.999])
            eps = optimizer_config.get("eps", 1e-8)

            optimizer = AdamW(
                self.model.parameters(),
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        return optimizer

    def _create_scheduler(self) -> Optional[object]:
        """Create learning rate scheduler from config."""
        scheduler_config = self.config.get("scheduler", {})
        scheduler_type = scheduler_config.get("type", "cosine_annealing")

        if scheduler_type == "none":
            return None

        if scheduler_type == "cosine_annealing":
            T_max = scheduler_config.get("T_max", self.epochs)
            eta_min = scheduler_config.get("eta_min", 1e-6)

            # Create main scheduler
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=T_max,
                eta_min=eta_min,
            )

            # Add warmup if configured
            warmup_epochs = scheduler_config.get("warmup_epochs", 0)

            if warmup_epochs > 0:
                warmup_start_lr = scheduler_config.get("warmup_start_lr", 1e-5)
                base_lr = self.optimizer.param_groups[0]["lr"]

                # Warmup scheduler
                warmup_scheduler = LambdaLR(
                    self.optimizer,
                    lr_lambda=lambda epoch: (base_lr - warmup_start_lr) * epoch / warmup_epochs + warmup_start_lr
                )

                # Combine warmup + main scheduler
                scheduler = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[warmup_epochs],
                )
            else:
                scheduler = main_scheduler
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")

        return scheduler

    def _create_early_stopping(self) -> Optional[EarlyStopping]:
        """Create early stopping from config."""
        es_config = self.config.get("early_stopping", {})

        if not es_config.get("enabled", False):
            return None

        patience = es_config.get("patience", 15)
        metric = es_config.get("metric", "macro_f1")
        mode = es_config.get("mode", "max")

        return EarlyStopping(
            patience=patience,
            metric=metric,
            mode=mode,
            verbose=True,
        )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        loss_components = {"loss_ce": 0.0, "loss_focal": 0.0, "loss_f1": 0.0}
        all_predictions = []
        all_labels = []
        all_probs = []

        # Apply fine-tuning schedule if enabled
        if self.finetuning_enabled and epoch >= self.finetuning_start_epoch:
            if hasattr(self.criterion, "update_weights"):
                finetuning_weights = self.finetuning_config.get("weights", {})
                ce_weight = finetuning_weights.get("cross_entropy")
                focal_weight = finetuning_weights.get("focal")
                f1_weight = finetuning_weights.get("soft_f1")

                self.criterion.update_weights(
                    ce_weight=ce_weight,
                    focal_weight=focal_weight,
                    f1_weight=f1_weight,
                )
                print(f"Fine-tuning schedule: Updated loss weights at epoch {epoch}")

        # Training loop
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]")

        for batch_idx, batch in enumerate(pbar):
            # Handle custom collate_fn output (list of tensors)
            frames_list = batch["frames"]
            labels = batch.get("label")

            if labels is None:
                raise ValueError("Training requires labels in batch")

            labels = labels.to(self.device)

            # Process each sample in batch (handle variable-length videos)
            batch_loss = 0.0
            batch_loss_components = {"loss_ce": 0.0, "loss_focal": 0.0, "loss_f1": 0.0}
            batch_predictions = []
            batch_probs = []

            for frames, label in zip(frames_list, labels):
                frames = frames.to(self.device)  # Shape: (N, 3, H, W)

                # Forward pass with mixed precision
                if self.mixed_precision:
                    with autocast():
                        # Get predictions for all frames
                        logits = self.model(frames)  # Shape: (N, 2)

                        # Aggregate predictions (mean pooling over frames)
                        logits_mean = logits.mean(dim=0, keepdim=True)  # Shape: (1, 2)

                        # Compute loss
                        label_tensor = label.unsqueeze(0)  # Shape: (1,)
                        loss, loss_dict = self.criterion(logits_mean, label_tensor)
                else:
                    logits = self.model(frames)
                    logits_mean = logits.mean(dim=0, keepdim=True)
                    label_tensor = label.unsqueeze(0)
                    loss, loss_dict = self.criterion(logits_mean, label_tensor)

                # Accumulate loss
                batch_loss += loss
                for key in batch_loss_components:
                    if key in loss_dict:
                        batch_loss_components[key] += loss_dict[key].item()

                # Get predictions
                probs = torch.softmax(logits_mean, dim=1)
                pred = torch.argmax(logits_mean, dim=1)

                batch_predictions.append(pred.item())
                batch_probs.append(probs.detach().cpu().numpy())

            # Average loss over batch
            batch_loss = batch_loss / len(frames_list)

            # Backward pass
            self.optimizer.zero_grad()

            if self.mixed_precision:
                self.scaler.scale(batch_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                batch_loss.backward()
                self.optimizer.step()

            # Accumulate metrics
            total_loss += batch_loss.item()
            for key in loss_components:
                loss_components[key] += batch_loss_components[key] / len(frames_list)

            all_predictions.extend(batch_predictions)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(batch_probs)

            # Update progress bar
            pbar.set_postfix({"loss": batch_loss.item()})

        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)

        # Convert to numpy
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probs = np.vstack(all_probs)

        # Compute metrics
        metrics = self.metrics_calculator.compute_all_metrics(
            y_true=all_labels,
            y_pred=all_predictions,
            y_probs=all_probs,
        )

        metrics["loss"] = avg_loss
        for key, value in loss_components.items():
            metrics[key] = value / len(self.train_loader)

        return metrics

    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate model.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()

        all_predictions = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.epochs} [Val]")

            for batch in pbar:
                frames_list = batch["frames"]
                labels = batch.get("label")

                if labels is None:
                    continue

                labels = labels.to(self.device)

                batch_predictions = []
                batch_probs = []

                for frames, label in zip(frames_list, labels):
                    frames = frames.to(self.device)

                    # Forward pass
                    if self.mixed_precision:
                        with autocast():
                            logits = self.model(frames)
                            logits_mean = logits.mean(dim=0, keepdim=True)
                    else:
                        logits = self.model(frames)
                        logits_mean = logits.mean(dim=0, keepdim=True)

                    # Get predictions
                    probs = torch.softmax(logits_mean, dim=1)
                    pred = torch.argmax(logits_mean, dim=1)

                    batch_predictions.append(pred.item())
                    batch_probs.append(probs.cpu().numpy())

                all_predictions.extend(batch_predictions)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(batch_probs)

        # Compute metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probs = np.vstack(all_probs)

        metrics = self.metrics_calculator.compute_all_metrics(
            y_true=all_labels,
            y_pred=all_predictions,
            y_probs=all_probs,
        )

        return metrics

    def train(self, epochs: Optional[int] = None) -> Dict[str, list]:
        """Train model for multiple epochs.

        Args:
            epochs: Number of epochs (overrides config)

        Returns:
            Dictionary of training history
        """
        if epochs is not None:
            self.epochs = epochs

        print(f"\n{'=' * 80}")
        print(f"Starting training for {self.epochs} epochs")
        print(f"{'=' * 80}\n")

        history = {
            "train_loss": [],
            "train_macro_f1": [],
            "val_macro_f1": [],
            "learning_rate": [],
        }

        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch) if self.val_loader else {}

            # Learning rate step
            if self.scheduler is not None:
                self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log metrics
            self._log_epoch(epoch, train_metrics, val_metrics, current_lr)

            # Update history
            history["train_loss"].append(train_metrics["loss"])
            history["train_macro_f1"].append(train_metrics["macro_f1"])
            history["learning_rate"].append(current_lr)

            if val_metrics:
                history["val_macro_f1"].append(val_metrics["macro_f1"])

                # Save best checkpoint
                if val_metrics["macro_f1"] > self.best_metric:
                    self.best_metric = val_metrics["macro_f1"]
                    self.save_checkpoint(epoch, is_best=True)

                # Early stopping check
                if self.early_stopping is not None:
                    should_stop = self.early_stopping(val_metrics["macro_f1"], epoch)
                    if should_stop:
                        print(f"\nEarly stopping triggered at epoch {epoch}")
                        break

            # Save periodic checkpoint
            checkpoint_config = self.config.get("checkpoint", {})
            save_frequency = checkpoint_config.get("save_frequency", 5)

            if epoch % save_frequency == 0:
                self.save_checkpoint(epoch, is_best=False)

        print(f"\n{'=' * 80}")
        print("Training completed!")
        print(f"Best validation Macro F1: {self.best_metric:.4f}")
        print(f"{'=' * 80}\n")

        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()

        return history

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        learning_rate: float,
    ) -> None:
        """Log epoch metrics to console and TensorBoard.

        Args:
            epoch: Epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
            learning_rate: Current learning rate
        """
        # Console output
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch}/{self.epochs}")
        print(f"{'-' * 80}")
        print(f"Learning Rate: {learning_rate:.6f}")
        print(f"\nTrain Metrics:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Macro F1: {train_metrics['macro_f1']:.4f}")
        print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  F1 (Real): {train_metrics['f1_real']:.4f}")
        print(f"  F1 (Fake): {train_metrics['f1_fake']:.4f}")

        if val_metrics:
            print(f"\nValidation Metrics:")
            print(f"  Macro F1: {val_metrics['macro_f1']:.4f}")
            print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  F1 (Real): {val_metrics['f1_real']:.4f}")
            print(f"  F1 (Fake): {val_metrics['f1_fake']:.4f}")

        print(f"{'=' * 80}")

        # TensorBoard logging
        if self.writer is not None:
            # Scalars
            self.writer.add_scalar("Learning_Rate", learning_rate, epoch)

            for key, value in train_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"Train/{key}", value, epoch)

            for key, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"Val/{key}", value, epoch)

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # Save checkpoint
        if is_best:
            checkpoint_path = self.checkpoint_dir / f"{self.experiment_name}_best.pth"
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved best checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = self.checkpoint_dir / f"{self.experiment_name}_epoch{epoch}.pth"
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> int:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Epoch number from checkpoint
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "scaler_state_dict" in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.best_metric = checkpoint.get("best_metric", -np.inf)
        epoch = checkpoint["epoch"]

        print(f"Loaded checkpoint from epoch {epoch}")
        print(f"Best metric: {self.best_metric:.4f}")

        return epoch
