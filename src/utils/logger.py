"""Logging utilities for deepfake detection project.

This module provides logging setup and utilities for tracking training,
validation, and inference progress.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


def setup_logger(
    name: str = "deepfake_detection",
    log_file: Optional[Union[str, Path]] = None,
    log_level: Union[str, int] = logging.INFO,
    console_output: bool = True,
    file_output: bool = True,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Setup logger with file and console handlers.

    Args:
        name: Logger name
        log_file: Path to log file. If None, uses default: logs/{name}_{timestamp}.log
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Whether to output to console
        file_output: Whether to output to file
        format_string: Custom format string. If None, uses default format.

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger("training", "logs/train.log", logging.INFO)
        >>> logger.info("Training started")
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Default format with timestamp, level, logger name, and message
    if format_string is None:
        format_string = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if file_output:
        if log_file is None:
            # Create default log file path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True, parents=True)
            log_file = log_dir / f"{name}_{timestamp}.log"
        else:
            log_file = Path(log_file)
            log_file.parent.mkdir(exist_ok=True, parents=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file}")

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str = "deepfake_detection") -> logging.Logger:
    """Get existing logger or create new one.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If logger has no handlers, set it up with defaults
    if not logger.handlers:
        logger = setup_logger(name)

    return logger


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that works with tqdm progress bars.

    This handler uses tqdm.write() to avoid breaking progress bars.
    """

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            from tqdm import tqdm

            msg = self.format(record)
            tqdm.write(msg)
        except ImportError:
            # Fall back to regular print if tqdm not available
            print(self.format(record))
        except Exception:
            self.handleError(record)


def setup_tqdm_logger(
    name: str = "deepfake_detection",
    log_file: Optional[Union[str, Path]] = None,
    log_level: Union[str, int] = logging.INFO,
) -> logging.Logger:
    """Setup logger compatible with tqdm progress bars.

    Args:
        name: Logger name
        log_file: Path to log file
        log_level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.handlers.clear()

    # Format
    format_string = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Tqdm-compatible console handler
    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (same as regular logger)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(exist_ok=True, parents=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False

    return logger


class MetricsLogger:
    """Logger for tracking training/validation metrics.

    Provides structured logging of metrics with optional file output.
    """

    def __init__(
        self,
        log_file: Optional[Union[str, Path]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize metrics logger.

        Args:
            log_file: Path to metrics log file (CSV format)
            logger: Existing logger instance. If None, creates new logger.
        """
        self.log_file = Path(log_file) if log_file else None
        self.logger = logger if logger else get_logger("metrics")

        if self.log_file:
            self.log_file.parent.mkdir(exist_ok=True, parents=True)

            # Write header if file doesn't exist
            if not self.log_file.exists():
                with open(self.log_file, 'w') as f:
                    f.write("timestamp,epoch,phase,metric,value\n")

    def log_metric(
        self,
        metric_name: str,
        value: float,
        epoch: Optional[int] = None,
        phase: str = "train",
    ) -> None:
        """Log a single metric.

        Args:
            metric_name: Name of metric (e.g., "loss", "accuracy", "f1")
            value: Metric value
            epoch: Training epoch number
            phase: Phase ("train", "val", "test")
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Console/file log
        epoch_str = f"Epoch {epoch}" if epoch is not None else "N/A"
        self.logger.info(f"{epoch_str} | {phase} | {metric_name}: {value:.4f}")

        # CSV log
        if self.log_file:
            epoch_val = epoch if epoch is not None else -1
            with open(self.log_file, 'a') as f:
                f.write(f"{timestamp},{epoch_val},{phase},{metric_name},{value}\n")

    def log_metrics(
        self,
        metrics: dict,
        epoch: Optional[int] = None,
        phase: str = "train",
    ) -> None:
        """Log multiple metrics.

        Args:
            metrics: Dictionary of metric_name: value pairs
            epoch: Training epoch number
            phase: Phase ("train", "val", "test")
        """
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log_metric(metric_name, value, epoch, phase)

    def log_epoch_summary(
        self,
        epoch: int,
        train_metrics: dict,
        val_metrics: Optional[dict] = None,
    ) -> None:
        """Log summary of an epoch.

        Args:
            epoch: Epoch number
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary (optional)
        """
        self.logger.info(f"=" * 80)
        self.logger.info(f"Epoch {epoch} Summary")
        self.logger.info(f"-" * 80)

        self.logger.info("Training Metrics:")
        for name, value in train_metrics.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"  {name}: {value:.4f}")
                self.log_metric(name, value, epoch, "train")

        if val_metrics:
            self.logger.info("Validation Metrics:")
            for name, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"  {name}: {value:.4f}")
                    self.log_metric(name, value, epoch, "val")

        self.logger.info(f"=" * 80)


def log_system_info(logger: Optional[logging.Logger] = None) -> None:
    """Log system information (Python, PyTorch, CUDA).

    Args:
        logger: Logger instance. If None, uses default logger.
    """
    if logger is None:
        logger = get_logger()

    import sys
    import platform

    logger.info("=" * 80)
    logger.info("System Information")
    logger.info("-" * 80)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")

    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        logger.warning("PyTorch not installed")

    logger.info("=" * 80)
