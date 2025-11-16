"""Metrics calculation for deepfake detection.

This module provides metrics computation focused on Macro F1-score,
which is the primary evaluation metric for the competition.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class MetricsCalculator:
    """Metrics computation for binary classification (Real vs Fake).

    Provides comprehensive metrics with focus on Macro F1-score.
    """

    @staticmethod
    def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Macro F1-score (primary competition metric).

        Macro F1 is the average of F1 scores for each class:
        Macro F1 = (F1_real + F1_fake) / 2

        Args:
            y_true: Ground truth labels, shape (n_samples,)
            y_pred: Predicted labels, shape (n_samples,)

        Returns:
            Macro-averaged F1-score

        Example:
            >>> y_true = np.array([0, 0, 1, 1])
            >>> y_pred = np.array([0, 1, 1, 1])
            >>> f1 = MetricsCalculator.compute_macro_f1(y_true, y_pred)
            >>> print(f"{f1:.4f}")
        """
        return f1_score(y_true, y_pred, average="macro")

    @staticmethod
    def compute_f1_per_class(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Tuple[float, float]:
        """Compute F1 score for each class separately.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels

        Returns:
            Tuple of (f1_real, f1_fake)
        """
        f1_scores = f1_score(y_true, y_pred, average=None)

        # f1_scores[0] = F1 for class 0 (Real)
        # f1_scores[1] = F1 for class 1 (Fake)
        f1_real = f1_scores[0] if len(f1_scores) > 0 else 0.0
        f1_fake = f1_scores[1] if len(f1_scores) > 1 else 0.0

        return f1_real, f1_fake

    @staticmethod
    def compute_precision_recall(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """Compute precision and recall for each class.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels

        Returns:
            Tuple of (precision_real, precision_fake, recall_real, recall_fake)
        """
        # Precision per class
        precision_scores = precision_score(y_true, y_pred, average=None, zero_division=0)
        precision_real = precision_scores[0] if len(precision_scores) > 0 else 0.0
        precision_fake = precision_scores[1] if len(precision_scores) > 1 else 0.0

        # Recall per class
        recall_scores = recall_score(y_true, y_pred, average=None, zero_division=0)
        recall_real = recall_scores[0] if len(recall_scores) > 0 else 0.0
        recall_fake = recall_scores[1] if len(recall_scores) > 1 else 0.0

        return precision_real, precision_fake, recall_real, recall_fake

    @staticmethod
    def compute_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute all evaluation metrics.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_probs: Prediction probabilities (optional, for AUC)
                    Shape: (n_samples, 2) or (n_samples,)

        Returns:
            Dictionary containing:
                - macro_f1: Macro F1-score (PRIMARY METRIC)
                - accuracy: Overall accuracy
                - precision_real: Precision for Real class
                - precision_fake: Precision for Fake class
                - recall_real: Recall for Real class
                - recall_fake: Recall for Fake class
                - f1_real: F1 for Real class
                - f1_fake: F1 for Fake class
                - auc: AUC-ROC (if y_probs provided)
                - confusion_matrix: 2x2 confusion matrix (as nested list)
        """
        metrics = {}

        # Primary metric: Macro F1
        metrics["macro_f1"] = MetricsCalculator.compute_macro_f1(y_true, y_pred)

        # Accuracy
        metrics["accuracy"] = accuracy_score(y_true, y_pred)

        # Per-class F1
        f1_real, f1_fake = MetricsCalculator.compute_f1_per_class(y_true, y_pred)
        metrics["f1_real"] = f1_real
        metrics["f1_fake"] = f1_fake

        # Precision and Recall
        prec_real, prec_fake, rec_real, rec_fake = MetricsCalculator.compute_precision_recall(
            y_true, y_pred
        )
        metrics["precision_real"] = prec_real
        metrics["precision_fake"] = prec_fake
        metrics["recall_real"] = rec_real
        metrics["recall_fake"] = rec_fake

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # AUC-ROC (if probabilities provided)
        if y_probs is not None:
            try:
                # If y_probs is 2D (n_samples, 2), use second column (fake probability)
                if y_probs.ndim == 2:
                    probs_fake = y_probs[:, 1]
                else:
                    probs_fake = y_probs

                metrics["auc"] = roc_auc_score(y_true, probs_fake)
            except Exception:
                # Skip AUC if calculation fails
                metrics["auc"] = 0.0

        return metrics

    @staticmethod
    def print_metrics_report(metrics: Dict[str, float]) -> None:
        """Print formatted metrics report.

        Args:
            metrics: Dictionary of metrics from compute_all_metrics()
        """
        print("=" * 80)
        print("METRICS REPORT")
        print("=" * 80)

        # Primary metric
        print(f"\n*** Macro F1-score (PRIMARY): {metrics.get('macro_f1', 0.0):.4f} ***\n")

        # Overall metrics
        print("Overall:")
        print(f"  Accuracy: {metrics.get('accuracy', 0.0):.4f}")

        if "auc" in metrics and metrics["auc"] > 0:
            print(f"  AUC-ROC:  {metrics['auc']:.4f}")

        # Real class (Class 0)
        print("\nReal Class (0):")
        print(f"  Precision: {metrics.get('precision_real', 0.0):.4f}")
        print(f"  Recall:    {metrics.get('recall_real', 0.0):.4f}")
        print(f"  F1-score:  {metrics.get('f1_real', 0.0):.4f}")

        # Fake class (Class 1)
        print("\nFake Class (1):")
        print(f"  Precision: {metrics.get('precision_fake', 0.0):.4f}")
        print(f"  Recall:    {metrics.get('recall_fake', 0.0):.4f}")
        print(f"  F1-score:  {metrics.get('f1_fake', 0.0):.4f}")

        # Confusion Matrix
        if "confusion_matrix" in metrics:
            cm = np.array(metrics["confusion_matrix"])
            print("\nConfusion Matrix:")
            print("                Predicted")
            print("              Real    Fake")
            print(f"Actual Real   {cm[0, 0]:4d}    {cm[0, 1]:4d}")
            print(f"      Fake   {cm[1, 0]:4d}    {cm[1, 1]:4d}")

        print("=" * 80)

    @staticmethod
    def compute_f1_from_cm(cm: np.ndarray) -> float:
        """Compute Macro F1 from confusion matrix.

        Args:
            cm: Confusion matrix, shape (2, 2)
                [[TN, FP],
                 [FN, TP]]

        Returns:
            Macro F1-score
        """
        # Extract values
        tn, fp = cm[0, 0], cm[0, 1]
        fn, tp = cm[1, 0], cm[1, 1]

        # F1 for Real class (class 0)
        prec_real = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        rec_real = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_real = (
            2 * prec_real * rec_real / (prec_real + rec_real)
            if (prec_real + rec_real) > 0
            else 0.0
        )

        # F1 for Fake class (class 1)
        prec_fake = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_fake = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_fake = (
            2 * prec_fake * rec_fake / (prec_fake + rec_fake)
            if (prec_fake + rec_fake) > 0
            else 0.0
        )

        # Macro F1
        macro_f1 = (f1_real + f1_fake) / 2

        return macro_f1

    @staticmethod
    def is_balanced_performance(
        metrics: Dict[str, float],
        threshold: float = 0.1,
    ) -> bool:
        """Check if performance is balanced across classes.

        Balanced performance means F1 scores for both classes are similar.
        Important for Macro F1 optimization.

        Args:
            metrics: Metrics dictionary
            threshold: Maximum allowed difference in F1 scores

        Returns:
            True if performance is balanced, False otherwise
        """
        f1_real = metrics.get("f1_real", 0.0)
        f1_fake = metrics.get("f1_fake", 0.0)

        f1_diff = abs(f1_real - f1_fake)

        return f1_diff <= threshold

    @staticmethod
    def get_weak_class(metrics: Dict[str, float]) -> str:
        """Identify which class has weaker performance.

        Args:
            metrics: Metrics dictionary

        Returns:
            "real" or "fake" depending on which has lower F1
        """
        f1_real = metrics.get("f1_real", 0.0)
        f1_fake = metrics.get("f1_fake", 0.0)

        return "real" if f1_real < f1_fake else "fake"
