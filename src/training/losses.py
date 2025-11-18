"""Loss functions for deepfake detection model training.

This module implements loss functions optimized for Macro F1-score:
1. SoftF1Loss: Differentiable approximation of F1-score
2. FocalLoss: Addresses class imbalance and hard examples
3. CombinedLoss: Weighted combination of CE, Focal, and Soft-F1

Reference: research.md lines 262-321 for implementation details
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftF1Loss(nn.Module):
    """Differentiable approximation of F1-score for binary classification.
    
    Traditional F1-score uses argmax (non-differentiable). This implementation
    uses soft predictions (probabilities) to enable gradient-based optimization.
    
    Reference: research.md lines 270-298
    """
    
    def __init__(self, epsilon: float = 1e-7):
        """Initialize Soft F1 Loss.
        
        Args:
            epsilon: Small value to avoid division by zero
        """
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Soft F1 Loss.
        
        Args:
            logits: Model outputs, shape (batch_size, num_classes)
            targets: Ground truth labels, shape (batch_size,) with values in {0, 1}
        
        Returns:
            Loss value (scalar tensor), computed as 1 - Macro F1
        """
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=1)  # Shape: (batch_size, 2)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=2).float()  # Shape: (batch_size, 2)
        
        # Compute soft TP, FP, FN for each class
        # TP: predicted as class i AND actual class i
        tp = (probs * targets_one_hot).sum(dim=0)  # Shape: (2,)
        
        # FP: predicted as class i BUT actual class j (j != i)
        fp = (probs * (1 - targets_one_hot)).sum(dim=0)  # Shape: (2,)
        
        # FN: NOT predicted as class i BUT actual class i
        fn = ((1 - probs) * targets_one_hot).sum(dim=0)  # Shape: (2,)
        
        # Compute precision and recall for each class
        precision = tp / (tp + fp + self.epsilon)  # Shape: (2,)
        recall = tp / (tp + fn + self.epsilon)  # Shape: (2,)
        
        # Compute F1 for each class
        f1_per_class = 2 * precision * recall / (precision + recall + self.epsilon)  # Shape: (2,)
        
        # Macro F1: average across classes
        macro_f1 = f1_per_class.mean()
        
        # Return loss (1 - F1 to minimize)
        return 1 - macro_f1


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance and hard examples.
    
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    
    where:
    - p_t is the probability of the correct class
    - gamma focuses on hard examples (typically gamma=2)
    - alpha balances classes (typically alpha=0.25 for positive class)
    
    Reference: research.md lines 257-264
    Paper: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for positive class (Fake)
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Focal Loss.
        
        Args:
            logits: Model outputs, shape (batch_size, num_classes)
            targets: Ground truth labels, shape (batch_size,)
        
        Returns:
            Loss value
        """
        # Compute cross-entropy loss without reduction
        ce_loss = F.cross_entropy(logits, targets, reduction="none")  # Shape: (batch_size,)
        
        # Get probabilities
        probs = torch.softmax(logits, dim=1)  # Shape: (batch_size, num_classes)
        
        # Get probability of correct class for each sample
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # Shape: (batch_size,)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute alpha weight
        # alpha for positive class (Fake=1), (1-alpha) for negative class (Real=0)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Compute focal loss
        focal_loss = alpha_t * focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class CombinedLoss(nn.Module):
    """Combined loss function for Macro F1 optimization.
    
    Combines three loss functions:
    1. Cross-Entropy Loss: Stable baseline
    2. Focal Loss: Handles hard examples and class imbalance
    3. Soft F1 Loss: Directly optimizes target metric
    
    Loss = λ_ce * L_CE + λ_focal * L_Focal + λ_f1 * L_SoftF1
    
    Reference: research.md lines 315-321
    """
    
    def __init__(
        self,
        ce_weight: float = 0.5,
        focal_weight: float = 0.3,
        f1_weight: float = 0.2,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        class_weights: Optional[list] = None,
    ):
        """Initialize Combined Loss.

        Args:
            ce_weight: Weight for Cross-Entropy loss
            focal_weight: Weight for Focal loss
            f1_weight: Weight for Soft F1 loss
            focal_gamma: Gamma parameter for Focal loss
            focal_alpha: Alpha parameter for Focal loss
            class_weights: Class weights for handling imbalance [weight_real, weight_fake]
        """
        super().__init__()

        # Validate weights sum to 1.0 (optional, for interpretability)
        total_weight = ce_weight + focal_weight + f1_weight
        if abs(total_weight - 1.0) > 0.01:
            import warnings
            warnings.warn(
                f"Loss weights sum to {total_weight:.3f}, not 1.0. "
                "Consider normalizing for better interpretability."
            )

        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.f1_weight = f1_weight

        # Convert class weights to tensor if provided
        weight_tensor = None
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

        # Initialize loss functions
        self.ce_loss = nn.CrossEntropyLoss(weight=weight_tensor)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.f1_loss = SoftF1Loss()
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute combined loss.
        
        Args:
            logits: Model outputs, shape (batch_size, num_classes)
            targets: Ground truth labels, shape (batch_size,)
        
        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains individual losses
        """
        # Compute individual losses
        ce = self.ce_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        f1 = self.f1_loss(logits, targets)
        
        # Weighted combination
        total_loss = (
            self.ce_weight * ce +
            self.focal_weight * focal +
            self.f1_weight * f1
        )
        
        # Return total loss and components (for logging)
        loss_dict = {
            "loss_ce": ce.detach(),
            "loss_focal": focal.detach(),
            "loss_f1": f1.detach(),
            "loss_total": total_loss.detach(),
        }
        
        return total_loss, loss_dict
    
    def update_weights(
        self,
        ce_weight: Optional[float] = None,
        focal_weight: Optional[float] = None,
        f1_weight: Optional[float] = None,
    ) -> None:
        """Update loss weights (useful for fine-tuning schedule).
        
        Example usage:
            # Start training with CE-heavy weights
            loss_fn = CombinedLoss(ce_weight=0.7, focal_weight=0.2, f1_weight=0.1)
            
            # After 80 epochs, increase F1 weight for fine-tuning
            loss_fn.update_weights(ce_weight=0.4, focal_weight=0.2, f1_weight=0.4)
        
        Args:
            ce_weight: New weight for CE loss (if provided)
            focal_weight: New weight for Focal loss (if provided)
            f1_weight: New weight for F1 loss (if provided)
        """
        if ce_weight is not None:
            self.ce_weight = ce_weight
        if focal_weight is not None:
            self.focal_weight = focal_weight
        if f1_weight is not None:
            self.f1_weight = f1_weight


# Factory function for easy instantiation
def create_loss_function(
    loss_type: str = "combined",
    **kwargs,
) -> nn.Module:
    """Factory function to create loss functions.
    
    Args:
        loss_type: Type of loss ('ce', 'focal', 'soft_f1', 'combined')
        **kwargs: Additional arguments passed to loss constructor
    
    Returns:
        Loss function module
    
    Example:
        >>> loss_fn = create_loss_function('combined', ce_weight=0.5, f1_weight=0.3)
        >>> logits = torch.randn(32, 2)
        >>> targets = torch.randint(0, 2, (32,))
        >>> loss, loss_dict = loss_fn(logits, targets)
    """
    if loss_type == "ce":
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_type == "focal":
        return FocalLoss(**kwargs)
    elif loss_type == "soft_f1":
        return SoftF1Loss(**kwargs)
    elif loss_type == "combined":
        return CombinedLoss(**kwargs)
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            "Choose from: 'ce', 'focal', 'soft_f1', 'combined'"
        )
