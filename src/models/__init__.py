"""Model architectures for deepfake detection."""

from .deepfake_detector import DeepfakeDetector, create_model_from_config, count_parameters
from .spatial_branch import SpatialBranch
from .frequency_branch import FrequencyBranch
from .fusion_layer import FusionLayer, SimpleFusionLayer, GatedFusionLayer

__all__ = [
    "DeepfakeDetector",
    "SpatialBranch",
    "FrequencyBranch",
    "FusionLayer",
    "SimpleFusionLayer",
    "GatedFusionLayer",
    "create_model_from_config",
    "count_parameters",
]
