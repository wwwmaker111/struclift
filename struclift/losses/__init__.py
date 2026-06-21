from .contrastive import InfoNCELoss, RegionInfoNCELoss
from .alignment import (
    SCOTAlignmentLoss,
    EdgeRecoveryLoss,
    CrossAttentionAlignmentLoss,
    RefinedEmbeddingAlignmentLoss,
    NodeHardNegativeContrastiveLoss,
)
from .structural import (
    GraphBinarySourceContrastiveLoss,
    NeighborContrastiveLoss,
    NeighborReconLoss,
    NeighborReconstructionLoss,
    PatternClassificationLoss,
)
