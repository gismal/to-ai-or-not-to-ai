"""
Measuring what "familiar" looks like to a trained model.

After training used by the Kaggle. No change on the model's weight
"""

import numpy as np
from sklearn.covariance import LedoitWolf
from torch import Tensor, nn

from src.familiarity import mahalanobis


class WithFeatures(nn.Module):
    """MobileNetV3 that returns its pooled faetures, so one ONNX run gives both"""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.model.avgpool(self.model.features(x)).flatten(1)
        return self.model.classifier(features), features


def fit(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    percentile: float = 95.0,
) -> dict[str, np.ndarray]:
    """
    Class centres and shares spread from training features, the threshold from validation ones.

    The threshold is the distance that 'percentile' of familiart validation images stay within, so
    by construction about (100 - percentile)% of familiar images get flagged. Unfamiliar data is never
    used to choose it to keep the evaluation honest
    """
    classes = np.unique(train_labels)
    means = np.stack([train_features[train_labels == c].mean(axis=0) for c in classes])
    centred = train_features - means[np.searchsorted(classes, train_labels)]

    # Ledoit-Wolf shrinkage keeps the covariance stable with 576 dimensions and a few thousand images
    precision = LedoitWolf().fit(centred).precision_
    threshold = np.percentile(mahalanobis(val_features, means, precision), percentile)
    return {
        "means": means.astype(np.float32),
        "precision": precision.astype(np.float32),
        "threshold": np.float32(threshold),
        "percentile": np.float32(percentile),
    }
