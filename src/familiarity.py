"""
How familiar does an image look to the model?

Before deciding, the model turns every image its pooled features (its pooled features).
Images like the training data land in a compact region of that space: very different images land far away.
The region is summarized by one centre per class and a shared spread and an image's
distance from the nearest centre (a Mahalanobis distance) says how unfamiliar it is.
I used Mahalanobis because it first learns how much each number usually varies and which ones move together, then it
measures distance relative to that. It catches unusual combinations.
Beyond a threshold, chosen on
familiar validation images only, the model's confidence isn't worth much, so the API stops giving
a definite verdict.

The statistics come from scripts/training/familiarity.py and live in models/familiarity.npz.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

FAMILIARITY_PATH = Path("models/familiarity.npz")


def mahalanobis(
    features: np.ndarray, means: np.ndarray, precision: np.ndarray
) -> np.ndarray:
    """
    Distance from each row of 'features' (n, d) to the nearest of 'means' (k,d)
    """
    diffs = features[:, None, :] - means[None, :, :]  # (n, k, d)
    squared = np.einsum("nkd,de,nke->nk", diffs, precision, diffs)
    return np.sqrt(np.maximum(squared, 0.0)).min(axis=1)


@dataclass(frozen=True)
class FamiliarityCheck:
    means: np.ndarray  # (classes, features)
    precision: np.ndarray  # (features, features): the inverse of the shared covariance
    threshold: float  # distances aboce this count as unfamiliar

    @classmethod
    def load(cls, path: Path = FAMILIARITY_PATH) -> "FamiliarityCheck | None":
        """
        The check from disk or None if the file is missing or unreadable
        """
        try:
            with np.load(path) as data:
                check = cls(data["means"], data["precision"], float(data["threshold"]))

        except (OSError, KeyError, ValueError):
            return None
        if check.means.shape[1] != check.precision.shape[0]:
            return None
        return check

    def distance(self, features: np.ndarray) -> np.ndarray:
        return mahalanobis(np.atleast_2d(features), self.means, self.precision)

    def is_unfamiliar(self, features: np.ndarray) -> bool:
        """For one image's features."""
        return bool(self.distance(features)[0] > self.threshold)
