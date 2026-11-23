"""
Datasets, loaders and image transforms
"""

import io

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from scripts.train_config import TrainConfig
from src.logger import logger

# Inference reads probs[0] as P(AI), so the class order must never change
EXPECTED_CLASSES = ["AI_GENERATED", "REAL"]

# ImageNet's average colour (mean) and spread (std) per RGB channel. MobileNetV3 was
# pretrained on inputs scaled this way, so ours must be too. inference.py and the
# GradCAM service use the same numbers; change them everywhere or nowhere.
_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class RandomJPEG:
    """
    Resaves a PIL image as JPEG at a random quality, with probability p.

    Many datatests store real photos as JPEG and AI images as PNG. A model can then learn
    "JPEG-compressed or not" instead of "AI or not", which scores well on the dataset and fails on the web,
    where AI images are reasaved as JPEG too. Recompressing traning images of both classes takes that shortcut away.

    A class, not a lambda, so DataLoader workers can pickle it. torch's random functions are used because each worker gets its
    own torch seed.
    """

    def __init__(self, p: float, quality: tuple[int, int]) -> None:
        self.p = p
        self.quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        if torch.rand(1).item() >= self.p:
            return img
        q = int(torch.randint(self.quality[0], self.quality[1] + 1, (1,)).item())
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def __repr__(self) -> str:
        return f"RandomJPEG(p={self.p}, quality={self.quality})"


def get_transforms(
    config: TrainConfig | None = None,
) -> tuple[transforms.Compose, transforms.Compose]:
    """
        Augmented transform for training, clean transform for validation and inference.

    No blur and no rotation: both blend neighbouring pixels, and the traces generators
    leave are exactly that fine detail. Whether light blur would still help against
    unseen generators is a question for the robustness check, not a guess.

    """
    config = config or TrainConfig()
    train_transform = transforms.Compose(
        [
            # Re-save as JPEG at full size, like images on the web, so "is it a JPEG?"
            # stops predicting the class (see TrainConfig.jpeg_prob)
            RandomJPEG(config.jpeg_prob, config.jpeg_quality),
            # MobileNet was built for 224x224; the API resizes uploads the same way
            transforms.Resize((224, 224)),
            # A mirrored image keeps its class; doubles the variety without changing pixels
            transforms.RandomHorizontalFlip(),
            # Small random changes, so exposure or colour mood can't decide the class
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            # PIL image (0-255) -> PyTorch tensor (0-1, channels first)
            transforms.ToTensor(),
            # Scale to what the pretrained weights expect
            _NORMALIZE,
        ]
    )

    val_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), _NORMALIZE]
    )
    return train_transform, val_transform


def get_dataloaders(config: TrainConfig) -> tuple[DataLoader, DataLoader, list[str]]:
    for path in (config.train_dir, config.val_dir):
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset directory: {path}")

        train_transform, val_transform = get_transforms(config)
        train_dataset = datasets.ImageFolder(
            config.train_dir, transform=train_transform
        )
        val_dataset = datasets.ImageFolder(config.val_dir, transform=val_transform)

        if (
            train_dataset.classes != EXPECTED_CLASSES
            or val_dataset.classes != EXPECTED_CLASSES
        ):
            raise ValueError(
                f"Expected {EXPECTED_CLASSES}, got train= {train_dataset.classes},"
                f"val={val_dataset.classes}"
            )

        loader_kwargs = {
            "batch_size": int(config.batch_size),
            "num_workers": int(config.num_workers),
            "pin_memory": torch.cuda.is_available(),
        }
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

        logger.info(
            f"Dataset ready: train {len(train_dataset)}, val {len(val_dataset)}, "
            f"classes {train_dataset.classes}"
        )
        return train_loader, val_loader, train_dataset.classes
