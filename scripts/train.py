"""
MLOps Image Classification Training Pipeline
Clean, production ready transfer learning wirh MobileNetV3-Small
"""

import copy
import json
import os
from datetime import datetime, timezone

import mlflow
import mlflow.pytorch
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from scripts.train_config import TrainConfig
from src.config import settings
from src.logger import logger

# -- Data ---------------


def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """
    Augmented transform for training
    Clean transform for validation
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return train_transform, val_transform


def get_dataloaders(config: TrainConfig) -> tuple[DataLoader, DataLoader, list[str]]:
    """Load ImageFolder datasets and return DataLoaders + class names"""
    for path in (config.train_dir, config.val_dir):
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset directory: {path}")

    train_transform, val_transform = get_transforms()

    train_dataset = datasets.ImageFolder(config.train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(config.val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        pin_memory=True,
    )

    logger.info(
        f"Dataset ready — train: {len(train_dataset)}, "
        f"val: {len(val_dataset)}, classes: {train_dataset.classes}"
    )
    return train_loader, val_loader, train_dataset.classes


# -- Model ------
def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """Load pretrained MobileNetv3-Small and swap the classifier head"""
    model = models.mobilenet_v3_small(weights="DEFAULT")

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    last_layer = model.classifier[-1]
    assert isinstance(last_layer, nn.Linear)

    in_features = last_layer.in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Model ready. Trainable params: {trainable:,} / {total:,}")

    return model


# -- Early Stopping -----
class EarlyStopping:
    """Halts training when validation loss stops improving"""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# -- Trainer  ----
class Trainer:
    """Owns the full training loop, checkpointing and early stopping"""

    def __init__(
        self, model: nn.Module, config: TrainConfig, device: torch.device
    ) -> None:
        self.model = model
        self.config = config
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )
        self.early_stopping = EarlyStopping(patience=config.patience)

        self._best_val_loss = float("inf")
        self._best_weights: dict = {}
        self._best_metrics: dict[str, float] = {}

    # -- Private Helpers ----
    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for inputs, labels in tqdm(loader, leave=False):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(inputs), labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def _validate_epoch(self, loader: DataLoader) -> tuple[float, dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for inputs, labels in loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)

            total_loss += self.criterion(outputs, labels).item()
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        metrics = {
            "accuracy": float(accuracy_score(all_labels, all_preds)),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }
        return total_loss / len(loader), metrics

    def _save_checkpoint(self, metrics: dict[str, float]) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
            },
            self.config.checkpoint_path,
        )

    # -- Public API ------

    def run(self, train_loader: DataLoader, val_loader: DataLoader) -> dict[str, float]:
        """Run the full training loop and return the best validation metrics"""
        logger.info(f"Training on {self.device} for up to {self.config.epochs}")

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss, metrics = self._validate_epoch(val_loader)
            self.scheduler.step()

            logger.info(
                f"Epoch {epoch:>3}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"F1: {metrics['f1_score']:.4f} | Acc: {metrics['accuracy']:.4f}"
            )

            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._best_weights = copy.deepcopy(self.model.state_dict())
                self._best_metrics = metrics
                self._save_checkpoint(metrics)
                logger.info("  ✓ New best model saved to disk ")

            if self.early_stopping.step(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        self.model.load_state_dict(self._best_weights)
        return self._best_metrics


# -- Export --------
def export_model(
    model: nn.Module,
    config: TrainConfig,
    metrics: dict[str, float],
    classes: list[str],
) -> None:
    """Export to ONNX + write metadata. Uses a deep copy — original model untouched."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    export_copy = copy.deepcopy(model).cpu().eval()
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        export_copy,
        (dummy_input,),
        config.onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    logger.info(f"ONNX model exported {config.onnx_path}")

    metadata = {
        "model_version": "v2.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "base_model": "MobileNetV3_Small",
        "classes": classes,
        "epochs_trained": config.epochs,
        "batch_size": config.batch_size,
        "threshold": settings.MODEL_THRESHOLD,
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
    }
    config.metadata_path.write_text(json.dumps(metadata, indent=4))
    logger.info(f"Metadata saved: {config.metadata_path}")


# -- Entry Point -------
def main() -> None:
    logger.info("MLOps Training Pipeline")
    config = TrainConfig()

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("ai-detector")

    try:
        train_loader, val_loader, classes = get_dataloaders(config)

        with mlflow.start_run():
            mlflow.log_params(
                {
                    "epochs": config.epochs,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "freeze_backbone": config.freeze_backbone,
                    "patience": config.patience,
                    "num_classes": len(classes),
                    "classes": str(classes),
                    "base_model": "MobileNetV3-Small",
                    "optimizer": "Adam",
                    "scheduler": "CosineAnnealingLR",
                }
            )

            model = build_model(len(classes), config.freeze_backbone).to(device)
            trainer = Trainer(model, config, device)

            best_metrics = trainer.run(train_loader, val_loader)

            # log final metrics
            mlflow.log_metrics({k: round(v, 4) for k, v in best_metrics.items()})

            export_model(model, config, best_metrics, classes)

            mlflow.log_artifact(str(config.onnx_path), artifact_path="models")
            mlflow.log_artifact(str(config.checkpoint_path), artifact_path="models")
            mlflow.log_artifact(str(config.metadata_path), artifact_path="models")

            mlflow.pytorch.log_model(model, "pytorch_model")

            logger.info("Pipeline complete.")
            logger.info(
                f"Best metrics: { {k: round(v, 4) for k, v in best_metrics.items()} }"
            )

    except FileNotFoundError as e:
        logger.warning(f"Aborted: {e}. Prepare the dataset and retry.")
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
