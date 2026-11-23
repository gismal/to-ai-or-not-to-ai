from pathlib import Path

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, model_validator


class TrainConfig(BaseModel):
    # Paths
    dataset_dir: Path = Path("data")
    output_dir: Path = Path("models")
    model_version: str = "v2.0.0"

    # Phase 1: train the new classifier head (backbone frozen)
    seed: PositiveInt = 42
    batch_size: PositiveInt = 16
    epochs: PositiveInt = 20
    learning_rate: PositiveFloat = 1e-3
    patience: PositiveInt = 5
    freeze_backbone: bool = True
    num_workers: PositiveInt = 4

    # Phase 2: fine-tune the whole network at a smaller learning rate.
    # The epochs are a ceiling. Early stopping cuts the phase when validation stops improving.
    finetune_epochs: PositiveInt = 10
    finetune_lr: PositiveFloat = 1e-4
    finetune_patience: PositiveInt = 3

    # JPEG re-compression of training images, both classes. Many datasets store real photos
    # Most of the AI images are PNG but they can be JPEG as well. Model can come to the conclusion, JPEG = real.
    # To force the model to  learn the generator's own fingerprints.
    # Test and validation sets are untouched
    # 1.0 because the face data is 98% PNG for AI and 71% JPEG for real (see the format check)
    jpeg_prob: float = Field(default=1.0, ge=0.0, le=1.0)
    jpeg_quality: tuple[int, int] = (60, 95)

    @model_validator(mode="after")
    def _check_jpeg_quality(self) -> "TrainConfig":
        low, high = self.jpeg_quality
        if not 1 <= low <= high <= 100:
            raise ValueError(
                f"jpeg_quality must be (low, high) within 1..100, got {self.jpeg_quality}"
            )
        return self

    # -- Derived paths -----------------
    @property
    def train_dir(self) -> Path:
        return self.dataset_dir / "train"

    @property
    def val_dir(self) -> Path:
        return self.dataset_dir / "val"

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / "best_checkpoint.pt"

    @property
    def onnx_path(self) -> Path:
        return self.output_dir / "model_v1.onnx"

    @property
    def metadata_path(self) -> Path:
        return self.output_dir / "metadata.json"

    # -- Serialization -----------------------------
    @classmethod
    def from_yaml(cls, path: Path) -> "TrainConfig":
        import yaml

        return cls(**yaml.safe_load(path.read_text()))

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=4))

    def to_mlflow_params(self) -> dict:
        return {
            "epochs": self.epochs,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            # phase 1
            "phase1_epochs": self.epochs,
            "phase1_lr": self.learning_rate,
            "phase1_patience": self.patience,
            "phase1_frozen_backbone": self.freeze_backbone,
            # pahse 2
            "phase2_epochs": self.finetune_epochs,
            "phase2_lr": self.finetune_lr,
            "phase2_patience": self.finetune_patience,
            # JPEG compression
            "jpeg_prob": self.jpeg_prob,
            "jpeg_quality": str(self.jpeg_quality),
        }
