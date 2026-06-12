from pathlib import Path
from pydantic import BaseModel, Field, PositiveInt, PositiveFloat

class TrainConfig(BaseModel):
    # Paths
    dataset_dir: Path = Path("data")
    output_dir: Path = Path("models")
    
    # Hyperparameter
    seed: PositiveInt = 42
    batch_size: PositiveInt = 16
    epochs: PositiveInt = 20
    learning_rate: PositiveFloat = 1e-3
    patience: PositiveInt = 5
    freeze_backbone: bool = True
    num_workers: PositiveInt = 4
    
    #-- Derived paths -----------------
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
        
    @classmethod
    def from_yaml(cls, path:Path) -> "TrainConfig":
        import yaml
        return cls(**yaml.safe_load(path.read_text()))
    
    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent= 4))