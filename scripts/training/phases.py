"""
The training plan and the small value types the trainer passes around
I keep this module cheap to test, no torch no sklearn
"""

from dataclasses import asdict, dataclass

from scripts.train_config import TrainConfig


@dataclass(frozen=True)
class Phase:
    """
    One stage of training.
    How long, how fast and whether the backbone learns
    """

    number: int
    name: str
    epochs: int
    lr: float
    patience: int
    train_backbone: bool


def training_plan(config: TrainConfig) -> list[Phase]:
    """
    1. Head: only the new classifier learns, so the random head cannot mess up
    the pretrained features
    2. Fine-tune: the whole network learns at a smaller learning rate, so backbone
    can pick up the low-level traces generators leave behind
    """
    return [
        Phase(
            1,
            "head",
            config.epochs,
            config.learning_rate,
            config.patience,
            train_backbone=not config.freeze_backbone,
        ),
        Phase(
            2,
            "finetune",
            config.finetune_epochs,
            config.finetune_lr,
            config.finetune_patience,
            train_backbone=True,
        ),
    ]


@dataclass(frozen=True)
class EvalResult:
    """
    Validation scores for one epoch. Precision, recall and F1 are class-weighted
    """

    loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float

    def rounded(self, digits: int = 4) -> dict[str, float]:
        return {k: round(v, digits) for k, v in asdict(self).items()}


@dataclass
class PhaseResult:
    phase: Phase
    epochs_run: int
    best: EvalResult | None  # None only if the phase ran 0 epochs

    def summary(self) -> dict:
        return {
            "name": self.phase.name,
            "epochs_run": self.epochs_run,
            "lr": self.phase.lr,
            "train_backbone": self.phase.train_backbone,
            "best_val": self.best.rounded() if self.best else None,
        }


class EarlyStopping:
    """
    Stops a phase when validation loss has not improved for `patience´ epochs
    """

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
