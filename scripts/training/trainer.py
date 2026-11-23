"""The training loop: runs phases and keeps the best weights across all of them."""

from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.training.model import set_backbone_trainable
from scripts.training.phases import EarlyStopping, EvalResult, Phase, PhaseResult
from scripts.training.tracking import mlflow
from src.logger import logger


class Trainer:
    """
    Runs training phases one after another.

    Everything that belongs to a single phase (optimizer, schedule, early stopping,
    the epoch history) lives inside run_phase. Only the overall best result is kept
    across phases, and the checkpoint on disk is its single copy: it is written when
    a result beats every earlier one, and loaded back when a phase ends. So a later
    phase only replaces the model if it really does better.
    """

    def __init__(
        self, model: nn.Module, device: torch.device, checkpoint_path: Path
    ) -> None:
        self.model = model
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.criterion = nn.CrossEntropyLoss()

        self.best: EvalResult | None = None
        self.best_phase: Phase | None = None

    # -- Public API ----

    def run_phase(
        self, phase: Phase, train_loader: DataLoader, val_loader: DataLoader
    ) -> PhaseResult:
        """Trains one phase. The model ends it holding the best weights so far."""
        set_backbone_trainable(self.model, phase.train_backbone)
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=phase.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase.epochs)
        stopper = EarlyStopping(phase.patience)
        history: list[EvalResult] = []

        logger.info(
            f"Phase {phase.number} ({phase.name}): up to {phase.epochs} epochs, "
            f"lr {phase.lr}, {sum(p.numel() for p in params):,} trainable parameters"
        )
        for epoch in range(1, phase.epochs + 1):
            train_loss = self._train_epoch(train_loader, optimizer)
            result = self._evaluate(val_loader)
            scheduler.step()
            history.append(result)

            self._log_epoch(phase, epoch, train_loss, result)
            self._keep_if_best(result, phase)
            if stopper.step(result.loss):
                logger.info(f"Phase {phase.number}: early stopping at epoch {epoch}")
                break

        self._restore_best()
        return PhaseResult(
            phase=phase,
            epochs_run=len(history),
            best=min(history, key=lambda r: r.loss, default=None),
        )

    # -- One epoch ----

    def _train_epoch(self, loader: DataLoader, optimizer: optim.Optimizer) -> float:
        self.model.train()
        total_loss = 0.0
        for inputs, labels in tqdm(loader, leave=False):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            loss = self.criterion(self.model(inputs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.inference_mode()
    def _evaluate(self, loader: DataLoader) -> EvalResult:
        self.model.eval()
        total_loss = 0.0
        preds, targets = [], []
        for inputs, labels in loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            total_loss += self.criterion(outputs, labels).item()
            preds.append(outputs.argmax(dim=1).cpu())
            targets.append(labels.cpu())

        y_pred, y_true = torch.cat(preds).numpy(), torch.cat(targets).numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        return EvalResult(
            loss=total_loss / len(loader),
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
        )

    def _log_epoch(
        self, phase: Phase, epoch: int, train_loss: float, result: EvalResult
    ) -> None:
        logger.info(
            f"Phase {phase.number} epoch {epoch:>3}/{phase.epochs} | "
            f"train loss {train_loss:.4f} | val loss {result.loss:.4f} | "
            f"F1 {result.f1_score:.4f} | acc {result.accuracy:.4f}"
        )
        mlflow.log_metrics(
            {
                f"phase{phase.number}_val_loss": result.loss,
                f"phase{phase.number}_val_f1": result.f1_score,
            },
            step=epoch,
        )

    # -- The best model across phases ----

    def _keep_if_best(self, result: EvalResult, phase: Phase) -> None:
        """Writes the checkpoint when this result beats every earlier one, in any phase."""
        if self.best is not None and result.loss >= self.best.loss:
            return
        self.best, self.best_phase = result, phase
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        # Only the weights: nothing resumes training from this file, and the API's
        # GradCAM service and the notebook read model_state_dict alone.
        torch.save(
            {"model_state_dict": self.model.state_dict(), "metrics": result.rounded()},
            self.checkpoint_path,
        )
        logger.info("  New best model saved")

    def _restore_best(self) -> None:
        """Puts the best weights so far back into the model."""
        if self.best is None:
            return
        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
