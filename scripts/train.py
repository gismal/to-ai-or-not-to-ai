"""
MLOps Image Classification Training Pipeline

MobileNetV3-Small, trained in two phases, then exported to ONNX

This file wires the live pieces in scripts/training folder. Loads the
data, runs each phase of the plan, export the best model and log to MLflow

    PYTHONPATH=. python scripts/train.py
"""

import os

import torch

from scripts.train_config import TrainConfig
from scripts.training.data import get_dataloaders, get_transforms
from scripts.training.export import export_model
from scripts.training.model import build_model
from scripts.training.phases import PhaseResult, training_plan
from scripts.training.tracking import mlflow
from scripts.training.trainer import Trainer
from src.logger import logger

__all__ = ["build_model", "get_transforms", "main"]


def main() -> None:
    """
    Runs the whole pipeline.
    Exits with code 1 if there's missing dataset, so
    a notebook or CI step never continues with a stale model
    """
    logger.info("MLOps training pipeline")
    config = TrainConfig()

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
    mlflow.set_experiment("ai_detector")

    try:
        train_loader, val_loader, classes = get_dataloaders(config)

        with mlflow.start_run():
            mlflow.log_params(
                {
                    **config.to_mlflow_params(),
                    "base_model": "MobileNetV3-Small",
                    "optimizer": "Adam",
                    "scheduler": "CosineAnnealingLR",
                    "classes": str(classes),
                }
            )

            model = build_model(len(classes), config.freeze_backbone).to(device)
            trainer = Trainer(model, device, config.checkpoint_path)
            results: list[PhaseResult] = [
                trainer.run_phase(phase, train_loader, val_loader)
                for phase in training_plan(config)
            ]

            assert trainer.best is not None and trainer.best_phase is not None
            export_model(
                model, config, trainer.best, trainer.best_phase.number, classes, results
            )

            for r in results:
                if r.best:
                    mlflow.log_metrics(
                        {
                            f"phase{r.phase.number}_best_{k}": v
                            for k, v in r.best.rounded().items()
                        }
                    )
            mlflow.log_metrics(
                {f"best_{k}": v for k, v in trainer.best.rounded().items()}
            )
            for path in (
                config.onnx_path,
                config.checkpoint_path,
                config.metadata_path,
            ):
                mlflow.log_artifact(str(path), artifact_path="models")

        baseline = results[0].best
        logger.info(
            f"Pipeline complete. Phase 1 best F1 {baseline.f1_score if baseline else 0:.4f}, "
            f"exported model (phase {trainer.best_phase.number}) F1 {trainer.best.f1_score:.4f}"
        )

    except FileNotFoundError as e:
        logger.warning(f"Aborted: {e}. Prepare the dataset and retry.")
        raise SystemExit(1) from e
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
