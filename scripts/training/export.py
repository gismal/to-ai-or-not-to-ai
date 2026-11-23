"""
ONNX export and metadata
"""

import copy
import json
from datetime import UTC, datetime

import torch
from torch import nn

from scripts.train_config import TrainConfig
from scripts.training.phases import EvalResult, PhaseResult
from src.config import settings
from src.logger import logger


def export_model(
    model: nn.Module,
    config: TrainConfig,
    best: EvalResult,
    best_phase: int,
    classes: list[str],
    phases: list[PhaseResult],
) -> None:
    """
    Exports to ONNX with a dynamic batch size and writes metadata- The model
    is not changed
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    export_copy = copy.deepcopy(model).cpu().eval()

    torch.onnx.export(
        export_copy,
        (torch.randn(1, 3, 224, 224),),
        config.onnx_path,
        export_params=True,
        # 18: what recent PyTorch exporters produce natively; asking for older opsets
        # makes them attempt a down-conversion that can fail. Needs onnxruntime >= 1.14.
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    logger.info(f"ONNX model exported to {config.onnx_path}")

    metadata = {
        "model_version": config.model_version,
        "trained_at": datetime.now(UTC).isoformat(),
        "base_model": "MobileNetV3_Small",
        "classes": classes,
        "batch_size": config.batch_size,
        "threshold": settings.MODEL_THRESHOLD,
        "training": {
            **{f"phase{r.phase.number}": r.summary() for r in phases},
            "exported_from_phase": best_phase,
            "jpeg_augmentation": {
                "p": config.jpeg_prob,
                "quality": list(config.jpeg_quality),
            },
        },
        "metrics": best.rounded(),
    }
    config.metadata_path.write_text(json.dumps(metadata, indent=4))
    logger.info(f"Metadata saved to {config.metadata_path}")
