"""
ExplainabilityService
Generates GradCAM heatmaps using the PyTorch checkpoint.

Why PyTorch and not ONNX?
ONNX is a static computation graph — it doesn't expose the gradient hooks
that GradCAM needs. We keep the PyTorch checkpoint alongside the ONNX model
specifically for this purpose. Inference stays fast via ONNX; explanations
use PyTorch on-demand.
"""

import io
import base64
import time
import asyncio
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.logger import logger
from src.core.enums import PredictionLabel, InferenceStatus
from src.core.exceptions import ModelInferenceError
from src.services.base_engine import BaseMLEngine

# Must match inference.py exactly — same normalization, same model sees same numbers
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ExplainabilityService(BaseMLEngine):
    """
    Loads the PyTorch checkpoint and runs GradCAM on demand.

    Focused on the `model.features[-1]` which is the last convolutional
    block before the classifier head, where the most semantically rich spatial
    features live.
    """

    def __init__(self, checkpoint_path: str | Path, num_classes: int = 2) -> None:
        """
        Args:
            - checkpoint_path: Path to best_checkpoint.pt saved by train.py
            - num_classes:     Must match what the model was trained with (1 for binary)

        Raises:
            - FileNotFoundError: If checkpoint doesn't exist — caught at startup so
                               we know immediately, not on the first explain request.
        """
        self.num_classes = num_classes
        self.model = None
        self.target_layers = None
        super().__init__(checkpoint_path)

    def _load(self) -> None:
        """Load PyTorch checkppoint and configure GradCAM target layers"""
        assert self.model is not None
        self.model = models.mobilenet_v3_small(weights=None)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, self.num_classes)  # type: ignore

        checkpoint = torch.load(str(self.model_path), map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # last conv block
        self.target_layers = [self.model.features[-1]]  # type: ignore

    # ── Private ───────────────────────────────────────────────────────────────

    def _preprocess(self, image_bytes: bytes) -> tuple[torch.Tensor, np.ndarray]:
        """
        Returns two things from the same image:
        - normalized tensor  to  feed  the model
        - float [0,1] array  for the base image for the heatmap overlay

        They must come from the same resize operation so the overlay aligns.
        """
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))

        img_array = np.array(img, dtype=np.float32) / 255.0

        normalized = (img_array - _MEAN) / _STD
        tensor = (
            torch.from_numpy(
                np.transpose(normalized, (2, 0, 1))  # HWC → CHW
            )
            .unsqueeze(0)
            .float()
        )
        return tensor, img_array

    def _run_gradcam(self, image_bytes: bytes) -> tuple[float, str]:
        """
        Synchronous GradCAM execution — called via asyncio.to_thread.

        Returns:
            - confidence:   raw softmax score from the model
            - heatmap_b64:  base64-encoded PNG of the overlay
        """
        input_tensor, original_image = self._preprocess(image_bytes)

        # GradCAM as context manager. it ensures hooks are cleaned up after use
        # targets=None → GradCAM automatically targets the highest-scoring class
        with GradCAM(model=self.model, target_layers=self.target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]  # type: ignore

        overlay = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)

        with torch.no_grad():  # type: ignore
            assert self.model is not None
            output = self.model(input_tensor)
            probs = torch.softmax(output[0], dim=0)
            confidence = float(probs[0])

        buffer = io.BytesIO()
        Image.fromarray(overlay).save(buffer, format="PNG")
        heatmap_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return confidence, heatmap_b64

    # ── Public API ────────────────────────────────────────────────────────────

    async def explain(
        self,
        image_bytes: bytes,
        filename: str,
        decide_class_fn: Callable[[float], PredictionLabel],
    ) -> dict:
        """
        Async entry point
        Args:
            - image_bytes:     Raw bytes of the uploaded image
            - filename:        Safe filename (already sanitized by the route)
            - decide_class_fn: Passed in from InferenceService._decide_class so
                             the uncertainty boundary logic isn't duplicated here

        Returns:
            - dict matching ExplainResponse schema

        Raises:
            - ModelInferenceError: If GradCAM fails for any reason
        """
        start = time.time()

        try:
            confidence, heatmap_b64 = await asyncio.to_thread(
                self._run_gradcam, image_bytes
            )
        except Exception as e:
            logger.error(f"GradCAM failed for {filename}: {e}", exc_info=True)
            raise ModelInferenceError(f"Explainability pipeline failed: {e}")

        prediction = decide_class_fn(confidence)

        if "UNCERTAIN" in prediction.value:
            logger.warning(
                f"UNCERTAIN on explain: {filename} (score: {confidence:.4f})"
            )

        return {
            "filename": filename,
            "confidence": round(confidence, 4),
            "prediction": prediction,
            "status": InferenceStatus.SUCCESS,
            "heatmap_base64": heatmap_b64,
            "processing_time_ms": round((time.time() - start) * 1000, 2),
            "cached": False,
        }
