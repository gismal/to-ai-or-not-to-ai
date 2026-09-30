"""
The inference pipeline
"""

import concurrent.futures
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from src.config import settings
from src.core.enums import InferenceStatus
from src.logger import logger
from src.services.base_engine import BaseMLEngine

# global mean and standart deviation values for further calculations
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class InferenceResult:
    """
    Data class representing the result of an image inference operation

    Attributes:
        - image (str): The absolute or relative path to the processed image
        - confidence (float): The model's confidence score, ranging from 0.0 to 1.0
        - detected (bool): True if the confidence score meets or exceeds the threshold
        - status (InferenceStatus): Inference status 'SUCCESS' or 'FAIL'
        - error (str | None): Exception message if an error occurred, otherwise None
    """

    image: str
    confidence: float = 0.0
    detected: bool = False
    status: InferenceStatus = InferenceStatus.SUCCESS
    error: str | None = None


def _preprocess(image_source):
    """
    Preprocess the images for healthier inference
    """

    try:
        with Image.open(image_source) as img:
            img = img.convert("RGB").resize((224, 224))

            img_data = np.array(img, dtype=np.float32) / 255.0
            img_data = (img_data - _MEAN) / _STD
            img_data = np.transpose(img_data, (2, 0, 1))

            return str(image_source), np.expand_dims(img_data, axis=0), None
    except Exception as e:
        return str(image_source), None, str(e)


class ONNXPredictor(BaseMLEngine):
    """
    This class handles:
        the initialization of the ONNX runtime session,
        preprocessing of input images (resizing, normalization, tensor formatting)
        the execution of the model
    to generate predictions
    """

    def __init__(self, model_path: str | Path = "models/model_v1.onnx") -> None:
        """
        Initializes the ONNX inference session and loads model configuration

        Args:
            - model (str | Path): The path to the serialized ONNX model file

        Raises:
            - FileNotFoundError: If the specified ONNX model file does not exist

        """
        self.threshold = settings.MODEL_THRESHOLD
        self._session: ort.InferenceSession | None = None
        self._input_name: str | None = None
        super().__init__(model_path)

    def _load(self) -> None:
        """
        Loads ONNX session, configure providers, warm up, validate output
        """

        # session settings
        session_options = ort.SessionOptions()
        # ORT enables all to maximize model execution speed
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        available = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )

        logger.info(f"Using providers: {providers}")

        self._session = ort.InferenceSession(
            str(self.model_path), sess_options=session_options, providers=providers
        )
        self._input_name = self._session.get_inputs()[0].name

        # warming up the session, no latency in the first user request
        logger.info("Warming up ONNX session")
        # preparing the dummy inference
        dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
        output = self._session.run(None, {self._input_name: dummy})  # type: ignore
        # prediction matrix
        preds = np.asarray(output[0])

        # sanity check
        # shape control since we have only 2 outputs
        # because of that predictions shape[1] should be 2
        # if no raise a ValueError
        if preds.shape[1] != 2:
            raise ValueError(
                f"Unexpected model output: {preds.shape}. "
                "Expected 2 classes (AI_GENERATED, REAL)."
            )
        logger.info("ONNX warmup validated, output shape confirmed.")

    def predict(
        self, image_source: str | Path | io.BytesIO, filename: str = "image"
    ) -> InferenceResult:
        """
        Executes the full inference pipeline on a given image

        Args:
        - image_source (str | Path | io.BytesIO): Image as a file path or byte stream
        - filename (str): Label for this image in the result

        Returns:
            - InferenceResult: A structured data containing prediction outcomes, confidence scores and status flags

        """
        # first, check the session
        if self._session is None or self._input_name is None:
            raise RuntimeError(
                "ONNXPredictor is not loaded. Call _load() before predict()"
            )

        _, input_data, error = _preprocess(image_source)

        if error:
            return InferenceResult(
                image=filename, status=InferenceStatus.FAILED, error=error
            )

        try:
            # run the session with the preprocessed inputs that we've provided
            outputs = self._session.run(None, {self._input_name: input_data})
            preds = np.asarray(outputs[0])
            # taking the raw scores
            logits = preds[0]

            # applying softmax to bring the scores in the range between 0 to 1
            exp = np.exp(logits - logits.max())
            probs = exp / exp.sum()
            confidence_score = float(probs[0])

            return InferenceResult(
                image=filename,
                confidence=round(confidence_score, 4),
                detected=confidence_score >= self.threshold,
            )
        except Exception as e:
            return InferenceResult(
                image=filename, status=InferenceStatus.FAILED, error=str(e)
            )

    def predict_batch(
        self, image_paths: list[str | Path], max_workers: int = 4, max_batch: int = 32
    ) -> list[InferenceResult]:
        """
        Batch Inference using multi-threading for preprocessing and vectorized batch execution
        for higher throughput

        Args:
            - image_path list(str | Path): list of paths to the image files to be analyzed
            - max_workers (int): Maximum concurrent threads for parallel image preprocessing
            - max_batch (int): Maximum allowable images per batch to prevent OOM.
        Returns:
            - list[InferenceResult]: A list containing the prediction outcomes, confidence scores and status flags
        """
        if self._session is None or self._input_name is None:
            raise RuntimeError(
                "ONNXPredictor is not loaded — call _load() before predict()"
            )

        # OOM Guard: Prevent memory overload by capping batch size
        original_count = len(image_paths)
        if original_count > max_batch:
            image_paths = image_paths[:max_batch]
            logger.warning(f"Truncating from {original_count} to {max_batch}")

        results, valid_batch = [], []

        # Preprocess images in parallel to avoid CPU bottleneck
        # I choose ThreadPoolExecutor because it's lightweight. It has no pickling costs, no overhead
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for path_str, tensor, error in executor.map(_preprocess, image_paths):
                if error:
                    results.append(
                        InferenceResult(
                            image=path_str, status=InferenceStatus.FAILED, error=error
                        )
                    )
                else:
                    valid_batch.append((path_str, tensor))

        if not valid_batch:
            return results

        valid_paths, valid_tensor = zip(*valid_batch)
        try:
            # Stacking individual tensors into a single batch matrix.
            # Thanks to this model will process them as a whole not each by one
            batch_data = np.stack([t.squeeze(0) for t in valid_tensor], axis=0)
            outputs = self._session.run(None, {self._input_name: batch_data})  # type: ignore

            batch_logits = np.asarray(outputs[0])

            # Softmax loop to get overall confidence score
            for path_str, logits in zip(valid_paths, batch_logits):
                exp = np.exp(logits - np.max(logits))
                probs = exp / np.sum(exp)
                confidence_score = float(probs[0])

                results.append(
                    InferenceResult(
                        image=path_str,
                        confidence=round(confidence_score, 4),
                        detected=confidence_score >= self.threshold,
                    )
                )
        except Exception as e:
            results.extend(
                [
                    InferenceResult(
                        image=p, status=InferenceStatus.FAILED, error=str(e)
                    )
                    for p in valid_paths
                ]
            )

        return results
