"""
Calibrates model confidence score Temperature Scaling.

What this solves:
Neural networks are systematically overconfident — a raw output of 0.9
doesn't mean the model is right 90% of the time. Temperature Scaling
fixes this with a single learnable parameter T.

How it works:
  calibrated_probs = softmax(logits / T)

T is found by minimizing NLL on the validation set via scalar optimization.
T > 1 → model was overconfident, now spread out
T < 1 → model was underconfident, now sharpened (rare)
T = 1 → no change needed (already well-calibrated)
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy.optimize import minimize_scalar
from sklearn.metrics import brier_score_loss

from src.logger import logger

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image_path: Path) -> np.ndarray:
    """Preprocesses a single image"""
    with Image.open(image_path) as img:
        img = img.convert("RGB").resize((224, 224))
        data = np.array(img, dtype=np.float32) / 255.0
        data = (data - _MEAN) / _STD
        data = np.transpose(data, (2, 0, 1))
        return np.expand_dims(data, axis=0)


# -- Temperature Scaling --------------------------
def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis"""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def nll_loss(T: float, logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Negative log-likelihood of true labels under the temperature scaled predictions
    Objective: minimize to find the optimal T
    """
    scaled = logits / T
    probs = softmax(scaled)
    true_probs = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(true_probs + 1e-8)))


def find_optimal_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """
        Finds T* = argmin NLL(logits/T, labels) via bounded scalar optimization.

    Search range (0.1, 10.0) covers all practical cases:
    - T ≈ 0.1 → extreme sharpening (very rare)
    - T ≈ 10  → near-uniform distribution (extreme overconfidence)
    """
    result = minimize_scalar(
        nll_loss, bounds=(0.1, 10.0), method="bounded", args=(logits, labels)
    )
    return float(result.x)


# -- Data collection ------------------------------------
def collect_logits(
    session: ort.InferenceSession, data_dir: Path, classes: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs the entire validation set through the model.
    Returns raw logits (before softmax) and integer true labels.
    """
    input_name = session.get_inputs()[0].name
    ai_class_idx = classes.index("AI_GENERATED")  # noqa: F841
    all_logits, all_labels = [], []

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        is_ai = class_dir.name == "AI_GENERATED"

        images = list(class_dir.glob("*jpg")) + list(class_dir.glob("*png"))
        logger.info(f"  {class_dir.name}: {len(images)} images")

        for img_path in images:
            try:
                tensor = preprocess(img_path)
                outputs = session.run(None, {input_name: tensor})
                all_logits.append(outputs[0][0])  # type: ignore
                all_labels.append(int(is_ai))
            except Exception as e:
                logger.warning(f"Skipping {img_path}: {e}")

    return np.array(all_logits), np.array(all_labels)


# -- Calibration Pipeline ------------------------------
def calibrate(
    model_path: Path,
    data_dir: Path,
    output_path: Path,
) -> None:
    """
    Calibration pipeline
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Validation data not found: {data_dir}")

    # -- Load ONNX Session ------------------------
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model_path), sess_options=opts)
    logger.info(f"Loaded model: {model_path}")

    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    logger.info(f"Classes: {classes}")

    # -- Collect raw outputs ----------------
    logger.info(f"Running validation set: {data_dir}")
    logits, labels = collect_logits(session, data_dir, classes)
    logger.info(f"Collected {len(logits)} samples")

    if len(logits) < 50:
        logger.warning(
            "Fewer than 50 validation samples. Calibration may be unreliable. "
            "Use a larger validation set for production calibration."
        )

    # -- Find optimal temperature -----------------------
    T_optimal = find_optimal_temperature(logits, labels)
    logger.info(f"Optimal temperature T = {T_optimal:.4f}")

    if T_optimal > 1.0:
        logger.info("T > 1: model was overconfident, confidence scores softened")
    elif T_optimal < 1.0:
        logger.info("T < 1: model was underconfident, confidence scores sharpened")
    else:
        logger.info("T ≈ 1: model is already well-calibrated")

    ai_class_idx = classes.index("AI_GENERATED")

    raw_probs = softmax(logits)[:, ai_class_idx]
    calibrated_probs = softmax(logits / T_optimal)[:, ai_class_idx]

    # Brier score: lower is the better (0 is perfect, 0.25 is useless)
    brier_before = brier_score_loss(labels, raw_probs)
    brier_after = brier_score_loss(labels, calibrated_probs)

    logger.info(f"Brier score before calibration: {brier_before:.4f}")
    logger.info(f"Brier score after  calibration: {brier_after:.4f}")
    logger.info(
        f"Improvement: {(brier_before - brier_after) / brier_before * 100:.1f}%"
    )

    # -- Save ----------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)

    calibrator = {
        "method": "temperature_scaling",
        "temperature": T_optimal,
        "classes": classes,
        "ai_class_idx": ai_class_idx,
    }

    joblib.dump(calibrator, output_path)
    logger.info(f"Calibrator saved: {output_path}")

    # -- Update metadata.json ----------------------
    metadata_path = model_path.parent / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        metadata["calibration"] = {
            "calibrator_path": str(output_path),
            "method": "temperature_scaling",
            "n_samples": len(logits),
            "brier_before": round(brier_before, 4),
            "brier_after": round(brier_after, 4),
        }
        metadata_path.write_text(json.dumps(metadata, indent=4))
        logger.info(f"Metadata updated: {metadata_path}")

    logger.info(
        "\nTo apply calibration in inference, load the calibrator with joblib.load()\n"
        "then apply: calibrated_conf = softmax(raw_logits / calibrator['temperature'])[ai_class_idx]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate ONNX model confidence scores"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/model_v1.onnx"),
        help="Path to the ONNX model",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/val"),
        help="Path to validation data directory (ImageFolder layout)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/calibrator.pkl"),
        help="Path to save the calibrator",
    )
    args = parser.parse_args()

    calibrate(args.model, args.data, args.output)


if __name__ == "__main__":
    main()
