"""
Quantizes the trained FP32 ONNX model to INT8

It converts the model's weigths from 32-bit floats to 8-bit integers.
The activations are quantized at runtime. Thanks to this:
    - ~4x smaller model file
    - ~2-3x faster CPU inference
    - Typically < 1% accuracy loss

Run after training, before deploying
"""

import argparse
import json
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

from src.logger import logger


def quantize(input_path: Path, output_path: Path) -> None:
    """
    Applies dynamic INT8 quantization to MobileNetV3 ONNX model
    Since it is dynamic no need for calibration dataset
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input model not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Quantizing {input_path} -> {output_path}")
    logger.info("Method: dynamic INT8 quantization")

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )

    original_mb = input_path.stat().st_size / 1_048_576
    quantized_mb = output_path.stat().st_size / 1_048_576
    reduction_pct = (1 - quantized_mb / original_mb) * 100

    logger.info(
        f"Done. "
        f"Original: {original_mb:.1f} MB → "
        f"Quantized: {quantized_mb:.1f} MB "
        f"({reduction_pct:.0f}% reduction)"
    )

    metadata_path = input_path.parent / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        metadata["quantized_model"] = str(output_path)
        metadata["quantization"] = {
            "method": "dynamic_int8",
            "tool": "onnxruntime.quantization",
            "original_mb": round(original_mb, 2),
            "quantized_mb": round(quantized_mb, 2),
            "size_reduction_pct": round(reduction_pct, 1),
        }
        metadata_path.write_text(json.dumps(metadata, indent=4))
        logger.info(f"Metadata updated: {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize ONNX model to INT8")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("models/model_v1.onnx"),
        help="Path to the FP32 ONNX model",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/model_v1_int8.onnx"),
        help="Output path for the INT8 model",
    )
    args = parser.parse_args()

    quantize(args.input, args.output)


if __name__ == "__main__":
    main()
