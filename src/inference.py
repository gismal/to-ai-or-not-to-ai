import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
import concurrent.futures
from dataclasses import dataclass
from src.config import settings
from src.logger import logger
from src.schemas.predict import InferenceStatus

@dataclass
class InferenceResult:
    """
    Data class representing the result of an image inference operation
    
    Attributes:
        - image (str): The absolute or relative path to the processed image
        - confidence (float): The model's confidence score, ranging from 0.0 to 1.0
        - detected (bool): True if the confidence score meets or exceeds the threshold
        - status (InferenceStatus): Operational status 'SUCCESS' or FAIL
        - error (str | None): Exception message if an error occurred, otherwise None
    """
    image: str
    confidence: float = 0.0
    detected: bool = False
    status: InferenceStatus = InferenceStatus.SUCCESS
    error: str | None = None
    
class ONNXPredictor:
    """
    This class handles the initialization of the ONNX runtime session, preprocessing of input images (resizing, normalization, tensor formatting) and the execution of the model
    to generate predictions    
    """
    TARGET_SIZE = (224, 224)
    
    def __init__(self, model_path: str | Path= "models/model_v1.onnx"):
        """
        Initializes the ONNX inference session and loads model configuration
        
        Args:
            - model (str | Path): The path to the serialized ONNX model file
        
        Raises: 
            - FileNotFoundError: If the specified ONNX model file does not exist
        
        """
        
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"No model found at: {self.model_path}")
        
        # Dynamic Hardware 
        available_providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            logger.info("GPU detected. Using CUDAExecutionProvider")
        else:
            providers = ['CPUExecutionProvider']
            logger.info("GPU not found. Falling back to CPUExecutionProvider")
                
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers = providers)
        self.threshold = settings.MODEL_THRESHOLD
        self.threshold = settings.MODEL_THRESHOLD
        self.input_name = self.session.get_inputs()[0].name
        
    def _preprocess(self, image_path: Path) -> np.ndarray:
        """ 
        Reads and preprocesses the image for the neural network
        
        The processing pipeline consists of:
        1. Load the image and converting it to the RGB color space
        2. Resizing the image to the model's target dimensions (224x224)
        3. Normalizing the pixel values from [0, 255] to [0.0, 1.0.]
        4. Transposing the matrix from HWC (Height, Width, Channels) to CHW form
        5. Expanding dimensions to include a batch axis, resulting in (1, 3, 224, 224)
        
        Args:
            - image_path (Path): Path object pointing to the target image file
        
        Returns: 
            - np.ndarray: A NumPy array formatted as a float32 tensor ready for inference
        
        Raises:
            - ValueError: If the image cannot be opened, ready or processed
        """
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB').resize(self.TARGET_SIZE) 
                
                # Normalization and format conversion (HWC -> CHW)
                img_data = np.array(img, dtype = np.float32) / 255.0
                img_data = np.transpose(img_data, (2, 0, 1))
                
                return np.expand_dims(img_data, axis = 0)
        except Exception as e:
            raise ValueError(f"Image cannot be preprocessed") from e

    def predict(self, image_path: str | Path) -> InferenceResult:
        """ 
        Executes the full inference pipeline on a given image
        
        Args:
            - image_path (str | Path): The path to the image file to be analyzed
            
        Returns:
            - InferenceResult: A structured data containing prediction outcomes, confidence scores and status flags
        """
        path_obj = Path(image_path)
        
        try:
            input_data = self._preprocess(path_obj)
            outputs = self.session.run(None, {self.input_name: input_data})
            
            confidence_score = float(outputs[0][0][0])
            is_detected = confidence_score >= self.threshold
            
            return InferenceResult(
                image = str(path_obj),
                confidence = round(confidence_score, 4),
                detected = is_detected
            )
        except Exception as e:
            return InferenceResult(
                image = str(path_obj),
                status = InferenceStatus.FAILED,
                error = str(e)
            )
        
    def predict_batch(self, image_paths: list[str | Path], max_workers: int = 4, max_batch: int = 32) -> list[InferenceResult]:
        """ 
        Batch Inference  using thread pools for max I/O performance
        Process multiple images simultaneously for higher throughput
        
        Args:
            - image_path list(str | Path): The path to the image file to be analyzed
            - max_workers (int): Maximum concurrent threads for I/O operations.
            - max_batch (int): Maximum allowable images per batch to prevent OOM.
        Returns:
            - list[InferenceResult]: A list containing the prediction outcomes, confidence scores and status flags
        """
        
        # OOM GUARD 
        if len(image_paths) > max_batch:
            logger.warning(f"Batch limit has been exceeded. Truncating from {len(image_paths)} to {max_batch}")
         
        results = []
        valid_batch = []
        
        def _process_single(path):
            path_obj = Path(path)
            try:
                return path_obj, self._preprocess(path_obj)[0], None
            except Exception as e:
                return path_obj, None, str(e)
            
        with concurrent.futures.ThreadPoolExecutor(max_workers = max_workers) as executor:
            for path_obj, tensor, error in executor.map(_process_single, image_paths):
                if error:
                    results.append(InferenceResult(image = str(path_obj), status= "FAILED", error = error))
                else:
                    valid_batch.append((path_obj, tensor))
        
        if not valid_batch:
            return results
        
        valid_paths, valid_tensors = zip(*valid_batch)
        
        try:
            batch_data = np.stack(valid_tensors, axis = 0)
            outputs = self.session.run(None, {self.input_name: batch_data})
            
            for path_obj, out in zip(valid_paths, outputs[0]):
                conf = float(out[0]) if np.ndim(out) > 0 else float(out)
                results.append(InferenceResult(
                    image = str(path_obj),
                    confidence = round(conf, 4),
                    detected = conf >= self.threshold
                ))
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            results.extend([InferenceResult(image=str(p), status= InferenceStatus.FAILED, error = str(e)) for p in valid_paths])
            
        return results
                