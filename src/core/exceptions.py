class InvalidImageFormatError(Exception):
    """
    Exception raised when the uploaded file format is not supported.
    API handles this by returning a 400 Bad Request response.
    """
    pass

class ModelInferenceError(Exception):
    """
    Exception raised when the ONNX predictor fails during inference.
    API handles this by returning a 500 Internal Server Error response.
    """
    pass