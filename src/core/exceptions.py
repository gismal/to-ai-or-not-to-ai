class AppError(Exception):
    """
    Base class for all application level errors
    Added for exception hierarchy
    """


class InvalidImageFormatError(AppError):
    """
    Exception raised when the uploaded file format is not supported.
    API handles this by returning a 400 Bad Request response.
    """


class ModelInferenceError(AppError):
    """
    Exception raised when the ONNX predictor fails during inference.
    API handles this by returning a 500 Internal Server Error response.
    """


class DatabaseError(AppError):
    """
    Exception raised when a database operation fails.
    API handles this by returning a 500 Internal Server Error response.
    """
