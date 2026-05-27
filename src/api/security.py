from fastapi import HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from src.config import settings
from src.logger import logger

api_key_header = APIKeyHeader(name="X-API-Key", auto_error = True)

def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Verifies the incoming API key against the environment settings
    
    Args:
        - api_key(str):
    
    Returns:
        - str: the validated API key
    
    Raises:
        - HTTPException: If the API key is missing or invalid (401 Unauthorized) 
    """
    if api_key != settings.API_KEY:
        logger.warning("Unauthorized API access attempt")
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Invalid API Key"
        )
    return api_key