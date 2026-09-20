import io

import imagehash
from PIL import Image

from src.logger import logger


def generate_phash(image_bytes: bytes) -> str | None:
    """
    Produces Perceptual Hash for the image. phash for better image understanding
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return str(imagehash.phash(image))
    except Exception as e:
        logger.warning(f"Can't generate perceptual hash: {e}")
        return None
