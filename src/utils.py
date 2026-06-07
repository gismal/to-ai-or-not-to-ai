import io
from PIL import Image
import imagehash

def generate_phash(image_bytes: bytes) -> str:
    """
    Produces Perceptual Hash for the image. phash for better image understanding 
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return str(imagehash.phash(image))