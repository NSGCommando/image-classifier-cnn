from pathlib import Path
from enum import Enum
from typing import TypedDict
class Paths(Enum):
    ModelsPath = Path(__file__).parents[1].resolve()/"models"
    ResultsPath = Path(__file__).parents[1].resolve()/"results"

class ImageData(TypedDict):
    filename: str
    content: bytes
    
class PortAddresses(Enum):
    http_dispatcher = f"http://localhost:8000"
    http_inferer = f"http://localhost:8001"

VALID_EXTENSIONS = {".webp", ".jpg", ".jpeg", ".png"}