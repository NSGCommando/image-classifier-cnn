from pathlib import Path
from enum import Enum
from typing import TypedDict
class Paths(Enum):
    ModelsPath = Path(__file__).parents[1].resolve()/"models"
    ResultsPath = Path(__file__).parents[1].resolve()/"results"
    RootPath = Path(__file__).parents[2].resolve()

class ImageData(TypedDict):
    filename: str
    content: bytes

VALID_EXTENSIONS = {".webp", ".jpg", ".jpeg", ".png"}