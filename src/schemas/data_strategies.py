import io
from typing import Tuple
from PIL import Image
RESAMPLER_DICT = {
    "nearest":Image.Resampling.NEAREST,
    "bilinear":Image.Resampling.BILINEAR,
    "box":Image.Resampling.BOX,
    "bicubic":Image.Resampling.BICUBIC,
    "hamming":Image.Resampling.HAMMING,
    "lanczos":Image.Resampling.LANCZOS,
}

def bytes_loader_strategy(img_data:bytes,resize_shape:Tuple,colour_scheme:str,resampler:str):
    return Image.open(io.BytesIO(img_data)).convert(colour_scheme).resize(resize_shape,resample=RESAMPLER_DICT.get(resampler))

def str_loader_strategy(img_path:str,resize_shape:Tuple,colour_scheme:str,resampler:str):
    return Image.open(img_path).convert(colour_scheme).resize(resize_shape,resample=RESAMPLER_DICT.get(resampler))

