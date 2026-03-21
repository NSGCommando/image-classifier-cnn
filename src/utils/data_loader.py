from src.schemas.data_strategies import bytes_loader_strategy, str_loader_strategy
from keras.preprocessing.image import img_to_array
from keras.ops import mean,expand_dims

STRATEGIES = {
        str:str_loader_strategy,
        bytes:bytes_loader_strategy
    }

class DataLoader():
    """Class for loading and preprocessing images during Inference.\n
    Default Args:\n 
    `target_size`=(28,28),\n
    `colour_scheme`="L" (Greyscale),\n
    `resampler`="bilinear".
    """
    def __init__(self,target_size=(28,28),colour_scheme:str="L",resampler="bilinear"):
        self.target_size = target_size
        self.colour_scheme = colour_scheme
        self.resampler = resampler
        
    def _load_data(self,image_data:str|bytes):
        """Internal method of DataLoader class to handle different types of data(image paths vs byte-streams).
        Returns resized and resampled data array"""
        strategy = STRATEGIES.get(type(image_data))
        if not strategy:
            raise ValueError(f"Unsupported data type: {type(image_data)}")
        return strategy(image_data,self.target_size,self.colour_scheme,self.resampler)

    def preprocess_data(self,image_data:str|bytes):
        """Method of DataLoader class to load and preprocess provided image source(path or bytes).
        Returns processed tensor image array"""
        loaded_data = self._load_data(image_data)
        img_array = img_to_array(loaded_data)/255.0
        if mean(img_array) > 0.5: # flip image luminance if it's mostly white, assuming mostly white means a white background
            img_array = 1.0 - img_array
        img_array = expand_dims(img_array, axis=0)
        return img_array

