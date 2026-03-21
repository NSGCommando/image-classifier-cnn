import keras.ops as ops
from src.utils.data_loader import DataLoader

new_dataloader =  DataLoader()
# names of classes from MNIST
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def infer_one(image_data, model):
    """
    Inference function to return predicted class and confidence.
    Returns a Dict of format: {"Predicted_class":prediction,"Confidence":confidence}.
    """
    if not isinstance(image_data,(str,bytes)):
        raise ValueError(f"The Image data is unsupported: {type(image_data)}")
    img = new_dataloader.preprocess_data(image_data)
    predictions = model.predict(x=img)
    pred_tensor = ops.argmax(predictions, axis=1)[0]
    conf_tensor = ops.max(predictions)

    predicted_class = int(ops.convert_to_numpy(pred_tensor))#type: ignore
    confidence = round(float(ops.convert_to_numpy(conf_tensor)),4)#type: ignore

    return {
        "predicted_class": CLASS_NAMES[predicted_class],
        "confidence": confidence
    }