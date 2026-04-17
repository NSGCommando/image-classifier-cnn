from src.utils.data_loader import DataLoader

new_dataloader =  DataLoader()
# names of classes from MNIST
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def infer_one(image_data, onnx_session):
    """
    Inference function to return predicted class and confidence.
    Runs via loading an ONNX runtime session.
    Returns a Dict of format: {"Predicted_class":prediction,"Confidence":confidence}.
    """
    if not isinstance(image_data,(str,bytes)):
        raise ValueError(f"The Image data is unsupported: {type(image_data)}")
    img = new_dataloader.preprocess_data(image_data)
    onnx_outputs = onnx_session.run(None,{"input_layer": img})
    predictions = onnx_outputs[0] # ONNX runtime will return a LIST of array outputs, we want the 0th array which has the probabilities
    class_pred_val = int(predictions.argmax(axis=1)[0])
    conf_val = round(float(predictions.max()),4)

    return {
        "predicted_class": CLASS_NAMES[class_pred_val],
        "confidence": conf_val
    }